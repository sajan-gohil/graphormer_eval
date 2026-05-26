"""
Idea C — k-hop concat baseline.

For each node v, build a structural fingerprint by attention-weighted
aggregation over each hop k=0..K-1, concatenate the per-hop slices into
a single (K * hop_dim)-vector per node, then map to predictions with an
MLP. No LRU, no transformer. This isolates how much of GRED's gain comes
from the per-hop aggregation versus the LRU mixing.

Pipeline:
    NodeEncoder
      -> (optional) shared dim projection from hidden_dim -> hop_dim
      -> KHopAttentionAggregator                             # (B, N, K, hop_dim)
      -> flatten over K                                      # (B, N, K*hop_dim)
      -> per-node MLP                                        # (B, N, hidden_head)
      -> task head (graph-level pool or per-node)            # logits

Datasets supported (via ``data.Task``):
    Peptides-func    (graph, multi-label, AP)
    Peptides-struct  (graph, regression, MAE)
    PascalVOC-SP     (node, multi-class, F1)

This module reuses ``models.build_node_encoder`` so Peptides keeps its
atom-categorical encoder and PascalVOC-SP gets the linear encoder
automatically based on the dataset registry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool

from khop_attention import KHopAttentionAggregator, flatten_per_hop
from models import build_node_encoder


class KHopConcatModel(nn.Module):
    """k-hop attention aggregation -> concat -> MLP -> task head.

    Args:
        hidden_dim:     node-encoder hidden dim. The aggregator runs at
                        ``hop_dim`` (often smaller — e.g. 16 — to keep
                        K*hop_dim manageable). If ``hop_dim != hidden_dim``
                        a learnable Linear projects hidden_dim -> hop_dim
                        before aggregation.
        hop_dim:        per-hop channel count.
        max_hops:       K — number of hop levels to aggregate over.
        num_heads:      attention heads inside the aggregator.
        head_hidden:    width of the MLP head on top of the concat tensor.
        head_layers:    depth of the MLP head (>=1).
        output_dim:     number of task output channels.
        dropout:        dropout used in head and aggregator.
        lap_pe_dim:     Laplacian PE dimension (0 disables).
        dataset_name:   passed through to build_node_encoder.
        task_level:     "graph" pools over real nodes; "node" applies the
                        head per real node and skips pooling.
        graph_pool:     "sum" (LRGB-Peptides default) or "mean".
        residual_self_in_concat: if True, hop-0 aggregation passes through
                        the un-attended self-features (encoder output
                        projected to hop_dim). If False, hop-0 still goes
                        through Q/K/V (which is fine but not informative
                        since it's u==v only — see KHopAttentionAggregator).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        hop_dim: int = 16,
        max_hops: int = 40,
        num_heads: int = 4,
        head_hidden: int = 256,
        head_layers: int = 2,
        output_dim: int = 10,
        dropout: float = 0.2,
        lap_pe_dim: int = 0,
        dataset_name: str = "Peptides-func",
        task_level: str = "graph",
        graph_pool: str = "sum",
        residual_self_in_concat: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hop_dim = hop_dim
        self.max_hops = max_hops
        self.dataset_name = dataset_name
        self.task_level = task_level
        self.graph_pool = graph_pool

        self.encoder = build_node_encoder(
            hidden_dim=hidden_dim,
            lap_pe_dim=lap_pe_dim,
            dataset_name=dataset_name,
        )

        # If hop_dim != hidden_dim, shrink before aggregation so the
        # concat tensor stays a manageable size (K * hop_dim).
        if hop_dim != hidden_dim:
            self.pre_proj = nn.Sequential(
                nn.Linear(hidden_dim, hop_dim),
                nn.GELU(),
            )
        else:
            self.pre_proj = nn.Identity()

        # Aggregator runs at hop_dim with hop embeddings on each hop slot.
        # Heads must divide hop_dim; clamp num_heads if user passed too many.
        eff_heads = num_heads
        while eff_heads > 1 and hop_dim % eff_heads != 0:
            eff_heads -= 1
        self.aggregator = KHopAttentionAggregator(
            hidden_dim=hop_dim,
            num_heads=eff_heads,
            dropout=dropout,
            add_hop_embedding=True,
            max_hops=max_hops,
            residual_self=residual_self_in_concat,
        )

        concat_dim = max_hops * hop_dim
        # MLP head — applied per node either way (graph-level only differs
        # in whether we pool before the final classifier).
        layers: list[nn.Module] = []
        in_dim = concat_dim
        for i in range(head_layers):
            out_dim = head_hidden if i < head_layers - 1 else head_hidden
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        # Final classifier
        layers.append(nn.Linear(in_dim, output_dim))
        self.head = nn.Sequential(*layers)

    # ----------------------------------------------------------------
    # Encoders
    # ----------------------------------------------------------------
    def encode_nodes(self, batch):
        lap_pe = getattr(batch, "lap_pe", None)
        return self.encoder(
            batch.x, batch.edge_index, batch.edge_attr, lap_pe=lap_pe
        )

    def encode_dense(self, batch):
        h = self.encode_nodes(batch)
        return to_dense_batch(h, batch.batch)

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------
    def forward(self, batch, dist_masks, node_masks):
        """
        Args:
            batch:       PyG Batch.
            dist_masks:  (B, K, max_N, max_N) float — k-hop boolean masks.
            node_masks:  (B, max_N) bool — True for real nodes.
        Returns:
            logits:           graph-level (B, output_dim) or node-level
                              (total_real_nodes, output_dim).
            node_embeddings:  (total_real_nodes, hop_dim*max_hops) flat
                              concat features (mostly diagnostic).
        """
        dense_x, dense_mask = self.encode_dense(batch)  # (B, max_N, d)
        # If the loader supplied a node_masks tensor, use it; otherwise fall
        # back to the dense_mask returned by to_dense_batch.
        nm = node_masks if node_masks is not None else dense_mask

        # Project to hop_dim before aggregation.
        h_small = self.pre_proj(dense_x)  # (B, max_N, hop_dim)

        # k-hop attention-weighted aggregation: (B, max_N, K, hop_dim).
        # Truncate dist_masks to max_hops if a longer K was preprocessed.
        K = min(dist_masks.shape[1], self.max_hops)
        agg = self.aggregator(h_small, dist_masks[:, :K], nm)
        if K < self.max_hops:
            # Pad on the K axis so the concat dimension always matches
            # the head's input width (max_hops * hop_dim).
            pad_shape = list(agg.shape)
            pad_shape[2] = self.max_hops - K
            agg = torch.cat([agg, agg.new_zeros(*pad_shape)], dim=2)

        concat = flatten_per_hop(agg)  # (B, max_N, max_hops * hop_dim)

        # Pull out only real-node rows for the head.
        flat = concat[nm]  # (total_real_N, K*hop_dim)

        if self.task_level == "node":
            # Per-node prediction (PascalVOC-SP).
            logits = self.head(flat)
            return logits, flat

        # Graph-level: pool first, then classify. We pool the concat tensor
        # itself rather than running the head per-node — cheaper and the
        # standard LRGB recipe.
        B = dense_x.shape[0]
        batch_vec = (
            torch.arange(B, device=dense_x.device)
            .unsqueeze(1)
            .expand_as(nm)[nm]
        )
        if self.graph_pool == "mean":
            pooled = global_mean_pool(flat, batch_vec)
        else:
            pooled = global_add_pool(flat, batch_vec)
        logits = self.head(pooled)
        return logits, flat
