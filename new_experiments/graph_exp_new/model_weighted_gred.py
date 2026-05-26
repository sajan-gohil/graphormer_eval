"""
Weighted GRED — replace GRED's uniform per-hop sum with attention-weighted
aggregation, then keep the LRU mixing over hops.

Each WeightedGREDLayer does:
    1. KHopAttentionAggregator over (B, K, N, N) hop masks  -> (B, N, K, d)
       (shared Q/K/V across hops; per-hop softmax restricted to hop-k pairs;
       learnable per-hop embedding added to outputs)
    2. Per-hop MLP (LayerNorm + GELU + residual on the aggregated features)
    3. Diagonal LRU scan over the K hop axis (far -> near), reusing
       ``models.DiagonalLRU`` unchanged
    4. Residual on input node features

Stacking N such layers updates h iteratively, with the same aggregator
parameters at each layer (or independent — controlled by ``share_layers``).

This addresses GRED's main weakness: per-hop ``sum(h_u)`` discards which
nodes at hop k are informative. Attention weighting lets the model down-
weight noise within each hop while the LRU still does the multi-scale
mixing along hops.

Datasets supported (graph and node level via ``data.Task``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool

from khop_attention import KHopAttentionAggregator
from models import DiagonalLRU, build_node_encoder


class WeightedGREDLayer(nn.Module):
    """One Weighted-GRED block: attention-weighted hop aggregation + LRU.

    Args:
        hidden_dim:   model channel count d.
        state_dim:    LRU complex state dimension.
        num_heads:    heads inside the per-hop aggregator.
        max_hops:     K — must match dist_masks at runtime (or larger).
        expand:       FFN expansion in the per-hop refinement MLP.
        dropout:      dropout shared across submodules.
        r_min/r_max/max_phase: LRU eigenvalue init.
        act:          GLU variant for DiagonalLRU.
        residual_self: whether the aggregator's hop-0 row is forced to a
                       clean copy of the input (recommended).
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        num_heads: int = 4,
        max_hops: int = 40,
        expand: int = 1,
        dropout: float = 0.2,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
        act: str = "full-glu",
        residual_self: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_hops = max_hops

        # Heads must divide hidden_dim; clamp down if necessary.
        eff_heads = num_heads
        while eff_heads > 1 and hidden_dim % eff_heads != 0:
            eff_heads -= 1
        self.aggregator = KHopAttentionAggregator(
            hidden_dim=hidden_dim,
            num_heads=eff_heads,
            dropout=dropout,
            add_hop_embedding=True,
            max_hops=max_hops,
            residual_self=residual_self,
        )

        # Per-hop MLP refinement, applied to each (B, N, K, d) slice.
        self.refine_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, expand * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # LRU over the hop axis (sequence of length K).
        self.lru = DiagonalLRU(
            input_dim=hidden_dim,
            state_dim=state_dim,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
            dropout=dropout,
            act=act,
        )

    def forward(
        self,
        h: torch.Tensor,
        dist_masks: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h:           (B, N, d) dense node features.
            dist_masks:  (B, K, N, N) float — k-hop boolean masks.
            node_mask:   (B, N) bool — True for real nodes.
        Returns:
            (B, N, d) updated node features.
        """
        B, N, d = h.shape
        K = dist_masks.shape[1]

        # 1. Attention-weighted per-hop aggregation: (B, N, K, d)
        agg = self.aggregator(h, dist_masks, node_mask)

        # 2. Per-hop refinement MLP with residual on the aggregated features.
        #    Reshape to (B*K*N, d) so the MLP applies element-wise.
        agg_flat = agg.reshape(B * N * K, d)
        agg_flat = agg_flat + self.refine_mlp(agg_flat)
        agg = agg_flat.reshape(B, N, K, d)

        # 3. LRU over the hop axis: order far -> near.
        #    DiagonalLRU expects (BN, K, d) with the *last* element being
        #    the near hop (index 0 in our K axis). So we flip K -> reverse,
        #    matching ``GREDLayer.forward`` semantics in models.py.
        hop_seq = agg.flip(dims=[2])              # (B, N, K, d) — far -> near
        hop_seq = hop_seq.reshape(B * N, K, d)
        h_out_flat = self.lru(hop_seq)            # (B*N, d)
        h_out = h_out_flat.reshape(B, N, d)

        # 4. Mask padded nodes and residual-add the input.
        if node_mask is not None:
            h_out = h_out * node_mask.unsqueeze(-1).float()
        return h + h_out


class WeightedGREDModel(nn.Module):
    """Stack of WeightedGREDLayers + task head.

    Args:
        hidden_dim:    model dim.
        state_dim:     LRU state dim.
        num_layers:    number of WeightedGRED blocks.
        max_hops:      K — distance-mask depth at runtime.
        num_heads:     heads inside the aggregator.
        share_layers:  if True, reuse a single WeightedGREDLayer for all
                       depths (no extra parameters with depth). If False,
                       each layer has independent parameters.
        output_dim:    task output channels.
        dropout:       dropout used in submodules and the head.
        graph_pool:    "sum" or "mean" (graph-level tasks).
        task_level:    "graph" or "node".
        dataset_name:  drives the node encoder choice.
        lap_pe_dim:    Laplacian PE dim (0 disables).
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        state_dim: int = 96,
        num_layers: int = 4,
        max_hops: int = 40,
        num_heads: int = 4,
        share_layers: bool = False,
        expand: int = 1,
        dropout: float = 0.2,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = 6.28,
        act: str = "full-glu",
        output_dim: int = 10,
        graph_pool: str = "sum",
        task_level: str = "graph",
        dataset_name: str = "Peptides-func",
        lap_pe_dim: int = 0,
        residual_self: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_hops = max_hops
        self.task_level = task_level
        self.graph_pool = graph_pool

        self.encoder = build_node_encoder(
            hidden_dim=hidden_dim,
            lap_pe_dim=lap_pe_dim,
            dataset_name=dataset_name,
        )

        if share_layers:
            shared = WeightedGREDLayer(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_heads=num_heads,
                max_hops=max_hops,
                expand=expand,
                dropout=dropout,
                r_min=r_min,
                r_max=r_max,
                max_phase=max_phase,
                act=act,
                residual_self=residual_self,
            )
            self.layers = nn.ModuleList([shared] * num_layers)
            self._shared = True
        else:
            self.layers = nn.ModuleList([
                WeightedGREDLayer(
                    hidden_dim=hidden_dim,
                    state_dim=state_dim,
                    num_heads=num_heads,
                    max_hops=max_hops,
                    expand=expand,
                    dropout=dropout,
                    r_min=r_min,
                    r_max=r_max,
                    max_phase=max_phase,
                    act=act,
                    residual_self=residual_self,
                )
                for _ in range(num_layers)
            ])
            self._shared = False

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_dense(self, batch):
        lap_pe = getattr(batch, "lap_pe", None)
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr, lap_pe=lap_pe)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, dist_masks, node_masks):
        """
        Args:
            batch:       PyG Batch.
            dist_masks:  (B, K, max_N, max_N) float k-hop masks.
            node_masks:  (B, max_N) bool real-node indicator.
        Returns:
            logits, node_emb_flat
        """
        dense_x, dense_mask = self.encode_dense(batch)
        nm = node_masks if node_masks is not None else dense_mask

        # Truncate / pass-through dist_masks to the layer.
        K = min(dist_masks.shape[1], self.max_hops)
        dm = dist_masks[:, :K]

        h = dense_x
        for layer in self.layers:
            h = layer(h, dm, nm)

        node_emb = h[nm]  # (total_real_N, d)

        if self.task_level == "node":
            logits = self.head(node_emb)
            return logits, node_emb

        # Graph-level pool over real nodes.
        B = dense_x.shape[0]
        batch_vec = (
            torch.arange(B, device=dense_x.device).unsqueeze(1).expand_as(nm)[nm]
        )
        if self.graph_pool == "mean":
            pooled = global_mean_pool(node_emb, batch_vec)
        else:
            pooled = global_add_pool(node_emb, batch_vec)
        logits = self.head(pooled)
        return logits, node_emb
