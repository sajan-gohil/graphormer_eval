"""
k-hop aggregation feeding the cross-attention router stack.

Motivation
----------
``model_cross_attn_only.py`` has no explicit access to long-range topology.
The only multi-hop signal it carries comes from (a) stacked GNN-based proxy
generators — which oversmooth with depth — or (b) the proxy bottleneck
itself, which only sees the current node features and not any hop-distance
structure.

This model wires the ``KHopAttentionAggregator`` (one-shot, parallel over K
hops) into the front of the cross-attention router pipeline:

    encoder
      -> (optional) hidden_dim -> hop_dim projection
      -> KHopAttentionAggregator                       (B, N, K, hop_dim)
      -> fuse K dim back to hidden_dim                 (B, N, hidden_dim)
      -> stack of L cross-attention router blocks
      -> head

Each node's features going into the proxy generator already encode the
graph's local-to-distance-K structure, so the generator's job is just to
choose / refine M proxies — not to propagate information K hops via deep
message passing.

Two refresh modes:
    * ``khop_per_block=False`` (default): k-hop aggregation runs once,
      before the first router block. Cheaper. Subsequent blocks see only
      router outputs.
    * ``khop_per_block=True``: every router block re-aggregates over the
      hop levels using the *current* node features and the *fixed* hop
      masks. More expensive (L * K attention passes) but lets every block
      consult hop-distance topology with up-to-date features.

Datasets supported (via the ``data.Task`` abstraction):
    Peptides-func, Peptides-struct (graph-level)
    PascalVOC-SP                    (node-level)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool

from khop_attention import KHopAttentionAggregator, flatten_per_hop
from models import build_node_encoder
from generators import (
    ScoreBasedGenerator,
    GNNPoolingGenerator,
    PMAGenerator,
    GraphCoarseningGenerator,
    CrossAttentionRouter,
)


# ---------------------------------------------------------------------------
# Generator factory (mirrors model_cross_attn_only._make_generator).
# ---------------------------------------------------------------------------
def _make_generator(name: str, hidden_dim: int, num_proxies: int,
                    dropout: float = 0.2,
                    gnn_layers: int = 3,
                    gnn_type: str = "GINE",
                    pool_types=("mean",),
                    decode_hidden: int = 64,
                    decode_layers: int = 3,
                    idx_emb_dim: int = 64,
                    decode_mode: str = "shared",
                    pma_query_mode: str = "farthest_point",
                    coarsen_gnn_type: str = "GIN",
                    coarsen_reg_weight: float = 0.0,
                    score_hidden: int = 128,
                    score_layers: int = 1,
                    score_heads: int = 4):
    if name == "score_based":
        return ScoreBasedGenerator(
            num_proxies=num_proxies, input_dim=hidden_dim,
            hidden_dim=score_hidden, num_layers=score_layers,
            num_heads=score_heads, dropout=dropout,
        )
    if name == "gnn_pooling":
        return GNNPoolingGenerator(
            num_proxies=num_proxies, input_dim=hidden_dim,
            gnn_layers=gnn_layers, gnn_type=gnn_type,
            pool_types=tuple(pool_types), decode_hidden=decode_hidden,
            decode_layers=decode_layers, idx_emb_dim=idx_emb_dim,
            dropout=dropout, decode_mode=decode_mode,
        )
    if name == "pma":
        return PMAGenerator(
            num_proxies=num_proxies, input_dim=hidden_dim,
            num_heads=score_heads, num_layers=score_layers,
            dropout=dropout, query_mode=pma_query_mode,
        )
    if name == "graph_coarsening":
        return GraphCoarseningGenerator(
            num_proxies=num_proxies, input_dim=hidden_dim,
            gnn_layers=gnn_layers, gnn_type=coarsen_gnn_type,
            dropout=dropout, reg_weight=coarsen_reg_weight,
            num_refine_layers=score_layers, num_heads=score_heads,
        )
    raise ValueError(f"Unknown generator: {name}")


# ---------------------------------------------------------------------------
# K-fusion: collapse (B, N, K, hop_dim) -> (B, N, hidden_dim).
# ---------------------------------------------------------------------------
class _KFusion(nn.Module):
    """Configurable fusion of per-hop features back to a single per-node vector.

    Modes:
        "concat_proj":  flatten K -> Linear(K*hop_dim, hidden_dim)
        "sum_proj":     sum over K  -> Linear(hop_dim, hidden_dim)
        "mean_proj":    mean over K -> Linear(hop_dim, hidden_dim)
    """

    def __init__(self, hop_dim: int, max_hops: int, hidden_dim: int,
                 mode: str = "concat_proj", dropout: float = 0.0):
        super().__init__()
        self.mode = mode
        self.max_hops = max_hops
        self.hop_dim = hop_dim
        if mode == "concat_proj":
            in_dim = max_hops * hop_dim
        elif mode in ("sum_proj", "mean_proj"):
            in_dim = hop_dim
        else:
            raise ValueError(f"Unknown fuse mode: {mode}")
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, agg: torch.Tensor) -> torch.Tensor:
        # agg: (B, N, K, hop_dim) — possibly with K < max_hops.
        if self.mode == "concat_proj":
            B, N, K, d = agg.shape
            if K < self.max_hops:
                pad = agg.new_zeros(B, N, self.max_hops - K, d)
                agg = torch.cat([agg, pad], dim=2)
            x = flatten_per_hop(agg)  # (B, N, max_hops * hop_dim)
        elif self.mode == "sum_proj":
            x = agg.sum(dim=2)
        else:  # mean_proj
            x = agg.mean(dim=2)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Single block = (optional re-aggregate) -> generator -> router.
# ---------------------------------------------------------------------------
class _Block(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_proxies: int,
        num_heads: int,
        dropout: float,
        generator_name: str,
        num_cross_layers: int,
        use_proxy_self_attn: bool,
        # generator-specific (forwarded):
        gnn_layers: int = 3,
        gnn_type: str = "GINE",
        pool_types=("mean",),
        decode_hidden: int = 64,
        decode_layers: int = 3,
        idx_emb_dim: int = 64,
        decode_mode: str = "shared",
        pma_query_mode: str = "farthest_point",
        coarsen_gnn_type: str = "GIN",
        coarsen_reg_weight: float = 0.0,
        score_hidden: int = 128,
        score_layers: int = 1,
        score_heads: int = 4,
    ):
        super().__init__()
        self.generator_name = generator_name
        self.generator = _make_generator(
            generator_name, hidden_dim, num_proxies,
            dropout=dropout,
            gnn_layers=gnn_layers, gnn_type=gnn_type, pool_types=pool_types,
            decode_hidden=decode_hidden, decode_layers=decode_layers,
            idx_emb_dim=idx_emb_dim, decode_mode=decode_mode,
            pma_query_mode=pma_query_mode,
            coarsen_gnn_type=coarsen_gnn_type,
            coarsen_reg_weight=coarsen_reg_weight,
            score_hidden=score_hidden, score_layers=score_layers,
            score_heads=score_heads,
        )
        self.router = CrossAttentionRouter(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_cross_layers=num_cross_layers,
            dropout=dropout,
            use_proxy_self_attn=use_proxy_self_attn,
        )
        self._gnn_based = generator_name in ("gnn_pooling", "graph_coarsening")

    def forward(self, dense_x, dense_mask, *,
                edge_index=None, batch_vec=None, edge_attr=None):
        if self._gnn_based:
            flat = dense_x[dense_mask]
            proxies, aux = self.generator(
                flat, mask=None, edge_index=edge_index,
                batch_vec=batch_vec, edge_attr=edge_attr,
            )
        else:
            proxies, aux = self.generator(dense_x, dense_mask)
        refined = self.router(dense_x, proxies, dense_mask)
        return refined, aux


# ---------------------------------------------------------------------------
# Top-level model.
# ---------------------------------------------------------------------------
class KHopCrossAttnModel(nn.Module):
    """Encoder -> k-hop aggregation -> fuse -> stacked cross-attn routers.

    Args:
        hidden_dim:        node/router channel dim.
        hop_dim:           per-hop channel dim inside the k-hop aggregator
                           (often smaller than hidden_dim to keep K*hop_dim
                           manageable for the concat fuse).
        max_hops:          K — number of hop levels.
        fuse_mode:         "concat_proj" (default) | "sum_proj" | "mean_proj".
        khop_per_block:    if True, re-aggregate over hops at every block
                           using the current node features. If False, run
                           the aggregator once at the top.
        khop_residual_self: passed to KHopAttentionAggregator.
        khop_num_heads:    attention heads inside the aggregator.
        num_proxies:       M proxies per block.
        num_layers:        number of router blocks.
        num_heads:         attention heads inside the router.
        num_cross_layers:  internal N->M->N iterations per block.
        use_proxy_self_attn: include the M-by-M self-attn refine step.
        generator_name:    "score_based" / "gnn_pooling" / "pma" / "graph_coarsening".
        share_blocks:      reuse one block across depths.
        graph_pool / task_level: as in other models.
        aux_loss_decay:    geometric decay on per-block aux losses.
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        hop_dim: int = 16,
        max_hops: int = 40,
        fuse_mode: str = "concat_proj",
        khop_per_block: bool = False,
        khop_residual_self: bool = True,
        khop_num_heads: int = 4,
        num_proxies: int = 32,
        num_layers: int = 4,
        num_heads: int = 4,
        num_cross_layers: int = 2,
        use_proxy_self_attn: bool = True,
        generator_name: str = "score_based",
        share_blocks: bool = False,
        output_dim: int = 10,
        dropout: float = 0.2,
        graph_pool: str = "sum",
        task_level: str = "graph",
        dataset_name: str = "Peptides-func",
        lap_pe_dim: int = 0,
        aux_loss_decay: float = 1.0,
        # Optional global self-attention AFTER the cross-attn block stack
        # and BEFORE the head. 0 (default) = off.
        final_sa_layers: int = 0,
        final_sa_heads: int | None = None,
        # generator-specific:
        gnn_layers: int = 3,
        gnn_type: str = "GINE",
        pool_types=("mean",),
        decode_hidden: int = 64,
        decode_layers: int = 3,
        idx_emb_dim: int = 64,
        decode_mode: str = "shared",
        pma_query_mode: str = "farthest_point",
        coarsen_gnn_type: str = "GIN",
        coarsen_reg_weight: float = 0.0,
        score_hidden: int = 128,
        score_layers: int = 1,
        score_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hop_dim = hop_dim
        self.max_hops = max_hops
        self.task_level = task_level
        self.graph_pool = graph_pool
        self.aux_loss_decay = aux_loss_decay
        self.khop_per_block = khop_per_block

        self.encoder = build_node_encoder(
            hidden_dim=hidden_dim,
            lap_pe_dim=lap_pe_dim,
            dataset_name=dataset_name,
        )

        # Project hidden_dim -> hop_dim before each aggregation pass.
        if hop_dim != hidden_dim:
            self.pre_proj = nn.Sequential(
                nn.Linear(hidden_dim, hop_dim),
                nn.GELU(),
            )
        else:
            self.pre_proj = nn.Identity()

        # Heads must divide hop_dim — clamp down if needed.
        eff_heads = khop_num_heads
        while eff_heads > 1 and hop_dim % eff_heads != 0:
            eff_heads -= 1
        self.aggregator = KHopAttentionAggregator(
            hidden_dim=hop_dim,
            num_heads=eff_heads,
            dropout=dropout,
            add_hop_embedding=True,
            max_hops=max_hops,
            residual_self=khop_residual_self,
        )

        # Fuse (B,N,K,hop_dim) back to (B,N,hidden_dim) for the router pipeline.
        self.fuse = _KFusion(
            hop_dim=hop_dim, max_hops=max_hops,
            hidden_dim=hidden_dim, mode=fuse_mode, dropout=dropout,
        )

        # Stack of (gen + router) blocks.
        block_kwargs = dict(
            hidden_dim=hidden_dim,
            num_proxies=num_proxies,
            num_heads=num_heads,
            dropout=dropout,
            generator_name=generator_name,
            num_cross_layers=num_cross_layers,
            use_proxy_self_attn=use_proxy_self_attn,
            gnn_layers=gnn_layers, gnn_type=gnn_type, pool_types=pool_types,
            decode_hidden=decode_hidden, decode_layers=decode_layers,
            idx_emb_dim=idx_emb_dim, decode_mode=decode_mode,
            pma_query_mode=pma_query_mode,
            coarsen_gnn_type=coarsen_gnn_type,
            coarsen_reg_weight=coarsen_reg_weight,
            score_hidden=score_hidden, score_layers=score_layers,
            score_heads=score_heads,
        )
        if share_blocks:
            shared = _Block(**block_kwargs)
            self.blocks = nn.ModuleList([shared] * num_layers)
        else:
            self.blocks = nn.ModuleList(
                [_Block(**block_kwargs) for _ in range(num_layers)]
            )

        # Optional final global self-attention stack.
        if final_sa_layers > 0:
            sa_heads = final_sa_heads if final_sa_heads is not None else num_heads
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=sa_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.final_sa = nn.TransformerEncoder(
                enc_layer, num_layers=final_sa_layers
            )
        else:
            self.final_sa = None

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.last_aux_loss = 0.0

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def encode_dense(self, batch):
        lap_pe = getattr(batch, "lap_pe", None)
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr, lap_pe=lap_pe)
        return to_dense_batch(h, batch.batch)

    def _khop_pass(self, dense_x, dist_masks, node_mask):
        """One k-hop aggregation -> fused (B, N, hidden_dim)."""
        h_small = self.pre_proj(dense_x)                       # (B,N,hop_dim)
        K = min(dist_masks.shape[1], self.max_hops)
        agg = self.aggregator(h_small, dist_masks[:, :K], node_mask)
        return self.fuse(agg)                                  # (B,N,hidden_dim)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(self, batch, dist_masks, node_masks):
        dense_x, dense_mask = self.encode_dense(batch)
        nm = node_masks if node_masks is not None else dense_mask

        edge_index = batch.edge_index
        batch_vec = batch.batch
        edge_attr = getattr(batch, "edge_attr", None)

        # Initial k-hop aggregation pass (always run once).
        h = self._khop_pass(dense_x, dist_masks, nm)

        total_aux = 0.0
        for i, block in enumerate(self.blocks):
            if self.khop_per_block and i > 0:
                # Re-aggregate using current node features. Same hop masks
                # because the graph's shortest-path structure is fixed.
                h = self._khop_pass(h, dist_masks, nm)
            h, aux = block(
                h, nm,
                edge_index=edge_index, batch_vec=batch_vec, edge_attr=edge_attr,
            )
            if aux is not None:
                total_aux = total_aux + aux * (self.aux_loss_decay ** i)
        self.last_aux_loss = total_aux

        # Optional final global self-attention before extracting nodes.
        if self.final_sa is not None:
            # nn.TransformerEncoder: src_key_padding_mask True → ignore key.
            h = self.final_sa(h, src_key_padding_mask=~nm)

        node_emb = h[nm]

        if self.task_level == "node":
            logits = self.head(node_emb)
            return logits, node_emb

        B = dense_x.shape[0]
        batch_vec_dense = (
            torch.arange(B, device=dense_x.device)
            .unsqueeze(1)
            .expand_as(nm)[nm]
        )
        if self.graph_pool == "mean":
            pooled = global_mean_pool(node_emb, batch_vec_dense)
        else:
            pooled = global_add_pool(node_emb, batch_vec_dense)
        logits = self.head(pooled)
        return logits, node_emb
