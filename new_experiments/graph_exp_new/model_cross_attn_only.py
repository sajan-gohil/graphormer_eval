"""
Idea A — cross-attention router only, no global self-attention.

Stack of L cross-attention "router" blocks, each of which:
    1. Generates fresh M proxies from the current N node embeddings via
       a configurable generator (score-based / GNN-pooling / PMA / coarsen).
    2. Runs a single N -> M -> N routing pass (Q=proxy, K=V=node, then
       proxy self-refine, then Q=node, K=V=proxy).
    3. Returns refined N node embeddings.

No global N-by-N self-attention is applied between layers — the only
global mixing is the M-proxy bottleneck. This is the end-to-end variant
that drops both the freezing of a pretrained transformer and the SA
layer that previously came after cross-attention.

Datasets:
    Peptides-func, Peptides-struct (graph-level)
    PascalVOC-SP                   (node-level)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool

from models import build_node_encoder
from generators import (
    ScoreBasedGenerator,
    GNNPoolingGenerator,
    PMAGenerator,
    GraphCoarseningGenerator,
    CrossAttentionRouter,
)


# Generator factory — kept here so the training script stays slim.
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


class CrossAttnOnlyBlock(nn.Module):
    """One end-to-end cross-attention router block (proxy gen + N->M->N)."""

    def __init__(
        self,
        hidden_dim: int,
        num_proxies: int,
        num_heads: int,
        dropout: float,
        generator_name: str,
        num_cross_layers: int,
        use_proxy_self_attn: bool,
        # Generator-specific:
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

        # Generators that operate on flat PyG node tensors (need edge_index/batch).
        self._gnn_based = generator_name in ("gnn_pooling", "graph_coarsening")

    def forward(self, dense_x, dense_mask, *, edge_index=None,
                batch_vec=None, edge_attr=None):
        """
        Args:
            dense_x:    (B, N, d) dense node embeddings.
            dense_mask: (B, N) bool real-node mask.
            edge_index/batch_vec/edge_attr: required only for GNN-based
                generators that consume flat-PyG tensors.
        Returns:
            refined_nodes: (B, N, d)
            aux_loss:      scalar tensor (or None) from the generator.
        """
        if self._gnn_based:
            flat = dense_x[dense_mask]  # (total_real_N, d)
            proxies, aux = self.generator(
                flat, mask=None, edge_index=edge_index,
                batch_vec=batch_vec, edge_attr=edge_attr,
            )
        else:
            proxies, aux = self.generator(dense_x, dense_mask)

        refined = self.router(dense_x, proxies, dense_mask)
        return refined, aux


class CrossAttnOnlyModel(nn.Module):
    """End-to-end model = encoder -> stacked cross-attn router blocks -> head.

    Args:
        hidden_dim:        node/encoder/router/proxy channel dim.
        num_proxies:       M, the proxy-bottleneck size (set << N).
        num_layers:        number of router blocks stacked.
        num_heads:         attention heads inside each router.
        num_cross_layers:  internal N->M->N iterations per block.
        use_proxy_self_attn: include the optional M-by-M self-attn refine
                             step inside each router (default True).
        generator_name:    which generator to use at every block.
        share_blocks:      reuse a single CrossAttnOnlyBlock across depths.
        output_dim:        task output channels.
        dropout:           shared dropout rate.
        graph_pool:        "sum" or "mean" for graph-level pooling.
        task_level:        "graph" or "node".
        aux_loss_decay:    geometric decay on per-block aux losses (to be
                           added to the main task loss by the trainer).
    """

    def __init__(
        self,
        hidden_dim: int = 96,
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
        # and BEFORE the head. 0 (default) = off → matches Idea A spec.
        final_sa_layers: int = 0,
        final_sa_heads: int | None = None,
        # Generator-specific (forwarded to _make_generator):
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
        self.task_level = task_level
        self.graph_pool = graph_pool
        self.aux_loss_decay = aux_loss_decay

        self.encoder = build_node_encoder(
            hidden_dim=hidden_dim,
            lap_pe_dim=lap_pe_dim,
            dataset_name=dataset_name,
        )

        block_kwargs = dict(
            hidden_dim=hidden_dim,
            num_proxies=num_proxies,
            num_heads=num_heads,
            dropout=dropout,
            generator_name=generator_name,
            num_cross_layers=num_cross_layers,
            use_proxy_self_attn=use_proxy_self_attn,
            gnn_layers=gnn_layers,
            gnn_type=gnn_type,
            pool_types=pool_types,
            decode_hidden=decode_hidden,
            decode_layers=decode_layers,
            idx_emb_dim=idx_emb_dim,
            decode_mode=decode_mode,
            pma_query_mode=pma_query_mode,
            coarsen_gnn_type=coarsen_gnn_type,
            coarsen_reg_weight=coarsen_reg_weight,
            score_hidden=score_hidden,
            score_layers=score_layers,
            score_heads=score_heads,
        )

        if share_blocks:
            shared = CrossAttnOnlyBlock(**block_kwargs)
            self.blocks = nn.ModuleList([shared] * num_layers)
        else:
            self.blocks = nn.ModuleList([
                CrossAttnOnlyBlock(**block_kwargs) for _ in range(num_layers)
            ])
        self._shared = share_blocks

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

        # Attribute set on every forward — exposed so the training loop
        # can add it to the main loss without threading aux through the
        # return signature of every model variant.
        self.last_aux_loss = 0.0

    def encode_dense(self, batch):
        lap_pe = getattr(batch, "lap_pe", None)
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr, lap_pe=lap_pe)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch):
        """
        Args:
            batch: PyG Batch.
        Returns:
            logits, node_emb_flat
        Side effect:
            self.last_aux_loss is set to the (decayed) sum of generator
            aux losses across blocks.
        """
        dense_x, dense_mask = self.encode_dense(batch)

        edge_index = batch.edge_index
        batch_vec = batch.batch
        edge_attr = getattr(batch, "edge_attr", None)

        total_aux = 0.0
        h = dense_x
        for i, block in enumerate(self.blocks):
            h, aux = block(
                h, dense_mask,
                edge_index=edge_index, batch_vec=batch_vec, edge_attr=edge_attr,
            )
            if aux is not None:
                decay = self.aux_loss_decay ** i
                total_aux = total_aux + aux * decay
        self.last_aux_loss = total_aux

        # Optional final global self-attention before extracting nodes.
        if self.final_sa is not None:
            # nn.TransformerEncoder: src_key_padding_mask True → ignore key.
            h = self.final_sa(h, src_key_padding_mask=~dense_mask)

        node_emb = h[dense_mask]

        if self.task_level == "node":
            logits = self.head(node_emb)
            return logits, node_emb

        # Graph-level pool
        B = dense_x.shape[0]
        batch_vec_dense = (
            torch.arange(B, device=dense_x.device)
            .unsqueeze(1)
            .expand_as(dense_mask)[dense_mask]
        )
        if self.graph_pool == "mean":
            pooled = global_mean_pool(node_emb, batch_vec_dense)
        else:
            pooled = global_add_pool(node_emb, batch_vec_dense)
        logits = self.head(pooled)
        return logits, node_emb
