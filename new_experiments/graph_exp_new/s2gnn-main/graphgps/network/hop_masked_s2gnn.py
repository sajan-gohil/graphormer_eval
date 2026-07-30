"""Hop-masked S^2GNN network.

Replaces the spatial GNN layers in S^2GNN with a hop-masked transformer while
keeping the spectral layers unchanged.  Each combined layer pairs one
``HopMaskedTransformerLayer`` (spatial replacement) with one
``FeatureBatchSpectralLayer`` (spectral, from s2gnn), mirroring the
``BatchS2GNNGNNLayer`` aggregation pattern.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.models.gnn import FeatureEncoder
from torch_geometric.utils import to_dense_batch
import torch_geometric.graphgym.register as register

from graphgps.layer.hop_masked_transformer_layer import (
    build_head_hop_sets,
    HopMaskedTransformerLayer,
)
from graphgps.layer.s2_spectral import FeatureBatchSpectralLayer


# ---------------------------------------------------------------------------
# Combined layer: hop-masked transformer (spatial) + spectral.
# ---------------------------------------------------------------------------
class BatchHopMaskedS2GNNLayer(nn.Module):
    """Combined [hop-masked transformer + spectral] layer.

    Analogous to ``BatchS2GNNGNNLayer`` in ``s2gnn.py`` but with a hop-masked
    transformer replacing the spatial GNN.  Both branches receive the same
    ``batch.x`` input; their outputs are summed and optionally combined with a
    residual connection and normalisation.

    Parameters
    ----------
    transformer_layers : nn.ModuleList
        One or more ``HopMaskedTransformerLayer`` instances that form the
        spatial replacement (usually just 1).
    spec_layer : FeatureBatchSpectralLayer
        The spectral filtering layer (unchanged from s2gnn).
    head_hop_sets : list
        Per-head hop-set assignment (output of ``build_head_hop_sets``).
    max_hops : int
        K — distance-mask depth.
    with_node_residual : bool
        Add a skip connection from the layer input.
    norm : bool
        Apply ``1/sqrt(norm_factor)`` scaling after aggregation.
    """

    def __init__(
        self,
        transformer_layers: nn.ModuleList,
        spec_layer: nn.Module,
        head_hop_sets: List[Optional[List[int]]],
        max_hops: int,
        with_node_residual: bool = True,
        norm: bool = True,
    ):
        super().__init__()
        self.transformer_layers = transformer_layers
        self.spec_layer = spec_layer
        self.head_hop_sets = head_hop_sets
        self.max_hops = max_hops
        self.num_heads = len(head_hop_sets)
        self.with_node_residual = with_node_residual
        self.norm_factor = 1 + norm * (with_node_residual + 1)  # +1 for sum

    # ----- helpers ----------------------------------------------------------
    def _build_per_head_mask(
        self,
        dist_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return (B, H, N, N) bool — True where head h is allowed to attend.

        For an unrestricted head (``hop_set is None``) all entries are True.
        """
        B, K_runtime, N, _ = dist_masks.shape
        H = self.num_heads
        out = dist_masks.new_zeros(B, H, N, N, dtype=torch.bool)
        for h, hop_set in enumerate(self.head_hop_sets):
            if hop_set is None:
                out[:, h] = True
                continue
            idx = [k for k in hop_set if k < K_runtime]
            if not idx:
                continue
            stacked = dist_masks[:, idx].bool().any(dim=1)  # (B, N, N)
            out[:, h] = stacked
        return out

    def _pad_dist_masks(
        self,
        batch,
        N_max: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Pad per-graph dist_mask arrays into (B, K, N_max, N_max)."""
        # batch.dist_mask is a list (length B) of numpy arrays, each (K_i, N_i, N_i)
        dist_mask_list = batch.dist_mask
        B = batch.num_graphs
        K = self.max_hops

        dm_padded = torch.zeros(
            B, K, N_max, N_max, dtype=torch.float32, device=device,
        )

        # Recover per-graph indices from the batched graph.
        # batch.ptr gives [0, n0, n0+n1, ...] for graph boundaries.
        ptr = batch.ptr  # (B+1,)
        for i in range(B):
            dm_i = dist_mask_list[i]
            if isinstance(dm_i, np.ndarray):
                dm_i = torch.from_numpy(dm_i.astype(np.float32))
            dm_i = dm_i.to(device)
            K_i, N_i, _ = dm_i.shape
            K_use = min(K_i, K)
            dm_padded[i, :K_use, :N_i, :N_i] = dm_i[:K_use]

        return dm_padded

    def _log_metrics(self, t_layer, dist_masks, spat_out, spec_out, node_mask):
        import logging
        import random
        # Only log 5% of the time during eval to avoid spam
        if self.training or random.random() > 0.05:
            return
            
        if not hasattr(t_layer.attn, '_last_attn') or not hasattr(t_layer.attn, '_last_out_pre_proj'):
            return

        attn = t_layer.attn._last_attn # (B, H, N, N)
        out_pre = t_layer.attn._last_out_pre_proj # (B, H, N, Dh)
        B, H, N, Dh = out_pre.shape

        # 1. Output Norm of each head (Hop Importance)
        out_pre_masked = out_pre * node_mask.view(B, 1, N, 1)
        head_norms = torch.linalg.norm(out_pre_masked, dim=-1).sum(dim=-1) / (node_mask.sum(dim=-1, keepdim=True).unsqueeze(1) + 1e-6)
        head_norms_mean = head_norms.mean(dim=0)
        logging.info(f"[HopMaskedS2GNN] Head Output Norms (Hop Importance): {['{:.4f}'.format(x) for x in head_norms_mean.cpu().tolist()]}")
        
        # 2. Self vs Neighbor Ratio within isolated heads
        self_mask = torch.eye(N, device=attn.device).unsqueeze(0).unsqueeze(0).bool()
        attn_masked = attn * node_mask.view(B, 1, N, 1) * node_mask.view(B, 1, 1, N)
        self_attn_sum = (attn_masked * self_mask).sum(dim=(2, 3))
        total_attn_sum = attn_masked.sum(dim=(2, 3)) + 1e-6
        self_ratio = (self_attn_sum / total_attn_sum).mean(dim=0)
        logging.info(f"[HopMaskedS2GNN] Self-Attention Ratio per head: {['{:.4f}'.format(x) for x in self_ratio.cpu().tolist()]}")
        
        # 3. Global Head's natural distribution across hops
        global_head_idx = None
        for i, h_set in enumerate(self.head_hop_sets):
            if h_set is None:
                global_head_idx = i
                break
                
        if global_head_idx is not None:
            global_attn = attn_masked[:, global_head_idx] # (B, N, N)
            hop_distributions = []
            K_runtime = dist_masks.shape[1]
            for k in range(K_runtime):
                hop_k_mask = dist_masks[:, k].bool() # (B, N, N)
                hop_k_attn = (global_attn * hop_k_mask).sum(dim=(1, 2))
                hop_distributions.append(hop_k_attn.mean().item())
            logging.info(f"[HopMaskedS2GNN] Global Head Attn Dist across hops 0 to {K_runtime-1}: {['{:.4f}'.format(x) for x in hop_distributions]}")

        # 4. spat_out vs spec_out norms
        spat_norm = torch.linalg.norm(spat_out, dim=-1).mean().item()
        spec_norm = torch.linalg.norm(spec_out, dim=-1).mean().item()
        spat_ratio = spat_norm / (spat_norm + spec_norm + 1e-6)
        logging.info(f"[HopMaskedS2GNN] Spat Out Norm: {spat_norm:.4f}, Spec Out Norm: {spec_norm:.4f}, Ratio Spat/(Spat+Spec): {spat_ratio:.4f}")

    # ----- forward ----------------------------------------------------------
    def forward(self, batch):
        x_in = batch.x  # (N_total, d)

        # --- Spectral branch (sparse, unchanged) ---------------------------
        spec_out = self.spec_layer(batch)  # returns features, not batch

        # --- Hop-masked transformer branch (dense) -------------------------
        dense_x, mask = to_dense_batch(x_in, batch.batch)  # (B, N_max, d)
        B, N_max, d = dense_x.shape

        # Build padded distance masks and per-head mask.
        dist_masks = self._pad_dist_masks(batch, N_max, dense_x.device)
        per_head_mask = self._build_per_head_mask(dist_masks)

        # Run transformer sub-layers.
        h = dense_x
        for t_layer in self.transformer_layers:
            h = t_layer(h, per_head_mask, mask)

        # Back to sparse.
        spat_out = h[mask]  # (N_total, d)

        # --- Aggregate -----------------------------------------------------
        # Log metrics before aggregation
        if len(self.transformer_layers) > 0:
            self._log_metrics(self.transformer_layers[0], dist_masks, spat_out, spec_out, mask)

        y = spec_out + spat_out  # sum aggregation (same as s2gnn default)
        if self.with_node_residual:
            y = y + x_in
        batch.x = (1.0 / math.sqrt(self.norm_factor)) * y

        return batch


# ---------------------------------------------------------------------------
# Top-level network.
# ---------------------------------------------------------------------------
@register_network('hop_masked_s2gnn')
class HopMaskedS2GNN(nn.Module):
    """S^2GNN with hop-masked transformer replacing spatial GNN layers.

    Architecture::

        FeatureEncoder  →  [HopMaskedTransformer + Spectral] × layers_mp  →  Head

    The ``FeatureEncoder`` and task head (``post_mp``) are identical to the
    original ``S2GNN``.  Only the spatial branch within each message-passing
    layer is replaced.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        hm = cfg.gnn.hop_masked
        hidden_dim = hm.hidden_dim
        num_heads = hm.num_heads
        num_hops = hm.num_hops
        num_tf_layers = hm.num_layers  # transformer sub-layers per combined layer
        ffn_ratio = hm.ffn_ratio

        # ---- Feature encoder (same as s2gnn) --------------------------------
        self.encoder = FeatureEncoder(cfg.gnn.dim_inner)
        dim_in = self.encoder.dim_in

        # ---- Optional pre-MP MLP -------------------------------------------
        if cfg.gnn.layers_pre_mp > 0:
            from graphgps.layer.s2_spectral import MLPMultiBatch
            self.pre_mp = MLPMultiBatch(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp,
            )
            dim_in = cfg.gnn.dim_inner

        # ---- Hop-set assignment (fixed for all layers) ----------------------
        self.head_hop_sets = build_head_hop_sets(
            max_hops=num_hops,
            num_heads=num_heads,
            mode="single",
            window=0,
            include_self=True,
            num_global_heads=1,
        )
        self.max_hops = num_hops

        # ---- Optional linear projections if dims mismatch ------------------
        self.proj_in = None
        if cfg.gnn.dim_inner != hidden_dim:
            self.proj_in = nn.Linear(cfg.gnn.dim_inner, hidden_dim)

        self.proj_out = None
        if hidden_dim != cfg.gnn.dim_inner:
            self.proj_out = nn.Linear(hidden_dim, cfg.gnn.dim_inner)

        # ---- Build combined layers -----------------------------------------
        dropout = cfg.gnn.dropout
        ffn_dim = hidden_dim * ffn_ratio

        spec_layer_skip = [
            i % cfg.gnn.layers_mp
            for i in cfg.gnn.spectral.layer_skip
            if i < cfg.gnn.layers_mp
        ]

        layers = []
        for i in range(cfg.gnn.layers_mp):
            is_first = (i == 0)
            dim_in_ = dim_in if is_first else cfg.gnn.dim_inner

            # For the spectral layer we need dim_in == dim_out (residual).
            # The spectral layer always works with cfg.gnn.dim_inner internally
            # when using combined mode.  Here, since we use hidden_dim for the
            # transformer, the spectral layer operates on cfg.gnn.dim_inner and
            # we project in/out around the transformer.
            spec_dim = cfg.gnn.dim_inner
            layer_cfg = new_layer_config(
                spec_dim, spec_dim, cfg.gnn.spectral.filter_layers,
                has_act=True, has_bias=True, cfg=cfg,
            )

            # Spectral layer (may be skipped at certain indices).
            if i not in spec_layer_skip:
                spec_layer = FeatureBatchSpectralLayer(
                    layer_cfg, is_first=is_first,
                    overwrite_x=False, with_node_residual=False,
                )
            else:
                # Identity: return batch.x unchanged.
                spec_layer = _IdentitySpectralLayer()

            # Transformer sub-layers.
            tf_layers = nn.ModuleList([
                HopMaskedTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    norm_type="layer",
                )
                for _ in range(num_tf_layers)
            ])

            layers.append(BatchHopMaskedS2GNNLayer(
                transformer_layers=tf_layers,
                spec_layer=spec_layer,
                head_hop_sets=self.head_hop_sets,
                max_hops=self.max_hops,
                with_node_residual=cfg.gnn.residual,
                norm=cfg.gnn.spectral.combine_with_spatial_norm
                     if cfg.gnn.spectral.combine_with_spatial is not None
                     else True,
            ))

        self.gnn_layers = nn.ModuleList(layers)

        # ---- Task head (same as s2gnn) -------------------------------------
        GNNHead = register.head_dict[cfg.gnn.head]
        is_first = cfg.gnn.layers_mp <= 0
        self.post_mp = GNNHead(cfg.gnn.dim_inner, dim_out, is_first)

    def forward(self, batch):
        # Set num_graphs if not available.
        if not hasattr(batch, 'num_graphs'):
            batch.num_graphs = 1

        # Encode features.
        batch = self.encoder(batch)

        # Optional pre-MP.
        if hasattr(self, 'pre_mp'):
            batch = self.pre_mp(batch)

        # Optional projection into transformer hidden dim.
        if self.proj_in is not None:
            batch.x = self.proj_in(batch.x)

        # Run combined [transformer + spectral] layers.
        for layer in self.gnn_layers:
            batch = layer(batch)

        # Optional projection back to dim_inner for the task head.
        if self.proj_out is not None:
            batch.x = self.proj_out(batch.x)

        # Task head.
        batch = self.post_mp(batch)
        return batch

    def last_layer_keys(self) -> list:
        """Returns parameter keys of last layer (for fine-tuning)."""
        proto = 'post_mp.layer_post_mp.Layer_'
        matches = [
            int(n.replace(proto, '').split('.')[0])
            for n, _ in self.named_parameters() if n.startswith(proto)
        ]
        if not matches:
            return []
        layer_idx = max(matches)
        return [
            f'model.{n}'
            for n, _ in self.named_parameters()
            if n.startswith(f'{proto}{layer_idx}')
        ]


class _IdentitySpectralLayer(nn.Module):
    """Placeholder for skipped spectral layers — returns ``batch.x``."""

    def forward(self, batch):
        return batch.x
