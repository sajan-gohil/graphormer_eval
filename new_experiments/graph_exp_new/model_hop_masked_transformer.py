"""
Hop-masked multi-head transformer.

Motivated by the empirical finding that the simple k-hop concat baseline
(model_khop_concat.py) reaches SOTA on LRGB while the routing / recurrent
variants underperform it. The common thread in the failures is that they
collapse the hop axis somewhere (LRU mixes it; cross-attention loses it
through the M-proxy bottleneck; GNN-pooling smears it through stacked
message passing). The concat baseline succeeds because per-hop identity
is preserved end-to-end — each hop has its own slot in the final per-node
feature and the MLP head can learn hop-specific transformations.

This model preserves per-hop identity in a transformer-native way: each
attention head's softmax is restricted to a chosen hop range, so the head
specialises at exchanging information among nodes at that distance scale.
A standard pre-norm transformer encoder stack is composed on top, with the
MHA output projection and FFN acting as learned mixers across hop-
specialised heads.

Pipeline:
    encoder
      -> stack of L HopMaskedTransformerLayers
      -> head (graph-level pool + classifier, or per-node)

Each layer (pre-norm encoder):
    norm -> hop-masked multi-head self-attention -> residual
    norm -> FFN -> residual

Each head h has a hop-set S_h that always contains 0 (self), so the
attention row is never empty. For node v, head h attends only to nodes u
with d(v, u) in S_h.

Datasets: Peptides-func, Peptides-struct, PascalVOC-SP.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool

from models import build_node_encoder


_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Head-to-hop assignment.
# ---------------------------------------------------------------------------
def _split_contiguous(lst: Sequence[int], k: int) -> List[List[int]]:
    """Split ``lst`` into ``k`` roughly-equal contiguous chunks."""
    n = len(lst)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out, i = [], 0
    for s in sizes:
        out.append(list(lst[i:i + s]))
        i += s
    return out


def build_head_hop_sets(
    max_hops: int,
    num_heads: int,
    mode: str = "contiguous",
    window: int = 1,
    include_self: bool = True,
    num_global_heads: int = 0,
) -> List[Optional[List[int]]]:
    """Assign a hop set to each attention head.

    Args:
        max_hops:           K — total number of hop levels available.
        num_heads:          H — total attention heads.
        mode:               "contiguous" | "window" | "single" | "interleaved".
        window:             half-window for "window" mode; stride for
                            "interleaved" mode.
        include_self:       if True, hop 0 is added to every restricted head
                            so the attention row always has at least one
                            valid key (no empty-softmax NaNs).
        num_global_heads:   the last N heads are unrestricted (hop_set=None
                            → free global attention).

    Returns a list of length ``num_heads``; each entry is either a list of
    int hop indices the head is allowed to attend to, or ``None`` for an
    unrestricted head.
    """
    K = max_hops
    H_restricted = num_heads - num_global_heads
    if H_restricted < 0:
        raise ValueError(
            f"num_global_heads={num_global_heads} > num_heads={num_heads}"
        )

    if H_restricted == 0:
        return [None] * num_heads

    if mode == "contiguous":
        # Partition [1, K-1] into H_restricted contiguous chunks.
        hops = list(range(1, K))
        sets = _split_contiguous(hops, H_restricted)
    elif mode == "window":
        # Evenly-spaced centres across [1, K-1], each head covers
        # [centre - window, centre + window].
        if H_restricted == 1:
            centres = [(1 + K - 1) // 2]
        else:
            centres = [
                int(round(1 + i * (K - 2) / (H_restricted - 1)))
                for i in range(H_restricted)
            ]
        sets = [
            list(range(max(1, c - window), min(K, c + window + 1)))
            for c in centres
        ]
    elif mode == "single":
        if H_restricted != K - 1:
            raise ValueError(
                f"single mode requires num_heads-num_global_heads == K-1; "
                f"got {H_restricted} restricted heads vs K-1={K-1}"
            )
        sets = [[i + 1] for i in range(H_restricted)]
    elif mode == "interleaved":
        # Each head gets a centre k evenly spaced across [0, K-1].
        # The head attends to every hop at k ± window*n that stays in [0, K).
        if H_restricted == 1:
            centres = [0]
        else:
            centres = [
                int(round(i * (K - 1) / (H_restricted - 1)))
                for i in range(H_restricted)
            ]
        sets = []
        for c in centres:
            hops = {c}
            n = 1
            while True:
                added = False
                lo = c - window * n
                hi = c + window * n
                if lo >= 0:
                    hops.add(lo)
                    added = True
                if hi < K:
                    hops.add(hi)
                    added = True
                if not added:
                    break
                n += 1
            sets.append(sorted(hops))
    else:
        raise ValueError(f"Unknown hop assignment mode: {mode}")

    if include_self:
        sets = [sorted(set([0] + s)) for s in sets]

    return sets + [None] * num_global_heads


# ---------------------------------------------------------------------------
# Block-diagonal linear: per-head projection with no cross-head mixing.
# ---------------------------------------------------------------------------
class BlockDiagLinear(nn.Module):
    """Block-diagonal linear: each head's Dh-dim slice is projected
    independently. Equivalent to H parallel (Dh -> Dh) Linears, stored
    as a single (H, Dh, Dh) weight tensor for batched einsum.

    Preserves per-head identity end-to-end through the output projection,
    so the dedicated cross-hop mixer is the only place where hop channels
    interact.
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.weight = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads, head_dim))
        for h in range(num_heads):
            nn.init.kaiming_uniform_(self.weight[h], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh = self.num_heads, self.head_dim
        x = x.view(B, N, H, Dh)
        out = torch.einsum("bnhd,hde->bnhe", x, self.weight) + self.bias
        return out.reshape(B, N, d)


# ---------------------------------------------------------------------------
# Dynamic cross-hop mixer: self-attention along the hop axis.
# ---------------------------------------------------------------------------
class DynamicCrossHopMixer(nn.Module):
    """Per-node attention across the H hop-tagged slabs.

    For each node v, treat its H head-outputs as a length-H sequence
    (each of dim Dh) and run single-head self-attention over them.
    The Q/K/V projections make the mixing *dynamic* (input-dependent)
    rather than a static learned matrix — different nodes produce
    different hop-mixing weights.

    Cost: O(B * N * H² * Dh) — negligible since H is small (e.g. 8).
    """

    def __init__(self, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.H = num_heads
        self.Dh = head_dim
        self.q = nn.Linear(head_dim, head_dim)
        self.k = nn.Linear(head_dim, head_dim)
        self.v = nn.Linear(head_dim, head_dim)
        self.out = nn.Linear(head_dim, head_dim)
        self.drop = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, H*Dh) -> (B, N, H*Dh) with dynamic cross-hop mixing."""
        B, N, d = x.shape
        x = x.view(B, N, self.H, self.Dh)          # (B, N, H, Dh)

        q = self.q(x)  # (B, N, H, Dh)
        k = self.k(x)
        v = self.v(x)

        # Attention along the H (hop) axis for each (batch, node) pair.
        # attn: (B, N, H_query, H_key)
        attn = torch.einsum('bnhd,bnkd->bnhk', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        # Mix: (B, N, H, H) @ (B, N, H, Dh) -> (B, N, H, Dh)
        out = torch.einsum('bnhk,bnkd->bnhd', attn, v)
        out = self.out(out)
        return out.reshape(B, N, d)


# ---------------------------------------------------------------------------
# Hop-masked multi-head attention.
# ---------------------------------------------------------------------------
class HopMaskedMHA(nn.Module):
    """Standard multi-head self-attention with per-head hop masking.

    Each head h is restricted to attend to (v, u) pairs whose shortest-path
    distance lies in the head's hop set. Heads whose hop set is "None" in
    the per-head mask receive all-ones masking (free global attention).

    Args:
        block_diag_out: if True, use a block-diagonal output projection
            (one (Dh, Dh) block per head) so heads are not mixed in
            out_proj. This preserves per-head/per-hop identity; the
            dynamic cross-hop mixer sublayer handles mixing instead.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 block_diag_out: bool = False):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.block_diag_out = block_diag_out
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        if block_diag_out:
            self.out_proj = BlockDiagLinear(num_heads, self.head_dim)
        else:
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,              # (B, N, d)
        per_head_mask: torch.Tensor,  # (B, H, N, N) bool — True where allowed
        node_mask: torch.Tensor,      # (B, N) bool — True for real nodes
    ) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)   # (B, H, N, Dh)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, N, N)

        # Hop masking — disallow (v, u) pairs not in the head's hop set.
        scores = scores.masked_fill(~per_head_mask, _NEG_INF)

        # Padding masks (real-node check).
        key_pad = (~node_mask).unsqueeze(1).unsqueeze(2)      # (B, 1, 1, N)
        query_pad = (~node_mask).unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)
        scores = scores.masked_fill(key_pad, _NEG_INF)
        scores = scores.masked_fill(query_pad, _NEG_INF)

        attn = F.softmax(scores, dim=-1)
        # Rows with no valid keys produce NaN — replace with zeros so the
        # node simply gets a zero contribution from this head.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                       # (B, H, N, Dh)
        out = out.transpose(1, 2).reshape(B, N, d)        # (B, N, d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Pre-norm transformer encoder layer with hop-masked self-attention.
# ---------------------------------------------------------------------------
class HopMaskedTransformerLayer(nn.Module):
    """Pre-norm encoder layer.

    Sublayer order with default flags:
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))

    With ``dynamic_cross_hop=True`` an extra sublayer is inserted between
    attention and FFN:
        x = x + attn(norm1(x))
        x = x + cross_hop_attn(norm_ch(x))   # dynamic mixing across H hop slabs
        x = x + ffn(norm2(x))                # pointwise channel mixer

    Pairing ``block_diag_out=True`` (no cross-head mixing inside attention)
    with ``dynamic_cross_hop=True`` gives a clean separation: attention is
    the "within-hop" mixer, the cross-hop attention is the "across-hop"
    mixer (dynamic, node-conditioned), and the FFN remains the pointwise
    channel mixer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        block_diag_out: bool = False,
        dynamic_cross_hop: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = HopMaskedMHA(
            hidden_dim, num_heads, dropout, block_diag_out=block_diag_out
        )
        self.drop1 = nn.Dropout(dropout)

        self.use_cross_hop = dynamic_cross_hop
        if dynamic_cross_hop:
            head_dim = hidden_dim // num_heads
            self.norm_ch = nn.LayerNorm(hidden_dim)
            self.cross_hop = DynamicCrossHopMixer(
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
            )
            self.drop_ch = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, per_head_mask, node_mask):
        x = x + self.drop1(self.attn(self.norm1(x), per_head_mask, node_mask))
        if self.use_cross_hop:
            x = x + self.drop_ch(self.cross_hop(self.norm_ch(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Top-level model.
# ---------------------------------------------------------------------------
class HopMaskedTransformerModel(nn.Module):
    """Encoder -> stack of hop-masked transformer layers -> task head.

    Args:
        hidden_dim:        node / attention channel dim.
        num_heads:         number of attention heads. Must divide hidden_dim.
        ffn_ratio:         FFN inner-dim multiplier (e.g. 4 → ffn = 4*hidden).
        num_layers:        number of stacked transformer layers.
        dropout:           shared dropout.
        max_hops:          K — total hop levels in the precomputed dist_masks.
        hop_mode:          "contiguous" / "window" / "single" — see
                           ``build_head_hop_sets``.
        hop_window:        half-window for "window" mode.
        num_global_heads:  number of heads that bypass hop masking.
        output_dim:        task output channels.
        graph_pool:        "sum" or "mean" pool for graph-level tasks.
        task_level:        "graph" or "node".
        dataset_name:      passed through to ``build_node_encoder``.
        lap_pe_dim:        Laplacian PE width if used.
        node_feat_dim:     optional override for linear node encoders.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        num_layers: int = 4,
        dropout: float = 0.2,
        max_hops: int = 40,
        hop_mode: str = "contiguous",
        hop_window: int = 1,
        num_global_heads: int = 0,
        output_dim: int = 10,
        graph_pool: str = "sum",
        task_level: str = "graph",
        dataset_name: str = "Peptides-func",
        lap_pe_dim: int = 0,
        node_feat_dim: Optional[int] = None,
        block_diag_out: bool = False,
        dynamic_cross_hop: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_hops = max_hops
        self.task_level = task_level
        self.graph_pool = graph_pool

        self.head_hop_sets = build_head_hop_sets(
            max_hops=max_hops,
            num_heads=num_heads,
            mode=hop_mode,
            window=hop_window,
            include_self=True,
            num_global_heads=num_global_heads,
        )

        self.encoder = build_node_encoder(
            hidden_dim=hidden_dim,
            lap_pe_dim=lap_pe_dim,
            dataset_name=dataset_name,
            node_feat_dim=node_feat_dim,
        )

        ffn_dim = hidden_dim * ffn_ratio
        self.layers = nn.ModuleList([
            HopMaskedTransformerLayer(
                hidden_dim, num_heads, ffn_dim, dropout,
                block_diag_out=block_diag_out,
                dynamic_cross_hop=dynamic_cross_hop,
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    # ----------------------------------------------------------------
    # Build the (B, H, N, N) boolean per-head mask once per forward.
    # ----------------------------------------------------------------
    def _build_per_head_mask(self, dist_masks: torch.Tensor) -> torch.Tensor:
        """Return (B, H, N, N) bool — True where head h is allowed to attend.

        For an unrestricted head (hop_set is None) all entries are True.
        For a restricted head, an entry is True iff the shortest-path
        distance falls in the head's hop set (capped at the runtime K).
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
                # Empty — head sees nothing; softmax row will be NaN→zero.
                # This happens only if all hops in the set exceed K_runtime,
                # which is the dataset's actual diameter cap.
                continue
            stacked = dist_masks[:, idx].bool().any(dim=1)  # (B, N, N)
            out[:, h] = stacked
        return out

    def encode_dense(self, batch):
        lap_pe = getattr(batch, "lap_pe", None)
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr, lap_pe=lap_pe)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, dist_masks, node_masks):
        dense_x, dense_mask = self.encode_dense(batch)
        nm = node_masks if node_masks is not None else dense_mask

        per_head_mask = self._build_per_head_mask(dist_masks)  # (B, H, N, N) bool

        x = dense_x
        for layer in self.layers:
            x = layer(x, per_head_mask, nm)

        node_emb = x[nm]

        if self.task_level == "node":
            return self.head(node_emb), node_emb

        # Graph-level pooling.
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
        return self.head(pooled), node_emb
