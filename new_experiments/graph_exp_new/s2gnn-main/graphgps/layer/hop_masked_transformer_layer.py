"""Hop-masked transformer layer for the S^2GNN framework.

Minimal subset of classes from ``model_hop_masked_transformer_final.py``,
stripped to the essentials needed for the ``single`` hop mode with one global
head and no optional features (no MoE, no cross-hop mixer, no multihop, no
block-diagonal output, no adj-power blend, no edge bias, no RRWP).

Classes
-------
- ``build_head_hop_sets``        — head-to-hop assignment
- ``LayerNormWrap`` / ``build_norm`` — normalization wrappers
- ``HopMaskedMHA``               — core hop-masked multi-head attention
- ``HopMaskedTransformerLayer``  — pre-norm encoder layer
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    mode: str = "single",
    window: int = 0,
    include_self: bool = True,
    num_global_heads: int = 1,
) -> List[Optional[List[int]]]:
    """Assign a hop set to each attention head.

    Parameters
    ----------
    max_hops : int
        K — total number of hop levels available.
    num_heads : int
        H — total attention heads.
    mode : str
        ``"contiguous"`` | ``"window"`` | ``"single"``.
    window : int
        Half-window for ``"window"`` mode.
    include_self : bool
        If True, hop 0 is added to every restricted head.
    num_global_heads : int
        The last N heads are unrestricted (``None`` → global attention).

    Returns
    -------
    list of (list[int] | None)
        Length ``num_heads``.  Each entry is either a list of hop indices the
        head is allowed to attend to, or ``None`` for an unrestricted head.
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
        hops = list(range(1, K))
        sets = _split_contiguous(hops, H_restricted)
    elif mode == "window":
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
    else:
        raise ValueError(f"Unknown hop assignment mode: {mode}")

    if include_self:
        sets = [sorted(set([0] + s)) for s in sets]

    return sets + [None] * num_global_heads


# ---------------------------------------------------------------------------
# Normalization.
# ---------------------------------------------------------------------------
class LayerNormWrap(nn.Module):
    """``nn.LayerNorm`` with a ``(x, node_mask)`` call signature."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, node_mask=None):
        return self.norm(x)


def build_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layer":
        return LayerNormWrap(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


# ---------------------------------------------------------------------------
# Hop-masked multi-head attention.
# ---------------------------------------------------------------------------
class HopMaskedMHA(nn.Module):
    """Standard multi-head self-attention with per-head hop masking.

    Each head *h* is restricted to attend to ``(v, u)`` pairs whose
    shortest-path distance lies in the head's hop set.  Heads whose hop set
    is ``None`` receive all-ones masking (free global attention).
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by "
                f"num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,              # (B, N, d)
        per_head_mask: torch.Tensor,   # (B, H, N, N) bool — True where allowed
        node_mask: torch.Tensor,       # (B, N) bool — True for real nodes
    ) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)  # (B,H,N,Dh)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        # Hop masking.
        scores = scores.masked_fill(~per_head_mask, _NEG_INF)

        # Padding masks.
        key_pad = (~node_mask).unsqueeze(1).unsqueeze(2)      # (B, 1, 1, N)
        query_pad = (~node_mask).unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)
        scores = scores.masked_fill(key_pad, _NEG_INF)
        scores = scores.masked_fill(query_pad, _NEG_INF)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                        # (B, H, N, Dh)
        
        # Save pre-projected output and attention for logging/analysis
        if not self.training:
            self._last_attn = attn.detach()
            self._last_out_pre_proj = out.detach()

        out = out.transpose(1, 2).reshape(B, N, d)         # (B, N, d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Pre-norm transformer encoder layer.
# ---------------------------------------------------------------------------
class HopMaskedTransformerLayer(nn.Module):
    """Pre-norm encoder layer with hop-masked self-attention.

    Sublayer order::

        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        norm_type: str = "layer",
    ):
        super().__init__()
        self.norm1 = build_norm(norm_type, hidden_dim)
        self.attn = HopMaskedMHA(hidden_dim, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = build_norm(norm_type, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, per_head_mask, node_mask):
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, N, d)
        per_head_mask : Tensor, shape (B, H, N, N) bool
        node_mask : Tensor, shape (B, N) bool
        """
        normed = self.norm1(x, node_mask)
        attn_out = self.attn(normed, per_head_mask, node_mask)
        x = x + self.drop1(attn_out)

        x = x + self.drop2(self.ffn(self.norm2(x, node_mask)))
        return x
