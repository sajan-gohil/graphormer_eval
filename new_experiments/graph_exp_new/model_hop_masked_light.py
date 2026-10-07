# model_hop_masked_light.py
"""
Lightweight hop-masked multi-head transformer.

Core features:
- Hop-masked attention (contiguous/window/single/interleaved/alternating modes)
- Multi-hop attention (every head × every hop)
- Global heads (unrestricted attention)
- Adjacency-power walk-count blending
- Dynamic cross-hop mixer
- Block-diagonal output projection
- Edge features via BondEncoder
- Post-transformer GATv2 layers
"""

from __future__ import annotations

import json
import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool, GATv2Conv

from models import build_node_encoder, build_bond_encoder, LinearBondEncoder


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
    hop_file: Optional[str] = None,
) -> List[Optional[List[int]]]:
    """Assign a hop set to each attention head."""
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
    elif mode == "interleaved":
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
    elif mode == "alternating":
        even_hops = [k for k in range(0, K) if k % 2 == 0]
        sets = [even_hops] * H_restricted
    elif mode == "file":
        if hop_file is None:
            raise ValueError(
                "hop_mode='file' requires --hop_file to be specified"
            )
        with open(hop_file, "r") as f:
            raw = json.load(f)
        all_sets: List[Optional[List[int]]] = [None] * num_heads
        for key, hops in raw.items():
            head_idx = int(key) - 1
            if head_idx < 0 or head_idx >= num_heads:
                raise ValueError(
                    f"Head key {key} (0-indexed: {head_idx}) out of range "
                    f"for num_heads={num_heads}"
                )
            all_sets[head_idx] = sorted(int(h) for h in hops)
        if include_self:
            all_sets = [
                sorted(set([0] + s)) if s is not None else None
                for s in all_sets
            ]
        return all_sets
    else:
        raise ValueError(f"Unknown hop assignment mode: {mode}")

    if include_self:
        sets = [sorted(set([0] + s)) for s in sets]

    return sets + [None] * num_global_heads


def _build_alternating_hop_sets(
    max_hops: int,
    num_heads: int,
    include_self: bool = True,
    num_global_heads: int = 0,
) -> Tuple[List[Optional[List[int]]], List[Optional[List[int]]]]:
    """Build two hop-set lists for the 'alternating' mode."""
    K = max_hops
    H_restricted = num_heads - num_global_heads
    even_hops = [k for k in range(0, K) if k % 2 == 0]
    odd_hops  = [k for k in range(0, K) if k % 2 == 1]
    if include_self and 0 not in odd_hops:
        odd_hops = sorted([0] + odd_hops)
    even_sets = [[0, i] for i in even_hops[1:H_restricted+1]] + [None] * num_global_heads
    odd_sets  = [[0, i] for i in odd_hops[1:H_restricted+1]]  + [None] * num_global_heads
    return even_sets, odd_sets


# ---------------------------------------------------------------------------
# Block-diagonal linear: per-head projection with no cross-head mixing.
# ---------------------------------------------------------------------------
class BlockDiagLinear(nn.Module):
    """Block-diagonal linear projection — no cross-head mixing."""

    def __init__(self, num_heads: int, in_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(num_heads, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads, out_dim))
        for h in range(num_heads):
            nn.init.kaiming_uniform_(self.weight[h], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H = self.num_heads
        x = x.view(B, N, H, self.in_dim)
        out = torch.einsum("bnhd,hde->bnhe", x, self.weight) + self.bias
        return out.reshape(B, N, H * self.out_dim)


# ---------------------------------------------------------------------------
# Normalization layers.
# ---------------------------------------------------------------------------
class LayerNormWrap(nn.Module):
    """nn.LayerNorm with a (x, node_mask) signature."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None):
        return self.norm(x)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class GraphNorm(nn.Module):
    """Masked GraphNorm over the dense node tensor."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None):
        if node_mask is None:
            m = x.new_ones(x.shape[0], x.shape[1], 1)
        else:
            m = node_mask.unsqueeze(-1).to(x.dtype)
        cnt = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(dim=1, keepdim=True) / cnt
        out = x - self.alpha * mean
        var = ((out * m) ** 2).sum(dim=1, keepdim=True) / cnt
        out = out / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out * m


def build_norm(norm_type: str, dim: int) -> nn.Module:
    if norm_type == "layer":
        return LayerNormWrap(dim)
    if norm_type == "rms":
        return RMSNorm(dim)
    if norm_type == "graph":
        return GraphNorm(dim)
    raise ValueError(f"Unknown norm_type: {norm_type}")


# ---------------------------------------------------------------------------
# Attention-based graph readout (Set-Transformer PMA).
# ---------------------------------------------------------------------------
class AttentionReadout(nn.Module):
    """Pool node embeddings into one graph vector via learnable query."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * hidden_dim ** -0.5)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d); node_mask: (B, N) bool True=real -> (B, d)."""
        B = x.shape[0]
        q = self.query.expand(B, 1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=~node_mask)
        return out.squeeze(1)


# ---------------------------------------------------------------------------
# Dynamic cross-hop mixer.
# ---------------------------------------------------------------------------
class DynamicCrossHopMixer(nn.Module):
    """Per-node attention across hop-tagged slabs."""

    def __init__(self, num_heads: int, head_dim: int, dropout: float = 0.0,
                 use_hop_embedding: bool = False, max_hops: Optional[int] = None,
                 hop_membership: Optional[torch.Tensor] = None):
        super().__init__()
        self.H = num_heads
        self.Dh = head_dim
        self.out = nn.Linear(head_dim, head_dim)
        self.drop = nn.Dropout(dropout)
        self.hop_tag_drop = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

        if not use_hop_embedding:
            self.hop_mode = "none"
        elif hop_membership is not None:
            self.hop_mode = "membership"
            self.hop_embedding = nn.Embedding(max_hops, head_dim)
            scaled_std = 0.02 * (head_dim ** -0.5)
            nn.init.normal_(self.hop_embedding.weight, mean=0.0, std=scaled_std)
            self.register_buffer("hop_membership", hop_membership.float())
        else:
            self.hop_mode = "head"
            self.hop_embedding = nn.Embedding(num_heads, head_dim)

        multiplier = 2 if self.hop_mode != "none" else 1
        self.q = nn.Linear(head_dim * multiplier, head_dim)
        self.k = nn.Linear(head_dim * multiplier, head_dim)
        self.v = nn.Linear(head_dim * multiplier, head_dim)

    def forward(self, x: torch.Tensor,
                head_valid_mask: Optional[torch.Tensor] = None,
                hop_tag: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, N, H*Dh) -> (B, N, H*Dh) with dynamic cross-hop mixing."""
        B, N, d = x.shape
        x = x.view(B, N, self.H, self.Dh)

        if self.hop_mode == "membership":
            tag = (self.hop_membership @ self.hop_embedding.weight)
            tag = self.hop_tag_drop(tag.view(1, 1, self.H, self.Dh).expand(B, N, self.H, self.Dh))
            x = torch.cat([x, tag], dim=-1)
        elif self.hop_mode == "head":
            tag = self.hop_tag_drop(self.hop_embedding.weight.view(1, 1, self.H, self.Dh).expand(B, N, self.H, self.Dh))
            x = torch.cat([x, tag], dim=-1)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn = torch.einsum('bnhd,bnkd->bnhk', q, k) * self.scale

        if head_valid_mask is not None:
            key_invalid = ~head_valid_mask
            attn = attn.masked_fill(
                key_invalid.unsqueeze(1).unsqueeze(2),
                _NEG_INF,
            )

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)

        out = torch.einsum('bnhk,bnkd->bnhd', attn, v)
        out = self.out(out)
        if hop_tag is not None:  # (H, Dh)
            out = out + hop_tag.view(1, 1, self.H, self.Dh)

        if head_valid_mask is not None:
            out = out * head_valid_mask.unsqueeze(1).unsqueeze(-1).to(out.dtype)

        return out.reshape(B, N, d)


# ---------------------------------------------------------------------------
# Path-aware embeddings added to K and V.
# ---------------------------------------------------------------------------
class PathEmbedder(nn.Module):
    """Dense (B, N, N, d) path embeddings, shared by all layers and heads.

    P[b,s,t] = mean_{r=1..k, edge (u->v) at position r on SP(s,t)}(
                   W_s x_u + W_t x_v + W_e BondEnc(e_uv) + pos_emb[r-1])
             + hop_emb[d(s,t)]
    hop_emb rows: 0..K-1 = distance, K = beyond K-1 / unreachable,
    K+1 = global-head tag for the cross-hop mixer.
    Edge term is used whenever batch.edge_attr is present.
    """

    def __init__(self, hidden_dim: int, max_hops: int,
                 dataset_name: str = "Peptides-func",
                 edge_feat_dim: Optional[int] = None):
        super().__init__()
        d, std = hidden_dim, 0.02 * hidden_dim ** -0.5
        self.K = max_hops
        self.pos_emb = nn.Embedding(max_hops, d)
        self.hop_emb = nn.Embedding(max_hops + 2, d)
        nn.init.normal_(self.pos_emb.weight, std=std)
        nn.init.normal_(self.hop_emb.weight, std=std)
        # Linear(e || x_u || x_v) split into blocks.
        self.W_s = nn.Linear(d, d, bias=False)
        self.W_t = nn.Linear(d, d)
        self.W_e = nn.Linear(d, d, bias=False)
        self.edge_enc = build_bond_encoder(d, dataset_name=dataset_name,
                                           edge_feat_dim=edge_feat_dim)
        # Continuous edge features: W_e(GELU(Linear(e))) = 2-layer MLP.
        # Categorical encoders (embedding lookups) stay W_e(Emb(e)).
        self.edge_act = (nn.GELU() if isinstance(self.edge_enc, LinearBondEncoder)
                         else nn.Identity())

    def hop_tags(self, hop_sets: List[Optional[List[int]]], H: int) -> torch.Tensor:
        """(H, Dh): mean hop_emb over each head's hop set, head-h slice."""
        W = self.hop_emb.weight
        Dh = W.shape[1] // H
        M = W.new_zeros(H, self.K + 2)
        for h, hs in enumerate(hop_sets):
            idx = [k for k in hs if 0 <= k < self.K] if hs is not None else []
            if idx:
                M[h, idx] = 1.0 / len(idx)
            else:
                M[h, self.K + 1] = 1.0
        full = (M @ W).view(H, H, Dh)
        ar = torch.arange(H, device=W.device)
        return full[ar, ar]

    def forward(self, x, edge_attr, edge_map, dist_masks, pred, node_mask):
        B, N, d = x.shape
        dev = x.device
        A, C = self.W_s(x), self.W_t(x)
        pred = pred.to(dev).long()

        E = None
        if edge_attr is not None and edge_map is not None:
            E = self.W_e(self.edge_act(self.edge_enc(edge_attr)))
            E = torch.cat([E, E.new_zeros(1, d)])  # index -1 -> zero row

        K = dist_masks.shape[1]
        L = min(K - 1, self.pos_emb.num_embeddings)
        pos_cum = self.pos_emb.weight[:L].cumsum(0)
        row_of = torch.full((B, N, N), -1, dtype=torch.long, device=dev)
        out = x.new_zeros(B, N, N, d)
        prev = None
        for k in range(1, L + 1):
            b, s, t = (dist_masks[:, k] > 0).nonzero(as_tuple=True)
            if b.numel() == 0:
                break
            p = pred[b, s, t]
            S = A[b, p] + C[b, t]
            if E is not None:
                S = S + E[edge_map[b, p, t]]
            if k > 1:
                S = S + prev[row_of[b, s, p]]
            row_of[b, s, t] = torch.arange(b.numel(), device=dev)
            prev = S
            out[b, s, t] = (S + pos_cum[k - 1]) / k

        # Pair-distance embedding d(s,t); K = beyond K-1 / unreachable.
        has = dist_masks.bool().any(1)
        didx = dist_masks.float().argmax(1)
        didx[~has] = self.K
        out = out + self.hop_emb(didx)
        pair = node_mask[:, :, None] & node_mask[:, None, :]
        return out * pair.unsqueeze(-1).to(out.dtype)


# ---------------------------------------------------------------------------
# Hop-masked multi-head attention.
# ---------------------------------------------------------------------------
class HopMaskedMHA(nn.Module):
    """Standard multi-head self-attention with per-head hop masking."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 block_diag_out: bool = False, blend_adj_power: bool = False,
                 use_edge_bias: bool = False):
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
            self.out_proj = BlockDiagLinear(num_heads, self.head_dim, self.head_dim)
        else:
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.blend_adj_power = blend_adj_power
        if blend_adj_power:
            self.adj_blend_gamma = nn.Parameter(torch.zeros(num_heads))

        self.use_edge_bias = use_edge_bias

    def forward(
        self,
        x: torch.Tensor,
        per_head_mask: torch.Tensor,
        node_mask: torch.Tensor,
        adj_blend_bias: Optional[torch.Tensor] = None,
        edge_bias: Optional[torch.Tensor] = None,
        path_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)

        scores = scores.masked_fill(~per_head_mask, _NEG_INF)

        if self.blend_adj_power and adj_blend_bias is not None:
            gamma = self.adj_blend_gamma.view(1, self.num_heads, 1, 1)
            scores = scores + gamma * adj_blend_bias

        if self.use_edge_bias and edge_bias is not None:
            scores = scores + edge_bias

        # Add path embeddings contribution to scores
        if path_embeddings is not None:
            # K_ij = k_j + P_ij  =>  extra score term q_i . P_ij (per head slice)
            path_emb_per_head = path_embeddings.view(B, N, N, H, Dh)  # (B, N, N, H, Dh)
            path_scores = torch.einsum('bhid,bijhd->bhij', q, path_emb_per_head) / math.sqrt(Dh)
            scores = scores + path_scores

        key_pad = (~node_mask).unsqueeze(1).unsqueeze(2)
        query_pad = (~node_mask).unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(key_pad, _NEG_INF)
        scores = scores.masked_fill(query_pad, _NEG_INF)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        if self.training and self.attn_drop.p > 0:
            p = self.attn_drop.p
            allowed = (per_head_mask
                       & node_mask[:, None, None, :]
                       & node_mask[:, None, :, None])
            keep = torch.rand_like(attn) >= p
            keep = keep | ~allowed
            attn = attn * keep.to(attn.dtype) / (1 - p)

        out = torch.matmul(attn, v)
        if path_embeddings is not None:
            out = out + torch.einsum('bhij,bijhd->bhid', attn, path_emb_per_head)
        out = out.transpose(1, 2).reshape(B, N, d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Multi-hop masked MHA: every head masked by every hop.
# ---------------------------------------------------------------------------
class MultiHopMaskedMHA(nn.Module):
    """Multi-head self-attention where EVERY head is masked by EVERY hop."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 block_diag_out: bool = False, readout: str = "sum",
                 include_global: bool = True):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if readout not in ("sum", "mean"):
            raise ValueError(f"readout must be 'sum' or 'mean', got {readout!r}")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.readout = readout
        self.include_global = include_global
        self.block_diag_out = block_diag_out
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        if block_diag_out:
            self.out_proj = BlockDiagLinear(num_heads, self.head_dim, self.head_dim)
        else:
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        dist_masks: torch.Tensor,
        node_mask: torch.Tensor,
        path_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh = self.num_heads, self.head_dim
        K = dist_masks.shape[1]

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        Ph = None
        if path_embeddings is not None:
            Ph = path_embeddings.view(B, N, N, H, Dh)
            scores = scores + torch.einsum('bhid,bijhd->bhij', q, Ph) / math.sqrt(Dh)

        key_pad   = (~node_mask).unsqueeze(1).unsqueeze(2)
        query_pad = (~node_mask).unsqueeze(1).unsqueeze(-1)

        def _attend(hop_mask: Optional[torch.Tensor]) -> torch.Tensor:
            """One masked-attention view."""
            s = scores
            if hop_mask is not None:
                s = s.masked_fill(~hop_mask, _NEG_INF)
            s = s.masked_fill(key_pad, _NEG_INF)
            s = s.masked_fill(query_pad, _NEG_INF)
            a = F.softmax(s, dim=-1)
            a = torch.nan_to_num(a, nan=0.0)
            return self.attn_drop(a)

        # Sum attention weights over views; a@v and a.P are linear in a.
        a_sum = scores.new_zeros(B, H, N, N)
        count = scores.new_zeros(B, N)

        for hop in range(K):
            hop_mask = dist_masks[:, hop] > 0
            if not hop_mask.any():
                continue
            a_sum = a_sum + _attend(hop_mask.unsqueeze(1))
            count = count + hop_mask.any(dim=-1).to(scores.dtype)

        if self.include_global:
            a_sum = a_sum + _attend(None)
            count = count + node_mask.to(scores.dtype)

        out_sum = torch.matmul(a_sum, v)
        if Ph is not None:
            out_sum = out_sum + torch.einsum('bhij,bijhd->bhid', a_sum, Ph)

        if self.readout == "mean":
            denom = count.clamp(min=1.0).view(B, 1, N, 1)
            out_sum = out_sum / denom

        out = out_sum.transpose(1, 2).reshape(B, N, d)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Pre-norm transformer encoder layer.
# ---------------------------------------------------------------------------
class HopMaskedTransformerLayer(nn.Module):
    """Pre-norm encoder layer with hop-masked attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        block_diag_out: bool = False,
        dynamic_cross_hop: bool = False,
        norm_type: str = "layer",
        blend_adj_power: bool = False,
        use_edge_bias: bool = False,
        multihop_attn: bool = False,
        multihop_readout: str = "sum",
        multihop_include_global: bool = True,
    ):
        super().__init__()
        self.multihop_attn = multihop_attn
        self.norm1 = build_norm(norm_type, hidden_dim)

        if multihop_attn:
            self.attn = MultiHopMaskedMHA(
                hidden_dim, num_heads, dropout,
                block_diag_out=block_diag_out,
                readout=multihop_readout, include_global=multihop_include_global,
            )
        else:
            self.attn = HopMaskedMHA(
                hidden_dim, num_heads, dropout, block_diag_out=block_diag_out,
                blend_adj_power=blend_adj_power, use_edge_bias=use_edge_bias,
            )
        self.drop1 = nn.Dropout(dropout)

        self.use_cross_hop = dynamic_cross_hop
        if dynamic_cross_hop:
            head_dim = hidden_dim // num_heads
            self.norm_ch = build_norm(norm_type, hidden_dim)
            self.cross_hop = DynamicCrossHopMixer(
                num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                use_hop_embedding=False,
            )
            self.drop_ch = nn.Dropout(dropout)

        self.norm2 = build_norm(norm_type, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, per_head_mask_or_dist_masks, node_mask,
                adj_blend_bias: Optional[torch.Tensor] = None,
                edge_bias: Optional[torch.Tensor] = None,
                path_embeddings: Optional[torch.Tensor] = None,
                head_valid_mask: Optional[torch.Tensor] = None,
                hop_tag: Optional[torch.Tensor] = None):
        """Forward pass."""
        normed = self.norm1(x, node_mask)
        if self.multihop_attn:
            attn_out = self.attn(normed, per_head_mask_or_dist_masks, node_mask,
                                 path_embeddings=path_embeddings)
            x = x + self.drop1(attn_out)
        else:
            attn_out = self.attn(normed, per_head_mask_or_dist_masks, node_mask,
                                 adj_blend_bias=adj_blend_bias, edge_bias=edge_bias,
                                 path_embeddings=path_embeddings)
            x = x + self.drop1(attn_out)
        if self.use_cross_hop:
            x = x + self.drop_ch(
                self.cross_hop(self.norm_ch(x, node_mask),
                               head_valid_mask=head_valid_mask,
                               hop_tag=hop_tag)
            )
        x = x + self.drop2(self.ffn(self.norm2(x, node_mask)))
        return x


# ---------------------------------------------------------------------------
# Post-transformer GATv2 block.
# ---------------------------------------------------------------------------
class PostGATv2Block(nn.Module):
    """GATv2 layers applied after transformer stack."""

    def __init__(
        self,
        hidden_dim: int,
        num_gat_heads: int = 4,
        num_gat_layers: int = 1,
        dropout: float = 0.1,
        use_edge_features: bool = False,
        dataset_name: str = "Peptides-func",
        edge_feat_dim: Optional[int] = None,
    ):
        super().__init__()
        assert hidden_dim % num_gat_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_gat_heads={num_gat_heads}"
        )
        self.num_gat_layers = num_gat_layers
        self.use_edge_features = use_edge_features
        if use_edge_features:
            self.bond_encoder = build_bond_encoder(
                hidden_dim, dataset_name=dataset_name, edge_feat_dim=edge_feat_dim,
            )
            edge_dim = hidden_dim
        else:
            edge_dim = None
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_gat_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_gat_heads,
                    heads=num_gat_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        edge_emb = None
        if self.use_edge_features and edge_attr is not None:
            edge_emb = self.bond_encoder(edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x, edge_index, edge_attr=edge_emb)
            out = self.drop(F.elu(out))
            x = norm(x + out)
        return x


# ---------------------------------------------------------------------------
# Top-level model.
# ---------------------------------------------------------------------------
class HopMaskedTransformerModel(nn.Module):
    """Encoder -> hop-masked transformer layers -> task head."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        num_layers: int = 4,
        dropout: float = 0.2,
        max_hops: int = 40,
        hop_mode: str = "contiguous",
        hop_window: int = 1,
        hop_file: Optional[str] = None,
        num_global_heads: int = 0,
        output_dim: int = 10,
        graph_pool: str = "sum",
        task_level: str = "graph",
        dataset_name: str = "Peptides-func",
        node_feat_dim: Optional[int] = None,
        block_diag_out: bool = False,
        dynamic_cross_hop: bool = False,
        norm_type: str = "layer",
        mask_type: str = "shortest_path",
        adj_self_loops: bool = False,
        num_post_gat_layers: int = 0,
        num_gat_heads: int = 4,
        use_edge_features: bool = False,
        edge_feat_dim: Optional[int] = None,
        blend_adj_power: bool = False,
        use_edge_bias: bool = False,
        multihop_attn: bool = False,
        multihop_readout: str = "sum",
        multihop_include_global: bool = True,
        embed_dropout: float = 0.0,
        use_path_embeddings: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if multihop_attn and blend_adj_power:
            raise ValueError(
                "multihop_attn and blend_adj_power are mutually exclusive"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.max_hops = max_hops
        self.task_level = task_level
        self.graph_pool = graph_pool
        self.mask_type = mask_type
        self.adj_self_loops = adj_self_loops
        self.multihop_attn = multihop_attn
        self.use_alternating = (hop_mode == "alternating")
        self.blend_adj_power = blend_adj_power
        self.use_edge_bias = use_edge_bias
        self.embed_drop = nn.Dropout(embed_dropout)
        self.dropout = dropout

        # Hop-to-head assignment
        if self.use_alternating:
            even_sets, odd_sets = _build_alternating_hop_sets(
                max_hops=max_hops,
                num_heads=num_heads,
                include_self=True,
                num_global_heads=num_global_heads,
            )
            self.per_layer_hop_sets = [
                even_sets if (i % 2 == 0) else odd_sets
                for i in range(num_layers)
            ]
            self.head_hop_sets = even_sets
        else:
            self.head_hop_sets = build_head_hop_sets(
                max_hops=max_hops,
                num_heads=num_heads,
                mode=hop_mode,
                window=hop_window,
                include_self=True,
                num_global_heads=num_global_heads,
                hop_file=hop_file,
            )
            self.per_layer_hop_sets = [self.head_hop_sets] * num_layers

        used = [
            k
            for hop_sets in self.per_layer_hop_sets
            for s in hop_sets if s is not None
            for k in s
        ]
        self._max_hop_index = max(used) if used else 0

        self.use_edge_features = use_edge_features
        self.encoder = build_node_encoder(
            hidden_dim=hidden_dim,
            lap_pe_dim=0,
            dataset_name=dataset_name,
            node_feat_dim=node_feat_dim,
            use_edge_features=use_edge_features,
            edge_feat_dim=edge_feat_dim,
        )

        # Edge bias
        if use_edge_bias:
            self.bond_encoder_attn = build_bond_encoder(
                hidden_dim, dataset_name=dataset_name, edge_feat_dim=edge_feat_dim
            )
            self.edge_bias_proj = nn.Linear(hidden_dim, num_heads, bias=False)
            if self.use_alternating:
                hop1_masks_per_layer = [
                    torch.tensor(
                        [1 in (hop_set or []) for hop_set in layer_sets],
                        dtype=torch.bool,
                    )
                    for layer_sets in self.per_layer_hop_sets
                ]
                self.register_buffer('_hop1_head_masks_per_layer',
                                    torch.stack(hop1_masks_per_layer))
            else:
                hop1_mask = torch.tensor(
                    [1 in (hop_set or []) for hop_set in self.head_hop_sets],
                    dtype=torch.bool,
                )
                self.register_buffer('_hop1_head_mask', hop1_mask)

        # Path-aware embeddings
        self.use_path_embeddings = use_path_embeddings
        self.path_embedder = None
        if use_path_embeddings:
            self.path_embedder = PathEmbedder(
                hidden_dim=hidden_dim,
                max_hops=max_hops,
                dataset_name=dataset_name,
                edge_feat_dim=edge_feat_dim,
            )

        self._use_cross_hop = dynamic_cross_hop

        ffn_dim = int(hidden_dim * ffn_ratio)
        layers = []
        for layer_idx in range(num_layers):
            layers.append(
                HopMaskedTransformerLayer(
                    hidden_dim, num_heads, ffn_dim, dropout,
                    block_diag_out=block_diag_out,
                    dynamic_cross_hop=dynamic_cross_hop,
                    norm_type=norm_type,
                    blend_adj_power=blend_adj_power,
                    use_edge_bias=use_edge_bias,
                    multihop_attn=multihop_attn,
                    multihop_readout=multihop_readout,
                    multihop_include_global=multihop_include_global,
                )
            )
        self.layers = nn.ModuleList(layers)

        self.readout = None
        if graph_pool == "attention":
            self.readout = AttentionReadout(hidden_dim, num_heads, dropout=dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        if num_post_gat_layers > 0:
            self.post_gat = PostGATv2Block(
                hidden_dim=hidden_dim,
                num_gat_heads=num_gat_heads,
                num_gat_layers=num_post_gat_layers,
                dropout=dropout,
                use_edge_features=use_edge_features,
                dataset_name=dataset_name,
                edge_feat_dim=edge_feat_dim,
            )
        else:
            self.post_gat = None

    def _build_adj_blend_bias(
        self,
        dist_masks: torch.Tensor,
        hop_sets: List[Optional[List[int]]],
    ) -> torch.Tensor:
        """Build (B, H, N, N) adj_power blend bias."""
        adj_power = self._build_adj_power_masks(dist_masks)
        B, K, N, _ = adj_power.shape
        H = self.num_heads
        blend_bias = adj_power.new_zeros(B, H, N, N)
        for h, hop_set in enumerate(hop_sets):
            if hop_set is None:
                continue
            idx = [k for k in hop_set if 0 < k < K]
            if not idx:
                continue
            blend_bias[:, h] = adj_power[:, idx].sum(dim=1)
        return blend_bias

    def _build_adj_blend_bias_from_power(
        self,
        adj_power: torch.Tensor,
        hop_sets: List[Optional[List[int]]],
    ) -> torch.Tensor:
        """Build adj_blend_bias from pre-computed adj_power."""
        B, K, N, _ = adj_power.shape
        H = self.num_heads
        blend_bias = adj_power.new_zeros(B, H, N, N)
        for h, hop_set in enumerate(hop_sets):
            if hop_set is None:
                continue
            idx = [k for k in hop_set if 0 < k < K]
            if not idx:
                continue
            blend_bias[:, h] = adj_power[:, idx].sum(dim=1)
        return blend_bias

    def _build_edge_bias(
        self,
        batch,
        B: int,
        N_max: int,
        device: torch.device,
        layer_idx: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Build (B, H, N_max, N_max) edge attention bias."""
        edge_attr = getattr(batch, 'edge_attr', None)
        if edge_attr is None:
            return None

        edge_index = batch.edge_index
        batch_vec  = batch.batch

        bond_emb    = self.bond_encoder_attn(edge_attr)
        edge_scalars = self.edge_bias_proj(bond_emb)

        src_global = edge_index[0]
        dst_global = edge_index[1]
        edge_graph = batch_vec[src_global]

        counts  = torch.bincount(batch_vec, minlength=B)
        offsets = torch.zeros(B, dtype=torch.long, device=device)
        if B > 1:
            offsets[1:] = counts[:-1].cumsum(0)

        src_local = src_global - offsets[edge_graph]
        dst_local = dst_global - offsets[edge_graph]

        flat_idx = (
            edge_graph.long() * (N_max * N_max)
            + src_local.long() * N_max
            + dst_local.long()
        )

        H = self.num_heads
        edge_bias_flat = torch.zeros(B * N_max * N_max, H, device=device, dtype=bond_emb.dtype)
        edge_bias_flat.scatter_add_(
            0,
            flat_idx.unsqueeze(1).expand(-1, H),
            edge_scalars,
        )

        edge_bias = (
            edge_bias_flat
            .view(B, N_max, N_max, H)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        if self.use_alternating and layer_idx is not None:
            hop1_mask = self._hop1_head_masks_per_layer[layer_idx].to(device)
        else:
            hop1_mask = self._hop1_head_mask.to(device)
        edge_bias = edge_bias * hop1_mask.view(1, H, 1, 1).to(edge_bias.dtype)

        return edge_bias

    def _build_adj_power_masks(self, dist_masks: torch.Tensor) -> torch.Tensor:
        """Build hop masks from powers of the adjacency matrix."""
        B, K, N, _ = dist_masks.shape
        eye = torch.eye(N, device=dist_masks.device, dtype=torch.bool)

        out = dist_masks.new_zeros(B, K, N, N, dtype=torch.float32)
        out[:, 0] = eye.float().unsqueeze(0)
        if K <= 1:
            return out

        A = dist_masks[:, 1] > 0
        base = (A | eye.unsqueeze(0)) if self.adj_self_loops else A
        base_f = base.float()
        k_max = min(K - 1, self._max_hop_index)

        cur_f = base_f.clone()
        seen = eye.unsqueeze(0).expand(B, -1, -1).clone()
        reach_count = eye.unsqueeze(0).expand(B, -1, -1).float()
        for k in range(1, k_max + 1):
            cur_bool = cur_f > 0
            reach_count = reach_count + cur_bool.float()
            new_mask = cur_bool & ~seen
            old_mask = cur_bool &  seen
            scores = torch.zeros(B, N, N, device=dist_masks.device, dtype=torch.float32)
            scores[cur_bool] =  1.0 / reach_count[cur_bool]
            scores[new_mask] = 1.0
            out[:, k] = scores
            seen = seen | cur_bool
            if k < k_max:
                cur_f = torch.bmm(cur_f, base_f)
        return out

    def _build_hop_membership(
        self,
        hop_sets: List[Optional[List[int]]],
        max_hops: int,
    ) -> torch.Tensor:
        """Return a (H, max_hops) {0,1} matrix."""
        H = self.num_heads
        membership = torch.zeros(H, max_hops)
        for h, hop_set in enumerate(hop_sets):
            if hop_set is None:
                continue
            for k in hop_set:
                if 0 <= k < max_hops:
                    membership[h, k] = 1.0
        return membership

    def _build_per_head_mask(
        self,
        dist_masks: torch.Tensor,
        hop_sets: Optional[List[Optional[List[int]]]] = None,
    ) -> torch.Tensor:
        """Return (B, H, N, N) bool per-head mask."""
        B, K_runtime, N, _ = dist_masks.shape
        H = self.num_heads
        if hop_sets is None:
            hop_sets = self.head_hop_sets
        out = dist_masks.new_zeros(B, H, N, N, dtype=torch.bool)
        for h, hop_set in enumerate(hop_sets):
            if hop_set is None:
                out[:, h] = True
                continue
            idx = [k for k in hop_set if k < K_runtime]
            if not idx:
                out[:, h] = True
                continue
            stacked = dist_masks[:, idx].bool().any(dim=1)
            out[:, h] = stacked

        eye = torch.eye(N, device=dist_masks.device, dtype=torch.bool)
        non_self = out & ~eye[None, None]
        has_nonself = non_self.any(dim=-1).any(dim=-1)
        out[~has_nonself] = True

        return out

    def _build_edge_map(self, batch, B: int, N: int) -> torch.Tensor:
        """(B, N, N) long: index of edge (u->v) in batch.edge_index, -1 if none."""
        bv, ei = batch.batch, batch.edge_index
        offsets = torch.zeros(B, dtype=torch.long, device=bv.device)
        if B > 1:
            offsets[1:] = torch.bincount(bv, minlength=B)[:-1].cumsum(0)
        g = bv[ei[0]]
        m = torch.full((B, N, N), -1, dtype=torch.long, device=bv.device)
        m[g, ei[0] - offsets[g], ei[1] - offsets[g]] = torch.arange(
            ei.shape[1], device=bv.device)
        return m

    def encode_dense(self, batch):
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr, lap_pe=None)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, dist_masks, node_masks):
        """Forward pass. Returns (logits, node_emb, aux_loss, None)."""
        dense_x, dense_mask = self.encode_dense(batch)
        dense_x = self.embed_drop(dense_x)
        nm = node_masks if node_masks is not None else dense_mask

        mask_source = (
            self._build_adj_power_masks(dist_masks)
            if self.mask_type == "adj_power"
            else dist_masks
        )

        B_orig, N_orig = dense_x.shape[:2]

        adj_blend_bias = None
        if self.blend_adj_power and not self.use_alternating:
            adj_blend_bias = self._build_adj_blend_bias(
                mask_source, self.head_hop_sets
            )

        edge_bias = None
        if self.use_edge_bias and not self.use_alternating:
            edge_bias = self._build_edge_bias(batch, B_orig, N_orig, dense_x.device)

        x = dense_x

        # Path embeddings: built once, shared by all layers (both attention modes).
        path_embeddings = None
        hop_tags = [None] * len(self.layers)
        if self.use_path_embeddings:
            edge_attr = getattr(batch, "edge_attr", None)
            edge_map = (self._build_edge_map(batch, B_orig, N_orig)
                        if edge_attr is not None else None)
            path_embeddings = self.path_embedder(
                dense_x, edge_attr, edge_map, dist_masks,
                batch.path_data["pred"], nm)
            if self._use_cross_hop:
                if self.multihop_attn:
                    all_hops = [list(range(dist_masks.shape[1]))] * self.num_heads
                    tag = self.path_embedder.hop_tags(all_hops, self.num_heads)
                    hop_tags = [tag] * len(self.layers)
                else:
                    hop_tags = [self.path_embedder.hop_tags(hs, self.num_heads)
                                for hs in self.per_layer_hop_sets]

        if self.multihop_attn:
            for layer_idx, layer in enumerate(self.layers):
                x = layer(x, mask_source, nm, path_embeddings=path_embeddings,
                          hop_tag=hop_tags[layer_idx])
            aux_loss = x.new_tensor(0.0)
        else:
            adj_power_mats = None
            if self.use_alternating and self.blend_adj_power:
                adj_power_mats = self._build_adj_power_masks(mask_source)

            for layer_idx, layer in enumerate(self.layers):
                per_head_mask = self._build_per_head_mask(
                    mask_source, self.per_layer_hop_sets[layer_idx],
                )

                if self.training and self.dropout > 0:
                    B_cur = per_head_mask.shape[0]
                    random_global = torch.rand(
                        B_cur, self.num_heads, device=x.device,
                    ) < (self.dropout / 2)
                    per_head_mask[random_global] = True

                head_valid_mask = None
                if self._use_cross_hop:
                    N_curr = x.shape[1]
                    eye = torch.eye(N_curr, device=x.device, dtype=torch.bool)
                    non_self = per_head_mask & ~eye.unsqueeze(0).unsqueeze(0)
                    head_valid_mask = non_self.any(dim=-1).any(dim=-1)

                adj_blend_bias_layer = adj_blend_bias
                if self.use_alternating and self.blend_adj_power and adj_power_mats is not None:
                    adj_blend_bias_layer = self._build_adj_blend_bias_from_power(
                        adj_power_mats, self.per_layer_hop_sets[layer_idx]
                    )

                edge_bias_layer = edge_bias
                if self.use_alternating and self.use_edge_bias:
                    edge_bias_layer = self._build_edge_bias(
                        batch, B_orig, N_orig, dense_x.device, layer_idx=layer_idx
                    )

                x = layer(x, per_head_mask, nm, adj_blend_bias=adj_blend_bias_layer,
                         edge_bias=edge_bias_layer, path_embeddings=path_embeddings,
                         head_valid_mask=head_valid_mask,
                         hop_tag=hop_tags[layer_idx])
            aux_loss = x.new_tensor(0.0)

        node_emb = x[nm]

        if self.post_gat is not None:
            edge_attr = getattr(batch, "edge_attr", None) if self.use_edge_features else None
            node_emb = self.post_gat(node_emb, batch.edge_index, edge_attr=edge_attr)

        if self.task_level == "node":
            return self.head(node_emb), node_emb, aux_loss, None

        B = dense_x.shape[0]
        batch_vec_dense = (
            torch.arange(B, device=dense_x.device)
            .unsqueeze(1)
            .expand_as(nm)[nm]
        )
        if self.graph_pool == "attention":
            pooled = self.readout(x, nm)
        elif self.graph_pool == "mean":
            pooled = global_mean_pool(node_emb, batch_vec_dense)
        else:
            pooled = global_add_pool(node_emb, batch_vec_dense)
        return self.head(pooled), node_emb, aux_loss, None

