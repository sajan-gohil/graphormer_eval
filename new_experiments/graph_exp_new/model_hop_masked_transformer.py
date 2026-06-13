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
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool, GATv2Conv

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
        mode:               "contiguous" | "window" | "single" | "interleaved"
                        | "alternating".
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
    elif mode == "alternating":
        # All restricted heads get all hops of a given parity.
        # Even-indexed layers use even hops [0, 2, 4, ...], odd-indexed
        # layers use odd hops [1, 3, 5, ...].  Since build_head_hop_sets
        # produces a *single* list, the model calls this function twice
        # (once for even parity, once for odd) and assigns per-layer.
        # Here we produce a single shared set that the caller picks from.
        # By convention: "alternating" returns even-parity hops.
        # The model swaps to odd at odd-indexed layers.
        even_hops = [k for k in range(0, K) if k % 2 == 0]
        sets = [even_hops] * H_restricted
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
    """Build two hop-set lists for the 'alternating' mode.

    Returns:
        (even_sets, odd_sets) — one for even-indexed layers, one for odd.
        Each list has length ``num_heads``.
    """
    K = max_hops
    H_restricted = num_heads - num_global_heads
    even_hops = [k for k in range(0, K) if k % 2 == 0]  # [0, 2, 4, ...]
    odd_hops  = [k for k in range(0, K) if k % 2 == 1]  # [1, 3, 5, ...]
    if include_self and 0 not in odd_hops:
        odd_hops = sorted([0] + odd_hops)
    even_sets = [[0, i] for i in even_hops[1:H_restricted+1]] + [None] * num_global_heads
    odd_sets  = [[0, i] for i in odd_hops[1:H_restricted+1]]  + [None] * num_global_heads
    return even_sets, odd_sets


# ---------------------------------------------------------------------------
# Block-diagonal linear: per-head projection with no cross-head mixing.
# ---------------------------------------------------------------------------
class BlockDiagLinear(nn.Module):
    """Block-diagonal linear: each head's slice is projected independently.
    Equivalent to H parallel (in_dim -> out_dim) Linears, stored as a single
    (H, in_dim, out_dim) weight tensor for batched einsum.

    Preserves per-head identity end-to-end through the output projection,
    so the dedicated cross-hop mixer is the only place where hop channels
    interact.

    ``out_dim`` defaults to ``in_dim`` (square, original behaviour). When the
    value head dim differs from the QK head dim (asymmetric V), this maps each
    head's value slab (in_dim = v_head_dim) back to the QK head dim
    (out_dim = head_dim) so the concatenated output stays at hidden_dim.
    """

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
        return out.reshape(B, N, H * self.out_dim) #+ x.view(B, N, H * self.out_dim)


# ---------------------------------------------------------------------------
# Normalization layers. All share a (x, node_mask) signature so the layer can
# call them uniformly; only GraphNorm uses node_mask.
# ---------------------------------------------------------------------------
class LayerNormWrap(nn.Module):
    """nn.LayerNorm with a (x, node_mask) signature (node_mask ignored)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None):
        return self.norm(x)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (LayerNorm without mean-centering)."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, node_mask: Optional[torch.Tensor] = None):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class GraphNorm(nn.Module):
    """Masked GraphNorm over the dense (B, N, d) node tensor.

    Subtracts a learnable fraction (alpha) of the per-graph feature mean, then
    normalises by the per-graph std, with affine (gamma, beta). Statistics are
    computed over the real nodes of each graph only (using node_mask), so
    padding does not contaminate the mean/variance.
    """

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
            m = node_mask.unsqueeze(-1).to(x.dtype)          # (B, N, 1)
        cnt = m.sum(dim=1, keepdim=True).clamp_min(1.0)       # (B, 1, 1)
        mean = (x * m).sum(dim=1, keepdim=True) / cnt         # (B, 1, d)
        out = x - self.alpha * mean
        var = ((out * m) ** 2).sum(dim=1, keepdim=True) / cnt  # (B, 1, d)
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
# Attention-based graph readout (Set-Transformer PMA with a single seed query).
# ---------------------------------------------------------------------------
class AttentionReadout(nn.Module):
    """Pool a set of node embeddings into one graph vector via a learnable
    query attending over the (masked) nodes. Replaces sum/mean pooling."""

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
                 block_diag_out: bool = False, v_head_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # Value head dim defaults to the QK head dim (symmetric, original).
        self.v_head_dim = self.head_dim if v_head_dim is None else v_head_dim
        self.v_dim = num_heads * self.v_head_dim
        self.block_diag_out = block_diag_out
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, self.v_dim)
        if block_diag_out:
            # Per-head value slab (v_head_dim) -> QK head dim, so the
            # concatenated output returns to hidden_dim with no cross-head mix.
            self.out_proj = BlockDiagLinear(num_heads, self.v_head_dim, self.head_dim)
        else:
            self.out_proj = nn.Linear(self.v_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,              # (B, N, d)
        per_head_mask: torch.Tensor,  # (B, H, N, N) bool — True where allowed
        node_mask: torch.Tensor,      # (B, N) bool — True for real nodes
    ) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh, Dv = self.num_heads, self.head_dim, self.v_head_dim

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)   # (B, H, N, Dh)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dv).transpose(1, 2)   # (B, H, N, Dv)

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

        out = torch.matmul(attn, v)                       # (B, H, N, Dv)
        out = out.transpose(1, 2).reshape(B, N, self.v_dim)  # (B, N, H*Dv)
        return self.out_proj(out)                         # (B, N, d)


# ---------------------------------------------------------------------------
# MoE (Mixture-of-Experts) hop gating — optional learned hop selection.
# ---------------------------------------------------------------------------
class HopGate(nn.Module):
    """Per-head gating network that produces soft weights over K hop masks.

    The gate is conditioned on a graph-level summary vector (mean-pool of
    node embeddings).

    Args:
        hidden_dim:  model hidden dimension (input to gate).
        max_hops:    K — number of hop levels.
        num_heads:   H — one gate distribution per head.
        top_k:       if > 0, only the top-k hops per head receive non-zero
                     weight (sparse gating à la Switch/Expert-Choice).
                     0 means dense (full softmax over all K hops).
        gate_noise:  if > 0, add Gaussian noise to logits before softmax
                     during training (encourages exploration).
    """

    def __init__(
        self,
        hidden_dim: int,
        max_hops: int,
        num_heads: int,
        top_k: int = 0,
        gate_noise: float = 0.1,
    ):
        super().__init__()
        self.max_hops = max_hops
        self.num_heads = num_heads
        self.top_k = top_k
        self.gate_noise = gate_noise

        # Small 2-layer MLP: hidden_dim -> H * K logits.
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads * max_hops),
        )

    def forward(
        self,
        x: torch.Tensor,           # (B, N, d)
        node_mask: torch.Tensor,    # (B, N) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return soft gate weights (B, H, K) and raw logits for aux loss."""
        # Graph-level summary: masked mean-pool.
        mask_f = node_mask.float().unsqueeze(-1)                   # (B, N, 1)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)  # (B, d)

        logits = self.gate_mlp(pooled)                             # (B, H*K)
        logits = logits.view(-1, self.num_heads, self.max_hops)    # (B, H, K)

        # Optional noise during training.
        if self.training and self.gate_noise > 0:
            noise = torch.randn_like(logits) * self.gate_noise
            logits = logits + noise

        if self.top_k > 0 and self.top_k < self.max_hops:
            # Sparse gating: keep only top-k logits, set rest to -inf.
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
            sparse_logits = torch.full_like(logits, _NEG_INF)
            sparse_logits.scatter_(-1, topk_idx, topk_vals)
            weights = F.softmax(sparse_logits, dim=-1)
        else:
            # weights = F.softmax(logits, dim=-1)
            weights = F.sigmoid(logits)

        return weights, logits   # (B, H, K), (B, H, K)


def compute_gate_aux_loss(
    gate_weights: torch.Tensor,     # (B, H, K)
    balance_coeff: float = 0.01,
    entropy_coeff: float = 0.01,
) -> torch.Tensor:
    """Compute load-balancing + entropy regularisation.

    Load-balancing (Switch-style):
        L_bal = K * sum_k( f_k * P_k )
        where f_k = fraction of heads choosing hop k as argmax,
              P_k = mean gate weight for hop k across heads.

    Entropy bonus (negative, to maximise):
        L_ent = -mean( H(g_h) )  over all heads and batch elements.
    """
    B, H, K = gate_weights.shape
    loss = gate_weights.new_tensor(0.0)

    if balance_coeff > 0:
        # f_k: fraction of (batch, head) pairs where hop k is argmax.
        assignments = gate_weights.argmax(dim=-1)                  # (B, H)
        counts = torch.zeros(B, K, device=gate_weights.device)
        for k in range(K):
            counts[:, k] = (assignments == k).float().sum(dim=-1)  # (B,)
        f_k = counts / H                                          # (B, K)
        P_k = gate_weights.mean(dim=1)                             # (B, K)
        load_balance = K * (f_k * P_k).sum(dim=-1).mean()
        loss = loss + balance_coeff * load_balance

    if entropy_coeff > 0:
        # Entropy of each gate distribution.
        log_w = torch.log(gate_weights + 1e-8)
        entropy = -(gate_weights * log_w).sum(dim=-1).mean()       # scalar
        loss = loss - entropy_coeff * entropy  # negative = encourage higher entropy

    return loss


class MoEHopMaskedMHA(nn.Module):
    """Multi-head self-attention where each head's hop mask is a *learned
    soft mixture* over all K distance masks, gated per input graph.

    Instead of a hard Boolean per-head mask, we compute:
        soft_mask[b, h, :, :] = sum_k gate_weights[b, h, k] * dist_masks[b, k]
    and use it as an additive bias to the attention logits.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Learnable temperature for the soft mask (per head).
        self.mask_temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(
        self,
        x: torch.Tensor,              # (B, N, d)
        gate_weights: torch.Tensor,    # (B, H, K)
        dist_masks: torch.Tensor,      # (B, K, N, N) float — hop distance masks
        node_mask: torch.Tensor,       # (B, N) bool — True for real nodes
    ) -> torch.Tensor:
        B, N, d = x.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)   # (B, H, N, Dh)
        k = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, N, N)

        # Build soft mask from gated mixture of distance masks.
        # gate_weights: (B, H, K) — dist_masks: (B, K, N, N)
        # soft_mask: (B, H, N, N) = einsum('bhk, bkij -> bhij')
        soft_mask = torch.einsum('bhk,bkij->bhij', gate_weights, dist_masks.float())

        # Apply soft mask as a multiplicative bias via temperature-scaled log.
        # Positions with soft_mask ≈ 0 will get large negative bias → masked out.
        # Positions with soft_mask ≈ 1 will be largely unaffected.
        mask_bias = torch.log(soft_mask.clamp(min=1e-6)) * self.mask_temperature
        scores = scores + mask_bias

        # Padding masks (real-node check).
        key_pad = (~node_mask).unsqueeze(1).unsqueeze(2)      # (B, 1, 1, N)
        query_pad = (~node_mask).unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)
        scores = scores.masked_fill(key_pad, _NEG_INF)
        scores = scores.masked_fill(query_pad, _NEG_INF)

        attn = F.softmax(scores, dim=-1)
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

    With ``use_moe_gating=True`` the deterministic hop-masked MHA is replaced
    by a MoE-gated variant where each head *learns* which hop masks to attend
    through via a soft gating network.  The gate weights are returned for
    auxiliary-loss computation upstream.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        block_diag_out: bool = False,
        dynamic_cross_hop: bool = False,
        norm_type: str = "layer",
        v_head_dim: Optional[int] = None,
        use_moe_gating: bool = False,
        max_hops: int = 40,
        top_k: int = 0,
        gate_noise: float = 0.1,
    ):
        super().__init__()
        self.use_moe_gating = use_moe_gating
        self.norm1 = build_norm(norm_type, hidden_dim)

        if use_moe_gating:
            self.gate = HopGate(hidden_dim, max_hops, num_heads, top_k, gate_noise)
            self.attn = MoEHopMaskedMHA(hidden_dim, num_heads, dropout)
        else:
            self.attn = HopMaskedMHA(
                hidden_dim, num_heads, dropout, block_diag_out=block_diag_out,
                v_head_dim=v_head_dim,
            )
        self.drop1 = nn.Dropout(dropout)

        self.use_cross_hop = dynamic_cross_hop
        if dynamic_cross_hop:
            head_dim = hidden_dim // num_heads
            self.norm_ch = build_norm(norm_type, hidden_dim)
            self.cross_hop = DynamicCrossHopMixer(
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
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

    def forward(self, x, per_head_mask_or_dist_masks, node_mask):
        """Forward pass.

        When ``use_moe_gating=False``:
            per_head_mask_or_dist_masks is (B, H, N, N) bool per-head mask.
            Returns: x  (B, N, d)

        When ``use_moe_gating=True``:
            per_head_mask_or_dist_masks is (B, K, N, N) float dist_masks.
            Returns: (x, gate_weights)  where gate_weights is (B, H, K)
        """
        normed = self.norm1(x, node_mask)
        if self.use_moe_gating:
            gate_weights, _ = self.gate(normed, node_mask)
            attn_out = self.attn(normed, gate_weights, per_head_mask_or_dist_masks, node_mask)
            x = x + self.drop1(attn_out)
        else:
            x = x + self.drop1(self.attn(normed, per_head_mask_or_dist_masks, node_mask))
            gate_weights = None
        if self.use_cross_hop:
            x = x + self.drop_ch(self.cross_hop(self.norm_ch(x, node_mask)))
        x = x + self.drop2(self.ffn(self.norm2(x, node_mask)))
        return (x, gate_weights) if self.use_moe_gating else x


# ---------------------------------------------------------------------------
# Post-transformer GATv2 block (optional).
# ---------------------------------------------------------------------------
class PostGATv2Block(nn.Module):
    """A GATv2 layer applied on the *sparse* PyG graph after the dense
    transformer stack.  Converts the dense (B, N, d) representation back
    to sparse per-node features, runs one or more GATv2Conv layers with
    residual connections, and returns per-node embeddings.

    This re-introduces explicit edge information that the hop-masked
    transformer only sees implicitly through distance masks.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_gat_heads: int = 4,
        num_gat_layers: int = 1,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        assert hidden_dim % num_gat_heads == 0, (
            f"hidden_dim={hidden_dim} must be divisible by num_gat_heads={num_gat_heads}"
        )
        self.num_gat_layers = num_gat_layers
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
                    concat=True,          # output = heads * out_channels = hidden_dim
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                # (total_nodes, d)  sparse
        edge_index: torch.Tensor,       # (2, E)
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x, edge_index, edge_attr=edge_attr)
            out = self.drop(F.elu(out))
            x = norm(x + out)           # residual + norm
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
        norm_type: str = "layer",
        v_head_dim: Optional[int] = None,
        mask_type: str = "shortest_path",
        adj_self_loops: bool = False,
        use_moe_gating: bool = False,
        top_k: int = 0,
        gate_noise: float = 0.1,
        balance_coeff: float = 0.01,
        entropy_coeff: float = 0.01,
        use_virtual_node: bool = False,
        num_post_gat_layers: int = 0,
        num_gat_heads: int = 4,
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
        self.mask_type = mask_type
        self.adj_self_loops = adj_self_loops
        self.use_moe_gating = use_moe_gating
        self.balance_coeff = balance_coeff
        self.entropy_coeff = entropy_coeff
        self.use_virtual_node = use_virtual_node
        self.use_alternating = (hop_mode == "alternating")

        # ----- Per-layer hop sets (alternating mode) or shared -----
        if self.use_alternating:
            even_sets, odd_sets = _build_alternating_hop_sets(
                max_hops=max_hops,
                num_heads=num_heads,
                include_self=True,
                num_global_heads=num_global_heads,
            )
            # Build per-layer list: layer 0 → even, layer 1 → odd, ...
            self.per_layer_hop_sets = [
                even_sets if (i % 2 == 0) else odd_sets
                for i in range(num_layers)
            ]
            self.head_hop_sets = even_sets  # for logging / compat
        else:
            self.head_hop_sets = build_head_hop_sets(
                max_hops=max_hops,
                num_heads=num_heads,
                mode=hop_mode,
                window=hop_window,
                include_self=True,
                num_global_heads=num_global_heads,
            )
            self.per_layer_hop_sets = [self.head_hop_sets] * num_layers

        # Largest hop index any head references — used to bound the number of
        # adjacency powers computed at runtime when mask_type="adj_power".
        used = [
            k
            for hop_sets in self.per_layer_hop_sets
            for s in hop_sets if s is not None
            for k in s
        ]
        self._max_hop_index = max(used) if used else 0

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
                norm_type=norm_type,
                v_head_dim=v_head_dim,
                use_moe_gating=use_moe_gating,
                max_hops=max_hops,
                top_k=top_k,
                gate_noise=gate_noise,
            )
            for _ in range(num_layers)
        ])

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

        if use_virtual_node:
            self.vn_embed = nn.Parameter(
                torch.randn(1, 1, hidden_dim) * hidden_dim ** -0.5
            )

        # Optional post-transformer GATv2 block.
        self.post_gat = None
        if num_post_gat_layers > 0:
            self.post_gat = PostGATv2Block(
                hidden_dim=hidden_dim,
                num_gat_heads=num_gat_heads,
                num_gat_layers=num_post_gat_layers,
                dropout=dropout,
            )

    # ----------------------------------------------------------------
    # Build the (B, H, N, N) boolean per-head mask once per forward.
    # ----------------------------------------------------------------
    def _build_adj_power_masks(self, dist_masks: torch.Tensor) -> torch.Tensor:
        """Build hop masks from powers of the adjacency matrix.

        Returns a *float* (B, K, N, N) tensor instead of bool:
        - slot 0  : identity (self), score = 1.0
        - slot k  : 1.0  for pairs whose shortest path = k  (first reachable hop)
                    1/w  for pairs already reachable at k' < k, where w is the
                        number of distinct walks of length k between them.
        Penalising by walk count suppresses high-multiplicity longer-range
        connections while preserving the exact-shortest-path signal at score 1.
        """
        B, K, N, _ = dist_masks.shape
        eye = torch.eye(N, device=dist_masks.device, dtype=torch.bool)

        # ── float output (non-binary scores) ──────────────────────────────────
        out = dist_masks.new_zeros(B, K, N, N, dtype=torch.float32)
        out[:, 0] = eye.float().unsqueeze(0)               # slot 0 = self, score 1
        if K <= 1:
            return out

        A = dist_masks[:, 1] > 0                      # (B, N, N) bool
        base = (A | eye.unsqueeze(0)) if self.adj_self_loops else A
        base_f = base.float()
        k_max = min(K - 1, self._max_hop_index)

        # cur_f  : float walk-count matrix  =  base^k  (NOT binarised)
        # seen   : cumulative bool mask of all (i,j) pairs already assigned a score.
        #          Pre-seeded with the identity so slot-0 self-edges are "used up".
        cur_f = base_f.clone()
        seen = eye.unsqueeze(0).expand(B, -1, -1).clone()  # (B, N, N) bool
        reach_count = eye.unsqueeze(0).expand(B, -1, -1).float()
        for k in range(1, k_max + 1):
            cur_bool = cur_f > 0
            reach_count = reach_count + cur_bool.float()
            new_mask = cur_bool & ~seen   # shortest-path hop for this pair → 1.0
            old_mask = cur_bool &  seen   # already connected at prior hop  → 1/w
            scores = torch.zeros(B, N, N, device=dist_masks.device, dtype=torch.float32)
            scores[cur_bool] =  1.0 / reach_count[cur_bool]
            scores[new_mask] = 1.0
            # Sentinel-fill positions we won't read to avoid ÷0, then index in
            #safe_f = cur_f.masked_fill(~old_mask, 1.0)
            #scores[old_mask] = (1.0 / safe_f)[old_mask]
            out[:, k] = scores
            seen = seen | cur_bool                         # mark these pairs as seen
            if k < k_max:
                cur_f = torch.bmm(cur_f, base_f)          # base^(k+1), keep as float
        return out

    def _build_per_head_mask(
        self,
        dist_masks: torch.Tensor,
        hop_sets: Optional[List[Optional[List[int]]]] = None,
    ) -> torch.Tensor:
        """Return (B, H, N, N) bool — True where head h is allowed to attend.

        For an unrestricted head (hop_set is None) all entries are True.
        For a restricted head, an entry is True iff the shortest-path
        distance falls in the head's hop set (capped at the runtime K).

        If ``hop_sets`` is provided it overrides ``self.head_hop_sets``
        (used for per-layer alternating assignment).
        """
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
                # Empty — head sees nothing; softmax row will be NaN→zero.
                # This happens only if all hops in the set exceed K_runtime,
                # which is the dataset's actual diameter cap.
                continue
            stacked = dist_masks[:, idx].bool().any(dim=1)  # (B, N, N)
            out[:, h] = stacked
        return out

    # ----------------------------------------------------------------
    # Virtual-node mask augmentation.
    # ----------------------------------------------------------------
    def _augment_dist_masks_vn(self, mask_source: torch.Tensor) -> torch.Tensor:
        """Add a virtual-node row and column to (B, K, N, N) mask tensor.

        The virtual node is placed at position 0.  Its row and column are set
        to 1.0 in **every** hop slot so that it is visible to every attention
        head regardless of its hop assignment.

        Returns: (B, K, N+1, N+1) tensor (same dtype as input).
        """
        B, K, N, _ = mask_source.shape
        aug = mask_source.new_zeros(B, K, N + 1, N + 1)
        aug[:, :, 1:, 1:] = mask_source          # original N×N block
        aug[:, :, 0, :] = 1                       # vn attends to everyone
        aug[:, :, :, 0] = 1                       # everyone attends to vn
        return aug

    def encode_dense(self, batch):
        lap_pe = getattr(batch, "lap_pe", None)
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr, lap_pe=lap_pe)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, dist_masks, node_masks):
        dense_x, dense_mask = self.encode_dense(batch)
        nm = node_masks if node_masks is not None else dense_mask

        mask_source = (
            self._build_adj_power_masks(dist_masks)
            if self.mask_type == "adj_power"
            else dist_masks
        )

        # ── Virtual node: prepend learnable embedding, augment masks ──
        if self.use_virtual_node:
            B, N, d = dense_x.shape
            vn = self.vn_embed.expand(B, 1, d)
            dense_x = torch.cat([vn, dense_x], dim=1)        # (B, N+1, d)
            nm = torch.cat([nm.new_ones(B, 1), nm], dim=1)   # (B, N+1)
            mask_source = self._augment_dist_masks_vn(mask_source)

        x = dense_x
        all_gate_weights = []

        if self.use_moe_gating:
            # MoE path: pass dist_masks directly to each layer's gating.
            for layer in self.layers:
                x, gate_weights = layer(x, mask_source, nm)
                all_gate_weights.append(gate_weights)

            # Compute auxiliary gate loss across all layers.
            aux_loss = x.new_tensor(0.0)
            if self.balance_coeff > 0 or self.entropy_coeff > 0:
                for gw in all_gate_weights:
                    aux_loss = aux_loss + compute_gate_aux_loss(
                        gw, self.balance_coeff, self.entropy_coeff,
                    )
                aux_loss = aux_loss / len(all_gate_weights)
        else:
            # Deterministic hop-mask path.
            # Build per-layer masks (differ only in alternating mode).
            for layer_idx, layer in enumerate(self.layers):
                per_head_mask = self._build_per_head_mask(
                    mask_source, self.per_layer_hop_sets[layer_idx],
                )
                x = layer(x, per_head_mask, nm)
            aux_loss = x.new_tensor(0.0)

        # ── Extract outputs ───────────────────────────────────────────
        if self.use_virtual_node:
            vn_out = x[:, 0, :]                                # (B, d)
            node_emb = x[:, 1:, :][nm[:, 1:]]                 # real nodes only
        else:
            vn_out = None
            node_emb = x[nm]

        # ── Optional post-transformer GATv2 on the sparse graph ───────
        if self.post_gat is not None:
            node_emb = self.post_gat(
                node_emb, batch.edge_index, edge_attr=batch.edge_attr,
            )

        if self.task_level == "node":
            return self.head(node_emb), node_emb, aux_loss

        # Graph-level pooling.
        if self.use_virtual_node:
            pooled = vn_out
        elif self.graph_pool == "attention":
            pooled = self.readout(x, nm)                   # (B, d), masked
        else:
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
        return self.head(pooled), node_emb, aux_loss

