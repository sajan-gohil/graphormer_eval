"""
K-hop attention-weighted aggregation for structure-aware node features.

For each target node v and each hop k, computes attention-weighted
aggregation over the set of source nodes at exactly distance k from v
using the precomputed boolean distance masks (the same dist_masks GRED
uses):

    agg[b, v, k, :] = sum_{u : d(v,u) == k} attn[b, v, u] * V[b, u, :]

The attention weights ``attn[b, v, u]`` come from a single shared
multi-head scaled-dot-product attention computation, with the per-hop
softmax restricted to (v, u) pairs at exactly distance k. Q/K/V are
shared across hops to keep parameter count low and to encode the prior
that "u is informative to v" is a structural property independent of
which hop u is at.

A learnable hop embedding is added to the per-hop output so downstream
sequence models (e.g. an LRU over hops) can distinguish the hop axis.

Used by:
    model_khop_concat.py     (Idea C: concat-then-MLP baseline)
    model_weighted_gred.py   (Weighted GRED replacing uniform per-hop sum)

Shape conventions (matching ``models.py`` and ``data.py``):
    h:           (B, N, d)            dense node embeddings
    dist_masks:  (B, K, N, N) float    1.0 where d(v, u) == k, else 0
    node_mask:   (B, N) bool           True for real nodes
    output:      (B, N, K, d)          per-hop aggregated features
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_NEG_INF = float("-inf")


class KHopAttentionAggregator(nn.Module):
    """Attention-weighted per-hop aggregation over a (B, K, N, N) hop mask.

    Computes shared multi-head Q/K/V projections, then for each hop ``k``:
      1. Mask the attention scores so only (v, u) pairs with d(v, u) == k
         contribute (other pairs set to -inf before softmax).
      2. Per-target softmax over hop-k sources.
      3. Multiply with value features and sum.

    Targets that have no source at hop k yield an all-(-inf) row, which
    softmax turns into NaN; we replace with zeros (the node is correctly
    un-updated for that hop).

    Args:
        hidden_dim: input/output channel count d.
        num_heads:  number of attention heads. Must divide hidden_dim.
        dropout:    dropout on attention weights.
        add_hop_embedding: if True, add a learnable per-hop embedding to
            the output of each hop. Requires ``max_hops`` >= the K dimension
            actually used at runtime.
        max_hops:   upper bound on K for sizing the hop-embedding table.
        residual_self: if True, the hop-0 output (which equals
            attention-weighted self-aggregation) is forced to a clean copy
            of the input. Hop-0 means d(v, u) == 0, i.e. u == v, so the
            mask is the identity and the attention output is just V[v].
            Setting this True bypasses any gating/dropout effects on the
            self-feature, which is useful as a residual signal.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        add_hop_embedding: bool = True,
        max_hops: int = 40,
        residual_self: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.residual_self = residual_self

        self.norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.add_hop_embedding = add_hop_embedding
        if add_hop_embedding:
            self.hop_embedding = nn.Embedding(max_hops, hidden_dim)
            nn.init.normal_(self.hop_embedding.weight, std=0.02)
        self.max_hops = max_hops

    def forward(
        self,
        h: torch.Tensor,
        dist_masks: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h:           (B, N, d) dense node embeddings.
            dist_masks:  (B, K, N, N) float — 1.0 at (v, u) iff d(v, u)==k.
            node_mask:   (B, N) bool — True for real nodes (used to zero out
                         padding rows/cols in attention).
        Returns:
            agg: (B, N, K, d) per-hop aggregated features (with hop embedding
                 added if configured).
        """
        B, N, d = h.shape
        K = dist_masks.shape[1]
        H = self.num_heads
        Dh = self.head_dim

        normed = self.norm(h)
        q = self.q_proj(normed).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        k = self.k_proj(normed).view(B, N, H, Dh).transpose(1, 2)
        v = self.v_proj(normed).view(B, N, H, Dh).transpose(1, 2)

        # Raw attention scores shared across hops: (B, H, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)

        if node_mask is not None:
            # Mask out attention to/from padded nodes.
            key_pad = (~node_mask).unsqueeze(1).unsqueeze(2)   # (B, 1, 1, N)
            query_pad = (~node_mask).unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            scores = scores.masked_fill(key_pad, _NEG_INF)
            scores = scores.masked_fill(query_pad, _NEG_INF)

        # Per-hop output buffer: (B, N, K, d)
        # We build it by iterating over hops to keep memory linear in K rather
        # than allocating a (B, H, K, N, N) attention tensor.
        out = h.new_zeros(B, N, K, d)

        for k_idx in range(K):
            mask_k = dist_masks[:, k_idx]  # (B, N, N), 1 at hop-k pairs
            # Restrict scores to hop-k (v, u) pairs.
            hop_scores = scores.masked_fill(
                mask_k.unsqueeze(1) == 0, _NEG_INF
            )  # (B, H, N, N)

            # Softmax — rows with no hop-k sources become NaN; zero them out.
            attn = F.softmax(hop_scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.attn_drop(attn)

            agg_k = torch.matmul(attn, v)  # (B, H, N, Dh)
            agg_k = agg_k.transpose(1, 2).reshape(B, N, d)
            agg_k = self.out_proj(agg_k)

            if self.residual_self and k_idx == 0:
                # d(v, u)==0 means u==v, so this is just self-aggregation.
                # Replace with a clean copy of the input.
                agg_k = h

            if self.add_hop_embedding and k_idx < self.max_hops:
                hop_id = torch.tensor(k_idx, device=h.device)
                agg_k = agg_k + self.hop_embedding(hop_id)

            out[:, :, k_idx, :] = agg_k

        return out  # (B, N, K, d)


def flatten_per_hop(agg: torch.Tensor) -> torch.Tensor:
    """Reshape (B, N, K, d) -> (B, N, K*d) by concatenating the hop axis."""
    B, N, K, d = agg.shape
    return agg.reshape(B, N, K * d)
