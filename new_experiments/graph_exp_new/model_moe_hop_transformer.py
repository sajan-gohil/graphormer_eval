"""
Mixture-of-Experts hop-masked multi-head transformer.

Extends the hop-masked transformer (model_hop_masked_transformer.py) by
replacing the *deterministic* hop-to-head allocation with a **learned
gating network**.  Each attention head dynamically selects which hop-
distance masks to attend through via a soft mixture gate, making the hop
selection input-dependent and end-to-end trainable.

Gate design
-----------
For each layer and each head h, a small gating MLP produces a weight vector
g_h ∈ R^K  (K = max_hops) from the *mean node embedding* of the current
graph (optionally enriched with a query summary).  These weights are passed
through softmax (or sparse top-k + softmax) and used to form a weighted
combination of the K per-hop Boolean distance masks.  The resulting soft
mask replaces the hard per-head mask in the standard HopMaskedMHA.

Auxiliary losses
----------------
Two optional regularisation terms encourage diversity and prevent gate
collapse:

1. **Load-balancing loss** (Switch-Transformer style): penalises uneven
   utilisation of hop levels across heads.
2. **Entropy bonus**: encourages each gate distribution to stay diffuse
   rather than collapsing to a single hop.

Pipeline:
    encoder
      -> stack of L MoEHopTransformerLayers
      -> head (graph-level pool + classifier, or per-node)

Datasets: Peptides-func, Peptides-struct, PascalVOC-SP.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool

from models import build_node_encoder


_NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Gating network for hop selection.
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


# ---------------------------------------------------------------------------
# Auxiliary losses for gate regularisation.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# MoE hop-masked multi-head attention.
# ---------------------------------------------------------------------------
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
# Pre-norm transformer encoder layer with MoE hop-masked self-attention.
# ---------------------------------------------------------------------------
class MoEHopTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        max_hops: int,
        ffn_dim: int,
        dropout: float = 0.1,
        top_k: int = 0,
        gate_noise: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gate = HopGate(hidden_dim, max_hops, num_heads, top_k, gate_noise)
        self.attn = MoEHopMaskedMHA(hidden_dim, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, dist_masks, node_mask):
        """
        Args:
            x:           (B, N, d) node features.
            dist_masks:  (B, K, N, N) hop distance masks.
            node_mask:   (B, N) bool.
        Returns:
            x:           (B, N, d) updated node features.
            gate_weights:(B, H, K) for aux loss computation.
        """
        normed = self.norm1(x)
        gate_weights, gate_logits = self.gate(normed, node_mask)
        attn_out = self.attn(normed, gate_weights, dist_masks, node_mask)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x, gate_weights


# ---------------------------------------------------------------------------
# Top-level model.
# ---------------------------------------------------------------------------
class MoEHopTransformerModel(nn.Module):
    """Encoder -> stack of MoE hop-masked transformer layers -> task head.

    Unlike the deterministic HopMaskedTransformerModel, hop selection is
    learned per-head per-layer via a gating network.

    Args:
        hidden_dim:        node / attention channel dim.
        num_heads:         number of attention heads. Must divide hidden_dim.
        ffn_ratio:         FFN inner-dim multiplier.
        num_layers:        number of stacked transformer layers.
        dropout:           shared dropout.
        max_hops:          K — total hop levels in the precomputed dist_masks.
        top_k:             sparse gating: keep only top-k hops per head.
                           0 = dense (full softmax).
        gate_noise:        Gaussian noise std added to gate logits during
                           training.
        balance_coeff:     weight for load-balancing auxiliary loss.
        entropy_coeff:     weight for entropy regularisation auxiliary loss.
        output_dim:        task output channels.
        graph_pool:        "sum" or "mean" pool for graph-level tasks.
        task_level:        "graph" or "node".
        dataset_name:      passed through to ``build_node_encoder``.
        lap_pe_dim:        Laplacian PE width if used.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        num_layers: int = 4,
        dropout: float = 0.2,
        max_hops: int = 40,
        top_k: int = 0,
        gate_noise: float = 0.1,
        balance_coeff: float = 0.01,
        entropy_coeff: float = 0.01,
        output_dim: int = 10,
        graph_pool: str = "sum",
        task_level: str = "graph",
        dataset_name: str = "Peptides-func",
        lap_pe_dim: int = 0,
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
        self.balance_coeff = balance_coeff
        self.entropy_coeff = entropy_coeff

        self.encoder = build_node_encoder(
            hidden_dim=hidden_dim,
            lap_pe_dim=lap_pe_dim,
            dataset_name=dataset_name,
        )

        ffn_dim = hidden_dim * ffn_ratio
        self.layers = nn.ModuleList([
            MoEHopTransformerLayer(
                hidden_dim, num_heads, max_hops, ffn_dim,
                dropout, top_k, gate_noise,
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

    def encode_dense(self, batch):
        lap_pe = getattr(batch, "lap_pe", None)
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr, lap_pe=lap_pe)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, dist_masks, node_masks):
        dense_x, dense_mask = self.encode_dense(batch)
        nm = node_masks if node_masks is not None else dense_mask

        x = dense_x
        all_gate_weights = []
        for layer in self.layers:
            x, gate_weights = layer(x, dist_masks, nm)
            all_gate_weights.append(gate_weights)

        # Compute auxiliary gate loss across all layers.
        aux_loss = x.new_tensor(0.0)
        if self.balance_coeff > 0 or self.entropy_coeff > 0:
            for gw in all_gate_weights:
                aux_loss = aux_loss + compute_gate_aux_loss(
                    gw, self.balance_coeff, self.entropy_coeff,
                )
            aux_loss = aux_loss / len(all_gate_weights)

        node_emb = x[nm]

        if self.task_level == "node":
            return self.head(node_emb), node_emb, aux_loss

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
        return self.head(pooled), node_emb, aux_loss
