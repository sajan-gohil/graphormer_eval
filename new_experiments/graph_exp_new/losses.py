"""
Proxy–node novelty and diversity losses for proxy generator training.

Primary loss:
    novelty_loss(proxies, node_emb, node_mask)
        → mean cosine similarity over the full (B, M, N) proxy–node matrix.
        Minimising pushes proxy embeddings away from existing node
        embeddings, forcing the generator to discover novel contributions.

    L = task_loss + alpha * novelty_loss(proxies, node_emb, node_mask)

Single-pass: no dual with/without-proxy inference required, so it works
identically in every pipeline (staged, self-novelty, adversarial distillation).

Complementary diversity loss:
    proxy_diversity_loss(proxies)
        → mean off-diagonal inter-proxy cosine similarity.
        Minimising pushes proxies apart from *each other*.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ================================================================
# NOVELTY LOSS (Proxy–Node Cosine Similarity)
# ================================================================

def novelty_loss(
    proxies: torch.Tensor,
    node_emb: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean cosine similarity across the full M×N proxy–node matrix.

    Computes the (B, M, N) pairwise cosine similarity between every proxy
    and every real node, then averages over all valid pairs.  The result
    is in [-1, 1] (cosine similarity's natural range); minimising it pushes
    proxies away from the node embedding manifold.

    Requires only proxy and node embeddings from a single forward pass —
    no dual with/without-proxy inference — so it works identically in every
    pipeline (staged, self-novelty, adversarial distillation).

    Complements ``proxy_diversity_loss`` which pushes proxies apart from
    *each other* but doesn't prevent them from clustering near existing
    nodes.

    Args:
        proxies:   (B, M, d) generated proxy embeddings.
        node_emb:  (B, N, d) node embeddings (dense, may be padded).
        node_mask: (B, N) boolean — True = real node.  Required when
                   ``node_emb`` contains padding.  If None, all positions
                   are treated as real.
        eps:       numerical stability for normalisation.

    Returns:
        Scalar in [-1, 1].
    """
    p_norm = F.normalize(proxies, p=2, dim=-1, eps=eps)    # (B, M, d)
    n_norm = F.normalize(node_emb, p=2, dim=-1, eps=eps)   # (B, N, d)

    # (B, M, N) cosine similarity for every proxy–node pair
    sim = torch.bmm(p_norm, n_norm.transpose(1, 2))

    if node_mask is not None:
        # Zero out padding columns so they don't contribute to the mean
        valid = node_mask.unsqueeze(1).float()          # (B, 1, N)
        sim = sim * valid
        return sim.sum() / (valid.sum() * proxies.shape[1]).clamp(min=1)
    else:
        return sim.mean()


def novelty_stats(
    proxies: torch.Tensor,
    node_emb: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Logging stats for the M×N proxy–node cosine similarity (detached).

    Returns:
        mean_sim:     mean over all valid (proxy, node) pairs.
        max_sim_mean: mean of per-proxy max similarity to any node.
        max_sim_std:  std of per-proxy max similarities.
    """
    p_norm = F.normalize(proxies, p=2, dim=-1, eps=eps)
    n_norm = F.normalize(node_emb, p=2, dim=-1, eps=eps)
    sim = torch.bmm(p_norm, n_norm.transpose(1, 2))  # (B, M, N)

    if node_mask is not None:
        valid = node_mask.unsqueeze(1).float()                # (B, 1, N)
        mean_sim = (sim * valid).sum() / (valid.sum() * proxies.shape[1]).clamp(min=1)
        # For max: mask padding with -inf so it can't win
        sim_for_max = sim.masked_fill(~node_mask.unsqueeze(1), -1e9)
    else:
        mean_sim = sim.mean()
        sim_for_max = sim

    max_per_proxy = sim_for_max.max(dim=-1).values  # (B, M)
    max_sim_mean = max_per_proxy.mean().detach()
    max_sim_std = max_per_proxy.std().detach()

    return mean_sim.detach(), max_sim_mean, max_sim_std


# ================================================================
# PROXY DIVERSITY LOSS
# ================================================================

def proxy_diversity_loss(
    proxies: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalise high cosine similarity among generated proxies.

    Computes the mean off-diagonal pairwise cosine similarity across the batch
    and returns it as a scalar loss.  Minimising this loss pushes proxies apart
    in the embedding space (towards orthogonality).

    Args:
        proxies: (B, M, d) generated proxy embeddings.
        eps:     numerical stability for normalisation.

    Returns:
        Scalar loss in [-1, 1] (typically [0, 1]).  When ``M < 2`` the result
        is 0 — no pairs to repel.
    """
    B, M, d = proxies.shape
    if M < 2:
        return proxies.new_zeros(())

    p_normed = F.normalize(proxies, p=2, dim=-1, eps=eps)   # (B, M, d)
    sim = torch.bmm(p_normed, p_normed.transpose(1, 2))      # (B, M, M)

    # Off-diagonal mask
    off_diag = ~torch.eye(M, dtype=torch.bool, device=proxies.device)
    off_diag_sim = sim[:, off_diag]                          # (B, M*(M-1))
    return off_diag_sim.mean()


# ================================================================
# LOGGING HELPERS
# ================================================================

def inter_proxy_cosine_stats(
    proxies: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and std of off-diagonal pairwise cosine similarity among proxies.

    Aggregated across the batch. Returns ``(mean, std)`` as scalar tensors.
    When ``M < 2`` the result is ``(0, 0)``.
    """
    B, M, d = proxies.shape
    if M < 2:
        zero = proxies.new_zeros(())
        return zero, zero

    p_normed = F.normalize(proxies, p=2, dim=-1, eps=eps)   # (B, M, d)
    sim = torch.bmm(p_normed, p_normed.transpose(1, 2))      # (B, M, M)

    off_diag = ~torch.eye(M, dtype=torch.bool, device=proxies.device)
    off_diag_sim = sim[:, off_diag]                          # (B, M*(M-1))
    return off_diag_sim.mean().detach(), off_diag_sim.std().detach()



