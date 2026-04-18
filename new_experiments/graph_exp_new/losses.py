"""
Self-novelty losses for proxy generator training.

Drops in as a pluggable replacement for reconstruction / static-target losses in
any pipeline: call ``novelty_loss(task_loss, logits_with, logits_without, ...)``
with a with-proxies forward and a (detached) without-proxies forward of the
same model, and add the returned total to the optimizer step.

Formulation (from NOVELTY_LOSS_PROPOSAL.md):
    output_novelty = || sigmoid(logits_with/T) - sigmoid(logits_without/T) ||_2 / sqrt(C)
    node_novelty   = mean over original nodes of (1 - cos(emb_with, emb_without))

    output_penalty = 1 - output_novelty           # in [0, 1]
    node_penalty   = 2 - node_novelty             # in [0, 2]

    L = task_loss + alpha * output_penalty + alpha_node * node_penalty

Sigmoid + L2 is the multi-label-correct analog of softmax + JSD: outputs are
independent per-class Bernoulli probabilities, not a simplex. The linear
penalty form (not hinge) always has a gradient pushing novelty upward; task
loss balances it against the "proxies must be useful" objective.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ================================================================
# NOVELTY PRIMITIVES
# ================================================================

def compute_output_novelty(
    logits_with: torch.Tensor,
    logits_without: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-sample L2 between sigmoid outputs, normalized by sqrt(C), averaged.

    Args:
        logits_with:    (B, C) logits from model with proxies.
        logits_without: (B, C) logits from model without proxies (typically detached).
        temperature:    sigmoid temperature. 1.0 = unscaled; >1 softens, <1 sharpens.

    Returns:
        Scalar in [0, 1].
    """
    p_with = torch.sigmoid(logits_with / temperature)
    p_without = torch.sigmoid(logits_without / temperature)
    diff = p_with - p_without                       # (B, C)
    C = diff.shape[-1]
    per_sample = torch.norm(diff, p=2, dim=-1) / (C ** 0.5)  # (B,)
    return per_sample.mean()


def compute_node_novelty(
    node_emb_with: torch.Tensor,
    node_emb_without: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean cosine distance (1 - cos_sim) over original-node positions.

    Accepts either dense ``(B, N, d)`` tensors (with ``mask``) or flat
    ``(total_N, d)`` tensors (mask ignored). Proxy-token positions must not
    appear in the inputs — pass the original-N slice only.

    Returns scalar in [0, 2].
    """
    if node_emb_with.dim() == 3:
        if mask is None:
            raise ValueError("compute_node_novelty: dense input requires a mask")
        a = node_emb_with[mask]
        b = node_emb_without[mask]
    else:
        a = node_emb_with
        b = node_emb_without

    cos_sim = F.cosine_similarity(a, b, dim=-1, eps=eps)
    return (1.0 - cos_sim).mean()


# ================================================================
# COMBINED LOSS
# ================================================================

def novelty_loss(
    task_loss: torch.Tensor,
    logits_with: torch.Tensor,
    logits_without: torch.Tensor,
    node_emb_with: Optional[torch.Tensor] = None,
    node_emb_without: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    alpha_node: float = 1.0,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Combine a precomputed task loss with the self-novelty penalties.

    Node-level term is included only when both ``node_emb_with`` and
    ``node_emb_without`` are provided.

    Args:
        task_loss:         scalar tensor (e.g. BCEWithLogitsLoss output).
        logits_with:       (B, C) tensor; carries gradient.
        logits_without:    (B, C) tensor; pass a detached value.
        node_emb_with:     (B, N, d) or (total_N, d) tensor; carries gradient.
        node_emb_without:  same shape as ``node_emb_with``; detached.
        mask:              (B, N) boolean mask if dense node embeddings are used.
        alpha:             weight on output_penalty (default 1.0).
        alpha_node:        weight on node_penalty (default 1.0).
        temperature:       sigmoid temperature for output novelty.

    Returns:
        (total_loss, metrics) where ``metrics`` is a dict of detached floats
        for logging: ``task_loss``, ``output_novelty``, ``output_penalty``,
        and, when node embeddings are supplied, ``node_novelty``, ``node_penalty``.
    """
    output_novelty = compute_output_novelty(logits_with, logits_without, temperature)
    output_penalty = 1.0 - output_novelty

    total = task_loss + alpha * output_penalty

    metrics = {
        "task_loss": task_loss.detach(),
        "output_novelty": output_novelty.detach(),
        "output_penalty": output_penalty.detach(),
    }

    if node_emb_with is not None and node_emb_without is not None:
        node_novelty = compute_node_novelty(
            node_emb_with, node_emb_without, mask=mask
        )
        node_penalty = 2.0 - node_novelty
        total = total + alpha_node * node_penalty
        metrics["node_novelty"] = node_novelty.detach()
        metrics["node_penalty"] = node_penalty.detach()

    return total, metrics


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
