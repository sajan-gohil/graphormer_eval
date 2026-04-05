"""
MMD (Maximum Mean Discrepancy) utilities.
Used in Phase 2 for proxy optimization regularization.
"""

from typing import Tuple, Union

import torch

def median_heuristic(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the median heuristic for the Gaussian kernel bandwidth.
    σ = median of pairwise distances.

    Args:
        X: (N, d) tensor of samples.

    Returns:
        σ: scalar bandwidth.
    """
    with torch.no_grad():
        dists = torch.cdist(X, X, p=2)
        # Get upper triangle (exclude diagonal)
        mask = torch.triu(torch.ones_like(dists, dtype=torch.bool), diagonal=1)
        pairwise_dists = dists[mask]
        if pairwise_dists.numel() == 0:
            return torch.tensor(1.0, device=X.device)
        sigma = torch.median(pairwise_dists)
        return torch.clamp(sigma, min=1e-5)


def gaussian_kernel(X: torch.Tensor, Y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Compute Gaussian (RBF) kernel matrix.
    k(x, y) = exp(-||x - y||² / (2σ²))

    Args:
        X: (N, d)
        Y: (M, d)
        sigma: scalar bandwidth

    Returns:
        K: (N, M) kernel matrix
    """
    dists_sq = torch.cdist(X, Y, p=2).pow(2)
    return torch.exp(-dists_sq / (2 * sigma ** 2))


def mmd_squared(P: torch.Tensor, Q: torch.Tensor, sigma: torch.Tensor = None) -> torch.Tensor:
    """
    Compute the squared MMD between two sets of samples using a Gaussian kernel.
    MMD²(P, Q) = E[k(p,p')] + E[k(q,q')] - 2E[k(p,q)]

    If sigma is None, uses the median heuristic on the concatenated samples.

    Args:
        P: (N, d) first set of samples (e.g., proxy embeddings).
        Q: (M, d) second set of samples (e.g., node embeddings).
        sigma: Gaussian kernel bandwidth. If None, computed via median heuristic.

    Returns:
        MMD² value (scalar tensor, differentiable w.r.t. P).
    """
    if sigma is None:
        combined = torch.cat([P, Q], dim=0)
        sigma = median_heuristic(combined)

    K_pp = gaussian_kernel(P, P, sigma)
    K_qq = gaussian_kernel(Q, Q, sigma)
    K_pq = gaussian_kernel(P, Q, sigma)

    n = P.shape[0]
    m = Q.shape[0]

    # Unbiased estimator (exclude diagonal for same-sample terms)
    if n > 1:
        term_pp = (K_pp.sum() - K_pp.diagonal().sum()) / (n * (n - 1))
    else:
        term_pp = torch.tensor(0.0, device=P.device)

    if m > 1:
        term_qq = (K_qq.sum() - K_qq.diagonal().sum()) / (m * (m - 1))
    else:
        term_qq = torch.tensor(0.0, device=Q.device)

    term_pq = K_pq.sum() / (n * m)

    return term_pp + term_qq - 2 * term_pq


def _proxy_moments(proxy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-graph first and second moments for proxy sets.

    Args:
        proxy: (B, M, d) proxy embeddings for B graphs.

    Returns:
        means: (B, d) per-graph means.
        variances: (B, d) per-graph variances (biased estimator).
    """
    if proxy.ndim != 3:
        raise ValueError(f"Expected proxy with shape (B, M, d), got {tuple(proxy.shape)}")
    means = proxy.mean(dim=1)
    variances = proxy.var(dim=1, unbiased=False)
    return means, variances


def cross_sample_moment_loss(proxy: torch.Tensor) -> torch.Tensor:
    """
    Intra-batch cross-sample moment matching loss.

    For each graph i in a batch, this penalizes deviation of its proxy mean/variance
    from batch-level average moments:
        ||mu_i - mu_batch||^2 + ||var_i - var_batch||^2

    Args:
        proxy: (B, M, d) proxy embeddings.

    Returns:
        Per-graph loss tensor of shape (B,).
    """
    means, variances = _proxy_moments(proxy)
    batch_mean = means.mean(dim=0, keepdim=True)
    batch_var = variances.mean(dim=0, keepdim=True)

    mean_term = (means - batch_mean).pow(2).mean(dim=1)
    var_term = (variances - batch_var).pow(2).mean(dim=1)
    return mean_term + var_term


def prior_moment_loss(
    proxy: torch.Tensor,
    target_variance: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """
    Moment-based regularization toward a fixed Gaussian prior.

    Penalizes each graph's proxies with:
        ||mean(proxy_i)||^2 + ||var(proxy_i) - target_variance||^2

    Args:
        proxy: (B, M, d) proxy embeddings.
        target_variance: scalar (or broadcastable tensor) target variance.

    Returns:
        Per-graph loss tensor of shape (B,).
    """
    means, variances = _proxy_moments(proxy)
    target_var = torch.as_tensor(target_variance, device=proxy.device, dtype=proxy.dtype)
    if target_var.ndim == 0:
        target_var = target_var.view(1, 1)
    elif target_var.ndim == 1:
        if target_var.numel() not in (1, proxy.size(-1)):
            raise ValueError(
                "target_variance vector must have size 1 or match proxy feature dimension"
            )
        target_var = target_var.view(1, -1)
    else:
        raise ValueError("target_variance must be a scalar or 1D tensor")

    mean_term = means.pow(2).mean(dim=1)
    var_term = (variances - target_var).pow(2).mean(dim=1)
    return mean_term + var_term
