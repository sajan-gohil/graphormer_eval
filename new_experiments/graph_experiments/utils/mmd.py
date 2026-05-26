"""
MMD (Maximum Mean Discrepancy) utilities.
Used in Phase 2 for proxy optimization regularization.
"""

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
