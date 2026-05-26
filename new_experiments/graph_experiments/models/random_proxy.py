"""
Phase 3.4: Random Proxy Embeddings.

Generate M random vectors from N(μ_X, σ_X), matching the node embedding
distribution. Insert into the frozen pretrained transformer and evaluate.

This ablates whether content matters at all, or if any additional tokens
help via regularization. Run 10 random seeds and report mean ± std of AP.
"""

import torch
from torch_geometric.nn import global_mean_pool


def generate_random_proxies(h_sparse, batch_idx, M, num_graphs, d, seed=None):
    """
    Generate M random proxy embeddings per graph, sampled from N(μ_X, σ_X)
    where μ_X and σ_X are computed from the node embeddings of each graph.

    Args:
        h_sparse: (N_total, d) node embeddings.
        batch_idx: (N_total,) batch assignment.
        M: Number of proxies per graph.
        num_graphs: B.
        d: Hidden dim.
        seed: Random seed for reproducibility. If None, uses current RNG state.

    Returns:
        proxies: (B, M, d)
    """
    device = h_sparse.device

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None

    proxies = torch.zeros(num_graphs, M, d, device=device)

    for g in range(num_graphs):
        mask = batch_idx == g
        nodes = h_sparse[mask]  # (N_g, d)

        mu = nodes.mean(dim=0)    # (d,)
        sigma = nodes.std(dim=0)  # (d,)
        # Clamp sigma to avoid degenerate case
        sigma = sigma.clamp(min=1e-6)

        # Sample from N(μ, σ) per dimension
        if generator is not None:
            noise = torch.randn(M, d, device=device, generator=generator)
        else:
            noise = torch.randn(M, d, device=device)

        proxies[g] = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise

    return proxies


def generate_random_proxies_global(h_sparse, batch_idx, M, num_graphs, d, seed=None):
    """
    Alternative: sample from the global N(μ_X, σ_X) computed across ALL nodes
    in the batch rather than per-graph. Simpler but less adapted.

    Returns:
        proxies: (B, M, d)
    """
    device = h_sparse.device

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None

    mu = h_sparse.mean(dim=0)    # (d,)
    sigma = h_sparse.std(dim=0)  # (d,)
    sigma = sigma.clamp(min=1e-6)

    if generator is not None:
        noise = torch.randn(num_graphs, M, d, device=device, generator=generator)
    else:
        noise = torch.randn(num_graphs, M, d, device=device)

    proxies = mu.unsqueeze(0).unsqueeze(0) + sigma.unsqueeze(0).unsqueeze(0) * noise
    return proxies
