"""
Phase 3.3: Mean Pooling Proxy Strategies.

Generate M proxy embeddings per graph by simple aggregation of node embeddings,
with NO additional learning. Insert into a frozen pretrained GPS transformer.

Four strategies:
  - Global mean:             All M proxies = mean(X). Degenerate floor.
  - K-means centroids:       Run k-means on X with k=M; use centroids.
  - Random node sampling:    Randomly select M nodes from X as proxies.
  - Degree-weighted sampling: Sample M nodes with probability ∝ node degree.
"""

import torch
import numpy as np
from torch_geometric.utils import degree


def generate_global_mean_proxies(h_sparse, batch_idx, M, num_graphs, d):
    """
    All M proxies per graph = mean of that graph's node embeddings.
    Degenerate but establishes a floor.

    Args:
        h_sparse: (N_total, d) node embeddings.
        batch_idx: (N_total,) batch assignment.
        M: Number of proxies per graph.
        num_graphs: B.
        d: Hidden dim.

    Returns:
        proxies: (B, M, d)
    """
    from torch_geometric.nn import global_mean_pool
    graph_means = global_mean_pool(h_sparse, batch_idx)  # (B, d)
    # Replicate M times
    proxies = graph_means.unsqueeze(1).expand(-1, M, -1).contiguous()  # (B, M, d)
    return proxies


def generate_kmeans_proxies(h_sparse, batch_idx, M, num_graphs, d, max_iter=50):
    """
    Run k-means on each graph's node embeddings with k=M; use centroids.

    Args:
        h_sparse: (N_total, d) node embeddings.
        batch_idx: (N_total,) batch assignment.
        M: Number of proxies (clusters) per graph.
        num_graphs: B.
        d: Hidden dim.
        max_iter: Max k-means iterations.

    Returns:
        proxies: (B, M, d)
    """
    device = h_sparse.device
    proxies = torch.zeros(num_graphs, M, d, device=device)

    for g in range(num_graphs):
        mask = batch_idx == g
        nodes = h_sparse[mask]  # (N_g, d)
        N_g = nodes.shape[0]

        if N_g <= M:
            # Fewer nodes than clusters: use all nodes, pad with mean
            proxies[g, :N_g] = nodes
            if N_g < M:
                mean_val = nodes.mean(dim=0)
                proxies[g, N_g:] = mean_val.unsqueeze(0).expand(M - N_g, -1)
            continue

        # K-means initialization: random subset
        perm = torch.randperm(N_g, device=device)[:M]
        centroids = nodes[perm].clone()  # (M, d)

        for _ in range(max_iter):
            # Assignment step
            dists = torch.cdist(nodes, centroids)  # (N_g, M)
            assignments = dists.argmin(dim=1)  # (N_g,)

            # Update step
            new_centroids = torch.zeros_like(centroids)
            for k in range(M):
                members = nodes[assignments == k]
                if len(members) > 0:
                    new_centroids[k] = members.mean(dim=0)
                else:
                    # Empty cluster: reinitialize to random node
                    new_centroids[k] = nodes[torch.randint(N_g, (1,))]

            # Check convergence
            shift = (new_centroids - centroids).norm()
            centroids = new_centroids
            if shift < 1e-6:
                break

        proxies[g] = centroids

    return proxies


def generate_random_node_proxies(h_sparse, batch_idx, M, num_graphs, d):
    """
    Randomly select M nodes from each graph as proxies.

    Args:
        h_sparse: (N_total, d) node embeddings.
        batch_idx: (N_total,) batch assignment.
        M: Number of proxies per graph.
        num_graphs: B.
        d: Hidden dim.

    Returns:
        proxies: (B, M, d)
    """
    device = h_sparse.device
    proxies = torch.zeros(num_graphs, M, d, device=device)

    for g in range(num_graphs):
        mask = batch_idx == g
        nodes = h_sparse[mask]  # (N_g, d)
        N_g = nodes.shape[0]

        if N_g <= M:
            proxies[g, :N_g] = nodes
            if N_g < M:
                # Sample with replacement
                idx = torch.randint(N_g, (M - N_g,), device=device)
                proxies[g, N_g:] = nodes[idx]
        else:
            perm = torch.randperm(N_g, device=device)[:M]
            proxies[g] = nodes[perm]

    return proxies


def generate_degree_weighted_proxies(h_sparse, batch_idx, edge_index, M, num_graphs, d):
    """
    Sample M nodes per graph with probability proportional to node degree.
    High-degree nodes are structural hubs — they may make good proxies.

    Args:
        h_sparse: (N_total, d) node embeddings.
        batch_idx: (N_total,) batch assignment.
        edge_index: (2, E) edge indices.
        M: Number of proxies per graph.
        num_graphs: B.
        d: Hidden dim.

    Returns:
        proxies: (B, M, d)
    """
    device = h_sparse.device
    N_total = h_sparse.shape[0]
    proxies = torch.zeros(num_graphs, M, d, device=device)

    # Compute degree for all nodes
    deg = degree(edge_index[0], num_nodes=N_total).float()
    # Add small epsilon to avoid zero probability
    deg = deg + 1e-6

    for g in range(num_graphs):
        mask = batch_idx == g
        nodes = h_sparse[mask]  # (N_g, d)
        node_deg = deg[mask]  # (N_g,)
        N_g = nodes.shape[0]

        if N_g <= M:
            proxies[g, :N_g] = nodes
            if N_g < M:
                probs = node_deg / node_deg.sum()
                idx = torch.multinomial(probs, M - N_g, replacement=True)
                proxies[g, N_g:] = nodes[idx]
        else:
            probs = node_deg / node_deg.sum()
            idx = torch.multinomial(probs, M, replacement=False)
            proxies[g] = nodes[idx]

    return proxies


# Registry for easy access
PROXY_STRATEGIES = {
    'global_mean': generate_global_mean_proxies,
    'kmeans': generate_kmeans_proxies,
    'random_nodes': generate_random_node_proxies,
    # degree_weighted has a different signature (needs edge_index)
}
