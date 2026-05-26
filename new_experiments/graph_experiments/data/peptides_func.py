"""
Phase 1 Data Pipeline: Peptides-func from LRGB with positional/structural encodings.
"""

import os
import torch
import numpy as np
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import (
    get_laplacian, to_scipy_sparse_matrix, add_self_loops, degree
)
from torch_geometric.data import Data
from scipy.sparse.linalg import eigsh
from scipy import sparse
import warnings


def compute_laplacian_pe(data: Data, k: int = 8) -> torch.Tensor:
    """
    Compute the k smallest non-trivial Laplacian eigenvectors as positional encodings.

    Args:
        data: PyG Data object with edge_index and num_nodes.
        k: Number of eigenvectors to compute.

    Returns:
        Tensor of shape (num_nodes, k). Padded with zeros if graph is too small.
    """
    num_nodes = data.num_nodes

    if num_nodes <= 1:
        return torch.zeros(num_nodes, k)

    # Compute normalized Laplacian
    edge_index, edge_weight = get_laplacian(
        data.edge_index, normalization='sym', num_nodes=num_nodes
    )
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)

    # Number of eigenvectors we can compute (need k+1 to skip trivial)
    max_k = min(k + 1, num_nodes)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Compute smallest eigenvalues/vectors
            eigenvalues, eigenvectors = eigsh(
                L.tocsc(), k=max_k, which='SM', tol=1e-5
            )

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Skip the trivial eigenvector (constant, eigenvalue ≈ 0)
        eigenvectors = eigenvectors[:, 1:]  # shape: (num_nodes, max_k-1)

        # Random sign flip for augmentation (eigenvectors have arbitrary sign)
        sign_flip = torch.bernoulli(
            torch.ones(eigenvectors.shape[1]) * 0.5
        ) * 2 - 1
        eigenvectors = torch.from_numpy(eigenvectors).float() * sign_flip.unsqueeze(0)

        # Pad if fewer eigenvectors than k
        if eigenvectors.shape[1] < k:
            pad = torch.zeros(num_nodes, k - eigenvectors.shape[1])
            eigenvectors = torch.cat([eigenvectors, pad], dim=1)

        return eigenvectors[:, :k]

    except Exception:
        return torch.zeros(num_nodes, k)


def compute_rwse(data: Data, walk_lengths: int = 20) -> torch.Tensor:
    """
    Compute Random Walk Structural Encoding (RWSE).
    Returns the diagonal of A^k (landing probabilities) for k=1..walk_lengths.

    Args:
        data: PyG Data object with edge_index and num_nodes.
        walk_lengths: Maximum random walk length.

    Returns:
        Tensor of shape (num_nodes, walk_lengths).
    """
    num_nodes = data.num_nodes

    if num_nodes <= 1:
        return torch.zeros(num_nodes, walk_lengths)

    # Build random walk transition matrix (D^{-1}A)
    edge_index = data.edge_index
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes).float()
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float('inf')] = 0.0

    # Sparse adjacency with D^{-1} normalization
    edge_weight = deg_inv[row]
    A = sparse.csr_matrix(
        (edge_weight.numpy(), (row.numpy(), col.numpy())),
        shape=(num_nodes, num_nodes)
    )

    # Compute landing probabilities
    pe = torch.zeros(num_nodes, walk_lengths)
    Ak = sparse.eye(num_nodes, format='csr')

    for step in range(walk_lengths):
        Ak = Ak @ A
        pe[:, step] = torch.from_numpy(Ak.diagonal()).float()

    return pe


def precompute_encodings(dataset, lap_k: int = 8, rwse_walk_lengths: int = 20,
                         cache_path: str = None):
    """
    Precompute and attach Laplacian PE and RWSE to each graph in the dataset.
    Optionally cache to disk for faster loading.

    Args:
        dataset: PyG dataset.
        lap_k: Number of Laplacian eigenvectors.
        rwse_walk_lengths: Max random walk length for RWSE.
        cache_path: Path to cache the processed data. If None, no caching.

    Returns:
        List of processed Data objects.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached encodings from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    processed = []
    for i, data in enumerate(dataset):
        # Compute positional encodings
        data.lap_pe = compute_laplacian_pe(data, k=lap_k)
        data.rwse = compute_rwse(data, walk_lengths=rwse_walk_lengths)
        processed.append(data)

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} graphs")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(processed, cache_path)
        print(f"Saved cached encodings to {cache_path}")

    return processed


def get_peptides_func_loaders(data_config):
    """
    Load Peptides-func dataset with positional encodings and return DataLoaders.

    Args:
        data_config: DataConfig instance.

    Returns:
        train_loader, val_loader, test_loader, dataset_info dict
    """
    print("Loading Peptides-func dataset...")
    train_dataset = LRGBDataset(root=data_config.root, name=data_config.dataset_name, split="train")
    val_dataset = LRGBDataset(root=data_config.root, name=data_config.dataset_name, split="val")
    test_dataset = LRGBDataset(root=data_config.root, name=data_config.dataset_name, split="test")

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Compute and cache positional/structural encodings
    cache_dir = os.path.join(data_config.root, "cached_encodings")
    os.makedirs(cache_dir, exist_ok=True)

    print("Computing positional encodings...")
    train_data = precompute_encodings(
        train_dataset, data_config.lap_eigvec_k, data_config.rwse_walk_lengths,
        cache_path=os.path.join(cache_dir, "train_pe.pt")
    )
    val_data = precompute_encodings(
        val_dataset, data_config.lap_eigvec_k, data_config.rwse_walk_lengths,
        cache_path=os.path.join(cache_dir, "val_pe.pt")
    )
    test_data = precompute_encodings(
        test_dataset, data_config.lap_eigvec_k, data_config.rwse_walk_lengths,
        cache_path=os.path.join(cache_dir, "test_pe.pt")
    )

    # Dataset statistics
    num_node_features = train_dataset[0].x.shape[1]
    num_edge_features = train_dataset[0].edge_attr.shape[1]
    num_classes = train_dataset[0].y.shape[0] if train_dataset[0].y.dim() == 1 else train_dataset[0].y.shape[1]

    avg_nodes = np.mean([d.num_nodes for d in train_data])
    avg_edges = np.mean([d.num_edges for d in train_data])

    dataset_info = {
        "num_node_features": num_node_features,
        "num_edge_features": num_edge_features,
        "num_classes": num_classes,
        "avg_nodes": avg_nodes,
        "avg_edges": avg_edges,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
    }

    print(f"  Node features: {num_node_features}, Edge features: {num_edge_features}")
    print(f"  Num classes: {num_classes}")
    print(f"  Avg nodes: {avg_nodes:.1f}, Avg edges: {avg_edges:.1f}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_data, batch_size=data_config.batch_size, shuffle=True,
        num_workers=data_config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=data_config.batch_size, shuffle=False,
        num_workers=data_config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=data_config.batch_size, shuffle=False,
        num_workers=data_config.num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, dataset_info
