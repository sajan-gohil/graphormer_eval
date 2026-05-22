# data.py
import os
import pickle
import numpy as np
import torch
import torch_geometric
from torch_geometric.datasets import LRGBDataset, GNNBenchmarkDataset, ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse.linalg import eigsh
from functools import partial
from multiprocessing import Pool


# ================================================================
# GRAPH DATASET REGISTRY (LRGB + GNNBenchmark + ZINC)
# ================================================================
#
# Single source of truth for everything that varies between graph datasets.
# Driven entirely by the --dataset CLI flag in the training scripts. Adding a
# new dataset means adding one entry here.
#
# Fields:
#   output_dim    : number of output channels for the task head.
#   task_type     : "multi_label" | "regression" | "multiclass".
#   level         : "graph" (one prediction per graph) | "node" (one per node).
#   node_encoder  : "atom_categorical" (peptides-style integer features) |
#                   "linear" (continuous float features projected via Linear).
#   node_feat_dim : in_dim hint for the linear node encoder. Ignored for
#                   "atom_categorical".
#   metric_name   : "macro_ap" | "mae" | "node_f1_macro" | "accuracy".
#   source        : "lrgb" | "gnn_benchmark" | "zinc".
#   pyg_name      : name string for the underlying PyG dataset (optional).
#   subset        : ZINC-specific flag (subset=True → ZINC-12k).
#
# Note: PascalVOC-SP graphs are large (~480 superpixels) and the full distance-
# mask cache scales as O(N^2 K). Default max_hops in get_loaders should be
# lowered when using VOC.

GRAPH_DATASETS = {
    "Peptides-func": {
        "output_dim": 10,
        "task_type": "multi_label",
        "level": "graph",
        "node_encoder": "atom_categorical",
        "node_feat_dim": 9,
        "metric_name": "macro_ap",
        "source": "lrgb",
    },
    "Peptides-struct": {
        "output_dim": 11,
        "task_type": "regression",
        "level": "graph",
        "node_encoder": "atom_categorical",
        "node_feat_dim": 9,
        "metric_name": "mae",
        "source": "lrgb",
    },
    "PascalVOC-SP": {
        "output_dim": 21,
        "task_type": "multiclass",
        "level": "node",
        "node_encoder": "linear",
        "node_feat_dim": 14,
        "metric_name": "node_f1_macro",
        "source": "lrgb",
    },
    "MNIST": {
        "output_dim": "auto",
        "task_type": "multiclass",
        "level": "graph",
        "node_encoder": "linear",
        "node_feat_dim": "auto",
        "metric_name": "accuracy",
        "source": "gnn_benchmark",
        "pyg_name": "MNIST",
    },
    "CIFAR10": {
        "output_dim": "auto",
        "task_type": "multiclass",
        "level": "graph",
        "node_encoder": "linear",
        "node_feat_dim": "auto",
        "metric_name": "accuracy",
        "source": "gnn_benchmark",
        "pyg_name": "CIFAR10",
    },
    "PATTERN": {
        "output_dim": "auto",
        "task_type": "multiclass",
        "level": "node",
        "node_encoder": "linear",
        "node_feat_dim": "auto",
        "metric_name": "accuracy",
        "source": "gnn_benchmark",
        "pyg_name": "PATTERN",
    },
    "CLUSTER": {
        "output_dim": "auto",
        "task_type": "multiclass",
        "level": "node",
        "node_encoder": "linear",
        "node_feat_dim": "auto",
        "metric_name": "accuracy",
        "source": "gnn_benchmark",
        "pyg_name": "CLUSTER",
    },
    "ZINC12k": {
        "output_dim": "auto",
        "task_type": "regression",
        "level": "graph",
        "node_encoder": "linear",
        "node_feat_dim": "auto",
        "metric_name": "mae",
        "source": "zinc",
        "pyg_name": "ZINC",
        "subset": True,
    },
}

DATASET_ALIASES = {
    "mnist": "MNIST",
    "cifar10": "CIFAR10",
    "pattern": "PATTERN",
    "cluster": "CLUSTER",
    "zinc12k": "ZINC12k",
}

DATASET_CHOICES = sorted(set(GRAPH_DATASETS.keys()) | set(DATASET_ALIASES.keys()))


def canonicalize_dataset_name(name: str) -> str:
    """Return the canonical dataset name (resolving aliases)."""
    if name in GRAPH_DATASETS:
        return name
    key = name.lower()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    alias_list = sorted(DATASET_ALIASES.keys())
    raise ValueError(
        f"Unknown dataset '{name}'. Available: {sorted(GRAPH_DATASETS.keys())} "
        f"(aliases: {alias_list})."
    )


def get_dataset_info(name):
    """Return the registry entry for ``name``.

    Raises ``ValueError`` if the dataset name is not registered.
    """
    canonical_name = canonicalize_dataset_name(name)
    # Return a shallow copy so callers can't accidentally mutate the registry.
    info = dict(GRAPH_DATASETS[canonical_name])
    info["name"] = canonical_name
    return info


def _infer_output_dim(info, dataset):
    if info.get("output_dim") not in ("auto", None):
        return info["output_dim"]
    if hasattr(dataset, "num_classes") and dataset.num_classes not in (None, -1, 0):
        return int(dataset.num_classes)
    if hasattr(dataset, "num_targets") and dataset.num_targets not in (None, 0):
        return int(dataset.num_targets)
    if hasattr(dataset, "num_tasks") and dataset.num_tasks not in (None, 0):
        return int(dataset.num_tasks)
    if hasattr(dataset, "data") and getattr(dataset.data, "y", None) is not None:
        y = dataset.data.y
        if y.numel() == 0:
            return 1
        if info.get("task_type") == "multiclass":
            y_flat = y.view(-1)
            if torch.is_tensor(y_flat):
                y_flat = y_flat[y_flat >= 0]
                if y_flat.numel() == 0:
                    return 1
                y_np = y_flat.cpu().numpy()
            else:
                y_np = np.asarray(y_flat)
                y_np = y_np[y_np >= 0]
            return int(np.unique(y_np).size) if y_np.size else 1
        return int(y.size(-1)) if y.dim() > 1 else 1
    return info.get("output_dim", 1)


def _infer_node_feat_dim(dataset):
    if hasattr(dataset, "num_node_features") and dataset.num_node_features is not None:
        return int(dataset.num_node_features)
    if hasattr(dataset, "num_features") and dataset.num_features is not None:
        return int(dataset.num_features)
    if hasattr(dataset, "data") and getattr(dataset.data, "x", None) is not None:
        return int(dataset.data.x.size(-1))
    try:
        # Some datasets disallow direct indexing or have empty splits.
        sample = dataset[0]
        if hasattr(sample, "x") and sample.x is not None:
            return int(sample.x.size(-1))
    except (IndexError, AttributeError, TypeError, RuntimeError):
        pass
    return None


# ================================================================
# DISTANCE MASK PREPROCESSING (for GRED backbone)
# ================================================================

def _compute_dist_mask_single(adj, max_hops=40):
    """Compute boolean distance masks for a single graph's adjacency matrix.
    Returns: (K, N, N) boolean array where K = min(diameter+1, max_hops)."""
    dist = floyd_warshall(adj, directed=False, unweighted=True)
    dist = np.where(np.isfinite(dist), dist, -1).astype(np.int32)
    actual_max = int(dist.max())
    K = min(actual_max + 1, max_hops)
    dist_mask = np.stack([(dist == k) for k in range(K)]).astype(np.bool_)
    return dist_mask  # (K, N, N)


def precompute_distance_masks(dataset, cache_path, max_hops=40, num_workers=8):
    """
    Precompute Floyd-Warshall distance masks for all graphs in a PyG dataset.
    Caches to disk. Returns list of (K_i, N_i, N_i) boolean arrays.
    """
    if os.path.exists(cache_path):
        print(f"  Loading cached distance masks from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  Computing distance masks for {len(dataset)} graphs (max_hops={max_hops})...", flush=True)
    # Build adjacency matrices
    adjs = []
    for g in dataset:
        num_nodes = g.x.shape[0]
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        ei = g.edge_index.numpy()
        adj[ei[0], ei[1]] = 1.0
        adjs.append(adj)

    compute_fn = partial(_compute_dist_mask_single, max_hops=max_hops)
    with Pool(min(num_workers, len(adjs))) as p:
        dist_masks = p.map(compute_fn, adjs)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dist_masks, f)
    print(f"  Saved distance masks to {cache_path}", flush=True)
    return dist_masks


class DistMaskDataset(torch.utils.data.Dataset):
    """Wraps a PyG dataset with precomputed distance masks."""
    def __init__(self, pyg_dataset, dist_masks):
        assert len(pyg_dataset) == len(dist_masks)
        self.pyg_dataset = pyg_dataset
        self.dist_masks = dist_masks

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        return self.pyg_dataset[idx], self.dist_masks[idx]


def collate_with_dist_masks(batch, max_hops=40):
    """
    Collate function for DistMaskDataset.
    Pads graphs and distance masks to the same max_N within the batch.

    Returns:
        pyg_batch: batched PyG Data object (standard)
        dist_masks_padded: (B, max_hops, max_N, max_N) float tensor
        node_masks: (B, max_N) boolean tensor
    """
    graphs, dist_masks_list = zip(*batch)
    pyg_batch = torch_geometric.data.Batch.from_data_list(list(graphs))

    B = len(graphs)
    max_N = max(g.x.shape[0] for g in graphs)

    # Pad distance masks to (B, max_hops, max_N, max_N)
    dm_padded = np.zeros((B, max_hops, max_N, max_N), dtype=np.float32)
    node_masks = np.zeros((B, max_N), dtype=np.bool_)

    for i, (g, dm) in enumerate(zip(graphs, dist_masks_list)):
        n = g.x.shape[0]
        K = dm.shape[0]  # actual number of hop levels for this graph
        K_use = min(K, max_hops)
        dm_padded[i, :K_use, :n, :n] = dm[:K_use].astype(np.float32)
        node_masks[i, :n] = True

    return (
        pyg_batch,
        torch.from_numpy(dm_padded),
        torch.from_numpy(node_masks),
    )


# ================================================================
# LAPLACIAN POSITIONAL ENCODING
# ================================================================

class AddLaplacianPE:
    """Transform that computes Laplacian eigenvector positional encodings.

    Stores ``data.lap_pe`` of shape ``(num_nodes, k)`` on each graph.
    Uses the *k* smallest non-trivial eigenvectors of the symmetric
    normalized graph Laplacian. Sign ambiguity is resolved deterministically
    per eigenvector to keep train/val/test evaluation stable.
    """

    def __init__(self, k: int = 8):
        self.k = k

    def __call__(self, data):
        num_nodes = data.x.size(0)
        k = self.k

        # Build symmetric normalized Laplacian
        edge_index, edge_weight = get_laplacian(
            data.edge_index, normalization="sym", num_nodes=num_nodes,
        )
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=num_nodes)

        # Compute k+1 smallest eigenpairs then drop the trivial (constant) one
        num_eig = min(k + 1, num_nodes)
        try:
            eigenvalues, eigenvectors = eigsh(
                L.tocsc(), k=num_eig, which="SM", return_eigenvectors=True,
            )
            # Sort by eigenvalue magnitude (eigsh doesn't guarantee order)
            idx = eigenvalues.argsort()
            eigenvectors = eigenvectors[:, idx]
            # Drop the first (trivial) eigenvector
            eigenvectors = eigenvectors[:, 1:]  # (N, num_eig-1)
        except Exception:
            eigenvectors = np.zeros((num_nodes, max(num_eig - 1, 0)), dtype=np.float32)

        # Pad if fewer than k eigenvectors available
        if eigenvectors.shape[1] < k:
            pad = np.zeros((num_nodes, k - eigenvectors.shape[1]), dtype=np.float32)
            eigenvectors = np.concatenate([eigenvectors, pad], axis=1)
        else:
            eigenvectors = eigenvectors[:, :k]

        # Deterministic sign disambiguation:
        # make the max-abs entry in each eigenvector non-negative.
        if eigenvectors.shape[1] > 0:
            anchor_idx = np.argmax(np.abs(eigenvectors), axis=0)
            signs = np.sign(eigenvectors[anchor_idx, np.arange(eigenvectors.shape[1])])
            signs[signs == 0] = 1.0
            eigenvectors = eigenvectors * signs

        data.lap_pe = torch.from_numpy(eigenvectors.astype(np.float32))
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


def get_loaders(batch_size=256, num_workers=4, use_dist_masks=False, max_hops=40,
                dist_mask_workers=8, use_lap_pe=False, lap_pe_dim=8,
                dataset_name="Peptides-func", return_info=False):
    """Load train/val/test splits and return loaders + datasets.

    Args:
        dataset_name: which dataset to load. Must be a key of
                      ``GRAPH_DATASETS`` (e.g. "Peptides-func", "MNIST",
                      "ZINC12k"). Drives the underlying PyG dataset selection
                      and templates the dist-mask cache directory.
        use_dist_masks: if True, precompute Floyd-Warshall distance masks and
                        return DataLoaders that yield (pyg_batch, dist_masks, node_masks).
        max_hops: maximum number of hop levels for distance masks. PascalVOC-SP
                  graphs are large; consider lowering this (e.g. 12-16) when
                  loading that dataset.
        dist_mask_workers: number of multiprocessing workers for Floyd-Warshall.
    """
    # Validate the dataset name early so callers fail fast on typos.
    info = get_dataset_info(dataset_name)
    dataset_name = info["name"]
    source = info.get("source", "lrgb")
    pyg_name = info.get("pyg_name", dataset_name)

    # Optional Laplacian PE transform (computed on every access)
    transform = AddLaplacianPE(k=lap_pe_dim) if use_lap_pe else None

    def _build_dataset(split):
        if source == "lrgb":
            return LRGBDataset(root="./data", name=pyg_name, split=split,
                               transform=transform)
        if source == "gnn_benchmark":
            return GNNBenchmarkDataset(root="./data", name=pyg_name, split=split,
                                       transform=transform)
        if source == "zinc":
            return ZINC(root="./data/ZINC", subset=bool(info.get("subset", False)),
                        split=split, transform=transform)
        raise ValueError(f"Unknown dataset source '{source}' for {dataset_name}.")

    train_ds = _build_dataset("train")
    val_ds = _build_dataset("val")
    test_ds = _build_dataset("test")

    # Fill dynamic fields like output_dim/node_feat_dim when marked as "auto".
    info = dict(info)
    info["output_dim"] = _infer_output_dim(info, train_ds)
    if info.get("node_encoder") == "linear":
        inferred = _infer_node_feat_dim(train_ds)
        if info.get("node_feat_dim") in ("auto", None):
            info["node_feat_dim"] = inferred if inferred is not None else 1

    if use_dist_masks:
        cache_dir = f"./data/{dataset_name}/dist_masks"
        os.makedirs(cache_dir, exist_ok=True)
        train_dm = precompute_distance_masks(
            train_ds, os.path.join(cache_dir, "train.pkl"), max_hops, dist_mask_workers)
        val_dm = precompute_distance_masks(
            val_ds, os.path.join(cache_dir, "val.pkl"), max_hops, dist_mask_workers)
        test_dm = precompute_distance_masks(
            test_ds, os.path.join(cache_dir, "test.pkl"), max_hops, dist_mask_workers)

        train_wrapped = DistMaskDataset(train_ds, train_dm)
        val_wrapped = DistMaskDataset(val_ds, val_dm)
        test_wrapped = DistMaskDataset(test_ds, test_dm)

        collate_fn = partial(collate_with_dist_masks, max_hops=max_hops)

        # Use standard PyTorch DataLoader (not PyG's) because our custom
        # collate_fn handles batching and PyG's Collater cannot handle the
        # numpy arrays returned by DistMaskDataset.
        from torch.utils.data import DataLoader as TorchDataLoader
        result = (
            TorchDataLoader(train_wrapped, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=collate_fn),
            TorchDataLoader(val_wrapped, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn),
            TorchDataLoader(test_wrapped, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn),
            train_ds, val_ds, test_ds,
        )
        if return_info:
            return (*result, info)
        return result

    result = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        train_ds, val_ds, test_ds,
    )
    if return_info:
        return (*result, info)
    return result


class ProxyTargetDataset(torch.utils.data.Dataset):
    """
    Pairs each graph with its best optimized proxy embedding set.

    Loads proxy pairs from a pickle file produced by Stage 2. For each graph
    (identified by sample_idx), keeps the entry with the lowest opt_loss
    across all optimization restarts.

    Graphs without any valid proxy pair are excluded.
    """
    def __init__(self, pyg_dataset, proxy_pairs_path):
        with open(proxy_pairs_path, "rb") as f:
            proxy_pairs = pickle.load(f)

        # Find best proxy per sample_idx (lowest opt_loss across restarts)
        best_by_idx = {}
        for p in proxy_pairs:
            sid = p["sample_idx"]
            if sid not in best_by_idx or p["opt_loss"] < best_by_idx[sid]["opt_loss"]:
                best_by_idx[sid] = p

        # Build aligned lists — only include graphs that have a valid proxy
        self.samples = []
        for sid in sorted(best_by_idx.keys()):
            pair = best_by_idx[sid]
            self.samples.append({
                "graph": pyg_dataset[sid],
                "encoder_emb": pair["encoder_emb"],      # (max_N, d)
                "mask": pair["mask"],                      # (max_N,)
                "proxy_emb": pair["proxy_emb"],            # (M, d)
                "sample_idx": sid,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["graph"], s["encoder_emb"], s["mask"], s["proxy_emb"]


def collate_with_proxies(batch):
    """
    Collate function for ProxyTargetDataset.

    Returns:
        pyg_batch: batched PyG Data object
        encoder_embs: (B, max_N, d) padded encoder embeddings
        emb_masks: (B, max_N) boolean masks
        proxy_embs: (B, M, d) target proxy embeddings
    """
    graphs, encoder_embs, masks, proxy_embs = zip(*batch)

    pyg_batch = torch_geometric.data.Batch.from_data_list(list(graphs))
    proxy_batch = torch.stack(proxy_embs, dim=0)

    # Pad encoder embeddings and masks to same max_N within this batch
    max_n = max(e.shape[0] for e in encoder_embs)
    d = encoder_embs[0].shape[1]
    B = len(encoder_embs)

    padded_embs = torch.zeros(B, max_n, d)
    padded_masks = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (emb, mask) in enumerate(zip(encoder_embs, masks)):
        n = emb.shape[0]
        padded_embs[i, :n] = emb
        padded_masks[i, :n] = mask

    return pyg_batch, padded_embs, padded_masks, proxy_batch
