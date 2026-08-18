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
#   metric_name   : "macro_ap" | "mae" | "node_f1_macro" | "accuracy" | "rocauc".
#   source        : "lrgb" | "gnn_benchmark" | "zinc" | "ogb_graph" |
#                   "transductive" | "unsupported".
#   pyg_name      : name string for the underlying PyG dataset (optional).
#   subset        : ZINC-specific flag (subset=True → ZINC-12k).
#   nan_labels    : True if targets contain NaN for unmeasured tasks
#                   (ogbg-molpcba). Loss and metric mask these positions.
#   transductive  : True for single-graph node classification. These are cut
#                   into subgraphs by ``transductive.py``; see --subgraph_mode.
#
# Note: PascalVOC-SP graphs are large (~480 superpixels) and the distance cache
# scales as O(N^2) per graph. Default max_hops in get_loaders should be lowered
# when using VOC or COCO.

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
    # COCO-SP: same superpixel featurisation as PascalVOC-SP (14 node feats)
    # but 81 classes and ~113k train graphs. The graph count is what makes the
    # distance cache expensive, not the graph size.
    "COCO-SP": {
        "output_dim": 81,
        "task_type": "multiclass",
        "level": "node",
        "node_encoder": "linear",
        "node_feat_dim": 14,
        "metric_name": "node_f1_macro",
        "source": "lrgb",
    },
    # ---- OGB graph-level (molecules; same 9-dim categorical atom features
    # ---- as Peptides, so `atom_categorical` applies unchanged) -------------
    "ogbg-molhiv": {
        "output_dim": 1,
        "task_type": "multi_label",     # single binary task
        "level": "graph",
        "node_encoder": "atom_categorical",
        "node_feat_dim": 9,
        "metric_name": "rocauc",
        "source": "ogb_graph",
    },
    "ogbg-molpcba": {
        "output_dim": 128,
        "task_type": "multi_label",
        "level": "graph",
        "node_encoder": "atom_categorical",
        "node_feat_dim": 9,
        "metric_name": "macro_ap",
        "source": "ogb_graph",
        "nan_labels": True,             # unmeasured (task, molecule) pairs
    },
    # ---- Transductive node classification (cut into subgraphs) -------------
    "ogbn-arxiv": {
        "output_dim": 40,
        "task_type": "multiclass",
        "level": "node",
        "node_encoder": "linear",
        "node_feat_dim": 128,
        "metric_name": "accuracy",
        "source": "transductive",
        "transductive": True,
    },
    "ogbn-products": {
        "output_dim": 47,
        "task_type": "multiclass",
        "level": "node",
        "node_encoder": "linear",
        "node_feat_dim": 100,
        "metric_name": "accuracy",
        "source": "transductive",
        "transductive": True,
    },
    "arxiv-year": {
        "output_dim": 5,
        "task_type": "multiclass",
        "level": "node",
        "node_encoder": "linear",
        "node_feat_dim": 128,
        "metric_name": "accuracy",
        "source": "transductive",
        "transductive": True,
    },
    # ---- Deferred ----------------------------------------------------------
    # PCQM-Contact is link-level (rank candidate contact pairs, filtered MRR).
    # That needs level="link", a pair-scoring head over node embeddings, BCE on
    # edge_label_index, and the filtered ranking protocol — none of which exist
    # yet. Registered so --dataset fails with a clear message, not a KeyError.
    "PCQM-Contact": {
        "output_dim": 1,
        "task_type": "multi_label",
        "level": "link",
        "node_encoder": "atom_categorical",
        "node_feat_dim": 9,
        "metric_name": "mrr",
        "source": "unsupported",
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
    "coco-sp": "COCO-SP",
    "cocosp": "COCO-SP",
    "voc": "PascalVOC-SP",
    "pascalvoc-sp": "PascalVOC-SP",
    "molhiv": "ogbg-molhiv",
    "molpcba": "ogbg-molpcba",
    "arxiv": "ogbn-arxiv",
    "products": "ogbn-products",
    "arxiv_year": "arxiv-year",
    "arxivyear": "arxiv-year",
    "pcqm-contact": "PCQM-Contact",
}

DATASET_CHOICES = sorted(set(GRAPH_DATASETS.keys()) | set(DATASET_ALIASES.keys()))


def canonicalize_dataset_name(name: str) -> str:
    """Return the canonical dataset name (resolving aliases)."""
    if name in GRAPH_DATASETS:
        print("Returning ================= ", name, flush=True)
        return name
    key = name.lower()
    if key in DATASET_ALIASES:
        print("Returning ===================", name, DATASET_ALIASES[key])
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
    print("Returning INFO ======================== ", info)
    return info


def _infer_output_dim(info, dataset):
    print("INFERRING OUTOUT DIM =========== ", info)
    if info.get("output_dim") not in ("auto", None):
        return info["output_dim"]
    # For regression tasks, num_classes is meaningless (e.g. ZINC returns the
    # count of unique float targets). Infer from target tensor shape instead.
    if info.get("task_type") == "regression":
        _ds_data = getattr(dataset, "_data", None) or getattr(dataset, "data", None)
        if _ds_data is not None and getattr(_ds_data, "y", None) is not None:
            y = _ds_data.y
            print("RETURning regression output dim ======================== ", int(y.size(-1)) if y.dim() > 1 else 1)
            return int(y.size(-1)) if y.dim() > 1 else 1
        return 1
    print("OUT DIM DATASET ==== ", dataset, "num classes", hasattr(dataset, "num_classes"), "targets", hasattr(dataset, "num_targets"), "num tasks",  hasattr(dataset, "num_tasks"))
    if hasattr(dataset, "num_classes") and dataset.num_classes not in (None, -1, 0):
        return int(dataset.num_classes)
    if hasattr(dataset, "num_targets") and dataset.num_targets not in (None, 0):
        return int(dataset.num_targets)
    if hasattr(dataset, "num_tasks") and dataset.num_tasks not in (None, 0):
        return int(dataset.num_tasks)
    if hasattr(dataset, "data") and getattr(dataset.data, "y", None) is not None:
        y = dataset.data.y
        if y.numel() == 0:
            print("OUT DIM RETURNED = ====== ", 1)
            return 1
        if info.get("task_type") == "multiclass":
            if torch.is_tensor(y):
                y_flat = y.reshape(-1)
                y_flat = y_flat[y_flat >= 0]
                if y_flat.numel() == 0:
                    print("OUT DIM RETURNED = multiclass task ====== ", 1)
                    return 1
                y_np = y_flat.cpu().numpy()
            else:
                y_np = np.asarray(y).reshape(-1)
                y_np = y_np[y_np >= 0]
            print("OUT DIM RETURNED = multiclass 2 ====== ", int(np.unique(y_np).size) if y_np.size else 1)
            return int(np.unique(y_np).size) if y_np.size else 1
        print("OUT DIM RETURNED = not multiclass 2 ====== ", int(y.size(-1)) if y.dim() > 1 else 1)
        return int(y.size(-1)) if y.dim() > 1 else 1
    print("OUT DIM RETURNED = final ================= ", info.get("output_dim", 1))
    return info.get("output_dim", 1)


def _infer_node_feat_dim(dataset):
    print("INFERRING NODE FEAT DIM ============== ")
    if hasattr(dataset, "num_node_features") and dataset.num_node_features is not None:
        print(int(dataset.num_node_features))
        return int(dataset.num_node_features)
    if hasattr(dataset, "num_features") and dataset.num_features is not None:
        print(int(dataset.num_features))
        return int(dataset.num_features)
    if hasattr(dataset, "data") and getattr(dataset.data, "x", None) is not None:
        print(int(dataset.data.x.size(-1)))
        return int(dataset.data.x.size(-1))
    try:
        # Some datasets disallow direct indexing or have empty splits.
        sample = dataset[0]
        if hasattr(sample, "x") and sample.x is not None:
            print("Returnign sample === ", int(sample.x.size(-1)))
            return int(sample.x.size(-1))
    except (IndexError, AttributeError, TypeError):
        pass
    return None


# ================================================================
# DISTANCE MASK PREPROCESSING (for GRED backbone)
# ================================================================

def _compute_dist_matrix_single(args, max_hops=40):
    """Shortest-path distance matrix for one graph, as int16.

    Returns an ``(N, N)`` int16 array: entry ``d`` for reachable pairs with
    ``d < max_hops``, and ``-1`` for unreachable pairs or pairs further than
    ``max_hops`` (those contribute to no hop mask, so the value is unused).

    Storing the distance matrix rather than the expanded ``(K, N, N)`` boolean
    stack cuts the cache by a factor of K. Masks are rebuilt in the collate,
    which costs one comparison per hop level and is negligible next to the
    forward pass.
    """
    num_nodes, edge_index = args
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[edge_index[0], edge_index[1]] = 1.0

    dist = floyd_warshall(adj, directed=False, unweighted=True)
    dist = np.where(np.isfinite(dist) & (dist < max_hops), dist, -1)
    return dist.astype(np.int16)


class MemmapDistStore:
    """Disk-backed store of variable-size int16 distance matrices.

    One flat ``.dat`` file holds every matrix end to end; a small pickle holds
    the per-graph offsets and sizes. Opened lazily per worker process so
    DataLoader workers don't share a file handle.
    """

    def __init__(self, dat_path, offsets, sizes):
        self.dat_path = dat_path
        self.offsets = offsets      # int64 element offsets into the flat file
        self.sizes = sizes          # per-graph N
        self._mm = None

    def _ensure_open(self):
        if self._mm is None:
            self._mm = np.memmap(self.dat_path, dtype=np.int16, mode="r")

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        self._ensure_open()
        n = int(self.sizes[idx])
        start = int(self.offsets[idx])
        return self._mm[start:start + n * n].reshape(n, n)

    def __getstate__(self):
        # Drop the memmap so the store can cross a process boundary.
        state = self.__dict__.copy()
        state["_mm"] = None
        return state


def precompute_distance_matrices(dataset, cache_dir, split, max_hops=40,
                                 num_workers=8, chunk_size=2048):
    """Compute and cache shortest-path distance matrices for a PyG dataset.

    Streams in chunks so neither the dense adjacencies nor the results are all
    held in RAM at once — required for COCO-SP (~113k graphs), which would need
    hundreds of GB under the previous pickle-everything approach.

    Returns a ``MemmapDistStore``.
    """
    os.makedirs(cache_dir, exist_ok=True)
    dat_path = os.path.join(cache_dir, f"{split}_dist.dat")
    idx_path = os.path.join(cache_dir, f"{split}_index.pkl")

    if os.path.exists(dat_path) and os.path.exists(idx_path):
        with open(idx_path, "rb") as f:
            meta = pickle.load(f)
        if meta.get("num_graphs") == len(dataset) and meta.get("max_hops") == max_hops:
            print(f"  Loading cached distance matrices from {dat_path}", flush=True)
            return MemmapDistStore(dat_path, meta["offsets"], meta["sizes"])
        print(f"  Cache at {dat_path} is stale "
              f"(graphs {meta.get('num_graphs')}->{len(dataset)}, "
              f"max_hops {meta.get('max_hops')}->{max_hops}); recomputing.",
              flush=True)

    n_graphs = len(dataset)

    # Distance matrices depend only on edge_index, so suppress any per-access
    # transform (e.g. AddLaplacianPE) for the duration. Otherwise every graph
    # would pay an eigendecomposition twice — once for the size pass, once for
    # the compute pass — which on COCO-SP is 113k needless eigsh calls.
    _saved_transform = getattr(dataset, "transform", None)
    if _saved_transform is not None:
        dataset.transform = None
    try:
        sizes = np.array([int(dataset[i].num_nodes) for i in range(n_graphs)],
                         dtype=np.int64)
        offsets = np.zeros(n_graphs, dtype=np.int64)
        np.cumsum(sizes[:-1] ** 2, out=offsets[1:])
        total = int((sizes ** 2).sum())

        print(f"  Computing distance matrices for {n_graphs} graphs "
              f"(max_hops={max_hops}, {total * 2 / 1e9:.2f} GB on disk)...",
              flush=True)

        mm = np.memmap(dat_path, dtype=np.int16, mode="w+", shape=(total,))
        compute_fn = partial(_compute_dist_matrix_single, max_hops=max_hops)

        with Pool(min(num_workers, max(n_graphs, 1))) as pool:
            for lo in range(0, n_graphs, chunk_size):
                hi = min(lo + chunk_size, n_graphs)
                payload = [(int(sizes[i]), dataset[i].edge_index.numpy())
                           for i in range(lo, hi)]
                for j, dm in enumerate(pool.imap(compute_fn, payload,
                                                 chunksize=16)):
                    i = lo + j
                    mm[offsets[i]:offsets[i] + sizes[i] ** 2] = dm.ravel()
                print(f"    {hi}/{n_graphs}", flush=True)

        mm.flush()
        del mm
    finally:
        if _saved_transform is not None:
            dataset.transform = _saved_transform

    with open(idx_path, "wb") as f:
        pickle.dump({"offsets": offsets, "sizes": sizes,
                     "num_graphs": n_graphs, "max_hops": max_hops}, f)
    print(f"  Saved distance matrices to {dat_path}", flush=True)
    return MemmapDistStore(dat_path, offsets, sizes)


class DistMaskDataset(torch.utils.data.Dataset):
    """Wraps a PyG dataset with a precomputed distance-matrix store."""
    def __init__(self, pyg_dataset, dist_store):
        assert len(pyg_dataset) == len(dist_store)
        self.pyg_dataset = pyg_dataset
        self.dist_store = dist_store

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        return self.pyg_dataset[idx], self.dist_store[idx]


def collate_with_dist_masks(batch, max_hops=40):
    """
    Collate function for DistMaskDataset.
    Pads graphs and expands distance matrices into hop masks at the batch's
    max_N.

    Returns:
        pyg_batch: batched PyG Data object (standard)
        dist_masks_padded: (B, max_hops, max_N, max_N) float tensor
        node_masks: (B, max_N) boolean tensor
    """
    graphs, dist_list = zip(*batch)
    pyg_batch = torch_geometric.data.Batch.from_data_list(list(graphs))

    B = len(graphs)
    max_N = max(g.num_nodes for g in graphs)

    dm_padded = np.zeros((B, max_hops, max_N, max_N), dtype=np.float32)
    node_masks = np.zeros((B, max_N), dtype=np.bool_)

    for i, (g, dist) in enumerate(zip(graphs, dist_list)):
        n = int(g.num_nodes)
        # dist is int16 (n, n) with -1 for unreachable / beyond max_hops.
        K_use = min(int(dist.max()) + 1, max_hops) if dist.size else 0
        for k in range(K_use):
            dm_padded[i, k, :n, :n] = (dist == k)
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


def _build_ogb_graph_splits(pyg_name, transform=None):
    """Split an ogbg-* graph-property dataset using its official index split.

    ``PygGraphPropPredDataset`` is a single dataset plus an index split rather
    than three dataset objects, so we slice it. Slices keep the parent's
    ``transform``, so we set it once on the parent.
    """
    try:
        from ogb.graphproppred import PygGraphPropPredDataset
    except ImportError as e:
        raise ImportError(
            f"{pyg_name} requires the `ogb` package: pip install ogb"
        ) from e

    ds = PygGraphPropPredDataset(name=pyg_name, root="./data")
    if transform is not None:
        ds.transform = transform
    split = ds.get_idx_split()
    return ds[split["train"]], ds[split["valid"]], ds[split["test"]]


def get_loaders(batch_size=256, num_workers=4, use_dist_masks=False, max_hops=40,
                dist_mask_workers=8, use_lap_pe=False, lap_pe_dim=8,
                dataset_name="Peptides-func", return_info=False,
                subgraph_mode="partition", num_parts=128, egonet_hops=2,
                egonet_max_nodes=1024, max_egonet_samples=None, seed=0):
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
        subgraph_mode: "partition" | "egonet" — only used by transductive
                       single-graph datasets (ogbn-arxiv, ogbn-products,
                       arxiv-year). See ``transductive.py``.
        num_parts: number of random partitions for subgraph_mode="partition".
        egonet_hops / egonet_max_nodes / max_egonet_samples: egonet mode knobs.
    """
    # Validate the dataset name early so callers fail fast on typos.
    info = get_dataset_info(dataset_name)
    dataset_name = info["name"]
    source = info.get("source", "lrgb")
    pyg_name = info.get("pyg_name", dataset_name)

    if source == "unsupported":
        raise NotImplementedError(
            f"{dataset_name} is registered but not implemented. It is a "
            f"{info.get('level')}-level task requiring a pair-scoring head and "
            f"the filtered {info.get('metric_name', '').upper()} protocol, "
            f"which the current Task abstraction (graph|node levels only) does "
            f"not cover."
        )

    # Optional Laplacian PE transform (computed on every access)
    transform = AddLaplacianPE(k=lap_pe_dim) if use_lap_pe else None
    print("TRANSFORMS ========== ", transform)

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

    if source == "ogb_graph":
        train_ds, val_ds, test_ds = _build_ogb_graph_splits(
            pyg_name, transform=transform)
    elif source == "transductive":
        from transductive import build_transductive_splits, ListGraphDataset

        tr, va, te = build_transductive_splits(
            dataset_name, root="./data", subgraph_mode=subgraph_mode,
            num_parts=num_parts, egonet_hops=egonet_hops,
            egonet_max_nodes=egonet_max_nodes,
            max_egonet_samples=max_egonet_samples, seed=seed)
        if transform is not None:
            tr = [transform(g) for g in tr]
            va = [transform(g) for g in va]
            te = [transform(g) for g in te]
        _wrap = partial(ListGraphDataset,
                        num_classes=info["output_dim"],
                        num_node_features=info["node_feat_dim"])
        train_ds, val_ds, test_ds = _wrap(tr), _wrap(va), _wrap(te)
    else:
        train_ds = _build_dataset("train")
        val_ds = _build_dataset("val")
        test_ds = _build_dataset("test")

    # Fill dynamic fields like output_dim/node_feat_dim when marked as "auto".
    info = dict(info)
    info["output_dim"] = _infer_output_dim(info, train_ds)
    print("INFO NODE ENCODER  ============================= ", info.get("node_encoder"), info.get("node_feat_dim"))
    if info.get("node_encoder") == "linear":
        inferred = _infer_node_feat_dim(train_ds)
        if info.get("node_feat_dim") in ("auto", None):
            info["node_feat_dim"] = inferred if inferred is not None else 1

    if use_dist_masks:
        # Transductive caches depend on the sampling config, so key on it —
        # otherwise switching --num_parts would silently reuse the wrong cache.
        cache_tag = "dist_cache"
        if info.get("transductive"):
            cache_tag = (f"dist_cache_{subgraph_mode}_p{num_parts}"
                         f"_h{egonet_hops}_m{egonet_max_nodes}_s{seed}")
        cache_dir = f"./data/{dataset_name}/{cache_tag}"
        os.makedirs(cache_dir, exist_ok=True)
        train_dm = precompute_distance_matrices(
            train_ds, cache_dir, "train", max_hops, dist_mask_workers)
        val_dm = precompute_distance_matrices(
            val_ds, cache_dir, "val", max_hops, dist_mask_workers)
        test_dm = precompute_distance_matrices(
            test_ds, cache_dir, "test", max_hops, dist_mask_workers)

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
