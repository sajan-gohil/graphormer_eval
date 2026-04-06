import os
import pickle
import numpy as np
import torch
import torch_geometric
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse.linalg import eigsh
from functools import partial
from multiprocessing import Pool


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
    normalized graph Laplacian.  A random sign flip is applied to each
    eigenvector to handle the sign-ambiguity of eigenvectors.
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

        # Random sign flip for sign ambiguity
        sign = 2.0 * (np.random.rand(k) > 0.5).astype(np.float32) - 1.0
        eigenvectors = eigenvectors * sign

        data.lap_pe = torch.from_numpy(eigenvectors.astype(np.float32))
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


def get_loaders(batch_size=256, num_workers=4, use_dist_masks=False, max_hops=40,
                dist_mask_workers=8, use_lap_pe=False, lap_pe_dim=8):
    """Load Peptides-func train/val/test splits and return loaders + datasets.

    Args:
        use_dist_masks: if True, precompute Floyd-Warshall distance masks and
                        return DataLoaders that yield (pyg_batch, dist_masks, node_masks).
        max_hops: maximum number of hop levels for distance masks.
        dist_mask_workers: number of multiprocessing workers for Floyd-Warshall.
    """
    # Optional Laplacian PE transform (computed on every access)
    transform = AddLaplacianPE(k=lap_pe_dim) if use_lap_pe else None

    train_ds = LRGBDataset(root="./data", name="Peptides-func", split="train",
                           transform=transform)
    val_ds = LRGBDataset(root="./data", name="Peptides-func", split="val",
                         transform=transform)
    test_ds = LRGBDataset(root="./data", name="Peptides-func", split="test",
                          transform=transform)

    if use_dist_masks:
        cache_dir = "./data/Peptides-func/dist_masks"
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

        return (
            DataLoader(train_wrapped, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, collate_fn=collate_fn),
            DataLoader(val_wrapped, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, collate_fn=collate_fn),
            DataLoader(test_wrapped, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, collate_fn=collate_fn),
            train_ds, val_ds, test_ds,
        )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        train_ds, val_ds, test_ds,
    )


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
