"""Distance-mask utilities for the hop-masked transformer.

Provides Floyd-Warshall based distance-mask precomputation with pickle
caching, a wrapper dataset, and a custom collate function that stores padded
distance masks directly on the PyG Batch object.
"""

import os
import pickle
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch_geometric.data
from scipy.sparse.csgraph import floyd_warshall


# ====================================================================
# Floyd-Warshall distance-mask computation.
# ====================================================================

def _compute_dist_mask_single(adj, max_hops=40):
    """Compute boolean distance masks for a single graph's adjacency matrix.

    Returns
    -------
    dist_mask : ndarray, shape (K, N, N), dtype bool
        ``K = min(diameter + 1, max_hops)``.  ``dist_mask[k, i, j]`` is True
        iff the shortest-path distance between nodes *i* and *j* equals *k*.
    """
    dist = floyd_warshall(adj, directed=False, unweighted=True)
    dist = np.where(np.isfinite(dist), dist, -1).astype(np.int32)
    actual_max = int(dist.max())
    K = min(actual_max + 1, max_hops)
    dist_mask = np.stack([(dist == k) for k in range(K)]).astype(np.bool_)
    return dist_mask  # (K, N, N)


def precompute_distance_masks(dataset, cache_path, max_hops=40,
                              num_workers=8):
    """Precompute Floyd-Warshall distance masks for every graph in *dataset*.

    Results are cached to *cache_path* as a pickled list of ``(K_i, N_i, N_i)``
    boolean arrays.  If the cache file already exists it is loaded directly.

    Parameters
    ----------
    dataset : PyG dataset (indexable, each element has ``.x`` and ``.edge_index``)
    cache_path : str
        Filesystem path for the pickle cache.
    max_hops : int
        Upper bound on the number of hop levels stored per graph.
    num_workers : int
        Number of parallel workers for Floyd-Warshall computation.

    Returns
    -------
    list of ndarray
        One ``(K_i, N_i, N_i)`` boolean array per graph.
    """
    cache_dir = os.path.dirname(cache_path)
    base_name = os.path.basename(cache_path).replace('.pkl', f'_max_hops_{max_hops}.pkl')
    cache_path = os.path.join(cache_dir, base_name)
    
    if os.path.exists(cache_path):
        print(f"  Loading cached distance masks from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Determine number of graphs from the original node slices.  The loader
    # now preserves the original per-graph slice layout for existing data
    # attributes, so this path remains valid even after split metadata has been
    # attached to the dataset.
    _data = getattr(dataset, '_data', None) or dataset.data
    _slices = dataset.slices

    x_slices = _slices['x']             # (num_graphs + 1,)
    all_ei = _data.edge_index           # (2, total_edges)
    num_graphs = len(x_slices) - 1

    print(
        f"  Computing distance masks for {num_graphs} graphs "
        f"(max_hops={max_hops})...",
        flush=True,
    )

    ei_slices = _slices['edge_index']
    adjs = []
    for i in range(num_graphs):
        node_start = int(x_slices[i])
        node_end = int(x_slices[i + 1])
        n = node_end - node_start

        ei_start = int(ei_slices[i])
        ei_end = int(ei_slices[i + 1])
        ei = all_ei[:, ei_start:ei_end].numpy()

        # If the edges are globally shifted, shift them back to local 0-based.
        # If the min node index in ei is >= node_start (for i > 0), they are shifted.
        if ei.size > 0 and np.min(ei) >= node_start and node_start > 0:
            ei = ei - node_start

        adj = np.zeros((n, n), dtype=np.float32)
        if ei.size > 0:
            adj[ei[0], ei[1]] = 1.0
        adjs.append(adj)

    compute_fn = partial(_compute_dist_mask_single, max_hops=max_hops)
    with Pool(min(num_workers, max(len(adjs), 1))) as p:
        dist_masks = p.map(compute_fn, adjs)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dist_masks, f)
    print(f"  Saved distance masks to {cache_path}", flush=True)
    return dist_masks


# ====================================================================
# Wrapper dataset + collate function for s2gnn DataLoader integration.
# ====================================================================

class DistMaskGraphDataset(torch.utils.data.Dataset):
    """Wraps a PyG dataset (or subset) to pair each graph with its dist_mask.

    ``__getitem__`` returns a ``(Data, dist_mask_numpy)`` tuple.  Use with
    :func:`collate_s2gnn_dist_masks` as the DataLoader's collate_fn.
    """

    def __init__(self, pyg_dataset, dist_masks):
        assert len(pyg_dataset) == len(dist_masks), (
            f"dataset length {len(pyg_dataset)} != dist_masks length "
            f"{len(dist_masks)}"
        )
        self.pyg_dataset = pyg_dataset
        self.dist_masks = dist_masks

    def __len__(self):
        return len(self.pyg_dataset)

    def __getitem__(self, idx):
        return self.pyg_dataset[idx], self.dist_masks[idx]


def collate_s2gnn_dist_masks(batch_list, max_hops=40):
    """Collate ``(Data, dist_mask)`` tuples into a single PyG Batch.

    The padded distance masks and node masks are stored as attributes on the
    returned Batch object so the training loop receives a single object.

    Attributes added to the batch
    -----------------------------
    dist_mask : list of numpy arrays
        Length B, each ``(K_i, N_i, N_i)`` — the raw per-graph masks for
        on-the-fly padding in the network's forward pass.
    """
    graphs, dist_masks_list = zip(*batch_list)
    pyg_batch = torch_geometric.data.Batch.from_data_list(list(graphs))

    # Store raw dist masks as a Python list on the batch.
    # The network's forward pass handles padding to (B, K, N_max, N_max).
    pyg_batch.dist_mask = list(dist_masks_list)

    return pyg_batch
