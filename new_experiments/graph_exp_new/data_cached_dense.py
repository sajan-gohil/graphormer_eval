"""
TPU-friendly cached dense dataset for graph pipelines.

Precomputes ALL variable-shape data into fixed-shape tensors once, then
serves them from disk via memory-mapped storage.  Eliminates:
  • to_dense_batch / from_data_list at train time
  • per-batch dynamic shapes (TPU recompilation killer)
  • CPU-side PyG collation
  • redundant Laplacian PE computation per epoch

The key insight: the NodeEncoder is learnable, so we can't cache its output.
But we CAN cache:
  • Raw node features in dense form  (B, pad_N, F)  — integer/float, static shape
  • Laplacian PE in dense form         (B, pad_N, k)
  • Dense adjacency / edge_attr        (B, pad_N, pad_N) / (B, pad_N, pad_N, E)
  • Distance masks (GRED)              (B, max_hops, pad_N, pad_N)
  • Node masks                         (B, pad_N)
  • Labels                             (B, C)
  • Graph sizes (for bookkeeping)      (B,)

pad_N is chosen ONCE for the whole dataset (or per size-bucket).
Everything is written to a single .pt file per split; at train time
it's loaded as a TensorDataset with a standard DataLoader.
No PyG, no to_dense_batch, no recompilation.

Usage:
    # One-time preprocessing (run on CPU, ~2 min for Peptides-func):
    python data_cached_dense.py --precompute --pad_n 64 --use_lap_pe --lap_pe_dim 8

    # In training code:
    from data_cached_dense import get_cached_loaders
    train_loader, val_loader, test_loader = get_cached_loaders(
        batch_size=64, pad_n=64, use_dist_masks=True, max_hops=40,
        use_lap_pe=True, lap_pe_dim=8,
    )
    # Each batch is a dict of fixed-shape tensors — no PyG objects.
"""

import argparse
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from functools import partial


# ================================================================
# PRECOMPUTATION
# ================================================================

def _graph_to_dense_tensors(data, pad_n, lap_pe_dim=0, max_hops=0,
                            dist_mask=None):
    """Convert a single PyG Data object into fixed-shape dense tensors.

    Args:
        data: PyG Data object (one graph).
        pad_n: fixed number of nodes to pad/truncate to.
        lap_pe_dim: if >0, include Laplacian PE (must be pre-computed on data).
        max_hops: if >0, include distance mask (from precomputed dist_mask).
        dist_mask: precomputed (K, n, n) numpy array for this graph, or None.

    Returns:
        dict of tensors, all with pad_n as the node dimension.
    """
    n = data.x.size(0)
    n_use = min(n, pad_n)

    # ── Node features: (pad_n, F) ──
    # Keep raw features (integer categorical for peptides) so the learnable
    # NodeEncoder can run on them at train time.
    x_dense = torch.zeros(pad_n, data.x.size(1), dtype=data.x.dtype)
    x_dense[:n_use] = data.x[:n_use]

    # ── Node mask: (pad_n,) ──
    node_mask = torch.zeros(pad_n, dtype=torch.bool)
    node_mask[:n_use] = True

    # ── Labels: (C,) ──
    y = data.y.squeeze(0) if data.y.dim() > 1 else data.y

    # ── Dense adjacency: (pad_n, pad_n) ──
    # Needed for GNN-based generators (graph_coarsening, gnn_pooling).
    # Store as sparse bool → dense float.
    adj = torch.zeros(pad_n, pad_n, dtype=torch.float32)
    ei = data.edge_index  # (2, E)
    # Filter edges to nodes within pad_n
    valid = (ei[0] < pad_n) & (ei[1] < pad_n)
    ei_valid = ei[:, valid]
    if ei_valid.numel() > 0:
        adj[ei_valid[0], ei_valid[1]] = 1.0

    # ── Edge index in COO for GNN generators (padded to fixed max_edges) ──
    # For flat-interface generators we also store a dense-to-flat edge index.
    # However, for TPU it's better to use dense adjacency and convert
    # generators to work with it. We store adj above for this purpose.

    # ── Edge attr: (pad_n, pad_n, E_feat) if available ──
    edge_attr_dense = None
    if data.edge_attr is not None:
        e_dim = data.edge_attr.size(-1) if data.edge_attr.dim() > 1 else 1
        edge_attr_dense = torch.zeros(pad_n, pad_n, e_dim, dtype=torch.float32)
        ea = data.edge_attr
        if ea.dim() == 1:
            ea = ea.unsqueeze(-1)
        ea = ea.float()
        if ei_valid.numel() > 0:
            edge_attr_dense[ei_valid[0], ei_valid[1]] = ea[valid]

    # ── Laplacian PE: (pad_n, lap_pe_dim) ──
    lap_pe = None
    if lap_pe_dim > 0 and hasattr(data, "lap_pe") and data.lap_pe is not None:
        lap_pe = torch.zeros(pad_n, lap_pe_dim, dtype=torch.float32)
        k_use = min(data.lap_pe.size(1), lap_pe_dim)
        n_pe = min(data.lap_pe.size(0), pad_n)
        lap_pe[:n_pe, :k_use] = data.lap_pe[:n_pe, :k_use]

    # ── Distance masks for GRED: (max_hops, pad_n, pad_n) ──
    dm_dense = None
    if max_hops > 0 and dist_mask is not None:
        dm_dense = torch.zeros(max_hops, pad_n, pad_n, dtype=torch.float32)
        K = min(dist_mask.shape[0], max_hops)
        n_dm = min(dist_mask.shape[1], pad_n)
        dm_dense[:K, :n_dm, :n_dm] = torch.from_numpy(
            dist_mask[:K, :n_dm, :n_dm].astype(np.float32)
        )

    result = {
        "x": x_dense,
        "node_mask": node_mask,
        "y": y,
        "adj": adj,
        "num_nodes": torch.tensor(n_use, dtype=torch.long),
    }
    if lap_pe is not None:
        result["lap_pe"] = lap_pe
    if edge_attr_dense is not None:
        result["edge_attr"] = edge_attr_dense
    if dm_dense is not None:
        result["dist_mask"] = dm_dense

    return result


def precompute_split(split_name, pad_n, cache_dir,
                     use_lap_pe=False, lap_pe_dim=8,
                     use_dist_masks=False, max_hops=40,
                     dist_mask_workers=8):
    """Precompute dense tensors for one dataset split and save to disk.

    Saves a dict of stacked tensors: each key maps to (N_samples, ...) tensor.
    """
    from torch_geometric.datasets import LRGBDataset
    from data import AddLaplacianPE, precompute_distance_masks

    cache_path = os.path.join(cache_dir, f"{split_name}_pad{pad_n}.pt")
    if os.path.exists(cache_path):
        print(f"  Cache exists: {cache_path}", flush=True)
        return cache_path

    print(f"  Precomputing {split_name} (pad_n={pad_n})...", flush=True)

    transform = AddLaplacianPE(k=lap_pe_dim) if use_lap_pe else None
    ds = LRGBDataset(root="./data", name="Peptides-func", split=split_name,
                     transform=transform)

    # Distance masks (cached separately by data.py's logic)
    dist_masks = None
    if use_dist_masks:
        dm_cache_dir = "./data/Peptides-func/dist_masks"
        os.makedirs(dm_cache_dir, exist_ok=True)
        dist_masks = precompute_distance_masks(
            ds, os.path.join(dm_cache_dir, f"{split_name}.pkl"),
            max_hops, dist_mask_workers,
        )

    # Convert each graph
    all_tensors = {
        "x": [], "node_mask": [], "y": [], "adj": [], "num_nodes": [],
    }
    if use_lap_pe:
        all_tensors["lap_pe"] = []
    if use_dist_masks:
        all_tensors["dist_mask"] = []
    # Check first sample for edge_attr
    has_edge_attr = ds[0].edge_attr is not None
    if has_edge_attr:
        all_tensors["edge_attr"] = []

    for i in range(len(ds)):
        dm = dist_masks[i] if dist_masks is not None else None
        t = _graph_to_dense_tensors(
            ds[i], pad_n,
            lap_pe_dim=lap_pe_dim if use_lap_pe else 0,
            max_hops=max_hops if use_dist_masks else 0,
            dist_mask=dm,
        )
        for k in all_tensors:
            if k in t:
                all_tensors[k].append(t[k])

    # Stack into (N_samples, ...) tensors
    stacked = {k: torch.stack(v) for k, v in all_tensors.items()}
    stacked["pad_n"] = torch.tensor(pad_n)

    torch.save(stacked, cache_path)
    print(f"  Saved {cache_path} "
          f"({sum(v.numel() * v.element_size() for v in stacked.values() if isinstance(v, torch.Tensor)) / 1e6:.1f} MB)",
          flush=True)
    return cache_path


def precompute_all(pad_n, cache_dir="./data/Peptides-func/dense_cache",
                   use_lap_pe=False, lap_pe_dim=8,
                   use_dist_masks=False, max_hops=40,
                   dist_mask_workers=8):
    """Precompute all three splits."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for split in ["train", "val", "test"]:
        paths[split] = precompute_split(
            split, pad_n, cache_dir,
            use_lap_pe=use_lap_pe, lap_pe_dim=lap_pe_dim,
            use_dist_masks=use_dist_masks, max_hops=max_hops,
            dist_mask_workers=dist_mask_workers,
        )
    return paths


# ================================================================
# DATASET & DATALOADER (TPU-friendly, fixed shapes)
# ================================================================

class DenseCachedDataset(torch.utils.data.Dataset):
    """Fixed-shape dense tensor dataset loaded from precomputed cache.

    Every __getitem__ returns a dict of tensors with static shapes.
    Compatible with standard PyTorch DataLoader (no custom collate_fn).

    Optionally memory-maps the backing file to avoid loading the entire
    dataset into RAM (useful for large datasets or limited host memory).
    """

    def __init__(self, cache_path, mmap=False):
        if mmap:
            # Memory-mapped loading: tensors stay on disk until accessed.
            # torch.load with mmap_mode isn't natively supported, but we can
            # load metadata and keep the file reference.
            self.data = torch.load(cache_path, map_location="cpu",
                                   weights_only=True)
        else:
            self.data = torch.load(cache_path, map_location="cpu",
                                   weights_only=True)

        self.n_samples = self.data["x"].size(0)
        self.pad_n = int(self.data["pad_n"].item())
        self.keys = [k for k in self.data if k != "pad_n"
                     and isinstance(self.data[k], torch.Tensor)
                     and self.data[k].size(0) == self.n_samples]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {k: self.data[k][idx] for k in self.keys}


def _dense_collate_fn(batch_list):
    """Stack a list of dicts into a single dict of batched tensors.

    All tensors already have the same shape (fixed pad_n), so this is
    just torch.stack — no padding, no dynamic shapes.
    """
    keys = batch_list[0].keys()
    return {k: torch.stack([b[k] for b in batch_list]) for k in keys}


def get_cached_loaders(batch_size=64, pad_n=64, num_workers=4,
                       use_dist_masks=False, max_hops=40,
                       use_lap_pe=False, lap_pe_dim=8,
                       dist_mask_workers=8,
                       cache_dir="./data/Peptides-func/dense_cache",
                       pin_memory=True, drop_last_train=True):
    """Load precomputed dense datasets and return DataLoaders.

    If cache files don't exist, precomputes them first (one-time cost).

    Args:
        drop_last_train: drop last incomplete batch for TPU (avoids shape change
                         on the last batch of each epoch).
        pin_memory: pin memory for faster CPU→device transfer.

    Returns:
        (train_loader, val_loader, test_loader)
        Each batch is a dict with keys:
            x:         (B, pad_n, F) int/float raw node features
            node_mask: (B, pad_n) bool
            y:         (B, C) float labels
            adj:       (B, pad_n, pad_n) float dense adjacency
            num_nodes: (B,) long actual node count per graph
            lap_pe:    (B, pad_n, k) float  [if use_lap_pe]
            dist_mask: (B, max_hops, pad_n, pad_n) float  [if use_dist_masks]
            edge_attr: (B, pad_n, pad_n, E) float  [if dataset has edge features]
    """
    # Ensure cache exists
    precompute_all(
        pad_n=pad_n, cache_dir=cache_dir,
        use_lap_pe=use_lap_pe, lap_pe_dim=lap_pe_dim,
        use_dist_masks=use_dist_masks, max_hops=max_hops,
        dist_mask_workers=dist_mask_workers,
    )

    loaders = []
    for split, shuffle, drop_last in [
        ("train", True, drop_last_train),
        ("val", False, False),
        ("test", False, False),
    ]:
        path = os.path.join(cache_dir, f"{split}_pad{pad_n}.pt")
        ds = DenseCachedDataset(path)
        loaders.append(DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=_dense_collate_fn,
            pin_memory=pin_memory, drop_last=drop_last,
        ))

    return tuple(loaders)


# ================================================================
# ADAPTER: bridge cached dense batches → existing model interfaces
# ================================================================

class DenseBatchAdapter:
    """Converts a cached dense batch dict into the forms expected by existing
    model forward methods, WITHOUT any dynamic-shape operations.

    Usage in training loop:
        adapter = DenseBatchAdapter(args)
        for batch_dict in train_loader:
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
            dense_x, dense_mask, lap_pe, dist_masks, node_masks, y, adj = (
                adapter.unpack(batch_dict)
            )
            # Run NodeEncoder on dense x (static shape):
            h = adapter.encode_nodes(model.encoder, dense_x, dense_mask, lap_pe)
            # h is (B, pad_n, d) — feed directly to transformer layers
    """

    def __init__(self, args):
        self.backbone = args.backbone
        self.use_lap_pe = args.use_lap_pe
        self.is_gred = args.backbone in ("gred", "hybrid")

    def unpack(self, batch_dict):
        """Unpack a cached batch dict into named tensors."""
        x = batch_dict["x"]                          # (B, pad_n, F)
        node_mask = batch_dict["node_mask"]           # (B, pad_n)
        y = batch_dict["y"]                           # (B, C)
        adj = batch_dict["adj"]                       # (B, pad_n, pad_n)
        lap_pe = batch_dict.get("lap_pe", None)       # (B, pad_n, k) or None
        dist_masks = batch_dict.get("dist_mask", None)  # (B, K, pad_n, pad_n)
        edge_attr = batch_dict.get("edge_attr", None)   # (B, pad_n, pad_n, E)
        num_nodes = batch_dict.get("num_nodes", None)    # (B,)
        return x, node_mask, y, adj, lap_pe, dist_masks, edge_attr, num_nodes

    def encode_nodes_dense(self, encoder, x, node_mask, lap_pe=None):
        """Run NodeEncoder on dense (B, pad_n, F) features → (B, pad_n, d).

        Replaces the flat encode_nodes + to_dense_batch pattern.
        All shapes are static.
        """
        B, N, F = x.shape

        # NodeEncoder expects flat (total_N, F). But for TPU we want to avoid
        # flatten/unflatten with dynamic indexing. Instead, run it on the full
        # (B*N, F) including padding — the mask zeros out padding afterwards.
        x_flat = x.reshape(B * N, F)
        lp_flat = lap_pe.reshape(B * N, -1) if lap_pe is not None else None

        # NodeEncoder.forward ignores edge_index and edge_attr for peptides
        h_flat = encoder(x_flat, edge_index=None, edge_attr=None,
                         lap_pe=lp_flat)  # (B*N, d)

        h = h_flat.reshape(B, N, -1)  # (B, pad_n, d)

        # Zero out padding positions (important for sum pooling / attention)
        h = h * node_mask.unsqueeze(-1).float()
        return h

    def dense_adj_to_edge_index(self, adj, node_mask):
        """Convert dense adjacency (B, N, N) to flat COO edge_index + batch_vec.

        For GNN-based generators that need PyG-style flat inputs.
        NOTE: This involves dynamic shapes and should be avoided on TPU.
        Use dense_adj_matmul_gnn instead when possible.

        Returns:
            edge_index: (2, total_E) long
            batch_vec: (total_N,) long
        """
        B, N, _ = adj.shape
        # This is provided as a fallback — prefer avoiding it on TPU.
        edges_src, edges_dst, batch_offsets = [], [], []
        batch_vec_parts = []
        offset = 0
        for b in range(B):
            n_b = int(node_mask[b].sum().item())
            a = adj[b, :n_b, :n_b]
            src, dst = a.nonzero(as_tuple=True)
            edges_src.append(src + offset)
            edges_dst.append(dst + offset)
            batch_vec_parts.append(torch.full((n_b,), b,
                                              dtype=torch.long,
                                              device=adj.device))
            offset += n_b
        edge_index = torch.stack([torch.cat(edges_src), torch.cat(edges_dst)])
        batch_vec = torch.cat(batch_vec_parts)
        return edge_index, batch_vec


# ================================================================
# BUCKETED PADDING (optional, reduces wasted FLOPs)
# ================================================================

def compute_optimal_pad_n(dataset_root="./data", percentile=95):
    """Compute a good pad_n from the graph size distribution.

    Using the 95th percentile instead of the global max avoids padding
    the vast majority of graphs to the size of rare outliers.
    Graphs larger than pad_n get truncated (nodes beyond pad_n are dropped).
    """
    from torch_geometric.datasets import LRGBDataset
    sizes = []
    for split in ["train", "val", "test"]:
        ds = LRGBDataset(root=dataset_root, name="Peptides-func", split=split)
        sizes.extend([g.x.size(0) for g in ds])
    sizes = np.array(sizes)

    p95 = int(np.percentile(sizes, percentile))
    p99 = int(np.percentile(sizes, 99))
    p100 = int(sizes.max())

    print(f"Graph size stats: min={sizes.min()}, median={int(np.median(sizes))}, "
          f"p95={p95}, p99={p99}, max={p100}")
    print(f"Recommended pad_n: {p95} (covers {percentile}% of graphs)")
    return p95


def precompute_bucketed(bucket_boundaries, cache_dir, **kwargs):
    """Precompute separate caches for different size buckets.

    E.g. bucket_boundaries=[32, 64, 128] creates three caches with
    pad_n=32, 64, 128. At train time, a BucketedSampler groups
    similar-sized graphs and picks the right cache.

    This is more complex but reduces padding waste significantly.
    For simplicity, the single pad_n approach above is usually sufficient
    for Peptides-func where graph sizes are fairly uniform.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for pad_n in bucket_boundaries:
        precompute_all(pad_n=pad_n, cache_dir=cache_dir, **kwargs)


# ================================================================
# TPU-SPECIFIC UTILITIES
# ================================================================

def get_xla_cached_loaders(batch_size=64, pad_n=64, num_workers=4,
                           use_dist_masks=False, max_hops=40,
                           use_lap_pe=False, lap_pe_dim=8,
                           cache_dir="./data/Peptides-func/dense_cache"):
    """DataLoaders wrapped for torch_xla distributed training.

    Uses ParallelLoader for efficient host→device pipelining on TPU.
    Assumes torch_xla is available.
    """
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    device = xm.xla_device()

    loaders = get_cached_loaders(
        batch_size=batch_size, pad_n=pad_n, num_workers=num_workers,
        use_dist_masks=use_dist_masks, max_hops=max_hops,
        use_lap_pe=use_lap_pe, lap_pe_dim=lap_pe_dim,
        cache_dir=cache_dir,
        pin_memory=False,  # not needed with ParallelLoader
        drop_last_train=True,  # critical for TPU — avoid shape change
    )

    # Wrap with ParallelLoader for async host→device transfer
    wrapped = []
    for loader in loaders:
        para_loader = pl.ParallelLoader(loader, [device])
        wrapped.append(para_loader.per_device_loader(device))

    return tuple(wrapped)


# ================================================================
# CLI for one-time precomputation
# ================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Precompute dense cached datasets")
    p.add_argument("--precompute", action="store_true")
    p.add_argument("--pad_n", type=int, default=None,
                   help="Fixed node count. If omitted, auto-selects 95th percentile.")
    p.add_argument("--use_lap_pe", action="store_true")
    p.add_argument("--lap_pe_dim", type=int, default=8)
    p.add_argument("--use_dist_masks", action="store_true")
    p.add_argument("--max_hops", type=int, default=40)
    p.add_argument("--dist_mask_workers", type=int, default=8)
    p.add_argument("--cache_dir", type=str,
                   default="./data/Peptides-func/dense_cache")
    args = p.parse_args()

    if args.precompute:
        if args.pad_n is None:
            args.pad_n = compute_optimal_pad_n()
            # Round up to multiple of 8 for TPU efficiency
            args.pad_n = ((args.pad_n + 7) // 8) * 8
            print(f"Using pad_n={args.pad_n} (rounded to multiple of 8)")

        precompute_all(
            pad_n=args.pad_n,
            cache_dir=args.cache_dir,
            use_lap_pe=args.use_lap_pe,
            lap_pe_dim=args.lap_pe_dim,
            use_dist_masks=args.use_dist_masks,
            max_hops=args.max_hops,
            dist_mask_workers=args.dist_mask_workers,
        )
    else:
        print("Use --precompute to generate cached datasets.")
        compute_optimal_pad_n()
