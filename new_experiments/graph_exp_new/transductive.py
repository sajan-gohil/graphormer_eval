# transductive.py
"""
Subgraph extraction for single-graph (transductive) node-classification
datasets: ogbn-arxiv, ogbn-products, arxiv-year.

Why this file exists
--------------------
The hop-masked transformer needs a dense ``(K, N, N)`` hop mask per graph.
For a 169k-node graph (arxiv) that is 2.9e10 entries; for 2.4M (products) it
is 5.8e12. Full-graph training is therefore impossible, and the graph must be
cut into pieces small enough that ``N^2`` is affordable.

Two modes, selected by ``--subgraph_mode``:

``partition``  (default, mirrors S2GNN)
    Randomly permute the node set and split it into ``num_parts`` contiguous
    chunks, then take the induced subgraph of each chunk. S2GNN does exactly
    this for OGB Products (§4.3: "we randomly divide the graph during training
    into 16 parts"). All three splits reuse the *same* partitions; only the
    label masking differs, so a partition seen during training reappears at
    eval time with its val/test labels revealed.

``egonet``
    For every labelled node in a split, take its ``egonet_hops``-hop
    neighbourhood, capped at ``egonet_max_nodes`` by random neighbour
    subsampling. Preserves local structure exactly but duplicates nodes across
    samples and biases the receptive field towards short range.

Deviation from S2GNN worth reporting
------------------------------------
S2GNN partitions only for *training* and runs inference on the entire graph in
one pass. A dense attention model cannot do that, so we partition at eval time
too. Predictions are therefore made without cross-partition edges, which is a
handicap relative to their protocol. Use ``num_parts`` as small as memory
allows and state the value in any comparison. With ``max_hops=8`` and float32
masks, a part of N nodes costs ``8 * N^2 * 4`` bytes per graph in the collate
buffer: N=1000 -> 32 MB, N=2000 -> 128 MB. Target 1000-2000 nodes per part
(arxiv: num_parts~128, products: num_parts~2048).
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected, k_hop_subgraph

IGNORE_INDEX = -1


# ================================================================
# BASE GRAPH LOADING
# ================================================================

def _load_ogbn(name: str, root: str) -> Tuple[Data, Dict[str, torch.Tensor]]:
    """Load an ogbn-* node-property-prediction graph with its official splits."""
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as e:
        raise ImportError(
            f"{name} requires the `ogb` package: pip install ogb"
        ) from e

    ds = PygNodePropPredDataset(name=name, root=root)
    data = ds[0]
    split = ds.get_idx_split()
    data.y = data.y.view(-1).long()
    return data, {k: v.view(-1) for k, v in split.items()}


def _load_arxiv_year(root: str, seed: int = 0,
                     num_classes: int = 5) -> Tuple[Data, Dict[str, torch.Tensor]]:
    """arxiv-year (Lim et al., 2021) built from ogbn-arxiv.

    Labels are publication year discretised into ``num_classes`` equal-frequency
    buckets. LINKX uses random 50/25/25 splits rather than the ogbn-arxiv
    temporal split, so we generate them deterministically from ``seed`` instead
    of downloading their .npz files (same protocol, no network dependency).
    """
    data, _ = _load_ogbn("ogbn-arxiv", root)

    if not hasattr(data, "node_year") or data.node_year is None:
        raise RuntimeError(
            "ogbn-arxiv is missing `node_year`; cannot construct arxiv-year."
        )
    year = data.node_year.view(-1).cpu().numpy()

    # Equal-frequency buckets. np.unique on the quantile edges guards against
    # duplicate boundaries when one year dominates.
    edges = np.unique(np.quantile(year, np.linspace(0, 1, num_classes + 1)[1:-1]))
    labels = np.digitize(year, edges)
    data.y = torch.from_numpy(labels).long()

    n = data.num_nodes
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train, n_val = int(0.5 * n), int(0.25 * n)
    split = {
        "train": torch.from_numpy(perm[:n_train]).long(),
        "valid": torch.from_numpy(perm[n_train:n_train + n_val]).long(),
        "test": torch.from_numpy(perm[n_train + n_val:]).long(),
    }
    return data, split


def load_transductive_graph(dataset_name: str, root: str,
                            seed: int = 0) -> Tuple[Data, Dict[str, torch.Tensor]]:
    """Dispatch to the right loader and symmetrise the edge index.

    Hop distances are computed on the undirected graph (``floyd_warshall`` is
    called with ``directed=False`` downstream), so we symmetrise here to keep
    the sparse ``edge_index`` consistent with the dense hop masks.
    """
    if dataset_name == "arxiv-year":
        data, split = _load_arxiv_year(root, seed=seed)
    elif dataset_name in ("ogbn-arxiv", "ogbn-products"):
        data, split = _load_ogbn(dataset_name, root)
    else:
        raise ValueError(f"{dataset_name} is not a transductive node dataset.")

    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    if data.x is not None:
        data.x = data.x.float()
    return data, split


# ================================================================
# PARTITION MODE
# ================================================================

def _random_partitions(num_nodes: int, num_parts: int,
                       seed: int = 0) -> List[torch.Tensor]:
    """Random node partition into ``num_parts`` roughly equal chunks."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_nodes)
    return [torch.from_numpy(np.sort(chunk)).long()
            for chunk in np.array_split(perm, num_parts)]


def _induced_subgraph(data: Data, node_idx: torch.Tensor) -> Data:
    """Induced subgraph on ``node_idx`` with node features and labels carried over."""
    edge_index, edge_attr = subgraph(
        node_idx, data.edge_index,
        edge_attr=getattr(data, "edge_attr", None),
        relabel_nodes=True, num_nodes=data.num_nodes,
    )
    sub = Data(
        x=data.x[node_idx],
        edge_index=edge_index,
        y=data.y[node_idx].clone(),
    )
    if edge_attr is not None:
        sub.edge_attr = edge_attr
    sub.n_id = node_idx           # original ids, for stitching predictions
    return sub


def _mask_labels_to_split(sub: Data, split_nodes: torch.Tensor,
                          num_nodes_total: int) -> Data:
    """Return a copy of ``sub`` with labels outside ``split_nodes`` set to -1.

    ``ignore_index=-1`` is already the convention in ``metrics.Task``, so both
    the loss and ``compute_accuracy`` skip these positions automatically.
    """
    keep = torch.zeros(num_nodes_total, dtype=torch.bool)
    keep[split_nodes] = True

    out = Data(x=sub.x, edge_index=sub.edge_index, y=sub.y.clone())
    if getattr(sub, "edge_attr", None) is not None:
        out.edge_attr = sub.edge_attr
    out.n_id = sub.n_id
    out.y[~keep[sub.n_id]] = IGNORE_INDEX
    return out


def build_partition_splits(data: Data, split: Dict[str, torch.Tensor],
                           num_parts: int, seed: int = 0,
                           drop_unlabelled: bool = True,
                           ) -> Tuple[List[Data], List[Data], List[Data]]:
    """Partition once, then emit three label-masked views of the same parts."""
    parts = _random_partitions(data.num_nodes, num_parts, seed=seed)
    subs = [_induced_subgraph(data, p) for p in parts]

    val_key = "valid" if "valid" in split else "val"
    out = []
    for nodes in (split["train"], split[val_key], split["test"]):
        views = [_mask_labels_to_split(s, nodes, data.num_nodes) for s in subs]
        if drop_unlabelled:
            # A part with no labelled node for this split yields no gradient and
            # no metric contribution — skip it so batches stay useful.
            views = [v for v in views if bool((v.y != IGNORE_INDEX).any())]
        out.append(views)
    return out[0], out[1], out[2]


# ================================================================
# EGONET MODE
# ================================================================

def _egonet(data: Data, seed_node: int, hops: int, max_nodes: int,
            rng: np.random.RandomState) -> Data:
    """k-hop ego net around ``seed_node``, capped at ``max_nodes``.

    When the neighbourhood overflows the cap we keep the seed and a random
    subset of the rest. Random rather than nearest-first because keeping only
    the closest nodes would systematically shrink the long-range structure this
    architecture is meant to exploit.
    """
    node_idx, edge_index, mapping, _ = k_hop_subgraph(
        int(seed_node), hops, data.edge_index,
        relabel_nodes=True, num_nodes=data.num_nodes,
    )

    if node_idx.numel() > max_nodes:
        seed_local = int(mapping[0])
        others = np.setdiff1d(np.arange(node_idx.numel()), [seed_local])
        keep_local = np.concatenate(
            [[seed_local], rng.choice(others, max_nodes - 1, replace=False)])
        keep_local = torch.from_numpy(np.sort(keep_local)).long()

        edge_index, _ = subgraph(keep_local, edge_index, relabel_nodes=True,
                                 num_nodes=node_idx.numel())
        node_idx = node_idx[keep_local]
        seed_pos = int((node_idx == seed_node).nonzero()[0, 0])
    else:
        seed_pos = int(mapping[0])

    y = torch.full((node_idx.numel(),), IGNORE_INDEX, dtype=torch.long)
    y[seed_pos] = data.y[seed_node]

    sub = Data(x=data.x[node_idx], edge_index=edge_index, y=y)
    sub.n_id = node_idx
    return sub


def build_egonet_splits(data: Data, split: Dict[str, torch.Tensor],
                        hops: int = 2, max_nodes: int = 1024, seed: int = 0,
                        max_samples_per_split: int = None,
                        ) -> Tuple[List[Data], List[Data], List[Data]]:
    """One ego net per labelled node. Only the seed node carries a label."""
    val_key = "valid" if "valid" in split else "val"
    out = []
    for si, nodes in enumerate((split["train"], split[val_key], split["test"])):
        rng = np.random.RandomState(seed + si)
        idx = nodes.cpu().numpy()
        if max_samples_per_split is not None and idx.size > max_samples_per_split:
            idx = rng.choice(idx, max_samples_per_split, replace=False)
        out.append([_egonet(data, int(v), hops, max_nodes, rng) for v in idx])
    return out[0], out[1], out[2]


# ================================================================
# ENTRY POINT
# ================================================================

def build_transductive_splits(dataset_name: str, root: str = "./data",
                              subgraph_mode: str = "partition",
                              num_parts: int = 128,
                              egonet_hops: int = 2,
                              egonet_max_nodes: int = 1024,
                              max_egonet_samples: int = None,
                              seed: int = 0,
                              ) -> Tuple[List[Data], List[Data], List[Data]]:
    """Return ``(train_list, val_list, test_list)`` of small PyG ``Data`` graphs."""
    data, split = load_transductive_graph(dataset_name, root, seed=seed)

    if subgraph_mode == "partition":
        avg = data.num_nodes / max(num_parts, 1)
        if avg > 4000:
            print(f"[transductive] WARNING: {dataset_name} with num_parts="
                  f"{num_parts} gives ~{avg:.0f} nodes/part. The dense hop mask "
                  f"is O(max_hops * N^2); raise --num_parts.", flush=True)
        train, val, test = build_partition_splits(
            data, split, num_parts=num_parts, seed=seed)
    elif subgraph_mode == "egonet":
        train, val, test = build_egonet_splits(
            data, split, hops=egonet_hops, max_nodes=egonet_max_nodes,
            seed=seed, max_samples_per_split=max_egonet_samples)
    else:
        raise ValueError(
            f"Unknown subgraph_mode '{subgraph_mode}' (partition | egonet)."
        )

    print(f"[transductive] {dataset_name} mode={subgraph_mode} "
          f"graphs: train={len(train)} val={len(val)} test={len(test)} "
          f"avg_nodes={np.mean([g.num_nodes for g in train]):.0f}", flush=True)
    return train, val, test


class ListGraphDataset(torch.utils.data.Dataset):
    """Minimal in-memory dataset exposing the bits ``data.py`` relies on."""

    def __init__(self, graphs: List[Data], num_classes: int, num_node_features: int):
        self.graphs = graphs
        self._num_classes = num_classes
        self._num_node_features = num_node_features

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_node_features(self):
        return self._num_node_features
