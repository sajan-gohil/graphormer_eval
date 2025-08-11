# Copyright (c) Microsoft Corporation and HuggingFace
# Licensed under the MIT License.

from collections.abc import Mapping
from typing import Any

import random
import numpy as np
import torch

from transformers.utils import is_cython_available, requires_backends
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from functools import lru_cache
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Union, List, Optional, Tuple

if is_cython_available():
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from . import algos_graphormer  # noqa E402


def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x

def k_hop_subgraph(
    node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False,
    sample_ratio_per_hop: Union[float, List[float]] = 1.0,
    rng: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """From: https://pytorch-geometric.readthedocs.io/en/stable/_modules/torch_geometric/utils/_subgraph.html#k_hop_subgraph"""
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx], device=row.device)
    elif isinstance(node_idx, (list, tuple)):
        node_idx = torch.tensor(node_idx, device=row.device)
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for hop in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        # torch.index_select(node_mask, 0, row, out=edge_mask)
        # subsets.append(col[edge_mask])
        # Sample only a subset of edges at this hop
        edge_mask_hop = node_mask[row]
        idx = edge_mask_hop.nonzero(as_tuple=False).view(-1)
        num_sample = int(sample_ratio_per_hop[hop] * idx.size(0))
        if num_sample < idx.size(0):
            perm = torch.randperm(idx.size(0), generator=rng, device=idx.device)[:num_sample]
            idx = idx[perm]

        edge_mask[idx] = True
        subsets.append(col[idx])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        mapping = row.new_full((num_nodes, ), -1)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = mapping[edge_index]

    return subset, edge_index, inv, edge_mask


@lru_cache(maxsize=512)
def preprocess_item(item, config, keep_features=True):
    requires_backends(preprocess_item, ["cython"])

    if keep_features and "edge_attr" in item.keys():  # edge_attr
        edge_attr = np.asarray(item["edge_attr"], dtype=np.int64)
    else:
        edge_attr = np.ones((len(item["edge_index"][0]), 1), dtype=np.int64)  # same embedding for all

    if keep_features and "x" in item.keys():  # input_nodes
        node_feature = np.asarray(item["x"], dtype=np.int64)
    else:
        raise Exception("NODE FEATURES NOT FOUND")
        node_feature = np.ones((item["x"].shape[0], 1), dtype=np.int64)  # same embedding for all

    edge_index = np.asarray(item["edge_index"], dtype=np.int64)

    input_nodes = node_feature
    if config and config.dataset_name in ["pcqm4mv2"]:
        input_nodes = convert_to_single_emb(node_feature) + 1

    num_nodes = item["x"].shape[0]

    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = np.zeros([num_nodes, num_nodes, edge_attr.shape[-1]], dtype=np.int64)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1

    # node adj matrix [num_nodes, num_nodes] bool
    adj = np.zeros([num_nodes, num_nodes], dtype=bool)
    adj[edge_index[0], edge_index[1]] = True

    shortest_path_result, path = algos_graphormer.floyd_warshall(adj)
    max_dist = np.amax(shortest_path_result)
    if max_dist > 0:
        input_edges = algos_graphormer.gen_edge_input(max_dist, path, attn_edge_type)
    else:
        input_edges = np.zeros([num_nodes, num_nodes, 1, attn_edge_type.shape[-1]], dtype=np.int64)
        # np.fill_diagonal(input_edges, 1)
    attn_bias = np.zeros([num_nodes + 1, num_nodes + 1], dtype=np.single)  # with graph token

    # combine
    item["input_nodes"] = input_nodes + 1  # we shift all indices by one for padding
    item["attn_bias"] = attn_bias
    item["attn_edge_type"] = attn_edge_type
    item["spatial_pos"] = shortest_path_result.astype(np.int64) + 1  # we shift all indices by one for padding
    item["in_degree"] = np.sum(adj, axis=1).reshape(-1) + 1  # we shift all indices by one for padding
    item["out_degree"] = item["in_degree"]  # for undirected graph
    item["input_edges"] = input_edges + 1  # we shift all indices by one for padding  # equal to max dist, encoding of edges along shortest path from i to j [edge 1 feat, edge 2 feat, ... 0,0,0]
    if "labels" not in item:
        item["labels"] = item["y"]

    return item


class GraphormerDataCollator:
    def __init__(self, spatial_pos_max=20, on_the_fly_processing=False, config=None):
        if not is_cython_available():
            raise ImportError("Graphormer preprocessing needs Cython (pyximport)")
        self.config = config
        self.spatial_pos_max = spatial_pos_max
        self.on_the_fly_processing = on_the_fly_processing

    def sample_subgraph(self, graphs):
        subgraphs = []
        for graph in graphs:
            node_set = set(list(range(graph.x.shape[0])))
            while node_set:
                node_idx = node_set.pop()
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(
                    node_idx,
                    num_hops=10,
                    edge_index=graph.edge_index,
                    relabel_nodes=False,
                    num_nodes=graph.x.shape[0],
                    flow="target_to_source",
                    directed=True,
                    sample_ratio_per_hop=0.5,
                )
                for i in subset:
                    if i in node_set:
                        node_set.remove(i)
                x_sub = graph.x[subset]
                y_sub = graph.y[subset]
                edge_attr = graph.edge_attr[edge_mask]
                # Optional masks (check if they exist)
                train_mask_sub = graph.train_mask[subset] if hasattr(graph, 'train_mask') else None
                val_mask_sub   = graph.val_mask[subset] if hasattr(graph, 'val_mask') else None
                test_mask_sub  = graph.test_mask[subset] if hasattr(graph, 'test_mask') else None

                sub_data = Data(
                    x=x_sub,
                    y=y_sub,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    train_mask=train_mask_sub,
                    val_mask=val_mask_sub,
                    test_mask=test_mask_sub
                )
                subgraphs.append(sub_data)
        return subgraphs

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        if self.config.create_subgraph:
            features = self.sample_subgraph(features)

        if self.on_the_fly_processing:
            features = [preprocess_item(i, config=self.config) for i in features]

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        batch = {}
        features = [i["_store"] for i in features]
        max_node_num = max(len(i["input_nodes"]) for i in features)
        node_feat_size = len(features[0]["input_nodes"][0])
        edge_feat_size = len(features[0]["attn_edge_type"][0][0])
        max_dist = max(len(i["input_edges"][0][0]) for i in features)
        edge_input_size = len(features[0]["input_edges"][0][0][0])
        batch_size = len(features)

        batch["attn_bias"] = torch.zeros(batch_size, max_node_num + 1, max_node_num + 1, dtype=torch.float)
        batch["attn_edge_type"] = torch.zeros(batch_size, max_node_num, max_node_num, edge_feat_size, dtype=torch.long)
        batch["spatial_pos"] = torch.zeros(batch_size, max_node_num, max_node_num, dtype=torch.long)
        batch["in_degree"] = torch.zeros(batch_size, max_node_num, dtype=torch.long)
        batch["input_nodes"] = torch.zeros(batch_size, max_node_num, node_feat_size, dtype=torch.long)
        batch["input_edges"] = torch.zeros(
            batch_size, max_node_num, max_node_num, max_dist, edge_input_size, dtype=torch.long
        )

        aug_added_edges = []  # List of (src, dst) tuples per graph
        aug_removed_edges = []
        aug_original_edges = []
        # Auxiliary edge augmentation: add/remove random edges and record them for loss
        for ix, f in enumerate(features):
            for k in ["attn_bias", "attn_edge_type", "spatial_pos", "in_degree", "input_nodes", "input_edges"]:
                f[k] = torch.tensor(f[k])

            if len(f["attn_bias"][1:, 1:][f["spatial_pos"] >= self.spatial_pos_max]) > 0:
                f["attn_bias"][1:, 1:][f["spatial_pos"] >= self.spatial_pos_max] = float("-inf")

            batch["attn_bias"][ix, : f["attn_bias"].shape[0], : f["attn_bias"].shape[1]] = f["attn_bias"]
            batch["attn_edge_type"][ix, : f["attn_edge_type"].shape[0], : f["attn_edge_type"].shape[1], :] = f[
                "attn_edge_type"
            ]
            batch["spatial_pos"][ix, : f["spatial_pos"].shape[0], : f["spatial_pos"].shape[1]] = f["spatial_pos"]
            batch["in_degree"][ix, : f["in_degree"].shape[0]] = f["in_degree"]
            batch["input_nodes"][ix, : f["input_nodes"].shape[0], :] = f["input_nodes"]
            batch["input_edges"][
                ix, : f["input_edges"].shape[0], : f["input_edges"].shape[1], : f["input_edges"].shape[2], :
            ] = f["input_edges"]

            # --- Augmentation ---
            if self.config.augment_edges:
                edge_index = torch.tensor(f["edge_index"], dtype=torch.long)
                num_nodes = f["input_nodes"].shape[0]
                # Make undirected for augmentation
                edge_index = to_undirected(edge_index)
                edge_set = set((int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1]))
                all_possible = set((i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j)
                non_edges = list(all_possible - edge_set)
                # Randomly add/remove edges
                n_add = max(1, int(0.05 * len(non_edges)))
                n_remove = max(1, int(0.05 * edge_index.shape[1]))
                added = random.sample(non_edges, min(n_add, len(non_edges))) if len(non_edges) > 0 else []
                removed = random.sample(list(edge_set), min(n_remove, len(edge_set))) if len(edge_set) > 0 else []
                aug_added_edges.append(torch.tensor(added, dtype=torch.long) if added else torch.empty((0,2), dtype=torch.long))
                aug_removed_edges.append(torch.tensor(removed, dtype=torch.long) if removed else torch.empty((0,2), dtype=torch.long))
                aug_original_edges.append(edge_index.clone())

        batch["out_degree"] = batch["in_degree"]
        batch["edge_index"] = [i["edge_index"] for i in features]

        batch["aug_added_edges"] = aug_added_edges if aug_added_edges else None
        batch["aug_removed_edges"] = aug_removed_edges if aug_removed_edges else None
        batch["aug_original_edges"] = aug_original_edges if aug_original_edges else None

        sample = features[0]["labels"]
        if len(sample) == 1:  # one task
            if isinstance(sample[0], float):  # regression
                batch["labels"] = torch.from_numpy(np.concatenate([i["labels"] for i in features]))
            else:  # binary classification
                batch["labels"] = torch.from_numpy(np.concatenate([i["labels"] for i in features]))
        else:  # multi task classification, left to float to keep the NaNs
            batch["labels"] = torch.from_numpy(np.stack([i["labels"] for i in features], axis=0))
        return batch
