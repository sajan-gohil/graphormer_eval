import torch
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from graphgps.loader.dataset.peptides_functional import PeptidesFunctionalDataset
from torch_geometric.graphgym.config import cfg

cfg.dataset.dir = 'datasets'
cfg.dataset.name = 'peptides-functional'

dataset = PeptidesFunctionalDataset('datasets')
_data = dataset.data
_slices = dataset.slices

x_slices = _slices['x']
all_ei = _data.edge_index
num_graphs = len(x_slices) - 1

print(f"Num graphs: {num_graphs}")

max_ks = []
for i in range(10):
    node_start = int(x_slices[i])
    node_end = int(x_slices[i + 1])
    n = node_end - node_start

    edge_mask = (
        (all_ei[0] >= node_start) & (all_ei[0] < node_end) &
        (all_ei[1] >= node_start) & (all_ei[1] < node_end)
    )
    ei = all_ei[:, edge_mask].numpy()
    ei = ei - node_start
    
    adj = np.zeros((n, n), dtype=np.float32)
    adj[ei[0], ei[1]] = 1.0
    
    dist = floyd_warshall(adj, directed=False, unweighted=True)
    dist = np.where(np.isfinite(dist), dist, -1).astype(np.int32)
    actual_max = int(dist.max())
    
    print(f"Graph {i}: {n} nodes, {ei.shape[1]} edges, max distance: {actual_max}")
    max_ks.append(actual_max)

print(f"Average max distance for first 10 graphs: {np.mean(max_ks)}")
