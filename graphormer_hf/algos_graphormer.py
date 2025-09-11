import numpy as np
import hashlib

UNREACHABLE_NODE_DISTANCE = 510

# Simple in-memory caches
_fw_cache = {}
_edge_cache = {}

def _hash_array(arr):
    """Hash a NumPy array's content, shape, and dtype for cache key."""
    h = hashlib.sha1()
    h.update(arr.shape.__repr__().encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()

def floyd_warshall(adjacency_matrix):
    adjacency_matrix = np.asarray(adjacency_matrix)
    key = _hash_array(adjacency_matrix)
    if key in _fw_cache:
        return _fw_cache[key]

    nrows, ncols = adjacency_matrix.shape
    assert nrows == ncols
    n = nrows

    M = adjacency_matrix.astype(np.int32, copy=True, order="C")
    path = -1 * np.ones((n, n), dtype=np.int32, order="C")

    for i in range(n):
        M[i, i] = 0
        for j in range(n):
            if i != j and M[i, j] == 0:
                M[i, j] = UNREACHABLE_NODE_DISTANCE

    for k in range(n):
        Mk = M[k, :]
        for i in range(n):
            Mik = M[i, k]
            for j in range(n):
                cost = Mik + Mk[j]
                if M[i, j] > cost:
                    M[i, j] = cost
                    path[i, j] = k

    for i in range(n):
        for j in range(n):
            if M[i, j] >= UNREACHABLE_NODE_DISTANCE:
                path[i, j] = UNREACHABLE_NODE_DISTANCE
                M[i, j] = UNREACHABLE_NODE_DISTANCE

    _fw_cache[key] = (M, path)
    return M, path

def get_all_edges(path, i, j):
    k = int(path[i, j])
    if k == -1:
        return []
    return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)

def gen_edge_input(max_dist, path_array, edge_feat):
    path_array = np.asarray(path_array)
    edge_feat = np.asarray(edge_feat)

    key = (max_dist, _hash_array(path_array), _hash_array(edge_feat))
    if key in _edge_cache:
        return _edge_cache[key]

    nrows, ncols = path_array.shape
    assert nrows == ncols
    n = nrows
    max_dist = int(max_dist)

    path_copy = path_array.astype(np.int64, copy=True, order="C")
    edge_feat_copy = edge_feat.astype(np.int64, copy=True, order="C")

    num_edge_features = edge_feat.shape[-1]
    edge_fea_all = -1 * np.ones((n, n, max_dist, num_edge_features), dtype=np.int32, order="C")

    for i in range(n):
        for j in range(n):
            if i == j or path_copy[i, j] == UNREACHABLE_NODE_DISTANCE:
                continue
            intermediates = get_all_edges(path_copy, i, j)
            full_path = [i] + intermediates + [j]
            for k in range(len(full_path) - 1):
                u, v = full_path[k], full_path[k + 1]
                edge_fea_all[i, j, k, :] = edge_feat_copy[u, v, :].astype(np.int32)

    _edge_cache[key] = edge_fea_all
    return edge_fea_all
