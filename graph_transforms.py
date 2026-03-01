"""
Graph transformation utilities for studying different structural approaches
to handling long-range dependencies in graph transformers.

Each function targets a specific structural aspect of the graph:

  add_random_edges          – add random shortcut edges to reduce effective distances
  add_edges_bounded_diameter – deterministically add edges so diameter < 6
  add_relay_nodes           – insert virtual relay nodes between distant node pairs
  compute_laplacian_pe      – compute Laplacian positional encodings (eigenvectors)
  add_spectral_attn_bias    – augment the Graphormer attention bias with spectral similarity
"""

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh


# ---------------------------------------------------------------------------
# 1. Random shortcut edges
# ---------------------------------------------------------------------------

def add_random_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    ratio: float = 0.1,
    seed: int = 42,
) -> torch.Tensor:
    """Add random shortcut edges to reduce long-range dependencies.

    Args:
        edge_index: [2, E] source/destination edge indices.
        num_nodes:  number of nodes in the graph.
        ratio:      fraction of existing edges to add as random shortcuts.
        seed:       random seed for reproducibility.

    Returns:
        New edge_index with additional random edges appended.
    """
    rng = np.random.default_rng(seed)
    existing = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    n_add = max(1, int(ratio * len(existing)))

    added: list[tuple[int, int]] = []
    # Allow up to 200× the target count attempts before giving up; this
    # gives ample room to find non-existing edges even in dense graphs.
    attempts, max_attempts = 0, n_add * 200
    while len(added) < n_add and attempts < max_attempts:
        u = int(rng.integers(0, num_nodes))
        v = int(rng.integers(0, num_nodes))
        if u != v and (u, v) not in existing:
            added.append((u, v))
            existing.add((u, v))
        attempts += 1

    if not added:
        return edge_index

    extra_src, extra_dst = zip(*added)
    extra = torch.tensor([list(extra_src), list(extra_dst)], dtype=torch.long)
    return torch.cat([edge_index, extra], dim=1)


# ---------------------------------------------------------------------------
# 2. Deterministic bounded-diameter edges
# ---------------------------------------------------------------------------

def add_edges_bounded_diameter(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_diameter: int = 5,
    max_new_edges: int = 1000,
) -> torch.Tensor:
    """Deterministically add edges so the graph diameter is at most *max_diameter*.

    For every source node *u* a BFS is run and any node *v* that is either
    unreachable or farther than *max_diameter* hops receives a direct edge
    (u, v).  The adjacency list is updated after each new edge so that later
    BFS calls benefit from previously added shortcuts.

    Args:
        edge_index:    [2, E] edge indices.
        num_nodes:     number of nodes.
        max_diameter:  target maximum shortest-path distance (exclusive upper bound).
        max_new_edges: cap on total new edges added (both directions count once).

    Returns:
        New edge_index with diameter <= max_diameter.
    """
    adj: list[set[int]] = [set() for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        adj[u].add(v)

    def bfs(start: int) -> dict[int, int]:
        dist = {start: 0}
        queue, head = [start], 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    new_edges: list[tuple[int, int]] = []

    for u in range(num_nodes):
        if len(new_edges) >= max_new_edges:
            break
        dists = bfs(u)
        for v in range(u + 1, num_nodes):  # u < v avoids duplicate pairs
            if len(new_edges) >= max_new_edges:
                break
            d = dists.get(v, num_nodes + 1)
            if d > max_diameter:
                if (u, v) not in edge_set:
                    new_edges.append((u, v))
                    edge_set.add((u, v))
                    # Update adjacency so subsequent BFS calls see the shortcut
                    adj[u].add(v)
                    adj[v].add(u)

    if not new_edges:
        return edge_index

    # Add in both directions so the graph remains undirected
    src = [e[0] for e in new_edges] + [e[1] for e in new_edges]
    dst = [e[1] for e in new_edges] + [e[0] for e in new_edges]
    extra = torch.tensor([src, dst], dtype=torch.long)
    return torch.cat([edge_index, extra], dim=1)


# ---------------------------------------------------------------------------
# 3. Virtual relay nodes
# ---------------------------------------------------------------------------

def add_relay_nodes(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    max_dist: int = 5,
    max_relays: int = 50,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Insert relay (virtual) nodes between pairs of nodes whose shortest-path
    distance exceeds *max_dist*, halving their effective communication distance.

    A relay node is created for each qualifying (u, v) pair.  Its feature
    vector is the element-wise average of u and v (rounded to integer for
    embedding index compatibility).  Four directed edges are added:
    u→relay, relay→u, v→relay, relay→v.

    Args:
        x:          [N, F] node feature tensor (integer-valued embedding indices).
        edge_index: [2, E] edge index tensor.
        edge_attr:  [E] or [E, A] edge attribute tensor.
        max_dist:   distance threshold above which relay nodes are inserted.
        max_relays: maximum number of relay nodes to insert.
        seed:       random seed for reproducible source-node sampling.

    Returns:
        (new_x, new_edge_index, new_edge_attr) with relay nodes appended.
    """
    rng = np.random.default_rng(seed)
    num_nodes = x.shape[0]

    adj: list[set[int]] = [set() for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        adj[u].add(v)

    def bfs(start: int) -> dict[int, int]:
        dist = {start: 0}
        queue, head = [start], 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    # Sample at most 100 source nodes to keep BFS tractable on large graphs;
    # empirically this finds enough distant pairs for most benchmark graphs.
    sample_size = min(num_nodes, 100)
    sample_nodes = rng.choice(num_nodes, sample_size, replace=False).tolist()

    relay_feats: list[torch.Tensor] = []
    relay_edges: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    relay_idx = num_nodes

    for u in sample_nodes:
        if len(relay_feats) >= max_relays:
            break
        dists = bfs(u)
        for v in range(num_nodes):
            if len(relay_feats) >= max_relays:
                break
            if v == u:
                continue
            pair = (min(u, v), max(u, v))
            if pair in seen_pairs:
                continue
            if dists.get(v, num_nodes + 1) > max_dist:
                seen_pairs.add(pair)
                relay_feat = ((x[u].float() + x[v].float()) / 2).long()
                relay_feats.append(relay_feat)
                relay_edges.extend([
                    (u, relay_idx), (relay_idx, u),
                    (v, relay_idx), (relay_idx, v),
                ])
                relay_idx += 1

    if not relay_feats:
        return x, edge_index, edge_attr

    new_x = torch.cat([x, torch.stack(relay_feats, dim=0)], dim=0)

    rsrc, rdst = zip(*relay_edges)
    relay_ei = torch.tensor([list(rsrc), list(rdst)], dtype=torch.long)
    new_ei = torch.cat([edge_index, relay_ei], dim=1)

    ea_cols = edge_attr.shape[1] if edge_attr.dim() > 1 else 1
    relay_ea = torch.ones(len(relay_edges), ea_cols, dtype=edge_attr.dtype)
    if edge_attr.dim() == 1:
        relay_ea = relay_ea.squeeze(1)
    new_ea = torch.cat([edge_attr, relay_ea], dim=0)

    return new_x, new_ei, new_ea


# ---------------------------------------------------------------------------
# 4. Laplacian positional encodings
# ---------------------------------------------------------------------------

def compute_laplacian_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 8,
) -> np.ndarray:
    """Compute Laplacian Positional Encodings (LapPE).

    Returns the *k* non-trivial eigenvectors of the (unnormalized) graph
    Laplacian L = D − A, sorted by ascending eigenvalue.  The trivial constant
    eigenvector (eigenvalue ≈ 0) is discarded.

    Args:
        edge_index: [2, E] edge index tensor.
        num_nodes:  number of nodes N.
        k:          number of eigenvectors to return.

    Returns:
        pe: float32 array of shape [N, k].
    """
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(len(row), dtype=np.float32)
    A = sparse.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    # Symmetrise and binarise
    A = ((A + A.T) > 0).astype(np.float32)
    deg = np.asarray(A.sum(axis=1)).flatten()
    L = sparse.diags(deg) - A

    k_actual = min(k + 1, num_nodes - 2)
    if k_actual < 1:
        return np.zeros((num_nodes, k), dtype=np.float32)

    try:
        eigenvalues, vecs = eigsh(
            L.astype(np.float64), k=k_actual, which="SM", tol=1e-5
        )
        order = np.argsort(eigenvalues)
        vecs = vecs[:, order]
        vecs = vecs[:, 1 : k + 1]  # skip trivial constant eigenvector
        if vecs.shape[1] < k:
            pad = np.zeros((num_nodes, k - vecs.shape[1]), dtype=np.float32)
            vecs = np.concatenate([vecs, pad], axis=1)
    except Exception:
        vecs = np.zeros((num_nodes, k), dtype=np.float32)

    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Spectral attention bias
# ---------------------------------------------------------------------------

def add_spectral_attn_bias(
    attn_bias: np.ndarray,
    edge_index: torch.Tensor,
    num_nodes: int,
    k: int = 8,
    scale: float = 1.0,
) -> np.ndarray:
    """Augment the Graphormer attention bias with a Laplacian spectral term.

    For each node pair (i, j) the following bias is added to
    attn_bias[i+1, j+1] (offset by 1 to skip the global graph token):

        Δbias(i, j) = −scale × ‖PE_i − PE_j‖² / k

    where PE is the Laplacian positional encoding matrix of shape [N, k].
    Closer nodes in spectral space receive a higher (less negative) attention
    bonus.

    Args:
        attn_bias:  [N+1, N+1] attention bias array (graph token at index 0).
        edge_index: [2, E] edge index tensor.
        num_nodes:  number of nodes N.
        k:          number of Laplacian eigenvectors to use.
        scale:      scaling coefficient for the spectral term.

    Returns:
        Modified attn_bias with the spectral term added in the [1:, 1:] block.
    """
    pe = compute_laplacian_pe(edge_index, num_nodes, k=k)  # [N, k]
    pe_t = torch.from_numpy(pe)
    diff = pe_t.unsqueeze(0) - pe_t.unsqueeze(1)  # [N, N, k]
    spectral_sim = -(diff ** 2).sum(dim=-1).numpy() / max(k, 1)  # [N, N]
    attn_bias[1 : num_nodes + 1, 1 : num_nodes + 1] += (
        scale * spectral_sim
    ).astype(np.float32)
    return attn_bias
