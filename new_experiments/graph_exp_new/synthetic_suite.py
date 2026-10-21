# synthetic_suite.py
"""Synthetic path-task suite: controlled graphs where the label depends on one
information source (node features, a direct edge, or the edges on the s->t
shortest path). Node-level target on t; continuous edge features.

Graphs come from library generators (networkx, scipy.spatial). Each dataset is
saved as train/val/test PyG files under --out and loaded with
``SyntheticPathDataset(root, split)``.

Tasks (label = 1[score > train median], label noise on train only):
  1  score = a.x_s + b.x_t                       node features only (control)
  2  score = w.e_st, k = 1                        direct edge
  3  score = mean_r w.e_r over the s->t path      path aggregation (no decoys)
  4  score = sum_r c_r w.e_r, c_r = linspace(-1,1) order dependence
  5  score = w.e_p, p one-hot in s's features     query-conditioned path lookup
  6  task 3 with decoy edges                      path vs neighbourhood
  7  multiple shortest paths: canonical path or   path-choice sensitivity
     mean over all shortest-path edges (--path_mode)

Families: tree, path, ring, grid, er, ba.
Presets (emulate real dataset groups): peptide_like, molecule_like, tsp_like,
superpixel_like, reddit_like, dense_like, community_like.

Example:
  python synthetic_suite.py --task 3 --family tree --n 200 --k 2,5,10,20 \\
      --edge_dim 8 --out synthetic
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from typing import List, Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


# ---------------------------------------------------------------------------
# Graph families (library generators; output: connected, labels 0..n-1).
# ---------------------------------------------------------------------------
def _seed(rng) -> int:
    return int(rng.integers(0, 2**31 - 1))


def _lcc(G: nx.Graph) -> nx.Graph:
    nodes = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(nodes).copy())


def _ring_closures(G: nx.Graph, n_rings: int, rng, max_deg: int = 4) -> nx.Graph:
    """Close 5- or 6-rings by linking nodes at distance 4 or 5 (molecule-like)."""
    nodes = list(G.nodes)
    for _ in range(n_rings * 10):
        if n_rings <= 0:
            break
        u = nodes[int(rng.integers(len(nodes)))]
        if G.degree(u) >= max_deg:
            continue
        near = nx.single_source_shortest_path_length(G, u, cutoff=5)
        cand = [v for v, d in near.items() if d in (4, 5) and G.degree(v) < max_deg]
        if cand:
            G.add_edge(u, cand[int(rng.integers(len(cand)))])
            n_rings -= 1
    return G


def _random_tree(n, rng):
    if hasattr(nx, "random_labeled_tree"):          # networkx >= 3.3
        return nx.random_labeled_tree(n, seed=_seed(rng))
    return nx.random_tree(n, seed=_seed(rng))


def _points(n, rng):
    return rng.random((n, 2))


def gen_tree(n, deg, rng):
    return _random_tree(n, rng)


def gen_path(n, deg, rng):
    return nx.path_graph(n)


def gen_ring(n, deg, rng):
    return nx.cycle_graph(n)


def gen_grid(n, deg, rng):
    side = max(2, int(round(math.sqrt(n))))
    return nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))


def gen_er(n, deg, rng):
    return _lcc(nx.gnp_random_graph(n, min(1.0, deg / max(n - 1, 1)), seed=_seed(rng)))


def gen_ba(n, deg, rng):
    return nx.barabasi_albert_graph(n, max(1, int(round(deg / 2))), seed=_seed(rng))


def gen_peptide_like(n, deg, rng):
    """Long backbone with short side chains (caterpillar/lobster) + few rings.
    Peptides-func: ~151 nodes, degree ~2, diameter ~57."""
    p1, p2 = 0.9, 0.67                         # nodes ~= backbone * (1 + p1 + p1 p2)
    backbone = max(2, int(round(n / (1 + p1 + p1 * p2))))
    G = nx.convert_node_labels_to_integers(nx.random_lobster(backbone, p1, p2, seed=_seed(rng)))
    return _ring_closures(G, max(0, int(round(0.03 * G.number_of_nodes()))), rng)


def gen_molecule_like(n, deg, rng):
    """Small tree + 5/6-ring closures (ogbg-mol*, QM9, PCQM: ~15-30 nodes)."""
    G = _random_tree(n, rng)
    return _ring_closures(G, max(1, n // 10), rng)


def gen_tsp_like(n, deg, rng):
    """Symmetrised k-NN graph of random 2D points (TSP benchmark)."""
    from scipy.spatial import cKDTree
    pts = _points(n, rng)
    kk = max(1, int(deg))
    _, nbr = cKDTree(pts).query(pts, k=min(kk + 1, n))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((i, int(j)) for i in range(n) for j in nbr[i, 1:])
    return _lcc(G)


def gen_superpixel_like(n, deg, rng):
    """Delaunay triangulation of random points: planar region adjacency,
    degree ~6 (PascalVOC-SP / COCO-SP rag-boundary: degree ~5.6)."""
    from scipy.spatial import Delaunay
    tri = Delaunay(_points(n, rng)).simplices
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for a, b, c in tri:
        G.add_edges_from([(int(a), int(b)), (int(b), int(c)), (int(a), int(c))])
    return G


def gen_reddit_like(n, deg, rng):
    """Preferential-attachment tree: power-law, degree ~2 (REDDIT-*, MalNet)."""
    return nx.barabasi_albert_graph(n, 1, seed=_seed(rng))


def gen_dense_like(n, deg, rng):
    """Power-law with clustering, short paths (ogbg-ppa, COLLAB)."""
    return nx.powerlaw_cluster_graph(n, max(1, int(round(deg / 2))), 0.3, seed=_seed(rng))


def gen_community_like(n, deg, rng, blocks: int = 4, ratio: float = 8.0):
    """Stochastic block model (subgraphs of large node-level graphs)."""
    size = n // blocks
    sizes = [size] * (blocks - 1) + [n - size * (blocks - 1)]
    p_out = deg / (ratio * size + (n - size))
    p_in = min(1.0, ratio * p_out)
    P = [[p_in if i == j else p_out for j in range(blocks)] for i in range(blocks)]
    return _lcc(nx.stochastic_block_model(sizes, P, seed=_seed(rng)))


FAMILIES = {
    "tree": gen_tree, "path": gen_path, "ring": gen_ring, "grid": gen_grid,
    "er": gen_er, "ba": gen_ba,
    "peptide_like": gen_peptide_like, "molecule_like": gen_molecule_like,
    "tsp_like": gen_tsp_like, "superpixel_like": gen_superpixel_like,
    "reddit_like": gen_reddit_like, "dense_like": gen_dense_like,
    "community_like": gen_community_like,
}

# Defaults (n, avg degree) used when --n / --deg are not given.
PRESET_DEFAULTS = {
    "peptide_like": (150, 2), "molecule_like": (25, 2), "tsp_like": (200, 8),
    "superpixel_like": (480, 6), "reddit_like": (430, 2), "dense_like": (240, 9),
    "community_like": (300, 8),
}


# ---------------------------------------------------------------------------
# Shortest-path helpers.
# ---------------------------------------------------------------------------
def _bfs(adj: List[List[int]], s: int):
    """Distances, shortest-path counts (capped at 2) and predecessor lists."""
    n = len(adj)
    dist = np.full(n, -1, dtype=np.int64)
    cnt = np.zeros(n, dtype=np.int64)
    preds: List[List[int]] = [[] for _ in range(n)]
    dist[s], cnt[s] = 0, 1
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
            if dist[v] == dist[u] + 1:
                cnt[v] = min(2, cnt[v] + cnt[u])
                preds[v].append(u)
    return dist, cnt, preds


def _canonical_path(preds, s: int, t: int) -> List[int]:
    """Node sequence s..t, taking the smallest-index predecessor at each step."""
    path = [t]
    while path[-1] != s:
        path.append(min(preds[path[-1]]))
    return path[::-1]


def _sample_pair(adj, k: int, need: str, rng, tries: int = 30):
    """Random (s, t) at distance k. need: 'any' | 'unique' | 'multi' paths."""
    n = len(adj)
    for s in rng.permutation(n)[:tries]:
        dist, cnt, preds = _bfs(adj, int(s))
        cand = np.flatnonzero(dist == k)
        if need == "unique":
            cand = cand[cnt[cand] == 1]
        elif need == "multi":
            cand = cand[cnt[cand] > 1]
        if cand.size:
            return int(s), int(cand[int(rng.integers(cand.size))]), dist, preds
    return None


# ---------------------------------------------------------------------------
# One labelled graph.
# ---------------------------------------------------------------------------
def make_graph(cfg: dict, rng, w: np.ndarray, a: np.ndarray, b: np.ndarray,
               max_regen: int = 50) -> Data:
    task, k = cfg["task"], (1 if cfg["task"] == 2 else cfg["k"])
    need = {3: "unique", 4: "unique", 5: "unique", 6: "unique", 7: "multi"}.get(task, "any")
    gen = FAMILIES[cfg["family"]]

    for _ in range(max_regen):
        G = gen(cfg["n"], cfg["deg"], rng)
        n = G.number_of_nodes()
        adj = [list(G.neighbors(u)) for u in range(n)]
        hit = _sample_pair(adj, k, need, rng)
        if hit is not None:
            break
    else:
        raise ValueError(
            f"Could not find an (s, t) pair at distance {k} with '{need}' shortest "
            f"paths in family={cfg['family']} n={cfg['n']} deg={cfg['deg']} "
            f"after {max_regen} graphs. Increase n or lower k.")
    s, t, dist_s, preds = hit

    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
    eid = {e: i for i, e in enumerate(edges)}
    D = cfg["edge_dim"]
    efeat = rng.normal(size=(len(edges), D)).astype(np.float32)

    # Edges on the labelled path (ordered from s) or on all shortest paths.
    path_nodes = _canonical_path(preds, s, t)
    path_eids = [eid[(min(u, v), max(u, v))] for u, v in zip(path_nodes[:-1], path_nodes[1:])]
    label_eids = path_eids
    if task == 7 and cfg["path_mode"] == "invariant":
        dist_t, _, _ = _bfs(adj, t)
        label_eids = [i for i, (u, v) in enumerate(edges)
                      if dist_s[u] + 1 + dist_t[v] == k or dist_s[v] + 1 + dist_t[u] == k]

    decoys = cfg["decoys"]
    if decoys == "auto":
        decoys = "off" if task == 3 else "on"
    if decoys == "off" and task != 1:
        keep = np.zeros(len(edges), dtype=bool)
        keep[label_eids] = True
        efeat[~keep] = 0.0

    z = efeat @ w                                                # (E,) per-edge score
    F = cfg["node_dim"]
    x_rand = rng.normal(size=(n, F)).astype(np.float32)
    extra = []
    p = -1
    if task == 1:
        score = float(x_rand[s] @ a + x_rand[t] @ b)
    elif task == 2:
        score = float(z[path_eids[0]])
    elif task in (3, 6):
        score = float(z[path_eids].mean())
    elif task == 4:
        c = np.linspace(-1.0, 1.0, k)
        score = float((c * z[path_eids]).sum() / np.sqrt((c ** 2).sum()))
    elif task == 5:
        p = int(rng.integers(k))
        onehot = np.zeros((n, k), dtype=np.float32)
        onehot[s, p] = 1.0                                       # query position on s
        extra.append(onehot)
        score = float(z[path_eids[p]])
    elif task == 7:
        score = float(z[label_eids].mean())
    else:
        raise ValueError(f"Unknown task {task}")

    flags = np.zeros((n, 2), dtype=np.float32)
    flags[s, 0], flags[t, 1] = 1.0, 1.0
    x = np.concatenate([x_rand, flags] + extra, axis=1)

    ei = np.array(edges, dtype=np.int64).T
    edge_index = np.concatenate([ei, ei[::-1]], axis=1)          # both directions
    edge_attr = np.concatenate([efeat, efeat], axis=0)
    y = torch.full((n,), -1, dtype=torch.long)                   # label set later on t
    return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr), y=y, num_nodes=n,
                score=torch.tensor([score]), src_node=torch.tensor([s]),
                tgt_node=torch.tensor([t]), query_pos=torch.tensor([p]))


# ---------------------------------------------------------------------------
# Dataset generation / loading.
# ---------------------------------------------------------------------------
def dataset_name(cfg: dict) -> str:
    k = 1 if cfg["task"] == 2 else cfg["k"]
    name = (f"t{cfg['task']}_{cfg['family']}_n{cfg['n']}_d{cfg['deg']}_k{k}"
            f"_e{cfg['edge_dim']}_noise{cfg['noise']}_dec{cfg['decoys']}")
    if cfg["task"] == 7:
        name += f"_{cfg['path_mode']}"
    return name


def generate(cfg: dict, out_root: str) -> str:
    rng = np.random.default_rng(cfg["seed"])
    D, F = cfg["edge_dim"], cfg["node_dim"]
    w = rng.normal(size=D); w /= np.linalg.norm(w)
    a = rng.normal(size=F); a /= np.linalg.norm(a)
    b = rng.normal(size=F); b /= np.linalg.norm(b)

    splits = {sp: [make_graph(cfg, rng, w, a, b) for _ in range(cfg[f"num_{sp}"])]
              for sp in ("train", "val", "test")}

    # Balanced binary labels: threshold at the train-split median score.
    thr = float(np.median([g.score.item() for g in splits["train"]]))
    for sp, graphs in splits.items():
        for g in graphs:
            lab = int(g.score.item() > thr)
            if sp == "train" and cfg["noise"] > 0 and rng.random() < cfg["noise"]:
                lab = 1 - lab                                    # clean val/test
            g.y[g.tgt_node.item()] = lab

    root = os.path.join(out_root, dataset_name(cfg))
    os.makedirs(root, exist_ok=True)
    for sp, graphs in splits.items():
        torch.save(InMemoryDataset.collate(graphs), os.path.join(root, f"{sp}.pt"))
    meta = dict(cfg, threshold=thr, node_feat_dim=int(splits["train"][0].x.size(1)),
                edge_feat_dim=D, stats=graph_stats(splits["train"]))
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[synthetic] {root}: " + json.dumps(meta["stats"]), flush=True)
    return root


class SyntheticPathDataset(InMemoryDataset):
    """Loads one split written by ``generate``."""

    def __init__(self, root: str, split: str = "train", transform=None):
        super().__init__(None, transform)
        path = os.path.join(root, f"{split}.pt")
        try:
            self.data, self.slices = torch.load(path, weights_only=False)
        except TypeError:                                        # torch < 1.13
            self.data, self.slices = torch.load(path)
        with open(os.path.join(root, "meta.json")) as f:
            self.meta = json.load(f)


# ---------------------------------------------------------------------------
# Statistics (apply to synthetic or real PyG graphs for comparison).
# ---------------------------------------------------------------------------
def graph_stats(graphs, max_graphs: int = 50, sources_per_graph: int = 20,
                seed: int = 0) -> dict:
    """Mean nodes, degree, diameter (lower bound from sampled BFS sources),
    shortest-path length and fraction of pairs with a unique shortest path."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(graphs))[:max_graphs]
    nodes, degs, diams, spl, uniq = [], [], [], [], []
    for i in idx:
        g = graphs[int(i)]
        n = int(g.num_nodes)
        adj = [[] for _ in range(n)]
        for u, v in g.edge_index.t().tolist():
            if u != v:
                adj[u].append(v)
        adj = [sorted(set(a)) for a in adj]
        nodes.append(n)
        degs.append(sum(len(a) for a in adj) / max(n, 1))
        best = 0
        for s in rng.permutation(n)[:sources_per_graph]:
            dist, cnt, _ = _bfs(adj, int(s))
            reach = dist > 0
            if reach.any():
                best = max(best, int(dist.max()))
                spl.append(float(dist[reach].mean()))
                uniq.append(float((cnt[reach] == 1).mean()))
        diams.append(best)
    m = lambda v: round(float(np.mean(v)), 3) if v else None
    return {"avg_nodes": m(nodes), "avg_degree": m(degs), "diameter": m(diams),
            "avg_spl": m(spl), "frac_unique_sp": m(uniq)}


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _ints(s: str) -> List[int]:
    return [int(v) for v in str(s).split(",") if v != ""]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", type=int, required=True, choices=range(1, 8))
    p.add_argument("--family", required=True, choices=sorted(FAMILIES))
    p.add_argument("--n", type=str, default=None, help="Comma list, e.g. 50,200,500")
    p.add_argument("--deg", type=str, default=None, help="Average degree, comma list")
    p.add_argument("--k", type=str, default="5", help="s->t distance, comma list")
    p.add_argument("--edge_dim", type=str, default="8", help="Comma list, e.g. 1,8,32")
    p.add_argument("--node_dim", type=int, default=8)
    p.add_argument("--noise", type=str, default="0", help="Train label noise, comma list")
    p.add_argument("--decoys", choices=["auto", "on", "off"], default="auto",
                   help="auto: off for task 3, on otherwise")
    p.add_argument("--path_mode", choices=["canonical", "invariant"], default="canonical",
                   help="Task 7 label: one canonical path or all shortest paths")
    p.add_argument("--num_train", type=int, default=2000)
    p.add_argument("--num_val", type=int, default=500)
    p.add_argument("--num_test", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="synthetic")
    args = p.parse_args()

    dn, dd = PRESET_DEFAULTS.get(args.family, (200, 4))
    grid = [(n, d, k, e, float(z))
            for n in (_ints(args.n) if args.n else [dn])
            for d in (_ints(args.deg) if args.deg else [dd])
            for k in _ints(args.k)
            for e in _ints(args.edge_dim)
            for z in args.noise.split(",")]
    for n, d, k, e, z in grid:
        cfg = dict(task=args.task, family=args.family, n=n, deg=d, k=k, edge_dim=e,
                   node_dim=args.node_dim, noise=z, decoys=args.decoys,
                   path_mode=args.path_mode, num_train=args.num_train,
                   num_val=args.num_val, num_test=args.num_test, seed=args.seed)
        try:
            generate(cfg, args.out)
        except ValueError as err:                               # infeasible config
            print(f"[synthetic] skipped {dataset_name(cfg)}: {err}", flush=True)


if __name__ == "__main__":
    main()

