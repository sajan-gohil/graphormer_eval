# synthetic_suite.py
"""Synthetic path-task suite: controlled graphs where the label depends on one
information source (node features, a direct edge, or the edges on the s->t
shortest path). Node-level target(s); continuous edge features.

Graphs come from library generators (networkx, scipy.spatial). Each dataset is
saved as train/val/test PyG files under --out and loaded with
``SyntheticPathDataset(root, split)``.

Tasks (label = 1[score > train median], label noise on train only;
z_e = w.e_e, path edges ordered from s):
  1   score = a.x_s + b.x_t                      node features only (control)
  2   score = z_st, k = 1                         direct edge
  3   score = mean_r z_r over the s->t path       path aggregation (no decoys)
  4b  score = 1[argmax_r z_r < argmin_r z_r]      order comparison (dataset "t4b_")
  5   score = z_p, p one-hot in s's features      query-conditioned path lookup
  6   task 3 with decoy edges                     path vs neighbourhood
  7   multiple shortest paths: canonical path or  path-choice sensitivity
      mean over all shortest-path edges (--path_mode)

Distance: --k (fixed) or --k_range lo,hi (drawn per graph / per pair).

Layouts:
  single (default)    one (s, t) pair, one labelled node t
  --pairs m           m (s_j, t_j) pairs at different k (from --k_range), linked
                      by a shared random pair-ID vector; m labelled nodes
  --sources m --rule nearest
                      one t, m sources; label from the nearest source, the
                      others strictly farther (distance in (k, 2k])
  --sources m --rule weighted
                      one t, m sources at distances in [1, k] (or --k_range);
                      score = sum_j gamma^d_j * term_j (task 1: + b.x_t)

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
import shutil
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


def gen_peptide_like(n, deg, rng, tol: float = 0.2, max_tries: int = 200):
    """Long backbone with short side chains (lobster) + few rings.
    Peptides-func: ~151 nodes, degree ~2, diameter ~57.

    nx.random_lobster adds leaves while U < p1 (geometric), so the expected
    leaves per backbone node are p1/(1-p1) and 2nd-level nodes per leaf
    p2/(1-p2). Its backbone length is uniform on [0, 2*n_arg], so graphs are
    resampled until the node count is within +-tol of n.
    """
    p1, p2 = 0.5, 0.4
    a, b = p1 / (1 - p1), p2 / (1 - p2)         # 1.0 and 0.67
    backbone = max(2, int(round(n / (1 + a + a * b))))
    G = None
    for _ in range(max_tries):
        G = nx.random_lobster(backbone, p1, p2, seed=_seed(rng))
        if abs(G.number_of_nodes() - n) <= tol * n:
            break
    G = nx.convert_node_labels_to_integers(G)
    return _ring_closures(G, max(0, int(round(0.03 * n))), rng)


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


def _need_mask(cnt: np.ndarray, need: str) -> np.ndarray:
    if need == "unique":
        return cnt == 1
    if need == "multi":
        return cnt > 1
    return np.ones(cnt.shape, dtype=bool)


def _pick(cands, rng) -> int:
    return int(cands[int(rng.integers(len(cands)))])


def _sample_layout(adj, mode: str, ks: List[int], m: int, need: str, rng,
                   tries: int = 30):
    """Choose targets and sources. Returns [(t, [sources], (dist, cnt, preds)
    of a BFS from t)], or None if this graph has no valid layout.

    single / pairs: one group per pair, source at distance ks[j] from its target,
                    all nodes distinct.
    nearest:        one target; source 0 at ks[0], m-1 other sources strictly
                    farther, at distance in (ks[0], 2*ks[0]].
    weighted:       one target; m sources at distances in ks, one per distinct
                    distance where possible.
    """
    n = len(adj)
    if mode in ("single", "pairs"):
        used, groups = set(), []
        for kj in ks:
            for t in rng.permutation(n)[:tries]:
                t = int(t)
                if t in used:
                    continue
                dist, cnt, preds = _bfs(adj, t)
                cand = [int(c) for c in np.flatnonzero((dist == kj) & _need_mask(cnt, need))
                        if int(c) not in used]
                if cand:
                    s = _pick(cand, rng)
                    groups.append((t, [s], (dist, cnt, preds)))
                    used.update((s, t))
                    break
            else:
                return None
        return groups

    for t in rng.permutation(n)[:tries]:
        t = int(t)
        dist, cnt, preds = _bfs(adj, t)
        ok = _need_mask(cnt, need)
        if mode == "nearest":
            k0 = ks[0]
            c0 = np.flatnonzero((dist == k0) & ok)
            far = np.flatnonzero((dist > k0) & (dist <= 2 * k0))
            if c0.size == 0 or far.size < m - 1:
                continue
            others = rng.choice(far, size=m - 1, replace=False).tolist() if m > 1 else []
            return [(t, [_pick(c0, rng)] + [int(o) for o in others], (dist, cnt, preds))]
        # weighted
        cand = np.flatnonzero(np.isin(dist, ks) & ok)
        if cand.size < m:
            continue
        by_d = {}
        for c in cand.tolist():
            by_d.setdefault(int(dist[c]), []).append(int(c))
        ds = list(by_d)
        rng.shuffle(ds)
        chosen = [_pick(by_d[d], rng) for d in ds[:m]]           # distinct distances first
        rest = [c for c in cand.tolist() if c not in chosen]
        while len(chosen) < m:
            chosen.append(rest.pop(int(rng.integers(len(rest)))))
        return [(t, chosen, (dist, cnt, preds))]
    return None


def _k_bounds(cfg: dict):
    """(lo, hi) range of s->t distances for this config."""
    if cfg["task"] == 2:
        return 1, 1
    lo, hi = cfg["k_range"] if cfg.get("k_range") else (cfg["k"], cfg["k"])
    if cfg["mode"] == "weighted" and not cfg.get("k_range"):
        lo = 1                                     # sources anywhere within k
    if cfg["task"] == 4:
        lo = max(lo, 2)                            # order needs >= 2 edges
    return lo, hi


def _draw_ks(cfg: dict, lo: int, hi: int, rng) -> List[int]:
    if cfg["mode"] == "weighted":
        return list(range(lo, hi + 1))             # allowed source distances
    if cfg["mode"] == "pairs":
        vals = np.arange(lo, hi + 1)
        rep = vals.size < cfg["m"]                 # distinct k per pair when possible
        return [int(v) for v in rng.choice(vals, size=cfg["m"], replace=rep)]
    return [int(rng.integers(lo, hi + 1))]


def _pair_score(task: int, z: np.ndarray, path_eids: List[int],
                label_eids: List[int], p: int) -> float:
    """Edge-based score of one (source, target) pair (tasks 2-7)."""
    if task == 2:
        return float(z[path_eids[0]])
    if task in (3, 6):
        return float(z[path_eids].mean())
    if task == 4:                                   # 4b: is max edge before min edge?
        zp = z[path_eids]
        return 0.5 if int(np.argmax(zp)) < int(np.argmin(zp)) else -0.5
    if task == 5:
        return float(z[path_eids[p]])
    if task == 7:
        return float(z[label_eids].mean())
    raise ValueError(f"Unknown task {task}")


# ---------------------------------------------------------------------------
# One labelled graph.
# ---------------------------------------------------------------------------
def make_graph(cfg: dict, rng, w: np.ndarray, a: np.ndarray, b: np.ndarray,
               max_regen: int = 50) -> Data:
    """One graph with one or more labelled targets (see module docstring)."""
    task, mode, m = cfg["task"], cfg["mode"], cfg["m"]
    need = {3: "unique", 4: "unique", 5: "unique", 6: "unique", 7: "multi"}.get(task, "any")
    gen = FAMILIES[cfg["family"]]
    lo, hi = _k_bounds(cfg)

    for _ in range(max_regen):
        G = gen(cfg["n"], cfg["deg"], rng)
        n = G.number_of_nodes()
        adj = [list(G.neighbors(u)) for u in range(n)]
        ks = _draw_ks(cfg, lo, hi, rng)
        groups = _sample_layout(adj, mode, ks, m, need, rng)
        if groups is not None:
            break
    else:
        raise ValueError(
            f"Could not place mode={mode} m={m} with distances {lo}..{hi} and "
            f"'{need}' shortest paths in family={cfg['family']} n={cfg['n']} "
            f"deg={cfg['deg']} after {max_regen} graphs. Increase n or lower k.")

    edges = [(min(u, v), max(u, v)) for u, v in G.edges()]
    eid = {e: i for i, e in enumerate(edges)}
    D = cfg["edge_dim"]
    efeat = rng.normal(size=(len(edges), D)).astype(np.float32)

    # Per (source, target): path edges ordered from the source, label edges.
    pairs = []                                     # (t, s, d, path_eids, label_eids)
    for t, sources, (dist_t, _, preds_t) in groups:
        for s in sources:
            d = int(dist_t[s])
            nodes = _canonical_path(preds_t, t, s)[::-1]          # s .. t
            path_eids = [eid[(min(u, v), max(u, v))] for u, v in zip(nodes[:-1], nodes[1:])]
            label_eids = path_eids
            if task == 7 and cfg["path_mode"] == "invariant":
                dist_s, _, _ = _bfs(adj, s)
                label_eids = [i for i, (u, v) in enumerate(edges)
                              if dist_s[u] + 1 + dist_t[v] == d or dist_s[v] + 1 + dist_t[u] == d]
            pairs.append((t, s, d, path_eids, label_eids))

    decoys = cfg["decoys"]
    if decoys == "auto":
        decoys = "off" if task == 3 else "on"
    if decoys == "off" and task != 1:
        keep = np.zeros(len(edges), dtype=bool)
        for _, s, _, _, le in pairs:
            if mode == "nearest" and s != groups[0][1][0]:
                continue                           # only the nearest source's path counts
            keep[le] = True
        efeat[~keep] = 0.0
    z = efeat @ w                                                 # (E,) per-edge score

    F = cfg["node_dim"]
    x_rand = rng.normal(size=(n, F)).astype(np.float32)
    flags = np.zeros((n, 2), dtype=np.float32)
    blocks = [x_rand, flags]

    # Task 5: query position p (one-hot, width = largest possible distance) on each source.
    ps = []
    if task == 5:
        width = 2 * hi if mode == "nearest" else hi
        onehot = np.zeros((n, width), dtype=np.float32)
        for _, s, d, _, _ in pairs:
            p = int(rng.integers(d))
            onehot[s, p] = 1.0
            ps.append(p)
        blocks.append(onehot)
    else:
        ps = [-1] * len(pairs)

    # Pair IDs (pairs mode, m > 1): shared random vector on s_j and t_j.
    if mode == "pairs" and m > 1:
        ids = rng.normal(size=(len(pairs), cfg["id_dim"])).astype(np.float32)
        ids /= np.linalg.norm(ids, axis=1, keepdims=True)
        idblk = np.zeros((n, cfg["id_dim"]), dtype=np.float32)
        for j, (t, s, _, _, _) in enumerate(pairs):
            idblk[s], idblk[t] = ids[j], ids[j]
        blocks.append(idblk)

    for t, s, _, _, _ in pairs:
        flags[s, 0], flags[t, 1] = 1.0, 1.0

    def src_term(j):
        t, s, _, pe, le = pairs[j]
        return float(x_rand[s] @ a) if task == 1 else _pair_score(task, z, pe, le, ps[j])

    targets, scores = [], []
    if mode in ("single", "pairs"):
        for j, (t, s, _, _, _) in enumerate(pairs):
            sc = src_term(j) + (float(x_rand[t] @ b) if task == 1 else 0.0)
            targets.append(t)
            scores.append(sc)
    else:
        t = groups[0][0]
        if mode == "nearest":
            sc = src_term(0)                      # source 0 is the nearest
        else:                                     # weighted: sum_j gamma^d_j * term_j
            sc = sum(cfg["gamma"] ** pairs[j][2] * src_term(j) for j in range(len(pairs)))
        sc += float(x_rand[t] @ b) if task == 1 else 0.0
        targets.append(t)
        scores.append(sc)

    x = np.concatenate(blocks, axis=1)
    ei = np.array(edges, dtype=np.int64).T
    edge_index = np.concatenate([ei, ei[::-1]], axis=1)           # both directions
    edge_attr = np.concatenate([efeat, efeat], axis=0)
    y = torch.full((n,), -1, dtype=torch.long)                    # labels set on targets later
    return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr), y=y, num_nodes=n,
                score=torch.tensor(scores, dtype=torch.float32),
                tgt_node=torch.tensor(targets, dtype=torch.long),
                src_node=torch.tensor([p_[1] for p_ in pairs], dtype=torch.long),
                src_dist=torch.tensor([p_[2] for p_ in pairs], dtype=torch.long),
                query_pos=torch.tensor(ps, dtype=torch.long))


GEN_VERSION = 2   # bump when generation changes; older datasets are rebuilt

# ---------------------------------------------------------------------------
# Dataset generation / loading.
# ---------------------------------------------------------------------------
def dataset_name(cfg: dict) -> str:
    task = "4b" if cfg["task"] == 4 else str(cfg["task"])     # 4b replaced the ramp task
    if cfg["task"] == 2:
        kpart = "k1"
    elif cfg.get("k_range"):
        kpart = f"k{cfg['k_range'][0]}-{cfg['k_range'][1]}"
    else:
        kpart = f"k{cfg['k']}"
    name = (f"t{task}_{cfg['family']}_n{cfg['n']}_d{cfg['deg']}_{kpart}"
            f"_e{cfg['edge_dim']}_noise{cfg['noise']}_dec{cfg['decoys']}")
    if cfg["task"] == 7:
        name += f"_{cfg['path_mode']}"
    if cfg["mode"] == "pairs" and cfg["m"] > 1:
        name += f"_pairs{cfg['m']}"
    elif cfg["mode"] in ("nearest", "weighted"):
        name += f"_src{cfg['m']}{cfg['mode']}"
        if cfg["mode"] == "weighted":
            name += f"_g{cfg['gamma']}"
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
    # Task 4b scores are +-0.5 (balanced by symmetry), so threshold at 0.
    if cfg["task"] == 4 and cfg["mode"] != "weighted":
        thr = 0.0
    else:
        thr = float(np.median(np.concatenate([g.score.numpy() for g in splits["train"]])))
    for sp, graphs in splits.items():
        for g in graphs:
            for t, sc in zip(g.tgt_node.tolist(), g.score.tolist()):
                lab = int(sc > thr)
                if sp == "train" and cfg["noise"] > 0 and rng.random() < cfg["noise"]:
                    lab = 1 - lab                                # clean val/test
                g.y[t] = lab

    root = os.path.join(out_root, dataset_name(cfg))
    os.makedirs(root, exist_ok=True)
    # Regenerated graphs invalidate the runner's distance/predecessor caches.
    for entry in os.listdir(root):
        if entry.startswith("dist_cache"):
            shutil.rmtree(os.path.join(root, entry))
    for sp, graphs in splits.items():
        torch.save(InMemoryDataset.collate(graphs), os.path.join(root, f"{sp}.pt"))
    all_graphs = [g for gs in splits.values() for g in gs]
    meta = dict(cfg, threshold=thr, node_feat_dim=int(splits["train"][0].x.size(1)),
                edge_feat_dim=D, stats=graph_stats(splits["train"]),
                max_dist=int(max(int(g.src_dist.max()) for g in all_graphs)),
                labels_per_graph=int(splits["train"][0].tgt_node.numel()),
                gen_version=GEN_VERSION)
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
    p.add_argument("--k_range", type=str, default=None,
                   help="'lo,hi': draw k per graph (per pair in --pairs mode) "
                        "uniformly from [lo, hi]; overrides --k")
    p.add_argument("--edge_dim", type=str, default="8", help="Comma list, e.g. 1,8,32")
    p.add_argument("--node_dim", type=int, default=8)
    p.add_argument("--noise", type=str, default="0", help="Train label noise, comma list")
    p.add_argument("--decoys", choices=["auto", "on", "off"], default="auto",
                   help="auto: off for task 3, on otherwise")
    p.add_argument("--path_mode", choices=["canonical", "invariant"], default="canonical",
                   help="Task 7 label: one canonical path or all shortest paths")
    p.add_argument("--pairs", type=int, default=1,
                   help="m (s, t) pairs per graph, linked by pair-ID vectors")
    p.add_argument("--id_dim", type=int, default=8)
    p.add_argument("--sources", type=int, default=1,
                   help="m sources and one target, partner defined by --rule")
    p.add_argument("--rule", choices=["nearest", "weighted"], default="nearest",
                   help="nearest: label from the closest source; weighted: "
                        "sum_j gamma^d_j * term_j over all sources")
    p.add_argument("--gamma", type=float, default=0.7, help="Decay for --rule weighted")
    p.add_argument("--num_train", type=int, default=2000)
    p.add_argument("--num_val", type=int, default=500)
    p.add_argument("--num_test", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="synthetic")
    p.add_argument("--skip_existing", action="store_true", default=False,
                   help="Skip datasets already generated with this GEN_VERSION")
    args = p.parse_args()

    if args.pairs > 1 and args.sources > 1:
        p.error("--pairs and --sources cannot both be > 1")
    if args.sources > 1:
        mode, m = args.rule, args.sources
    elif args.pairs > 1:
        mode, m = "pairs", args.pairs
    else:
        mode, m = "single", 1
    k_range = tuple(_ints(args.k_range)) if args.k_range else None
    if k_range is not None and (len(k_range) != 2 or k_range[0] > k_range[1]):
        p.error("--k_range must be 'lo,hi' with lo <= hi")

    dn, dd = PRESET_DEFAULTS.get(args.family, (200, 4))
    ks = [k_range[1]] if k_range else _ints(args.k)
    grid = [(n, d, k, e, float(z))
            for n in (_ints(args.n) if args.n else [dn])
            for d in (_ints(args.deg) if args.deg else [dd])
            for k in ks
            for e in _ints(args.edge_dim)
            for z in args.noise.split(",")]
    for n, d, k, e, z in grid:
        cfg = dict(task=args.task, family=args.family, n=n, deg=d, k=k,
                   k_range=list(k_range) if k_range else None, edge_dim=e,
                   node_dim=args.node_dim, noise=z, decoys=args.decoys,
                   path_mode=args.path_mode, mode=mode, m=m, id_dim=args.id_dim,
                   gamma=args.gamma, num_train=args.num_train,
                   num_val=args.num_val, num_test=args.num_test, seed=args.seed)
        meta_path = os.path.join(args.out, dataset_name(cfg), "meta.json")
        if args.skip_existing and os.path.exists(meta_path):
            with open(meta_path) as fh:
                if json.load(fh).get("gen_version") == GEN_VERSION:
                    print(f"[synthetic] exists {dataset_name(cfg)}", flush=True)
                    continue
        try:
            generate(cfg, args.out)
        except ValueError as err:                               # infeasible config
            print(f"[synthetic] skipped {dataset_name(cfg)}: {err}", flush=True)


if __name__ == "__main__":
    main()

