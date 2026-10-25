# real_graph_stats.py
"""Structure statistics (synthetic_suite.graph_stats) for real benchmark datasets,
to map them onto the synthetic families.

Run from hop_masked/:
  python real_graph_stats.py                       # all default datasets
  python real_graph_stats.py --only Cora PATTERN PCQM-Contact
  python real_graph_stats.py --heavy               # also large downloads / graphs
Writes real_graph_stats.json and prints a table. Datasets that fail to load
(missing package, download error) are reported and skipped.
"""

import argparse
import json
import traceback

import numpy as np
from torch_geometric.utils import to_undirected

from synthetic_suite import graph_stats

ROOT = "./data"


def _ds():
    """name -> (loader, kind, heavy). kind: 'multi' (many graphs) or 'single'."""
    from torch_geometric import datasets as D

    def ogbg(name):
        from ogb.graphproppred import PygGraphPropPredDataset
        return PygGraphPropPredDataset(name=name, root=ROOT)

    def ogbn(name):
        from ogb.nodeproppred import PygNodePropPredDataset
        return PygNodePropPredDataset(name=name, root=ROOT)

    lrgb = lambda n: (lambda: D.LRGBDataset(ROOT, name=n, split="train"))
    gnnb = lambda n: (lambda: D.GNNBenchmarkDataset(ROOT, name=n, split="train"))
    tu = lambda n: (lambda: D.TUDataset(f"{ROOT}/TU", name=n))
    het = lambda n: (lambda: D.HeterophilousGraphDataset(f"{ROOT}/hetero", name=n))
    plan = lambda n: (lambda: D.Planetoid(f"{ROOT}/Planetoid", name=n))

    return {
        # LRGB
        "Peptides-func":   (lrgb("Peptides-func"), "multi", False),
        "Peptides-struct": (lrgb("Peptides-struct"), "multi", False),
        "PascalVOC-SP":    (lrgb("PascalVOC-SP"), "multi", False),
        "COCO-SP":         (lrgb("COCO-SP"), "multi", True),
        "PCQM-Contact":    (lrgb("PCQM-Contact"), "multi", False),
        # Benchmarking GNNs
        "MNIST":    (gnnb("MNIST"), "multi", False),
        "CIFAR10":  (gnnb("CIFAR10"), "multi", False),
        "PATTERN":  (gnnb("PATTERN"), "multi", False),
        "CLUSTER":  (gnnb("CLUSTER"), "multi", False),
        "TSP":      (gnnb("TSP"), "multi", True),
        "CSL":      (lambda: D.GNNBenchmarkDataset(ROOT, name="CSL"), "multi", False),
        "ZINC":     (lambda: D.ZINC(f"{ROOT}/ZINC", subset=True, split="train"), "multi", False),
        # OGB graph-level
        "ogbg-molhiv":  (lambda: ogbg("ogbg-molhiv"), "multi", False),
        "ogbg-molpcba": (lambda: ogbg("ogbg-molpcba"), "multi", True),
        "ogbg-ppa":     (lambda: ogbg("ogbg-ppa"), "multi", True),
        "ogbg-code2":   (lambda: ogbg("ogbg-code2"), "multi", True),
        "PCQM4Mv2":     (lambda: D.PCQM4Mv2(f"{ROOT}/PCQM4Mv2", split="train"), "multi", True),
        # TU / social / bio
        "REDDIT-BINARY":    (tu("REDDIT-BINARY"), "multi", False),
        "REDDIT-MULTI-5K":  (tu("REDDIT-MULTI-5K"), "multi", False),
        "REDDIT-MULTI-12K": (tu("REDDIT-MULTI-12K"), "multi", False),
        "IMDB-BINARY":      (tu("IMDB-BINARY"), "multi", False),
        "COLLAB":           (tu("COLLAB"), "multi", False),
        "DD":               (tu("DD"), "multi", False),
        "PROTEINS":         (tu("PROTEINS"), "multi", False),
        "MalNet-Tiny":      (lambda: D.MalNetTiny(f"{ROOT}/MalNetTiny"), "multi", False),
        # Single-graph node-level
        "Cora":           (plan("Cora"), "single", False),
        "CiteSeer":       (plan("CiteSeer"), "single", False),
        "PubMed":         (plan("PubMed"), "single", False),
        "Roman-empire":   (het("Roman-empire"), "single", False),
        "Amazon-ratings": (het("Amazon-ratings"), "single", False),
        "Minesweeper":    (het("Minesweeper"), "single", False),
        "Tolokers":       (het("Tolokers"), "single", False),
        "Questions":      (het("Questions"), "single", False),
        "ogbn-proteins":  (lambda: ogbn("ogbn-proteins"), "single", True),
        # Knowledge graphs (relation type = edge feature)
        "WN18RR":     (lambda: D.WordNet18RR(f"{ROOT}/WN18RR"), "single", False),
        "FB15k-237":  (lambda: D.FB15k_237(f"{ROOT}/FB15k-237", split="train"), "single", True),
    }


def _dim(t):
    if t is None:
        return 0
    return 1 if t.dim() == 1 else int(t.size(-1))


def stats_for(name, loader, kind, max_graphs, sources, seed=0):
    ds = loader()
    rng = np.random.default_rng(seed)
    if kind == "single":
        graphs = [ds[0]]
        src = max(1, sources // 4)            # BFS over the whole graph is slow
    else:
        idx = rng.permutation(len(ds))[:max_graphs]
        graphs = [ds[int(i)] for i in idx]
        src = sources
    # Undirected copies: KGs and some datasets store one direction only.
    und = []
    for g in graphs:
        g = g.clone()
        g.edge_index = to_undirected(g.edge_index, num_nodes=g.num_nodes)
        und.append(g)
    s = graph_stats(und, max_graphs=len(und), sources_per_graph=src, seed=seed)
    g0 = graphs[0]
    s.update(n_graphs=len(ds), kind=kind,
             node_feat_dim=_dim(getattr(g0, "x", None)),
             edge_feat_dim=_dim(getattr(g0, "edge_attr", None)),
             directed_input=bool(g0.edge_index.size(1) != und[0].edge_index.size(1)))
    if getattr(g0, "edge_type", None) is not None:          # KGs: relation ids
        s["edge_types"] = int(g0.edge_type.max()) + 1
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None, help="Dataset names to run")
    p.add_argument("--heavy", action="store_true",
                   help="Include large downloads / graphs (COCO-SP, TSP, molpcba, "
                        "ppa, code2, PCQM4Mv2, ogbn-proteins, FB15k-237)")
    p.add_argument("--max_graphs", type=int, default=200)
    p.add_argument("--sources", type=int, default=20)
    p.add_argument("--out", default="real_graph_stats.json")
    args = p.parse_args()

    registry = _ds()
    names = args.only or [n for n, (_, _, heavy) in registry.items()
                          if args.heavy or not heavy]
    results = {}
    for n in names:
        if n not in registry:
            print(f"[skip] unknown dataset {n}")
            continue
        loader, kind, _ = registry[n]
        try:
            results[n] = stats_for(n, loader, kind, args.max_graphs, args.sources)
            print(f"[ok] {n}: {results[n]}", flush=True)
        except Exception as e:                           # missing package / download
            print(f"[fail] {n}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=1)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    cols = ["kind", "n_graphs", "avg_nodes", "avg_degree", "diameter", "avg_spl",
            "frac_unique_sp", "node_feat_dim", "edge_feat_dim"]
    w = max([len(n) for n in results] + [8])
    print(f"\n{'dataset':<{w}}  " + "  ".join(f"{c:>14}" for c in cols))
    for n, s in results.items():
        print(f"{n:<{w}}  " + "  ".join(f"{str(s.get(c)):>14}" for c in cols))
    print(f"\nSaved to {args.out}. diameter is a lower bound (sampled BFS sources).")


if __name__ == "__main__":
    main()

