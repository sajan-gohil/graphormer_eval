#!/usr/bin/env python3
"""
Measure the three approximation defects of the hop-masked architecture on REAL data.

    python hop_masked/measure_defects.py --dataset Peptides-func --n_graphs 400 --H 30

Theory: HOP_MASKED_LEARNING_THEORY.md, Theorem A. For a target pair-kernel K*,

    d_row   = std_u( (K* 1)_u ) / ||K*||_F
              -> lower bound on the error of ANY row-stochastic/constant-row-sum
                 operator (i.e. plain masked attention). Removed by a node-wise gate.

    d_shell = ||K* - shell-mean(K*)||_F / ||K*||_F   (over shells h <= H)
              -> error of a BINARY mask that cannot resolve pairs within a shell.
                 Removed by a shell-conditioned log walk-profile bias.

    d_hor   = ||Pi_{>H} K*||_F / ||K*||_F
              -> mass beyond the hop horizon. Removed by a far-field / spectral branch.

K* is NOT assumed. It is fitted: the spectrum is split into bands, a ridge probe
predicts the labels from band-projected graph readouts, and the fitted band weights
give the task's own spectral profile. Defects are then reported per band AND for the
probe-weighted K*.

Only torch + torch_geometric + numpy are required (scipy/sklearn optional, not used).
"""
import argparse, json, sys, time
import numpy as np

# ----------------------------------------------------------------------------- graph utils
def bfs_hops(A_bool):
    """All-pairs shortest-path hop matrix by boolean BFS. -1 = unreachable."""
    n = A_bool.shape[0]
    D = np.full((n, n), -1, dtype=np.int32)
    eye = np.eye(n, dtype=bool)
    D[eye] = 0
    cur, seen, k = eye.copy(), eye.copy(), 0
    while True:
        k += 1
        cur = (cur @ A_bool) & ~seen
        if not cur.any():
            break
        D[cur] = k
        seen |= cur
    return D


def sym_norm_lap(A):
    d = A.sum(1)
    inv = np.zeros_like(d)
    nz = d > 0
    inv[nz] = 1.0 / np.sqrt(d[nz])
    At = A * inv[:, None] * inv[None, :]
    return np.eye(len(A)) - At, At


# ----------------------------------------------------------------------------- defects
def defects(K, D, H):
    """Theorem A. Returns (d_row, d_shell, d_hor), all relative to ||K||_F."""
    F = np.linalg.norm(K)
    if F < 1e-12:
        return np.nan, np.nan, np.nan
    d_row = np.std(K.sum(1)) / F
    far = (D > H) | (D < 0)
    d_hor = np.linalg.norm(K[far]) / F
    ss = 0.0
    for h in range(0, H + 1):
        m = D == h
        c = int(m.sum())
        if c == 0:
            continue
        v = K[m]
        ss += float(((v - v.mean()) ** 2).sum())
    return d_row, np.sqrt(ss) / F, d_hor


def within_shell_cv(At, D, H, lag=2, max_h=6):
    """CV of A~^{h+lag} across pairs inside shell h -- the signal a binary mask discards."""
    out = {}
    P = At.copy()
    pows = {1: At}
    for s in range(2, max_h + lag + 1):
        P = P @ At
        pows[s] = P.copy()
    for h in range(1, min(H, max_h) + 1):
        m = D == h
        if m.sum() < 4 or (h + lag) not in pows:
            continue
        v = pows[h + lag][m]
        mu = v.mean()
        if abs(mu) > 1e-12:
            out[h] = float(v.std() / abs(mu))
    return out


# ----------------------------------------------------------------------------- convex fit of omega
def fit_omega(K, D, At, H=6, r=4, eps=1e-8, lam=1e-6):
    """Closed-form CONVEX fit of the shell-conditioned log walk-profile bias.

    A (2-head + per-node gate + bias) block realises, on shell h:
        Khat[u,v] = sign(K[u,v]) * m_u * softmax_v( <w_h, log(eps + rho_uv)> ),
        m_u = sum_{v in shell h(u)} |K[u,v]|.
    softmax is shift-invariant per row, so the identifiable target is log|K| row-centred
    within the shell. The fit is then ordinary least squares in w_h -- convex, closed
    form, O((r+1)^2). Baseline = the same formula at w_h = 0, i.e. what a binary mask
    can do. Returns (w_per_shell, rel_baseline, rel_fitted, within_base, within_fit).

    This converts delta_shell from an upper bound on what is removable into the
    fraction actually removed, and gives a warm start that costs no gradient steps.
    """
    pw = {s: np.linalg.matrix_power(At, s) for s in range(0, H + r + 2)}
    num_b = num_f = den = 0.0
    Ws = {}
    for h in range(1, H + 1):
        rows, Phi, tgt = [], [], []
        for u in range(len(K)):
            idx = np.nonzero(D[u] == h)[0]
            if len(idx) < 3:
                continue
            k = K[u, idx]
            if np.abs(k).sum() < 1e-14:
                continue
            P = np.stack([np.log(eps + np.abs(pw[h + j][u, idx])) for j in range(r + 1)], 1)
            t = np.log(eps + np.abs(k))
            Phi.append(P - P.mean(0)); tgt.append(t - t.mean())    # quotient the row shift
            rows.append((u, idx, k, P))
        if not rows:
            continue
        Phi = np.concatenate(Phi); tgt = np.concatenate(tgt)
        w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(r + 1), Phi.T @ tgt)   # the convex solve
        Ws[h] = w
        for (u, idx, k, P) in rows:
            m = np.abs(k).sum()
            def rec(sc):
                e = np.exp(sc - sc.max()); return np.sign(k) * m * e / e.sum()
            num_f += ((rec(P @ w) - k) ** 2).sum()
            num_b += ((rec(np.zeros(len(idx))) - k) ** 2).sum()
            den += (k ** 2).sum()
    F = np.linalg.norm(K)
    if den == 0:
        return Ws, np.nan, np.nan, np.nan, np.nan
    return Ws, np.sqrt(num_b) / F, np.sqrt(num_f) / F, np.sqrt(num_b / den), np.sqrt(num_f / den)


# ----------------------------------------------------------------------------- direct .pt loader
def find_pt(dataset, root):
    """Locate a pre-collated PyG .pt for `dataset` anywhere under `root`.
    Handles the graphgps layout (geometric_data_processed.pt) and any directory
    naming (peptides_func / peptides-func / peptides-functional / ...)."""
    import os.path as osp, glob, re
    # Only filenames that are actually a collated (data, slices) payload. This
    # deliberately excludes PyG's own {train,val,test}_data.pt and raw/*.pt, so
    # datasets the repo loader already handles keep using it.
    COLLATED = {"geometric_data_processed.pt", "data.pt"}
    toks = re.split(r"[-_]", dataset.lower())            # Peptides-func -> [peptides, func]
    cands = []
    for p in glob.glob(osp.join(root, "**", "*.pt"), recursive=True):
        if osp.basename(p).lower() not in COLLATED:
            continue
        rel = osp.relpath(p, root).lower()
        if osp.sep + "raw" + osp.sep in osp.sep + rel:
            continue
        if all(t[:4] in rel for t in toks if len(t) >= 3):
            cands.append((osp.getsize(p), p))
    if not cands:
        return None
    return max(cands)[1]                                  # biggest match = the data file


def load_pt_dataset(path):
    """Wrap a torch-saved (data, slices) collation, or a list of Data, as a dataset."""
    import torch
    from torch_geometric.data import InMemoryDataset, Data

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, (tuple, list)) and len(obj) >= 2 and isinstance(obj[1], dict):
        data, slices = obj[0], obj[1]
    elif isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], Data):
        return list(obj)                                  # already a plain list of graphs
    elif isinstance(obj, dict) and "data" in obj:
        data, slices = obj["data"], obj.get("slices")
    else:
        raise SystemExit(f"[pt] unrecognised payload in {path}: {type(obj)}")

    class _PtDataset(InMemoryDataset):
        def __init__(self):
            super().__init__(None, None, None, None)
            # PyG >= 2.4 stores the collation on _data; older versions on data.
            if hasattr(self, "_data"):
                self._data, self.slices = data, slices
            else:
                self.data, self.slices = data, slices
        def _download(self): pass
        def _process(self): pass

    return _PtDataset()


# ----------------------------------------------------------------------------- download repair
def repair_lrgb_download(dataset, info, root):
    """PyG's download_url uses urllib with no browser User-Agent; Dropbox answers it
    with an HTML interstitial, which then fails as `BadZipFile`. Detect that, and
    re-fetch with a browser UA. Paths/URLs are read off the installed LRGBDataset
    class rather than hardcoded, so this survives PyG version changes."""
    import os, os.path as osp, shutil, zipfile, urllib.request
    from torch_geometric.datasets import LRGBDataset

    name = info.get("pyg_name", dataset).lower()
    valid = getattr(LRGBDataset, "names", None)
    if valid is not None and name not in valid:
        raise SystemExit(f"'{name}' not in LRGBDataset.names = {list(valid)}")

    raw_dir = osp.join(root, name, "raw")
    needed = getattr(LRGBDataset, "raw_file_names", ["train.pt", "val.pt", "test.pt"])
    needed = needed if isinstance(needed, (list, tuple)) else ["train.pt", "val.pt", "test.pt"]
    if osp.isdir(raw_dir) and all(osp.exists(osp.join(raw_dir, f)) for f in needed):
        return                                        # already good
    if osp.isdir(osp.join(root, name, "processed")):
        return                                        # already processed

    urls = getattr(LRGBDataset, "urls", None)
    stems = getattr(LRGBDataset, "dwnld_file_name", None)
    if not urls or not stems:
        print("[repair] cannot introspect LRGBDataset.urls/dwnld_file_name; "
              "letting PyG try on its own")
        return
    url, stem = urls[name], stems[name]
    zpath = osp.join(root, f"{stem}.zip")

    if osp.exists(zpath) and not zipfile.is_zipfile(zpath):
        print(f"[repair] {zpath} is not a zip (Dropbox HTML interstitial) — removing")
        os.remove(zpath)
    if not osp.exists(zpath):
        os.makedirs(root, exist_ok=True)
        print(f"[repair] fetching {url}")
        req = urllib.request.Request(url, headers={
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"),
            "Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=180) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
    if not zipfile.is_zipfile(zpath):
        os.remove(zpath)
        raise SystemExit(
            "[repair] Dropbox still refused. Fetch it by hand, then re-run:\n"
            f"  curl -L -A 'Mozilla/5.0' -o {zpath} '{url}'\n"
            f"  file {zpath}     # must say: Zip archive data\n"
            f"  unzip -q {zpath} -d {root} && mkdir -p {osp.dirname(raw_dir)} && "
            f"mv {osp.join(root, stem)} {raw_dir}")

    print(f"[repair] extracting -> {raw_dir}")
    shutil.rmtree(raw_dir, ignore_errors=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    os.makedirs(osp.dirname(raw_dir), exist_ok=True)
    src = osp.join(root, stem)
    if osp.isdir(src):
        shutil.move(src, raw_dir)
    else:                                             # zip already had the right layout
        os.makedirs(raw_dir, exist_ok=True)
    os.remove(zpath)
    print(f"[repair] ok: {sorted(os.listdir(raw_dir))[:6]}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Peptides-func")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_graphs", type=int, default=400)
    ap.add_argument("--H", type=int, default=30, help="hop horizon under test (= max_hops)")
    ap.add_argument("--n_bands", type=int, default=8, help="spectral bands for the probe")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--loader", default="auto", choices=["auto", "pt", "repo", "graphgps"],
                    help="'auto' = use a pre-collated .pt under --root if one matches, "
                         "else fall back to the repo loader. 'pt' = force the .pt path. "
                         "'repo' = hop_masked/data.get_loaders (LRGBDataset). "
                         "'graphgps' = graphgps PeptidesFunctional/StructuralDataset "
                         "(different download source).")
    ap.add_argument("--pt_file", default=None,
                    help="explicit path to a torch-saved (data, slices) collation")
    ap.add_argument("--gg_root", default="datasets", help="root for --loader graphgps")
    ap.add_argument("--fit_omega", action="store_true", default=True,
                    help="closed-form convex fit of the walk-profile bias (default on)")
    ap.add_argument("--no_fit_omega", dest="fit_omega", action="store_false")
    ap.add_argument("--fit_H", type=int, default=6, help="shells to fit omega on")
    ap.add_argument("--r", type=int, default=4, help="walk-profile order")
    ap.add_argument("--also_H", type=int, nargs="*", default=[5, 10, 20],
                    help="extra horizons for the d_hor sweep")
    ap.add_argument("--out", default="defects_report.json")
    args = ap.parse_args()

    import torch
    from torch_geometric.utils import to_dense_adj

    # --- load: reuse the repo's own loader (hop_masked/data.py) ----------------
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from data import get_loaders, get_dataset_info  # noqa

    info = get_dataset_info(args.dataset)
    t0 = time.time()

    pt = args.pt_file
    if args.loader in ("auto", "pt") and pt is None:
        pt = find_pt(args.dataset, args.root)
    if args.loader == "pt" and pt is None:
        raise SystemExit(f"[pt] no .pt found for {args.dataset} under {args.root}; "
                         f"pass --pt_file explicitly")

    if pt is not None and args.loader in ("auto", "pt"):
        print(f"[pt] loading {pt}")
        ds = load_pt_dataset(pt)
        print(f"[pt] no official split file alongside it -> measuring on ALL {len(ds)} "
              f"graphs. Fine here: these are structural/statistical quantities, and the "
              f"ridge probe is a diagnostic, not a model being selected on.")
    elif args.loader == "graphgps":
        # Independent source: builds from a SMILES CSV via ogb.smiles2graph.
        # Different Dropbox files from LRGBDataset's, so it survives when those
        # are rate-limited. Needs rdkit + ogb. Peptides only.
        import pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        mod = {"Peptides-func": ("peptides_functional", "PeptidesFunctionalDataset"),
               "Peptides-struct": ("peptides_structural", "PeptidesStructuralDataset")}[args.dataset]
        m = __import__(f"graphgps.loader.dataset.{mod[0]}", fromlist=[mod[1]])
        full = getattr(m, mod[1])(root=args.gg_root)
        sp = full.get_idx_split()[{"train": "train", "val": "val", "test": "test"}[args.split]]
        ds = [full[int(i)] for i in sp]
    else:
        if info.get("source", "lrgb") == "lrgb":
            repair_lrgb_download(args.dataset, info, args.root)
        *_loaders, train_ds, val_ds, test_ds, info = get_loaders(
            batch_size=1, num_workers=0, use_dist_masks=False,
            dataset_name=args.dataset, return_info=True)
        ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]

    print(f"[load] {args.dataset}/{args.split}: {len(ds)} graphs  ({time.time()-t0:.1f}s)")

    task_type = info.get("task_type", "multi_label")
    n_cls = info.get("output_dim")
    if not isinstance(n_cls, int):
        n_cls = int(max(int(ds[i].y.max()) for i in range(min(200, len(ds)))) + 1)
    if task_type == "multiclass" and info.get("level") == "node":
        print("[probe] node-level multiclass: graph-readout probe is weak here; "
              "band weights fall back toward uniform. Read the per-band rows.")

    idx = np.linspace(0, len(ds) - 1, min(args.n_graphs, len(ds))).astype(int)
    bands = np.linspace(0.0, 2.0, args.n_bands + 1)

    Hs = sorted({args.H, *args.also_H})
    diam, n_nodes = [], []
    band_def = {b: [] for b in range(args.n_bands)}          # per-band (d_row,d_shell,d_hor) @ args.H
    band_hor = {(b, H): [] for b in range(args.n_bands) for H in Hs}   # per-band d_hor @ each H
    HEAT_T = [0.5, 2.0, 8.0]
    smooth_def = {t: [] for t in HEAT_T}
    omega_fit = []
    cv_acc, feats, labels = {}, [], []

    for gi in idx:
        g = ds[int(gi)]
        n = int(g.num_nodes)
        if n < 4:
            continue
        A = to_dense_adj(g.edge_index, max_num_nodes=n)[0].numpy()
        A = ((A + A.T) > 0).astype(np.float64)
        np.fill_diagonal(A, 0.0)
        if A.sum() == 0:
            continue
        D = bfs_hops(A > 0)
        L, At = sym_norm_lap(A)
        w, V = np.linalg.eigh(L)

        n_nodes.append(n)
        diam.append(int(D.max()))

        # per-band spectral projectors P_b = V_b V_b^T  (exactly the S2GNN filter form)
        X = g.x.numpy().astype(np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        phi = []
        Pbs = []
        for b in range(args.n_bands):
            sel = (w >= bands[b]) & (w < bands[b + 1] if b < args.n_bands - 1 else w <= bands[b + 1] + 1e-9)
            if sel.sum() == 0:
                phi.append(np.zeros(2 * X.shape[1])); continue
            Vb = V[:, sel]
            Pb = Vb @ Vb.T
            Y = Pb @ X
            phi.append(np.concatenate([Y.mean(0), (Y ** 2).mean(0)]))   # linear + quadratic readout
            band_def[b].append(defects(Pb, D, args.H))
            for H in Hs:                       # streamed: never cache Pb across graphs
                band_hor[(b, H)].append(defects(Pb, D, H)[2])
            del Pb, Vb, Y
        # smooth filters: sharp band indicators are delocalised and over-state d_hor.
        # S2GNN uses a Tukey window precisely to avoid that, so heat kernels bracket
        # the realistic case from the other side.
        for t in HEAT_T:
            smooth_def[t].append(defects(V @ np.diag(np.exp(-t * w)) @ V.T, D, args.H))

        if args.fit_omega:      # convex closed-form solve for the walk-profile bias
            Kref = V @ np.diag(np.exp(-2.0 * w)) @ V.T
            _, rb, rf, wb, wf = fit_omega(Kref, D, At, H=args.fit_H, r=args.r)
            if np.isfinite(rb):
                omega_fit.append((rb, rf, wb, wf))

        feats.append(np.concatenate(phi))
        y = g.y.numpy().reshape(-1).astype(np.float64)
        if task_type == "multiclass" and y.size == 1:
            oh = np.zeros(n_cls); oh[int(y[0])] = 1.0; y = oh    # class index -> one-hot
        elif info.get("level") == "node":
            y = np.array([np.bincount(y.astype(int), minlength=n_cls).astype(float)])
            y = y.reshape(-1) / max(y.sum(), 1)                  # node-label histogram
        labels.append(y)

        for h, c in within_shell_cv(At, D, args.H).items():
            cv_acc.setdefault(h, []).append(c)

    feats = np.array(feats); labels = np.array(labels)
    print(f"[graphs] used {len(feats)}   n: mean {np.mean(n_nodes):.0f} max {max(n_nodes)}"
          f"   diameter: mean {np.mean(diam):.1f} median {np.median(diam):.0f} max {max(diam)} "
          f"p95 {np.percentile(diam,95):.0f}")
    print(f"[horizon] P[diam > H={args.H}] = {np.mean(np.array(diam) > args.H):.3f}")

    # --- ridge probe: which spectral bands does the task actually use? ---------
    Z = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    Y = labels - labels.mean(0)
    lam = 1e-2 * len(Z)
    W = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)
    per_feat = np.linalg.norm(W, axis=1)
    k = feats.shape[1] // args.n_bands
    imp = np.array([per_feat[b * k:(b + 1) * k].sum() for b in range(args.n_bands)])
    imp = imp / imp.sum()
    r2 = 1 - ((Z @ W - Y) ** 2).sum() / (Y ** 2).sum()
    print(f"[probe] ridge R^2 on {args.dataset} labels = {r2:.3f}  (sanity: >0 means the profile is meaningful)")

    # --- report ---------------------------------------------------------------
    print(f"\n=== Theorem A defects, H = {args.H} (relative to ||K*||_F) ===")
    print(f"{'band [lo,hi)':<16}{'weight':>8}{'d_row':>9}{'d_shell':>10}{'d_hor':>9}")
    agg = np.zeros(3)
    for b in range(args.n_bands):
        if not band_def[b]:
            continue
        m = np.nanmean(np.array(band_def[b]), axis=0)
        agg += imp[b] * m
        print(f"[{bands[b]:.2f},{bands[b+1]:.2f})".ljust(16)
              + f"{imp[b]:>8.3f}{m[0]:>9.3f}{m[1]:>10.3f}{m[2]:>9.3f}")
    print("-" * 52)
    print(f"{'probe-weighted':<16}{1.0:>8.3f}{agg[0]:>9.3f}{agg[1]:>10.3f}{agg[2]:>9.3f}")

    # d_hor vs horizon, for the probe-weighted target
    print(f"\n=== d_hor vs horizon (probe-weighted) ===")
    for H in Hs:
        s = sum(imp[b] * np.nanmean(band_hor[(b, H)])
                for b in range(args.n_bands) if band_hor[(b, H)])
        print(f"  H = {H:>3}:  d_hor = {s:.4f}")

    print(f"\n=== smooth filters e^(-tL), H = {args.H}   (band indicators above are the "
          f"delocalised extreme; these bracket d_hor from below) ===")
    print(f"{'filter':<16}{'d_row':>9}{'d_shell':>10}{'d_hor':>9}")
    for t in HEAT_T:
        m = np.nanmean(np.array(smooth_def[t]), axis=0)
        print(f"heat t={t:<9.1f}{m[0]:>9.3f}{m[1]:>10.3f}{m[2]:>9.3f}")

    if cv_acc:
        print(f"\n=== within-shell CV of A~^(h+2)  (signal a binary mask discards) ===")
        for h in sorted(cv_acc):
            print(f"  h = {h}: CV = {np.mean(cv_acc[h]):.3f}   (n_graphs={len(cv_acc[h])})")

    # --- decision -------------------------------------------------------------
    names = ["node-wise signed gate", "log walk-profile bias (shell-conditioned)",
             "far-field / spectral branch"]
    costs = np.array([15500, 150, 5000])

    # Rank on the SMOOTH filter, not the band indicators. Sharp spectral cutoffs are
    # oscillatory and globally supported, so they inflate d_shell and d_hor -- badly on
    # low-diameter graphs, where one huge outer shell dominates. S2GNN windows its
    # filters (Tukey) precisely to stay smooth, so heat t=2 is the defensible K*.
    ref = np.nanmean(np.array(smooth_def[2.0]), axis=0)
    print("\n=== ranking (K* = heat e^-2L, the defensible target) ===")
    print(f"{'component':<44}{'defect':>9}{'params':>9}{'per-param':>12}")
    for i in np.argsort(-(ref / costs)):
        print(f"{names[i]:<44}{ref[i]:>9.3f}{costs[i]:>9d}{ref[i]/costs[i]:>12.2e}")
    print(f"  [upper bracket, sharp band indicators: d_row {agg[0]:.3f}  "
          f"d_shell {agg[1]:.3f}  d_hor {agg[2]:.3f}]")

    if omega_fit:
        m = np.mean(np.array(omega_fit), axis=0)
        print(f"\n=== convex closed-form fit of the walk-profile bias "
              f"(K*=e^-2L, shells 1..{args.fit_H}, r={args.r}) ===")
        print(f"  binary mask (omega=0)   : within-shell rel err = {m[2]:.4f}")
        print(f"  + log walk-profile bias : within-shell rel err = {m[3]:.4f}")
        print(f"  -> {1 - m[3]/max(m[2],1e-12):.1%} of the shell defect removed by "
              f"{(args.r+1)} params/shell, solved in closed form (no gradient steps)")

    if cv_acc:
        cvm = float(np.mean([np.mean(v) for v in cv_acc.values()]))
        print(f"\nmean within-shell CV = {cvm:.3f}   ", end="")
        if cvm < 0.15:
            print("LOW -> little within-shell signal exists; the walk-profile bias\n"
                  "  should NOT help here. Use this dataset as a falsification test.")
        else:
            print("HIGH -> real within-shell signal the binary mask discards;\n"
                  "  the walk-profile bias is the change to try.")
    print(f"P[diam > H] = {np.mean(np.array(diam) > args.H):.3f}"
          f"   (d_hor can only be non-zero when this is > 0)")
    print("\nAct on a defect only if it plausibly exceeds your seed sigma on the test metric.")
    print("Measure that sigma first (4-6 seeds of the current best config).")

    json.dump({"dataset": args.dataset, "H": args.H, "n_graphs": int(len(feats)),
               "diam_mean": float(np.mean(diam)), "diam_p95": float(np.percentile(diam, 95)),
               "diam_max": int(max(diam)), "p_diam_gt_H": float(np.mean(np.array(diam) > args.H)),
               "probe_r2": float(r2), "band_weights": imp.tolist(),
               "d_row": float(agg[0]), "d_shell": float(agg[1]), "d_hor": float(agg[2]),
               "within_shell_cv": {int(h): float(np.mean(v)) for h, v in cv_acc.items()}},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

