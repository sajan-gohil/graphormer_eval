# analyze_checkpoint.py
"""
Standalone post-hoc analysis of a trained hop-masked transformer checkpoint.

Three independent reports, each selectable with a flag (default: all):

  --do_covariates   Per-graph structural features vs. per-graph score.
                    Spearman correlation, plus a partial correlation that
                    controls for node count, plus binned tables in the style of
                    the existing per-diameter breakdown.

  --do_loho         Leave-one-hop-out ablation. Zeroes hop shell h in the input
                    masks and re-evaluates the split, per hop. Purely a
                    data-side intervention, so the model is untouched.

  --do_head_diag    Per-head attention diagnostics aggregated over the split
                    (entropy, participation ratio, transported norm, cross-hop
                    mixing matrix).

Usage
-----
    python analyze_checkpoint.py --checkpoint checkpoints_hop_masked/best_Peptides-func.pt
    python analyze_checkpoint.py --checkpoint ... --split test --bins 8
    python analyze_checkpoint.py --checkpoint ... --do_loho --no_covariates

Everything is rebuilt from the ``args`` dict stored inside the checkpoint, so no
architecture flags need repeating on the command line.

WHY THE SCORE IS RANK-BASED
---------------------------
On multi-label targets, BCE and AP diverge late in training: the loss rises on
val/test while AP keeps improving, because the model grows overconfident on the
examples it already ranks correctly. Confidence drift moves the loss; it does
not move the ranking. Correlating structural features against per-graph *loss*
therefore mostly measures calibration, which is why per-diameter losses look
flat while the metric does not.

So for multi-label tasks the per-graph score is LRAP (label ranking average
precision): for one graph, rank its classes by predicted score and average the
precision at each true label. It is defined for a single graph — unlike AP,
which needs a set — and it is invariant to monotone rescaling of the logits.
Per-graph loss is still reported alongside it as ``loss`` so the two can be
compared directly.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

from data import get_loaders
from metrics import build_task
from model_hop_masked_transformer_final_2 import (HopMaskedTransformerModel,
                                               set_attn_diagnostics)
from train_hop_masked_transformer_final_2 import _move_batch_to_device


# ================================================================
# CHECKPOINT -> MODEL
# ================================================================

def _build_model_from_args(saved_args, task, dataset_info, dataset_name):
    """Reconstruct the model using only keys the constructor actually accepts.

    Signature-driven rather than a copy of the training script's call site, so
    adding a constructor argument does not silently break this file.
    """
    import inspect

    params = inspect.signature(HopMaskedTransformerModel.__init__).parameters
    kwargs = {k: v for k, v in saved_args.items() if k in params}

    # Values the training script derives rather than passing through verbatim.
    kwargs.update(
        output_dim=task.output_dim,
        task_level=task.level,
        dataset_name=dataset_name,
        node_feat_dim=dataset_info.get("node_feat_dim"),
        edge_feat_dim=dataset_info.get("edge_feat_dim"),
        lap_pe_dim=(saved_args.get("lap_pe_dim", 0)
                    if saved_args.get("use_lap_pe") else 0),
    )
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return HopMaskedTransformerModel(**kwargs)


def load_checkpoint(path, split, device, batch_size=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = ckpt["args"]
    dataset_name = saved["dataset"]

    loaders = get_loaders(
        batch_size=batch_size or saved.get("batch_size", 64),
        num_workers=saved.get("num_workers", 4),
        use_dist_masks=True,
        max_hops=saved["max_hops"],
        dist_mask_workers=saved.get("dist_mask_workers", 8),
        use_lap_pe=saved.get("use_lap_pe", False),
        lap_pe_dim=saved.get("lap_pe_dim", 8),
        dataset_name=dataset_name,
        return_info=True,
        subgraph_mode=saved.get("subgraph_mode", "partition"),
        num_parts=saved.get("num_parts", 128),
        egonet_hops=saved.get("egonet_hops", 2),
        egonet_max_nodes=saved.get("egonet_max_nodes", 1024),
        max_egonet_samples=saved.get("max_egonet_samples"),
        seed=saved.get("seed", 0),
    )
    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]
    dataset_info = loaders[-1]
    loader = {"train": train_loader, "val": val_loader,
              "test": test_loader}[split]

    task = build_task(dataset_info["name"], dataset_info=dataset_info)
    task.loss_fn = task.loss_fn.to(device)

    model = _build_model_from_args(saved, task, dataset_info,
                                  dataset_info["name"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    print(f"Loaded {path}\n  dataset={dataset_info['name']} split={split} "
          f"epoch={ckpt.get('epoch')} "
          f"val={ckpt.get('val_metric')} test={ckpt.get('test_metric')}",
          flush=True)
    return model, loader, task, dataset_info, saved


# ================================================================
# PER-GRAPH SCORING
# ================================================================

def _per_graph_scores(logits, y, task, batch_vec, num_graphs):
    """Return (scores, losses, pos_rate) as lists of length num_graphs.

    ``scores`` is higher-is-better for every task type, so that the sign of a
    correlation always reads the same way. For regression we return negative
    MAE for that reason.
    """
    import torch.nn.functional as F
    from sklearn.metrics import label_ranking_average_precision_score

    scores, losses, pos = [], [], []

    if task.level == "node":
        # Node-level: group the concatenated node predictions by graph.
        pred = logits.argmax(-1).cpu().numpy()
        true = y.view(-1).cpu().numpy()
        ce = F.cross_entropy(logits, y.view(-1).long(), ignore_index=-1,
                             reduction="none").cpu().numpy()
        bvec = batch_vec.cpu().numpy()
        for g in range(num_graphs):
            m = (bvec == g) & (true >= 0)
            if m.sum() == 0:
                scores.append(np.nan); losses.append(np.nan); pos.append(np.nan)
                continue
            scores.append(float((pred[m] == true[m]).mean()))
            losses.append(float(ce[m].mean()))
            pos.append(float(m.sum()))
        return scores, losses, pos

    if task.task_type == "multi_label":
        prob = torch.sigmoid(logits).cpu().numpy()
        true = y.view(logits.shape).cpu().numpy()
        bce = F.binary_cross_entropy_with_logits(
            logits, y.view(logits.shape).float(), reduction="none")
        bce = bce.mean(-1).cpu().numpy()
        for i in range(logits.shape[0]):
            t = true[i:i + 1]
            valid = ~np.isnan(t)
            # LRAP is undefined without at least one positive label.
            if not valid.any() or np.nansum(t) == 0:
                scores.append(np.nan)
            else:
                t0 = np.where(valid, np.nan_to_num(t), 0)
                scores.append(float(
                    label_ranking_average_precision_score(t0, prob[i:i + 1])))
            losses.append(float(bce[i]))
            pos.append(float(np.nansum(t)))
        return scores, losses, pos

    # Regression: negate so that larger is always better.
    pred = logits.detach().cpu().numpy()
    true = y.view(logits.shape).cpu().numpy()
    mae = np.abs(pred - true).mean(-1)
    return (-mae).tolist(), mae.tolist(), [np.nan] * len(mae)


@torch.no_grad()
def evaluate_split(model, loader, task, device, zero_hop=None):
    """Forward the split once.

    zero_hop: if set, hop shell ``h`` is removed from the input masks before the
    forward. Because hop masks are consumed as data, this ablates the hop for
    every head that uses it without touching the model.

    Returns (per_graph dict of lists, aggregate_metric).
    """
    out = defaultdict(list)
    preds_acc, labels_acc = [], []

    for batch in loader:
        pyg_batch, dist_masks, node_masks = _move_batch_to_device(batch, device)
        if zero_hop is not None:
            dist_masks = dist_masks.clone()
            dist_masks[:, zero_hop] = 0.0

        logits, _, _, _ = model(pyg_batch, dist_masks, node_masks)

        num_graphs = int(node_masks.shape[0])
        bvec = getattr(pyg_batch, "batch", None)
        if bvec is None:
            bvec = torch.zeros(pyg_batch.num_nodes, dtype=torch.long,
                               device=device)
        s, l, p = _per_graph_scores(logits, pyg_batch.y, task, bvec, num_graphs)
        out["score"].extend(s)
        out["loss"].extend(l)
        out["pos"].extend(p)
        out["n_nodes"].extend(node_masks.sum(-1).cpu().numpy().tolist())

        preds_acc.append(task.predict(logits))
        labels_acc.append(task.labels_to_numpy(pyg_batch.y))

    agg = task.compute_metric(np.concatenate(preds_acc, 0),
                              np.concatenate(labels_acc, 0))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}, agg


# ================================================================
# STRUCTURAL COVARIATES (from the cached distance matrices)
# ================================================================

def _gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    if x.sum() == 0:
        return 0.0
    idx = np.arange(1, x.size + 1)
    return float((2 * idx - x.size - 1).dot(x) / (x.size * x.sum()))


def graph_covariates(dist):
    """Structural features for one graph, derived from its SPD matrix.

    ``dist`` is the cached int16 matrix: hop distance, or -1 for unreachable
    pairs and pairs beyond max_hops. Everything here comes from that matrix, so
    no recomputation and no second pass over the dataset is needed. The hop-1
    shell is the adjacency.
    """
    d = np.asarray(dist, dtype=np.int32)
    n = d.shape[0]
    f = {"n_nodes": float(n)}

    off = ~np.eye(n, dtype=bool)
    reach = (d >= 0) & off
    f["disconnected_frac"] = float((~reach & off).sum() / max(off.sum(), 1))

    if reach.any():
        dr = d[reach].astype(np.float64)
        f["diameter"] = float(dr.max())
        f["mean_spd"] = float(dr.mean())
        f["median_spd"] = float(np.median(dr))
        f["spd_std"] = float(dr.std())
        # Hop-mass distribution: how the N^2 pair budget is spread over shells.
        # A head assigned to a hop carrying almost no pairs is structurally idle,
        # so spread and tail location matter more than the diameter alone.
        cnt = np.bincount(d[reach].astype(np.int64),
                          minlength=int(dr.max()) + 1).astype(np.float64)
        pmf = cnt / cnt.sum()
        nz = pmf[pmf > 0]
        f["hop_mass_entropy"] = float(-(nz * np.log(nz)).sum())
        f["hop_mass_peak"] = float(pmf.argmax())
        cdf = np.cumsum(pmf)
        f["hop_p50"] = float(np.searchsorted(cdf, 0.50))
        f["hop_p90"] = float(np.searchsorted(cdf, 0.90))
        # Fraction of pairs beyond a 3-hop MPNN-style receptive field.
        f["frac_pairs_gt3"] = float(pmf[4:].sum()) if pmf.size > 4 else 0.0
    else:
        for k in ("diameter", "mean_spd", "median_spd", "spd_std",
                  "hop_mass_entropy", "hop_mass_peak", "hop_p50", "hop_p90",
                  "frac_pairs_gt3"):
            f[k] = np.nan

    # ---- adjacency-derived ------------------------------------------------
    A = (d == 1).astype(np.float64)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(1)
    f["mean_degree"] = float(deg.mean())
    f["max_degree"] = float(deg.max()) if n else np.nan
    f["degree_gini"] = _gini(deg)

    # Local clustering coefficient, averaged over nodes with degree >= 2.
    tri = np.einsum("ij,jk,ki->i", A, A, A)
    denom = deg * (deg - 1)
    ok = denom > 0
    f["clustering_mean"] = float((tri[ok] / denom[ok]).mean()) if ok.any() else 0.0
    # Global transitivity: 3 * triangles / connected triples.
    f["transitivity"] = float(tri.sum() / denom.sum()) if denom.sum() > 0 else 0.0

    # Degree assortativity over edges.
    ei = np.nonzero(np.triu(A, 1))
    if ei[0].size > 1:
        du, dv = deg[ei[0]], deg[ei[1]]
        # Degree-regular graphs have zero variance here, so corrcoef returns
        # NaN with a warning. Silence it and report 0 correlation instead.
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(np.r_[du, dv], np.r_[dv, du])[0, 1]
        f["assortativity"] = float(0.0 if np.isnan(c) else c)
    else:
        f["assortativity"] = np.nan

    # lambda_2 of the symmetric normalised Laplacian (algebraic connectivity):
    # small values mean a bottlenecked graph. Dense eigh is fine at these sizes.
    if n > 2 and deg.sum() > 0:
        dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
        L = np.eye(n) - (dinv[:, None] * A * dinv[None, :])
        try:
            ev = np.linalg.eigvalsh(L)
            f["lambda2"] = float(np.sort(ev)[1])
            f["spectral_gap"] = float(np.sort(ev)[-1] - np.sort(ev)[1])
        except np.linalg.LinAlgError:
            f["lambda2"] = f["spectral_gap"] = np.nan
    else:
        f["lambda2"] = f["spectral_gap"] = np.nan

    return f


def collect_covariates(loader, limit=None):
    """Pull covariates for every graph in the loader's underlying dataset."""
    ds = loader.dataset
    store = getattr(ds, "dist_store", None)
    if store is None:
        raise RuntimeError(
            "Loader is not a DistMaskDataset; rerun with use_dist_masks=True.")

    total = len(store) if limit is None else min(limit, len(store))
    rows = []
    for i in range(total):
        rows.append(graph_covariates(store[i]))
        if (i + 1) % 2000 == 0:
            print(f"  covariates {i + 1}/{total}", flush=True)
    keys = sorted(rows[0].keys())
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}


# ================================================================
# CORRELATION + BINNED REPORTS
# ================================================================

def _partial_spearman(x, y, control):
    """Spearman between x and y after linearly removing `control` from both ranks.

    Node count correlates with diameter and with nearly every other structural
    feature, so a raw correlation cannot distinguish "long graphs are harder"
    from "big graphs are harder". This regresses the control out of both
    rank-transformed variables and correlates the residuals.
    """
    from scipy.stats import rankdata

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(control)
    if ok.sum() < 10:
        return np.nan, np.nan
    rx, ry, rc = (rankdata(v[ok]) for v in (x, y, control))
    C = np.c_[np.ones(rc.size), rc]
    rx = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
    ry = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    r, p = spearmanr(rx, ry)
    return float(r), float(p)


def report_covariates(cov, per_graph, out_csv=None, bins=6, top_feature=None):
    score = per_graph["score"]
    loss = per_graph["loss"]
    n = cov["n_nodes"]

    print("\n" + "=" * 92)
    print("STRUCTURAL COVARIATES vs PER-GRAPH SCORE")
    print("  rho        Spearman(feature, score)")
    print("  rho|n      same, controlling for node count")
    print("  rho_loss   Spearman(feature, per-graph loss)  [expect weak/flat if")
    print("             the loss has decoupled from the ranking metric]")
    print("=" * 92)
    print(f"{'feature':<22}{'rho':>9}{'p':>11}{'rho|n':>9}{'p':>11}"
          f"{'rho_loss':>11}{'n_valid':>9}")
    print("-" * 92)

    ranked = []
    for k in sorted(cov.keys()):
        x = cov[k]
        ok = np.isfinite(x) & np.isfinite(score)
        if ok.sum() < 10 or np.nanstd(x) == 0:
            continue
        r, p = spearmanr(x[ok], score[ok])
        rp, pp = _partial_spearman(x, score, n)
        ok2 = np.isfinite(x) & np.isfinite(loss)
        rl = spearmanr(x[ok2], loss[ok2])[0] if ok2.sum() >= 10 else np.nan
        print(f"{k:<22}{r:>9.3f}{p:>11.2e}{rp:>9.3f}{pp:>11.2e}"
              f"{rl:>11.3f}{int(ok.sum()):>9}")
        ranked.append((abs(rp) if np.isfinite(rp) else 0.0, k))

    ranked.sort(reverse=True)
    print("-" * 92)
    print("Strongest size-controlled associations: "
          + ", ".join(k for _, k in ranked[:5]))

    # Binned tables, in the style of the existing per-diameter breakdown.
    feats = [top_feature] if top_feature else [k for _, k in ranked[:3]]
    for feat in feats:
        if feat not in cov:
            continue
        _report_bins(feat, cov[feat], per_graph, bins)

    if out_csv:
        keys = sorted(cov.keys())
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["graph_idx", "score", "loss", "pos"] + keys)
            for i in range(len(score)):
                w.writerow([i, score[i], loss[i], per_graph["pos"][i]]
                           + [cov[k][i] for k in keys])
        print(f"\nPer-graph table written to {out_csv}")


def _report_bins(feat, x, per_graph, bins):
    """Equal-count bins over `feat`, with the label prior shown per bin.

    The prior column matters: a metric like AP moves with class balance, so a
    bin-to-bin difference is only a model effect if the priors are comparable.
    """
    score, loss, pos = per_graph["score"], per_graph["loss"], per_graph["pos"]
    ok = np.isfinite(x) & np.isfinite(score)
    if ok.sum() < bins * 5:
        return
    qs = np.quantile(x[ok], np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)

    print(f"\n  Binned by {feat} (equal-count):")
    print(f"  {'range':<20}{'count':>7}{'score':>9}{'loss':>9}"
          f"{'label_prior':>13}")
    for i in range(len(qs) - 1):
        lo, hi = qs[i], qs[i + 1]
        m = ok & (x >= lo) & ((x <= hi) if i == len(qs) - 2 else (x < hi))
        if m.sum() == 0:
            continue
        pr = np.nanmean(pos[m]) if np.isfinite(pos[m]).any() else np.nan
        print(f"  [{lo:8.3f},{hi:8.3f}]{int(m.sum()):>7}"
              f"{np.nanmean(score[m]):>9.4f}{np.nanmean(loss[m]):>9.4f}"
              f"{pr:>13.3f}")


# ================================================================
# LEAVE-ONE-HOP-OUT
# ================================================================

def report_loho(model, loader, task, device, max_hops, baseline, saved_args):
    if saved_args.get("mask_type") != "shortest_path":
        print(f"\n[loho] SKIPPED: mask_type={saved_args.get('mask_type')}. "
              f"With adj_power the masks are rebuilt from the hop-1 shell "
              f"inside the model, so zeroing a shell does not ablate that hop.")
        return

    print("\n" + "=" * 92)
    print("LEAVE-ONE-HOP-OUT (hop shell zeroed in the input masks at eval)")
    print(f"  baseline {task.metric_label} = {baseline:.4f}")
    print("  A hop whose removal costs nothing was not carrying signal, no")
    print("  matter how much attention mass it received.")
    print("=" * 92)
    print(f"{'hop':>5}{'metric':>12}{'delta':>12}{'rel_%':>10}")
    print("-" * 92)
    rows = []
    for h in range(max_hops):
        _, m = evaluate_split(model, loader, task, device, zero_hop=h)
        delta = m - baseline
        if not task.higher_is_better:
            delta = -delta
        rel = 100.0 * delta / abs(baseline) if baseline else np.nan
        print(f"{h:>5}{m:>12.4f}{delta:>12.4f}{rel:>10.2f}")
        rows.append((h, m, delta))
    print("-" * 92)
    worst = sorted(rows, key=lambda r: r[2])[:5]
    print("Most load-bearing hops (largest drop when removed): "
          + ", ".join(f"h={h}({d:+.4f})" for h, _, d in worst))
    return rows


# ================================================================
# HEAD DIAGNOSTICS
# ================================================================

@torch.no_grad()
def report_head_diag(model, loader, task, device, max_batches=20):
    """Aggregate per-head attention statistics over several batches."""
    set_attn_diagnostics(True)
    acc, cross = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    try:
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            pyg_batch, dist_masks, node_masks = _move_batch_to_device(batch, device)
            model(pyg_batch, dist_masks, node_masks)
            for li, layer in enumerate(getattr(model, "layers", [])):
                d = getattr(getattr(layer, "attn", None), "_diag", None)
                if d is not None:
                    for k, v in d.items():
                        acc[li][k].append(v)
                cd = getattr(getattr(layer, "cross_hop", None), "_diag", None)
                if cd is not None:
                    cross[li].append(cd)
    finally:
        set_attn_diagnostics(False)

    if not acc:
        print("\n[head_diag] No statistics captured. Per-head diagnostics are "
              "instrumented for HopMaskedMHA; MoE/multihop attention variants "
              "are not covered.")
        return

    hop_sets = getattr(model, "head_hop_sets", None)
    print("\n" + "=" * 92)
    print(f"PER-HEAD ATTENTION DIAGNOSTICS (mean over {max_batches} batches)")
    print("  ent_n  normalised entropy: 1.0 = uniform over allowed keys, 0 = one key")
    print("  part   participation ratio = effective number of keys attended")
    print("  keys   mean allowed keys per query row (the hop's structural capacity)")
    print("  self   attention mass that stays on the query node")
    print("  |out|  norm of what the head transports; ~0 means it moves nothing")
    print("  live   fraction of query rows where the head had any allowed key")
    print("=" * 92)
    for li in sorted(acc):
        st = {k: torch.stack(v).mean(0) for k, v in acc[li].items()}
        print(f"[layer {li}]")
        for h in range(st["entropy"].numel()):
            lbl = "global"
            if hop_sets is not None and h < len(hop_sets):
                hs = hop_sets[h]
                lbl = "global" if hs is None else ",".join(map(str, hs))
            print(f"  head {h:02d} | hops={lbl:<12s} "
                  f"ent_n={float(st['entropy_norm'][h]):.3f} "
                  f"part={float(st['participation'][h]):6.2f} "
                  f"keys={float(st['allowed_keys'][h]):6.2f} "
                  f"self={float(st['self_mass'][h]):.3f} "
                  f"|out|={float(st['out_norm'][h]):.3f} "
                  f"live={float(st['live_frac'][h]):.3f}")

        if cross.get(li):
            mix = torch.stack([c["mix_matrix"] for c in cross[li]]).mean(0)
            ent = torch.stack([c["entropy"] for c in cross[li]]).mean(0)
            print(f"[layer {li}] cross-hop mixing (row=query head, col=key head):")
            for h in range(mix.shape[0]):
                row = " ".join(f"{float(v):.2f}" for v in mix[h])
                print(f"  head {h:02d} | ent={float(ent[h]):.3f} | {row}")


# ================================================================
# MAIN
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Post-hoc analysis of a hop-masked transformer checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override the checkpoint's batch size.")
    p.add_argument("--bins", type=int, default=6,
                   help="Number of equal-count bins in the binned tables.")
    p.add_argument("--bin_feature", type=str, default=None,
                   help="Force the binned table onto this feature "
                        "(e.g. diameter). Default: top 3 by |rho|n|.")
    p.add_argument("--limit_graphs", type=int, default=None,
                   help="Only compute covariates for the first N graphs.")
    p.add_argument("--diag_batches", type=int, default=20)
    p.add_argument("--out_csv", type=str, default=None,
                   help="Write the per-graph feature/score table here.")
    p.add_argument("--no_covariates", dest="do_covariates",
                   action="store_false", default=True)
    p.add_argument("--no_loho", dest="do_loho",
                   action="store_false", default=True)
    p.add_argument("--no_head_diag", dest="do_head_diag",
                   action="store_false", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, loader, task, dataset_info, saved = load_checkpoint(
        args.checkpoint, args.split, device, batch_size=args.batch_size)

    per_graph, baseline = evaluate_split(model, loader, task, device)
    print(f"\nSplit {args.split}: {task.metric_label} = {baseline:.4f} "
          f"over {len(per_graph['score'])} graphs "
          f"(mean per-graph score {np.nanmean(per_graph['score']):.4f})")

    if args.do_covariates:
        cov = collect_covariates(loader, limit=args.limit_graphs)
        k = len(cov["n_nodes"])
        pg = {kk: vv[:k] for kk, vv in per_graph.items()}
        report_covariates(cov, pg, out_csv=args.out_csv, bins=args.bins,
                          top_feature=args.bin_feature)

    if args.do_loho:
        report_loho(model, loader, task, device, saved["max_hops"],
                    baseline, saved)

    if args.do_head_diag:
        report_head_diag(model, loader, task, device,
                         max_batches=args.diag_batches)


if __name__ == "__main__":
    main()
