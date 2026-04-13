"""
Analyze Stage 2 proxy optimization outputs and generate diagnostic plots.

Expected pickle format: a list of dicts with keys used by Stage 2 in
main_staged.py, including:
  - base_loss, opt_loss, mmd_loss, cross_moment_loss, prior_moment_loss
  - base_correct, best_correct
  - encoder_emb, mask, proxy_emb

Example:
    python plot_stage2_proxy_analysis.py \
        --proxy_pairs_path checkpoints_staged/proxy_pairs.pkl \
        --out_dir checkpoints_staged/stage2_analysis
"""

import argparse
import json
import os
import pickle
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
import torch



def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot diagnostics from Stage 2 proxy-pairs pickle"
    )
    parser.add_argument("--proxy_pairs_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (will be auto-capped for tiny datasets)",
    )
    parser.add_argument("--tsne_max_iter", type=int, default=1000)
    parser.add_argument(
        "--tsne_max_points",
        type=int,
        default=20000,
        help="Max total points used by t-SNE; <=0 means use all points",
    )
    parser.add_argument(
        "--dedupe_by_sample_idx",
        action="store_true",
        help="Keep only the lowest-opt_loss entry per sample_idx before plotting",
    )
    return parser.parse_args()


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_float(x):
    if x is None:
        return np.nan
    if torch.is_tensor(x):
        if x.numel() == 0:
            return np.nan
        return float(x.detach().cpu().reshape(-1)[0].item())
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return np.nan
        return float(x.reshape(-1)[0])
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _to_int(x):
    v = _to_float(x)
    if not np.isfinite(v):
        return -1
    return int(v)


def _extract_float_array(records, key):
    vals = []
    missing = 0
    for rec in records:
        if key not in rec:
            missing += 1
        vals.append(_to_float(rec.get(key, np.nan)))
    return np.asarray(vals, dtype=np.float64), missing


def _extract_int_array(records, key):
    vals = []
    missing = 0
    for rec in records:
        if key not in rec:
            missing += 1
        vals.append(_to_int(rec.get(key, -1)))
    return np.asarray(vals, dtype=np.int64), missing


def _safe_stats(arr):
    finite = arr[np.isfinite(arr)]
    stats = {
        "count_total": int(arr.size),
        "count_finite": int(finite.size),
        "count_nan": int(np.isnan(arr).sum()),
        "count_inf": int(np.isinf(arr).sum()),
    }
    if finite.size == 0:
        stats.update(
            {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "q25": np.nan,
                "median": np.nan,
                "q75": np.nan,
                "max": np.nan,
            }
        )
        return stats

    stats.update(
        {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "q25": float(np.percentile(finite, 25)),
            "median": float(np.median(finite)),
            "q75": float(np.percentile(finite, 75)),
            "max": float(np.max(finite)),
        }
    )
    return stats


def _fmt_stats_line(name, stats, missing=0):
    return (
        f"{name}: total={stats['count_total']} finite={stats['count_finite']} "
        f"nan={stats['count_nan']} inf={stats['count_inf']} missing_key={missing} | "
        f"mean={stats['mean']:.6f} std={stats['std']:.6f} min={stats['min']:.6f} "
        f"q25={stats['q25']:.6f} median={stats['median']:.6f} q75={stats['q75']:.6f} "
        f"max={stats['max']:.6f}"
    )


def _save_hist(values, title, xlabel, path, bins=60, dpi=160):
    finite = values[np.isfinite(values)]
    plt.figure(figsize=(7.5, 5.5))
    if finite.size > 0:
        plt.hist(finite, bins=bins, color="#2a9d8f", edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _corr_stats(x, y):
    finite_mask = np.isfinite(x) & np.isfinite(y)
    xf = x[finite_mask]
    yf = y[finite_mask]
    out = {"count_finite_pairs": int(xf.size)}
    if xf.size < 2:
        out.update({"pearson": np.nan, "spearman": np.nan})
        return out

    if np.std(xf) == 0 or np.std(yf) == 0:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(xf, yf)[0, 1])

    spear = spearmanr(xf, yf, nan_policy="omit")
    spearman = float(spear.statistic) if spear is not None else np.nan
    out.update({"pearson": pearson, "spearman": spearman})
    return out


def _save_scatter(x, y, title, xlabel, ylabel, path, dpi=160, add_identity=False):
    finite_mask = np.isfinite(x) & np.isfinite(y)
    xf = x[finite_mask]
    yf = y[finite_mask]

    plt.figure(figsize=(6.5, 6.0))
    if xf.size > 0:
        plt.scatter(xf, yf, s=10, alpha=0.35, color="#1d3557", edgecolors="none")
    if add_identity and xf.size > 0:
        low = min(float(np.min(xf)), float(np.min(yf)))
        high = max(float(np.max(xf)), float(np.max(yf)))
        plt.plot([low, high], [low, high], linestyle="--", color="#e63946", linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _save_box_mmd_vs_correct(mmd, correct, title, xlabel, path, dpi=160):
    finite_mask = np.isfinite(mmd) & (correct >= 0)
    mmd_f = mmd[finite_mask]
    corr_f = correct[finite_mask]

    unique_corr = sorted(np.unique(corr_f).tolist())
    grouped = [mmd_f[corr_f == c] for c in unique_corr]

    plt.figure(figsize=(8.0, 5.5))
    if len(grouped) > 0:
        plt.boxplot(grouped, tick_labels=[str(c) for c in unique_corr], showfliers=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("MMD Loss")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

    group_stats = {}
    for c, g in zip(unique_corr, grouped):
        if g.size == 0:
            continue
        group_stats[int(c)] = {
            "count": int(g.size),
            "mean": float(np.mean(g)),
            "median": float(np.median(g)),
            "q25": float(np.percentile(g, 25)),
            "q75": float(np.percentile(g, 75)),
        }
    return group_stats


def _extract_tsne_vectors(records):
    encoder_vecs = []
    proxy_vecs = []

    for rec in records:
        enc = _to_numpy(rec.get("encoder_emb", None))
        mask = _to_numpy(rec.get("mask", None))
        proxy = _to_numpy(rec.get("proxy_emb", None))

        if enc is None or proxy is None:
            continue
        if enc.ndim != 2 or proxy.ndim != 2:
            continue

        if mask is not None and np.asarray(mask).ndim == 1 and mask.shape[0] == enc.shape[0]:
            valid = enc[np.asarray(mask).astype(bool)]
            if valid.shape[0] == 0:
                valid = enc
        else:
            valid = enc

        if valid.shape[0] == 0 or proxy.shape[0] == 0:
            continue

        # One vector per sample keeps t-SNE tractable for large datasets.
        encoder_vecs.append(np.mean(valid, axis=0))
        proxy_vecs.append(np.mean(proxy, axis=0))

    if len(encoder_vecs) == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)

    enc_arr = np.asarray(encoder_vecs, dtype=np.float32)
    proxy_arr = np.asarray(proxy_vecs, dtype=np.float32)
    return enc_arr, proxy_arr


def _save_tsne_plot(enc_vecs, proxy_vecs, out_path, args):
    if enc_vecs.size == 0 or proxy_vecs.size == 0:
        return {
            "status": "skipped_empty",
            "num_encoder_points": int(enc_vecs.shape[0] if enc_vecs.ndim == 2 else 0),
            "num_proxy_points": int(proxy_vecs.shape[0] if proxy_vecs.ndim == 2 else 0),
        }

    if enc_vecs.shape[1] != proxy_vecs.shape[1]:
        return {
            "status": "skipped_dim_mismatch",
            "encoder_dim": int(enc_vecs.shape[1]),
            "proxy_dim": int(proxy_vecs.shape[1]),
        }

    rng = np.random.default_rng(args.seed)
    n = min(enc_vecs.shape[0], proxy_vecs.shape[0])
    idx = np.arange(n)

    used_all = True
    if args.tsne_max_points > 0:
        max_per_type = max(args.tsne_max_points // 2, 1)
        if n > max_per_type:
            idx = rng.choice(idx, size=max_per_type, replace=False)
            used_all = False

    enc_sel = enc_vecs[idx]
    proxy_sel = proxy_vecs[idx]

    X = np.concatenate([enc_sel, proxy_sel], axis=0)
    y = np.concatenate(
        [
            np.zeros(enc_sel.shape[0], dtype=np.int64),
            np.ones(proxy_sel.shape[0], dtype=np.int64),
        ]
    )

    n_points = X.shape[0]
    if n_points < 2:
        return {"status": "skipped_too_few_points", "n_points": int(n_points)}

    max_perplexity = max((n_points - 1) / 3.0, 1.0)
    perplexity = min(args.tsne_perplexity, max_perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=args.seed,
        max_iter=args.tsne_max_iter,
    )
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(7.0, 6.0))
    enc_mask = y == 0
    prox_mask = y == 1
    plt.scatter(
        Z[enc_mask, 0],
        Z[enc_mask, 1],
        s=9,
        alpha=0.45,
        c="#457b9d",
        label="encoder_emb (sample mean)",
        edgecolors="none",
    )
    plt.scatter(
        Z[prox_mask, 0],
        Z[prox_mask, 1],
        s=9,
        alpha=0.45,
        c="#e76f51",
        label="proxy_emb (sample mean)",
        edgecolors="none",
    )
    plt.title("t-SNE: Encoder vs Proxy Embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()

    enc_centroid = np.mean(Z[enc_mask], axis=0)
    prox_centroid = np.mean(Z[prox_mask], axis=0)
    centroid_distance = float(np.linalg.norm(enc_centroid - prox_centroid))

    return {
        "status": "ok",
        "num_encoder_points": int(enc_sel.shape[0]),
        "num_proxy_points": int(proxy_sel.shape[0]),
        "used_all_points": bool(used_all),
        "perplexity_used": float(perplexity),
        "centroid_distance_2d": centroid_distance,
    }


def _dedupe_by_best_opt_loss(records):
    best_by_idx = {}
    kept_without_sample_idx = 0

    for rec in records:
        sid = rec.get("sample_idx", None)
        if sid is None:
            best_by_idx[f"__no_sid_{kept_without_sample_idx}"] = rec
            kept_without_sample_idx += 1
            continue
        key = int(_to_int(sid))
        if key not in best_by_idx:
            best_by_idx[key] = rec
        else:
            cur = _to_float(rec.get("opt_loss", np.nan))
            old = _to_float(best_by_idx[key].get("opt_loss", np.nan))
            if np.isfinite(cur) and (not np.isfinite(old) or cur < old):
                best_by_idx[key] = rec

    return list(best_by_idx.values())


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.proxy_pairs_path, "rb") as f:
        records = pickle.load(f)

    if not isinstance(records, list):
        raise ValueError("Expected pickle content to be a list of dict records.")

    original_count = len(records)
    if args.dedupe_by_sample_idx:
        records = _dedupe_by_best_opt_loss(records)

    # Extract metrics requested by the user.
    base_loss, miss_base = _extract_float_array(records, "base_loss")
    opt_loss, miss_opt = _extract_float_array(records, "opt_loss")
    mmd_loss, miss_mmd = _extract_float_array(records, "mmd_loss")
    cross_loss, miss_cross = _extract_float_array(records, "cross_moment_loss")
    prior_loss, miss_prior = _extract_float_array(records, "prior_moment_loss")
    base_correct, miss_base_correct = _extract_int_array(records, "base_correct")
    best_correct, miss_best_correct = _extract_int_array(records, "best_correct")

    stats_log = []
    stats_json = {
        "input": {
            "proxy_pairs_path": args.proxy_pairs_path,
            "out_dir": args.out_dir,
            "num_records_loaded": int(original_count),
            "num_records_used": int(len(records)),
            "dedupe_by_sample_idx": bool(args.dedupe_by_sample_idx),
        }
    }

    metric_map = {
        "base_loss": (base_loss, miss_base),
        "opt_loss": (opt_loss, miss_opt),
        "mmd_loss": (mmd_loss, miss_mmd),
        "cross_moment_loss": (cross_loss, miss_cross),
        "prior_moment_loss": (prior_loss, miss_prior),
    }

    for name, (arr, miss) in metric_map.items():
        s = _safe_stats(arr)
        stats_json[name] = {"missing_key": int(miss), **s}
        line = _fmt_stats_line(name, s, missing=miss)
        stats_log.append(line)
        print(line, flush=True)

    # 1) Distributions + stats logs
    _save_hist(
        base_loss,
        title="Distribution of Base Loss",
        xlabel="base_loss",
        path=os.path.join(args.out_dir, "distribution_base_loss.png"),
        bins=args.bins,
        dpi=args.dpi,
    )
    _save_hist(
        opt_loss,
        title="Distribution of Opt Loss",
        xlabel="opt_loss",
        path=os.path.join(args.out_dir, "distribution_opt_loss.png"),
        bins=args.bins,
        dpi=args.dpi,
    )
    _save_hist(
        mmd_loss,
        title="Distribution of MMD Loss",
        xlabel="mmd_loss",
        path=os.path.join(args.out_dir, "distribution_mmd_loss.png"),
        bins=args.bins,
        dpi=args.dpi,
    )
    _save_hist(
        cross_loss,
        title="Distribution of Cross Moment Loss",
        xlabel="cross_moment_loss",
        path=os.path.join(args.out_dir, "distribution_cross_moment_loss.png"),
        bins=args.bins,
        dpi=args.dpi,
    )
    _save_hist(
        prior_loss,
        title="Distribution of Prior Moment Loss",
        xlabel="prior_moment_loss",
        path=os.path.join(args.out_dir, "distribution_prior_moment_loss.png"),
        bins=args.bins,
        dpi=args.dpi,
    )

    # 2) Scatter plots + stats logs
    scatter_base_opt = _corr_stats(base_loss, opt_loss)
    scatter_mmd_opt = _corr_stats(mmd_loss, opt_loss)
    stats_json["scatter_base_loss_vs_opt_loss"] = scatter_base_opt
    stats_json["scatter_mmd_loss_vs_opt_loss"] = scatter_mmd_opt
    stats_log.append(
        "base_loss vs opt_loss: "
        f"finite_pairs={scatter_base_opt['count_finite_pairs']} "
        f"pearson={scatter_base_opt['pearson']:.6f} "
        f"spearman={scatter_base_opt['spearman']:.6f}"
    )
    stats_log.append(
        "mmd_loss vs opt_loss: "
        f"finite_pairs={scatter_mmd_opt['count_finite_pairs']} "
        f"pearson={scatter_mmd_opt['pearson']:.6f} "
        f"spearman={scatter_mmd_opt['spearman']:.6f}"
    )

    _save_scatter(
        base_loss,
        opt_loss,
        title="Base Loss vs Opt Loss",
        xlabel="base_loss",
        ylabel="opt_loss",
        path=os.path.join(args.out_dir, "scatter_base_loss_vs_opt_loss.png"),
        dpi=args.dpi,
        add_identity=True,
    )
    _save_scatter(
        mmd_loss,
        opt_loss,
        title="MMD Loss vs Opt Loss",
        xlabel="mmd_loss",
        ylabel="opt_loss",
        path=os.path.join(args.out_dir, "scatter_mmd_loss_vs_opt_loss.png"),
        dpi=args.dpi,
        add_identity=False,
    )

    # Improvement stats (helpful for interpreting base vs opt scatter).
    improvement = base_loss - opt_loss
    rel_improvement = improvement / (np.abs(base_loss) + 1e-12)
    imp_stats = _safe_stats(improvement)
    rel_imp_stats = _safe_stats(rel_improvement)
    stats_json["absolute_improvement_base_minus_opt"] = imp_stats
    stats_json["relative_improvement"] = rel_imp_stats
    stats_log.append(_fmt_stats_line("absolute_improvement(base-opt)", imp_stats))
    stats_log.append(_fmt_stats_line("relative_improvement", rel_imp_stats))

    # 3) Box plots + grouped stats
    best_group_stats = _save_box_mmd_vs_correct(
        mmd_loss,
        best_correct,
        title="MMD Loss vs Best Correct",
        xlabel="best_correct",
        path=os.path.join(args.out_dir, "box_mmd_vs_best_correct.png"),
        dpi=args.dpi,
    )
    base_group_stats = _save_box_mmd_vs_correct(
        mmd_loss,
        base_correct,
        title="MMD Loss vs Base Correct",
        xlabel="base_correct",
        path=os.path.join(args.out_dir, "box_mmd_vs_base_correct.png"),
        dpi=args.dpi,
    )
    stats_json["mmd_vs_best_correct_group_stats"] = best_group_stats
    stats_json["mmd_vs_base_correct_group_stats"] = base_group_stats

    stats_log.append(
        "mmd_loss vs best_correct groups: "
        + ", ".join(
            [
                f"{k}->n={v['count']}, mean={v['mean']:.6f}, median={v['median']:.6f}"
                for k, v in sorted(best_group_stats.items())
            ]
        )
    )
    stats_log.append(
        "mmd_loss vs base_correct groups: "
        + ", ".join(
            [
                f"{k}->n={v['count']}, mean={v['mean']:.6f}, median={v['median']:.6f}"
                for k, v in sorted(base_group_stats.items())
            ]
        )
    )

    # 4) t-SNE (encoder_emb vs proxy_emb from all samples)
    enc_vecs, proxy_vecs = _extract_tsne_vectors(records)
    tsne_info = _save_tsne_plot(
        enc_vecs,
        proxy_vecs,
        out_path=os.path.join(args.out_dir, "tsne_encoder_vs_proxy.png"),
        args=args,
    )
    stats_json["tsne"] = tsne_info
    stats_log.append(f"t-SNE: {tsne_info}")

    # Save logs
    log_path = os.path.join(args.out_dir, "statistics_log.txt")
    with open(log_path, "w") as f:
        for line in stats_log:
            f.write(line + "\n")

    json_path = os.path.join(args.out_dir, "statistics.json")
    with open(json_path, "w") as f:
        json.dump(stats_json, f, indent=2)

    print(f"\nSaved plots and stats to: {args.out_dir}", flush=True)
    print(f"Statistics log: {log_path}", flush=True)
    print(f"Statistics json: {json_path}", flush=True)


if __name__ == "__main__":
    main()
