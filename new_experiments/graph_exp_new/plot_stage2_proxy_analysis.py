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
from sklearn.mixture import GaussianMixture
import torch



def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot diagnostics from Stage 2 proxy-pairs pickle"
    )
    parser.add_argument("--proxy_pairs_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--gmm_k_max",
        type=int,
        default=10,
        help="Maximum number of GMM components to try for BIC selection",
    )
    parser.add_argument(
        "--gmm_n_init",
        type=int,
        default=5,
        help="Number of GMM random initialisations per K (higher = more stable)",
    )
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
        default=10000,
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


# ---------------------------------------------------------------------------
# GMM / BIC helpers
# ---------------------------------------------------------------------------

def _fit_gmm_bic(data_1d, k_max, n_init, seed):
    """Fit GMMs with K=1..k_max on a 1-D array and return BIC scores.

    Returns
    -------
    ks        : list[int]   – component counts tried
    bics      : list[float] – BIC for each K
    best_k    : int         – K with the lowest BIC
    best_gmm  : GaussianMixture fitted with best_k components
    """
    finite = data_1d[np.isfinite(data_1d)].reshape(-1, 1)
    if finite.shape[0] < 2:
        return [], [], None, None

    k_max_eff = min(k_max, finite.shape[0])
    ks, bics = [], []
    best_bic = np.inf
    best_k = 1
    best_gmm = None

    for k in range(1, k_max_eff + 1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=n_init,
            random_state=seed,
            max_iter=300,
        )
        gm.fit(finite)
        bic = gm.bic(finite)
        ks.append(k)
        bics.append(float(bic))
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gm

    return ks, bics, best_k, best_gmm


def _save_gmm_bic_plot(ks, bics, best_k, title, path, dpi=160):
    """Save a BIC-vs-K elbow plot with the selected K annotated."""
    if not ks:
        return
    plt.figure(figsize=(7.0, 5.0))
    plt.plot(ks, bics, marker="o", color="#2a9d8f", linewidth=2, markersize=7,
             label="BIC")
    plt.axvline(x=best_k, color="#e63946", linestyle="--", linewidth=1.5,
                label=f"Best K={best_k}")
    plt.scatter([best_k], [bics[best_k - 1]], color="#e63946", s=80, zorder=5)
    plt.title(title)
    plt.xlabel("Number of GMM components (K)")
    plt.ylabel("BIC")
    plt.xticks(ks)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _save_gmm_overlay_hist(data_1d, best_gmm, best_k, title, xlabel, path,
                           bins=60, dpi=160):
    """Histogram of data with fitted GMM density overlaid."""
    finite = data_1d[np.isfinite(data_1d)]
    if finite.size == 0 or best_gmm is None:
        return
    plt.figure(figsize=(7.5, 5.5))
    counts, bin_edges, _ = plt.hist(
        finite, bins=bins, density=True,
        color="#457b9d", edgecolor="black", alpha=0.65, label="data"
    )
    x_grid = np.linspace(finite.min(), finite.max(), 500).reshape(-1, 1)
    log_prob = best_gmm.score_samples(x_grid)
    plt.plot(x_grid.ravel(), np.exp(log_prob), color="#e76f51", linewidth=2.5,
             label=f"GMM K={best_k}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


# ---------------------------------------------------------------------------
# Learnability analysis
# ---------------------------------------------------------------------------

# Thresholds (1-D case; used for the heuristic verdict)
_SEP_HARD      = 3.0   # min pairwise (mean_i - mean_j) / avg_std > this => well-separated modes
_SEP_EASY      = 1.0   # < this => modes heavily overlap (like one blurry blob)
_IMBAL_HARD    = 5.0   # max_weight / min_weight above this => rare modes exist
_ENT_NORM_LOW  = 0.5   # normalised entropy below this => very unbalanced weights
_K_VERY_HARD   = 5     # BIC selects this many or more components
_K_HARD        = 3
_CV_HIGH       = 1.0   # coefficient of variation of a component > this => very spread


def _gmm_learnability_stats(best_k, best_gmm):
    """Compute interpretable learnability stats for the best GMM.

    Returns a dict with:
      components        – per-component stats (mean, std, weight, cv, range_1sigma)
      separation        – pairwise |mean_i - mean_j| / avg_sigma for all pairs
      min_separation    – smallest of the above (worst case)
      max_separation    – largest of the above
      weight_entropy    – Shannon entropy of the weight vector (nats)
      weight_entropy_norm – entropy / log(K)  in [0,1]; 1 = perfectly balanced
      imbalance_ratio   – max_weight / min_weight
      dominant_weight   – weight of the heaviest component
      rare_mode_exists  – True if any component has weight < 0.05
      modes_overlap     – True if min_separation < _SEP_EASY
      modes_separated   – True if min_separation > _SEP_HARD
      verdict           – plain-English learnability assessment
      verdict_flags     – list of specific concern strings
    """
    if best_gmm is None or best_k < 1:
        return {"status": "no_gmm"}

    means   = best_gmm.means_.ravel()          # (K,)
    weights = best_gmm.weights_                # (K,)
    # For full covariance 1-D the shape is (K,1,1)
    stds = np.sqrt(best_gmm.covariances_.reshape(best_k, -1)[:, 0])  # (K,)

    # --- Per-component stats ------------------------------------------------
    components = []
    for i in range(best_k):
        mu, sigma, w = float(means[i]), float(stds[i]), float(weights[i])
        cv = abs(sigma / mu) if abs(mu) > 1e-12 else float("inf")
        components.append({
            "index":        i,
            "mean":         round(mu, 6),
            "std":          round(sigma, 6),
            "weight":       round(w, 6),
            "cv":           round(cv, 4),          # coefficient of variation
            "range_1sigma": [round(mu - sigma, 6), round(mu + sigma, 6)],
            "range_2sigma": [round(mu - 2*sigma, 6), round(mu + 2*sigma, 6)],
        })

    # --- Pairwise separation ------------------------------------------------
    sep_pairs = []
    for i in range(best_k):
        for j in range(i + 1, best_k):
            dist = abs(float(means[i]) - float(means[j]))
            avg_sigma = (float(stds[i]) + float(stds[j])) / 2.0
            ratio = dist / (avg_sigma + 1e-12)
            sep_pairs.append({
                "pair": [i, j],
                "mean_distance": round(dist, 6),
                "avg_sigma":     round(avg_sigma, 6),
                "separation_ratio": round(ratio, 4),
            })

    all_ratios = [p["separation_ratio"] for p in sep_pairs] if sep_pairs else []
    min_sep = float(min(all_ratios)) if all_ratios else float("nan")
    max_sep = float(max(all_ratios)) if all_ratios else float("nan")

    # --- Weight diversity ---------------------------------------------------
    eps = 1e-12
    w_entropy = float(-np.sum(weights * np.log(weights + eps)))
    w_entropy_norm = w_entropy / (np.log(best_k) + eps) if best_k > 1 else 1.0
    imbalance_ratio = float(np.max(weights) / (np.min(weights) + eps))
    dominant_weight = float(np.max(weights))
    rare_mode_exists = bool(np.any(weights < 0.05))

    # --- Verdict ------------------------------------------------------------
    flags = []
    score = 0  # higher = harder

    if best_k >= _K_VERY_HARD:
        flags.append(f"HIGH MODE COUNT: K={best_k} (≥{_K_VERY_HARD}) — generator must cover many distinct target regions")
        score += 3
    elif best_k >= _K_HARD:
        flags.append(f"MULTIMODAL: K={best_k} — generator needs to produce distinct clusters")
        score += 1

    if np.isfinite(min_sep) and min_sep > _SEP_HARD:
        flags.append(f"WELL-SEPARATED MODES: min separation ratio={min_sep:.2f} (>{_SEP_HARD}) — "
                     f"hard for a single-output generator; consider mixture-of-experts or multi-step sampling")
        score += 2
    elif np.isfinite(min_sep) and min_sep < _SEP_EASY:
        flags.append(f"OVERLAPPING MODES: min separation ratio={min_sep:.2f} (<{_SEP_EASY}) — "
                     f"blurry boundary; generator may blend modes (mean collapse risk)")
        score += 1

    if rare_mode_exists:
        flags.append(f"RARE MODES PRESENT: min weight={np.min(weights):.4f} — "
                     f"generator will likely under-sample these; expect mode drop")
        score += 2

    if imbalance_ratio > _IMBAL_HARD:
        flags.append(f"SEVERE WEIGHT IMBALANCE: ratio={imbalance_ratio:.1f} — "
                     f"training signal dominated by heavy components; rare modes starved")
        score += 1

    high_cv_comps = [c for c in components if np.isfinite(c["cv"]) and c["cv"] > _CV_HIGH]
    if high_cv_comps:
        flags.append(
            f"HIGH WITHIN-COMPONENT SPREAD: components {[c['index'] for c in high_cv_comps]} have CV>{_CV_HIGH} — "
            f"each mode itself is diffuse; high output variance needed"
        )
        score += 1

    if score == 0:
        verdict = "EASY — unimodal / well-overlapping, low imbalance: a standard MLP/GNN generator should handle this well"
    elif score <= 2:
        verdict = "MODERATE — some multimodality or imbalance; a larger generator with careful loss weighting is recommended"
    elif score <= 4:
        verdict = "HARD — multiple distinct modes and/or severe imbalance; consider: (a) per-mode loss weighting, " \
                  "(b) flow-matching / diffusion generator, (c) conditional generation with mode label"
    else:
        verdict = "VERY HARD — distribution is highly complex (many separated modes, severe imbalance, or diffuse components); " \
                  "a single deterministic generator will likely collapse to the mean. Use a stochastic / latent-variable model."

    return {
        "status":              "ok",
        "best_k":              int(best_k),
        "components":          components,
        "separation_pairs":    sep_pairs,
        "min_separation":      round(min_sep, 4) if np.isfinite(min_sep) else None,
        "max_separation":      round(max_sep, 4) if np.isfinite(max_sep) else None,
        "weight_entropy":      round(w_entropy, 6),
        "weight_entropy_norm": round(float(w_entropy_norm), 4),
        "imbalance_ratio":     round(imbalance_ratio, 4),
        "dominant_weight":     round(dominant_weight, 4),
        "rare_mode_exists":    rare_mode_exists,
        "modes_overlap":       bool(np.isfinite(min_sep) and min_sep < _SEP_EASY),
        "modes_separated":     bool(np.isfinite(min_sep) and min_sep > _SEP_HARD),
        "verdict":             verdict,
        "verdict_flags":       flags,
        "learnability_score":  score,   # 0=easy … >=5=very hard
    }


def _save_gmm_component_profile(data_1d, best_gmm, best_k, learnability,
                                title, xlabel, path, dpi=160):
    """Bell-curve profile plot: one Gaussian per component, coloured by weight.

    Annotates each component with its mean ± σ and weight.
    A small rug of the raw data is drawn at the bottom.
    """
    if best_gmm is None:
        return

    finite = data_1d[np.isfinite(data_1d)]
    if finite.size == 0:
        return

    means   = best_gmm.means_.ravel()
    weights = best_gmm.weights_
    stds    = np.sqrt(best_gmm.covariances_.reshape(best_k, -1)[:, 0])

    x_min = min(finite.min(), (means - 3 * stds).min())
    x_max = max(finite.max(), (means + 3 * stds).max())
    x_grid = np.linspace(x_min, x_max, 600)

    cmap   = plt.cm.get_cmap("plasma", best_k)
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    # Rug plot (thin, at y~0)
    ax.plot(finite, np.full_like(finite, -0.002 * (1.0 / (stds.mean() + 1e-8))),
            '|', color='#333333', alpha=0.15, markersize=4, label='data (rug)')

    # Mixture density
    log_prob = best_gmm.score_samples(x_grid.reshape(-1, 1))
    ax.plot(x_grid, np.exp(log_prob), color='#222222', linewidth=2.0,
            linestyle='--', label='mixture density', zorder=10)

    # Per-component Gaussians
    from scipy.stats import norm as _norm
    for i in range(best_k):
        mu, sigma, w = float(means[i]), float(stds[i]), float(weights[i])
        y_comp = w * _norm.pdf(x_grid, mu, sigma)
        col = cmap(i)
        ax.fill_between(x_grid, y_comp, alpha=0.25, color=col)
        ax.plot(x_grid, y_comp, color=col, linewidth=1.8,
                label=f"K{i}: μ={mu:.3g}, σ={sigma:.3g}, w={w:.3f}")
        # Annotate mean
        peak_y = float(w * _norm.pdf(mu, mu, sigma))
        ax.annotate(
            f"K{i}\n" + ("★" if w == weights.max() else ""),
            xy=(mu, peak_y), xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=8, color=col,
            arrowprops=dict(arrowstyle='->', color=col, lw=0.8),
        )

    # Verdict banner
    ls = learnability.get("learnability_score", 0)
    colors_banner = ["#2a9d8f", "#e9c46a", "#f4a261", "#e63946"]
    banner_col = colors_banner[min(ls // 2, 3)]
    verdict_short = learnability.get("verdict", "").split("—")[0].strip()
    ax.set_title(f"{title}\n{verdict_short}", color=banner_col, fontweight='bold')

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Weighted density")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _run_gmm_analysis(data_1d, feature_name, out_dir, args):
    """Full GMM pipeline for one scalar series; returns a result dict."""
    ks, bics, best_k, best_gmm = _fit_gmm_bic(
        data_1d, k_max=args.gmm_k_max, n_init=args.gmm_n_init, seed=args.seed
    )
    result = {"feature": feature_name, "k_max": args.gmm_k_max}
    if not ks:
        result["status"] = "skipped_insufficient_data"
        return result

    result["status"] = "ok"
    result["ks"] = ks
    result["bics"] = bics
    result["best_k"] = int(best_k)
    result["best_bic"] = float(bics[best_k - 1])
    result["means"] = best_gmm.means_.ravel().tolist()
    result["weights"] = best_gmm.weights_.tolist()
    result["covariances"] = best_gmm.covariances_.ravel().tolist()

    # --- Learnability analysis ---
    learn = _gmm_learnability_stats(best_k, best_gmm)
    result["learnability"] = learn

    slug = feature_name.replace(" ", "_")
    _save_gmm_bic_plot(
        ks, bics, best_k,
        title=f"GMM BIC – {feature_name}",
        path=os.path.join(out_dir, f"gmm_bic_{slug}.png"),
        dpi=args.dpi,
    )
    _save_gmm_overlay_hist(
        data_1d, best_gmm, best_k,
        title=f"GMM Fit – {feature_name} (K={best_k})",
        xlabel=feature_name,
        path=os.path.join(out_dir, f"gmm_overlay_{slug}.png"),
        bins=args.bins,
        dpi=args.dpi,
    )
    _save_gmm_component_profile(
        data_1d, best_gmm, best_k, learn,
        title=f"Component Profile – {feature_name}",
        xlabel=feature_name,
        path=os.path.join(out_dir, f"gmm_profile_{slug}.png"),
        dpi=args.dpi,
    )
    return result


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

    # 5) GMM / BIC analysis
    print("\nFitting GMMs (K=1..{}) for BIC model selection …".format(args.gmm_k_max),
          flush=True)
    gmm_results = {}

    # Scalar loss series
    for feat_name, arr in [
        ("opt_loss",            opt_loss),
        ("mmd_loss",            mmd_loss),
        ("base_loss",           base_loss),
        ("cross_moment_loss",   cross_loss),
        ("prior_moment_loss",   prior_loss),
        ("absolute_improvement", improvement),
    ]:
        print(f"  GMM BIC: {feat_name} …", flush=True)
        r = _run_gmm_analysis(arr, feat_name, args.out_dir, args)
        gmm_results[feat_name] = r
        if r["status"] == "ok":
            learn = r.get("learnability", {})
            stats_log.append(
                f"GMM BIC [{feat_name}]: best_k={r['best_k']} "
                f"best_bic={r['best_bic']:.4f} "
                f"means={[round(m,6) for m in r['means']]} "
                f"weights={[round(w,4) for w in r['weights']]} "
                f"| LEARNABILITY: {learn.get('verdict','?')} "
                f"| min_sep={learn.get('min_sep_ratio', learn.get('min_separation','?'))} "
                f"imbalance={learn.get('imbalance_ratio','?')} "
                f"score={learn.get('learnability_score','?')}"
            )
            for flag in learn.get("verdict_flags", []):
                stats_log.append(f"  FLAG [{feat_name}]: {flag}")
            print(
                f"    best_k={r['best_k']}  bic={r['best_bic']:.4f}  "
                f"means={[round(m,4) for m in r['means']]}",
                flush=True,
            )
            print(f"    LEARNABILITY: {learn.get('verdict','?')}", flush=True)
            for flag in learn.get("verdict_flags", []):
                print(f"      ⚑ {flag}", flush=True)
        else:
            stats_log.append(f"GMM BIC [{feat_name}]: {r['status']}")
            print(f"    skipped – {r['status']}", flush=True)

    # Proxy embedding norms (one scalar per sample)
    if proxy_vecs.size > 0:
        proxy_norms = np.linalg.norm(proxy_vecs, axis=1).astype(np.float64)
        print("  GMM BIC: proxy_emb_norm …", flush=True)
        r_pnorm = _run_gmm_analysis(proxy_norms, "proxy_emb_norm", args.out_dir, args)
        gmm_results["proxy_emb_norm"] = r_pnorm
        if r_pnorm["status"] == "ok":
            learn_pn = r_pnorm.get("learnability", {})
            stats_log.append(
                f"GMM BIC [proxy_emb_norm]: best_k={r_pnorm['best_k']} "
                f"best_bic={r_pnorm['best_bic']:.4f} "
                f"| LEARNABILITY: {learn_pn.get('verdict','?')} "
                f"score={learn_pn.get('learnability_score','?')}"
            )
            for flag in learn_pn.get("verdict_flags", []):
                stats_log.append(f"  FLAG [proxy_emb_norm]: {flag}")
            print(
                f"    best_k={r_pnorm['best_k']}  bic={r_pnorm['best_bic']:.4f}",
                flush=True,
            )
            print(f"    LEARNABILITY: {learn_pn.get('verdict','?')}", flush=True)
            for flag in learn_pn.get("verdict_flags", []):
                print(f"      ⚑ {flag}", flush=True)

    # Encoder embedding norms
    if enc_vecs.size > 0:
        enc_norms = np.linalg.norm(enc_vecs, axis=1).astype(np.float64)
        print("  GMM BIC: encoder_emb_norm …", flush=True)
        r_enorm = _run_gmm_analysis(enc_norms, "encoder_emb_norm", args.out_dir, args)
        gmm_results["encoder_emb_norm"] = r_enorm
        if r_enorm["status"] == "ok":
            learn_en = r_enorm.get("learnability", {})
            stats_log.append(
                f"GMM BIC [encoder_emb_norm]: best_k={r_enorm['best_k']} "
                f"best_bic={r_enorm['best_bic']:.4f} "
                f"| LEARNABILITY: {learn_en.get('verdict','?')} "
                f"score={learn_en.get('learnability_score','?')}"
            )
            for flag in learn_en.get("verdict_flags", []):
                stats_log.append(f"  FLAG [encoder_emb_norm]: {flag}")
            print(
                f"    best_k={r_enorm['best_k']}  bic={r_enorm['best_bic']:.4f}",
                flush=True,
            )
            print(f"    LEARNABILITY: {learn_en.get('verdict','?')}", flush=True)
            for flag in learn_en.get("verdict_flags", []):
                print(f"      ⚑ {flag}", flush=True)

    stats_json["gmm_bic"] = gmm_results

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
