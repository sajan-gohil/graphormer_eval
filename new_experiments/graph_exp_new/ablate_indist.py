"""
Ablation: evaluate how performance scales with increasing proxy multiples.

Loads the best models produced by main_indist.py from a given checkpoint folder
and evaluates val / test AP at proxy_multiplier = 0, 1, 2, ... k,
where k * num_proxies ≤ avg_N  (i.e. total proxies stay ≤ the original node count)
or until AP plateaus for `--plateau_patience` consecutive steps.

The effective token count at each step:
    k=0 : N                    (baseline, no proxies)
    k=1 : N + 1 * num_proxies
    k=2 : N + 2 * num_proxies
    ...
    k=K : N + K * num_proxies  (K * num_proxies ≤ avg_N)

Oversample + Diversity Selection (--oversample_factor K)
--------------------------------------------------------
When --oversample_factor K is set (K > 1), each ablation step that wants
`target` proxies will:
  1. Generate K * target proxies  (K calls to the generator)
  2. Select the `target` most diverse proxies via farthest-point sampling
  3. Feed only those `target` proxies to the backbone

This tests whether oversampling and then cherry-picking the most unique
proxies improves quality vs. using all generated proxies directly.

Usage
-----
    # Standard ablation (no oversampling):
    python ablate_indist.py --ckpt_dir checkpoints_indist2/vanilla_gt/score_based \\
                            --shared_dir checkpoints_indist2/vanilla_gt/shared \\
                            --eval_model phase5                                \\
                            --backbone vanilla_gt --generator score_based      \\
                            --num_proxies 64 --hidden_dim 256 --num_layers 8

    # With 4× oversampling + diversity selection:
    python ablate_indist.py --ckpt_dir checkpoints_indist2/vanilla_gt/score_based \\
                            --shared_dir checkpoints_indist2/vanilla_gt/shared \\
                            --eval_model phase5 --oversample_factor 4          \\
                            --backbone vanilla_gt --generator score_based      \\
                            --num_proxies 64

    # Phase 1 baseline only (no generator, just model at k=0):
    python ablate_indist.py --shared_dir checkpoints_indist2/vanilla_gt/shared \\
                            --eval_model phase1                                \\
                            --backbone vanilla_gt --num_proxies 64
"""

import argparse
import json
import os
import time
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn

from data import get_loaders
from models import GraphTransformer, GREDEncoder, GREDHybridTransformer
from generators import (
    ScoreBasedGenerator, GNNPoolingGenerator, PMAGenerator, GraphCoarseningGenerator,
)
from metrics import compute_macro_ap


# ================================================================
# CONFIG (reuses main_indist build helpers)
# ================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Proxy-count ablation for In-Distribution Proxy Learning")

    # --- Checkpoint paths ---
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="Directory containing phase3_generator.pt / phase5_best.pt etc.")
    p.add_argument("--shared_dir", type=str, default=None,
                   help="Directory containing phase1_best.pt / phase2_best.pt "
                        "(shared backbone checkpoints).")
    p.add_argument("--eval_model", type=str, default="phase5",
                   choices=["phase1", "phase1+gen", "phase5"],
                   help="Which model(s) to evaluate.  "
                        "phase1: baseline only (k=0).  "
                        "phase1+gen: Phase-1 model + Phase-3 generator.  "
                        "phase5: Phase-5 joint-trained model+generator.")

    # --- Architecture (must match training) ---
    p.add_argument("--backbone", type=str, default="vanilla_gt",
                   choices=["vanilla_gt", "gred", "hybrid"])
    p.add_argument("--generator", type=str, default="score_based",
                   choices=["score_based", "pma", "graph_coarsening", "gnn_pooling"])

    # Model dims (must match training)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_proxies", type=int, default=32,
                   help="M — base number of proxies per generator call")

    # Lap PE
    p.add_argument("--use_lap_pe", action="store_true", default=False)
    p.add_argument("--lap_pe_dim", type=int, default=8)

    # GRED / hybrid specific
    p.add_argument("--state_dim", type=int, default=88)
    p.add_argument("--num_gred_layers", type=int, default=8)
    p.add_argument("--num_transformer_layers", type=int, default=2)
    p.add_argument("--gred_expand", type=int, default=1)
    p.add_argument("--r_min", type=float, default=0.0)
    p.add_argument("--r_max", type=float, default=1.0)
    p.add_argument("--max_phase_lru", type=float, default=6.28)
    p.add_argument("--gred_act", type=str, default="full-glu",
                   choices=["full-glu", "half-glu"])
    p.add_argument("--max_hops", type=int, default=40)
    p.add_argument("--dist_mask_workers", type=int, default=8)

    # Generator dims (must match training)
    p.add_argument("--gen_hidden_dim", type=int, default=256)
    p.add_argument("--gen_num_layers", type=int, default=6)
    p.add_argument("--gen_num_heads", type=int, default=8)
    p.add_argument("--gen_dropout", type=float, default=0.2)
    p.add_argument("--pma_query_mode", type=str, default="farthest_point",
                   choices=["farthest_point", "soft_kmeans"])
    p.add_argument("--coarsen_gnn_type", type=str, default="GIN",
                   choices=["GIN", "GCN"])
    p.add_argument("--coarsen_reg_weight", type=float, default=0.1)
    p.add_argument("--coarsen_reg_type", type=str, default="mincut")
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--gnn_type", type=str, default="GINE",
                   choices=["GCN", "GIN", "GINE", "GAT"])
    p.add_argument("--pool_types", type=str, nargs="+", default=["max"])
    p.add_argument("--decode_hidden", type=int, default=128)
    p.add_argument("--decode_layers", type=int, default=3)
    p.add_argument("--idx_emb_dim", type=int, default=128)
    p.add_argument("--decode_mode", type=str, default="shared",
                   choices=["shared", "grouped"])

    # --- Ablation parameters ---
    p.add_argument("--max_k", type=int, default=None,
                   help="Maximum proxy multiplier (default: auto = avg_N / num_proxies)")
    p.add_argument("--plateau_patience", type=int, default=3,
                   help="Stop if val AP does not improve for this many consecutive k steps")
    p.add_argument("--num_runs", type=int, default=1,
                   help="Number of evaluation runs per k (for stochastic generators, "
                        "e.g. PMA). Results are mean ± std.")
    p.add_argument("--oversample_factor", type=int, default=1,
                   help="Generate oversample_factor * target proxies, then keep only "
                        "the target count by selecting the most diverse ones via "
                        "farthest-point sampling. 1 = no oversampling (default).")

    # --- Data / device ---
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--results_file", type=str, default=None,
                   help="Path to save results JSON (default: <ckpt_dir>/ablation_results.json)")

    return p


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.shared_dir is None:
        args.shared_dir = args.ckpt_dir
    return args


# ================================================================
# MODEL / GENERATOR BUILDERS (identical to main_indist)
# ================================================================

def build_model(args):
    lap_pe_dim = args.lap_pe_dim if args.use_lap_pe else 0
    if args.backbone == "vanilla_gt":
        return GraphTransformer(
            num_layers=args.num_layers, num_heads=args.num_heads,
            hidden_dim=args.hidden_dim, output_dim=args.output_dim,
            dropout=args.dropout, lap_pe_dim=lap_pe_dim,
        )
    elif args.backbone == "gred":
        return GREDEncoder(
            hidden_dim=args.hidden_dim, state_dim=args.state_dim,
            num_layers=args.num_gred_layers, expand=args.gred_expand,
            r_min=args.r_min, r_max=args.r_max,
            max_phase=args.max_phase_lru,
            dropout=args.dropout, act=args.gred_act,
            output_dim=args.output_dim, lap_pe_dim=lap_pe_dim,
        )
    elif args.backbone == "hybrid":
        return GREDHybridTransformer(
            hidden_dim=args.hidden_dim, state_dim=args.state_dim,
            num_gred_layers=args.num_gred_layers,
            num_transformer_layers=args.num_transformer_layers,
            num_heads=args.num_heads, expand=args.gred_expand,
            r_min=args.r_min, r_max=args.r_max,
            max_phase=args.max_phase_lru,
            dropout=args.dropout, act=args.gred_act,
            output_dim=args.output_dim, lap_pe_dim=lap_pe_dim,
        )
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")


def build_generator(args):
    if args.generator == "score_based":
        return ScoreBasedGenerator(
            num_proxies=args.num_proxies, input_dim=args.hidden_dim,
            hidden_dim=args.gen_hidden_dim, num_layers=args.gen_num_layers,
            num_heads=args.gen_num_heads, dropout=args.gen_dropout,
        )
    elif args.generator == "pma":
        return PMAGenerator(
            num_proxies=args.num_proxies, input_dim=args.hidden_dim,
            num_heads=args.gen_num_heads, num_layers=args.gen_num_layers,
            dropout=args.gen_dropout, query_mode=args.pma_query_mode,
        )
    elif args.generator == "graph_coarsening":
        return GraphCoarseningGenerator(
            num_proxies=args.num_proxies, input_dim=args.hidden_dim,
            gnn_layers=args.gen_num_layers, gnn_type=args.coarsen_gnn_type,
            dropout=args.gen_dropout, reg_type=args.coarsen_reg_type,
            reg_weight=args.coarsen_reg_weight,
            num_refine_layers=1, num_heads=args.gen_num_heads,
        )
    elif args.generator == "gnn_pooling":
        return GNNPoolingGenerator(
            num_proxies=args.num_proxies, input_dim=args.hidden_dim,
            gnn_layers=args.gnn_layers, gnn_type=args.gnn_type,
            pool_types=tuple(args.pool_types), decode_hidden=args.decode_hidden,
            decode_layers=args.decode_layers, idx_emb_dim=args.idx_emb_dim,
            dropout=args.gen_dropout, decode_mode=args.decode_mode,
        )
    else:
        raise ValueError(f"Unknown generator: {args.generator}")


def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)


def _uses_flat_interface(generator_name):
    return generator_name in ("graph_coarsening", "gnn_pooling")


# ================================================================
# PROXY GENERATION (same logic as main_indist._generate_proxies)
# ================================================================

def _generate_proxies(model, generator, batch, dense_x, dense_mask, args,
                      gred_h=None, num_proxy_sets=1):
    """Generate proxy embeddings for a batch. Handles flat vs dense interface."""
    gen_input = gred_h if gred_h is not None else dense_x
    gen_mask = dense_mask
    num_proxy_sets = max(int(num_proxy_sets), 1)

    proxy_list = []
    for _ in range(num_proxy_sets):
        if _uses_flat_interface(args.generator):
            flat_emb = gen_input[gen_mask]
            proxies, _ = generator(
                flat_emb, mask=None,
                edge_index=batch.edge_index, batch_vec=batch.batch,
                edge_attr=getattr(batch, "edge_attr", None),
            )
        else:
            proxies, _ = generator(gen_input, gen_mask)
        proxy_list.append(proxies)

    proxies = proxy_list[0] if len(proxy_list) == 1 else torch.cat(proxy_list, dim=1)
    return proxies


# ================================================================
# DIVERSITY SELECTION  (farthest-point sampling)
# ================================================================

def select_most_diverse(proxies, target_count):
    """
    Given an oversampled set of proxies, select the `target_count` most
    diverse ones per graph using greedy farthest-point sampling (FPS).

    Args:
        proxies: (B, C, d)  where C = oversample_factor * target_count
        target_count: int   number of proxies to keep  (C >= target_count)

    Returns:
        selected: (B, target_count, d)
    """
    B, C, d = proxies.shape
    if C <= target_count:
        return proxies  # nothing to select

    device = proxies.device
    selected = torch.zeros(B, target_count, d, device=device)

    for b in range(B):
        pts = proxies[b]  # (C, d)

        # Pairwise distance matrix (squared) — only computed once
        # Using cdist is memory-efficient enough for typical C < 1000
        dists = torch.cdist(pts, pts, p=2)  # (C, C)

        # Start from the point closest to the centroid (most "central")
        centroid = pts.mean(dim=0, keepdim=True)  # (1, d)
        first_idx = torch.cdist(centroid, pts).squeeze(0).argmax().item()

        chosen = [first_idx]
        # min_dist_to_chosen[i] = min distance from point i to any chosen point
        min_dist = dists[first_idx].clone()  # (C,)

        for _ in range(target_count - 1):
            # Pick the candidate farthest from the current chosen set
            # Mask already-chosen points so they aren't re-selected
            min_dist[chosen[-1]] = -1.0
            next_idx = min_dist.argmax().item()
            chosen.append(next_idx)
            # Update min distances
            min_dist = torch.min(min_dist, dists[next_idx])

        chosen_idx = torch.tensor(chosen, device=device)
        selected[b] = pts[chosen_idx]

    return selected


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate_with_proxy_multiplier(model, generator, loader, device, args,
                                   proxy_multiplier=0, oversample_factor=1):
    """
    Evaluate model with `proxy_multiplier` sets of generated proxies.
    If proxy_multiplier == 0, evaluates without any proxies (baseline).

    When oversample_factor > 1:
      - Generates oversample_factor * proxy_multiplier sets of M proxies
      - Selects the proxy_multiplier * M most diverse via FPS
      - Feeds those to the backbone
    """
    model.eval()
    if generator is not None:
        generator.eval()

    loss_fn = nn.BCEWithLogitsLoss()
    all_preds, all_labels, losses = [], [], []
    node_counts = []
    is_gred = args.backbone in ("gred", "hybrid")
    oversample_factor = max(int(oversample_factor), 1)

    for batch_data in loader:
        if is_gred:
            batch, dist_masks_batch, node_masks_batch = batch_data
            batch = batch.to(device)
            dist_masks_batch = dist_masks_batch.to(device)
            node_masks_batch = node_masks_batch.to(device)
        else:
            batch = batch_data.to(device)
            dist_masks_batch = None
            node_masks_batch = None

        dense_x, dense_mask = model.encode_dense(batch)

        # Track node counts for computing avg_N
        node_counts.append(dense_mask.sum(dim=1).float().cpu())

        if proxy_multiplier > 0 and generator is not None:
            gred_h = None
            if args.backbone == "hybrid":
                gred_h = model.encode_gred(dense_x, dist_masks_batch,
                                           node_masks_batch)

            # Total generator calls = oversample_factor * proxy_multiplier
            total_sets = oversample_factor * proxy_multiplier
            proxies = _generate_proxies(
                model, generator, batch, dense_x, dense_mask, args,
                gred_h=gred_h, num_proxy_sets=total_sets)

            # If oversampling, select the target count of most diverse proxies
            if oversample_factor > 1:
                target_count = proxy_multiplier * args.num_proxies
                proxies = select_most_diverse(proxies, target_count)

            if args.backbone == "vanilla_gt":
                logits, _ = model(batch, proxy_embeddings=proxies,
                                  precomputed_dense=(dense_x, dense_mask))
            elif args.backbone == "hybrid":
                logits, _ = model(batch, dist_masks_batch, node_masks_batch,
                                  proxy_embeddings=proxies,
                                  precomputed_dense=(dense_x, dense_mask),
                                  precomputed_gred=gred_h)
            elif args.backbone == "gred":
                logits, _ = model(batch, dist_masks_batch, node_masks_batch)
        else:
            # No proxies
            if args.backbone == "vanilla_gt":
                logits, _ = model(batch)
            else:
                logits, _ = model(batch, dist_masks_batch, node_masks_batch)

        losses.append(loss_fn(logits, batch.y).item())
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    ap = compute_macro_ap(
        np.concatenate(all_preds), np.concatenate(all_labels))
    avg_loss = float(np.mean(losses))
    avg_n = float(torch.cat(node_counts).mean())
    return ap, avg_loss, avg_n


# ================================================================
# CHECKPOINT LOADING
# ================================================================

def load_models(args):
    """Load model and (optionally) generator from checkpoints."""
    model = build_model(args).to(args.device)
    generator = None

    if args.eval_model == "phase5":
        ckpt_path = os.path.join(args.ckpt_dir, "phase5_best.pt")
        assert os.path.exists(ckpt_path), f"Phase 5 checkpoint not found: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        generator = build_generator(args).to(args.device)
        generator.load_state_dict(ckpt["generator_state"])
        print(f"Loaded Phase 5 model+generator from {ckpt_path}", flush=True)
        print(f"  (trained epoch {ckpt.get('epoch', '?')}, "
              f"val_ap={ckpt.get('val_ap', '?'):.4f}, "
              f"test_ap={ckpt.get('test_ap', '?'):.4f})", flush=True)

    elif args.eval_model == "phase1+gen":
        # Phase 1 model
        p1_path = os.path.join(args.shared_dir, "phase1_best.pt")
        assert os.path.exists(p1_path), f"Phase 1 checkpoint not found: {p1_path}"
        ckpt = torch.load(p1_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded Phase 1 model from {p1_path}", flush=True)

        # Phase 3 generator
        gen_path = os.path.join(args.ckpt_dir, "phase3_generator.pt")
        assert os.path.exists(gen_path), f"Phase 3 generator not found: {gen_path}"
        gen_ckpt = torch.load(gen_path, map_location=args.device, weights_only=False)
        generator = build_generator(args).to(args.device)
        generator.load_state_dict(gen_ckpt["generator_state"])
        print(f"Loaded Phase 3 generator from {gen_path}", flush=True)

    elif args.eval_model == "phase1":
        p1_path = os.path.join(args.shared_dir, "phase1_best.pt")
        assert os.path.exists(p1_path), f"Phase 1 checkpoint not found: {p1_path}"
        ckpt = torch.load(p1_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded Phase 1 model from {p1_path}", flush=True)
        print("  (baseline only — no generator)", flush=True)

    _freeze(model)
    model.eval()
    if generator is not None:
        _freeze(generator)
        generator.eval()

    return model, generator


# ================================================================
# MAIN ABLATION LOOP
# ================================================================

def run_ablation(args):
    is_gred = args.backbone in ("gred", "hybrid")
    oversample = args.oversample_factor

    _, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    model, generator = load_models(args)

    # --- Determine max_k ---
    # Quick pass over val set to get avg_N
    print("\nEstimating average node count from val set...", flush=True)
    _, _, avg_n = evaluate_with_proxy_multiplier(
        model, generator, val_loader, args.device, args, proxy_multiplier=0)
    print(f"  avg_N ≈ {avg_n:.1f} nodes per graph", flush=True)

    if args.max_k is not None:
        max_k = args.max_k
    else:
        max_k = int(np.ceil(avg_n / args.num_proxies))
    print(f"  num_proxies (M) = {args.num_proxies}", flush=True)
    print(f"  max_k = {max_k}  (max total proxies = {max_k * args.num_proxies})",
          flush=True)
    if oversample > 1:
        print(f"  oversample_factor = {oversample}×  "
              f"(generate {oversample}×target, keep most diverse)", flush=True)

    has_generator = generator is not None and args.eval_model != "phase1"

    # --- Results collection ---
    results = []

    # Header
    os_col = "  generated" if oversample > 1 else ""
    print("\n" + "=" * (90 + (12 if oversample > 1 else 0)), flush=True)
    print(f"{'k':>4s} | {'kept_proxies':>14s} |{os_col + ' |' if os_col else ''}"
          f" {'val_AP':>10s} | {'val_loss':>10s} | "
          f"{'test_AP':>10s} | {'test_loss':>10s} | "
          f"{'Δval_AP':>10s} | {'time':>8s}", flush=True)
    print("-" * (90 + (12 if oversample > 1 else 0)), flush=True)

    best_val_ap = -1.0
    plateau_count = 0

    for k in range(0, max_k + 1):
        target_proxies = k * args.num_proxies
        generated_proxies = target_proxies * oversample

        if k > 0 and not has_generator:
            print(f"\nSkipping k={k}+ (no generator loaded, baseline only).", flush=True)
            break

        t0 = time.time()

        if args.num_runs > 1 and k > 0:
            # Multiple runs for stochastic generators
            val_aps, val_losses, test_aps, test_losses = [], [], [], []
            for run_i in range(args.num_runs):
                va, vl, _ = evaluate_with_proxy_multiplier(
                    model, generator, val_loader, args.device, args,
                    proxy_multiplier=k, oversample_factor=oversample)
                ta, tl, _ = evaluate_with_proxy_multiplier(
                    model, generator, test_loader, args.device, args,
                    proxy_multiplier=k, oversample_factor=oversample)
                val_aps.append(va)
                val_losses.append(vl)
                test_aps.append(ta)
                test_losses.append(tl)

            val_ap = float(np.mean(val_aps))
            val_loss = float(np.mean(val_losses))
            test_ap = float(np.mean(test_aps))
            test_loss = float(np.mean(test_losses))
            val_std = float(np.std(val_aps))
            test_std = float(np.std(test_aps))
        else:
            val_ap, val_loss, _ = evaluate_with_proxy_multiplier(
                model, generator, val_loader, args.device, args,
                proxy_multiplier=k, oversample_factor=oversample)
            test_ap, test_loss, _ = evaluate_with_proxy_multiplier(
                model, generator, test_loader, args.device, args,
                proxy_multiplier=k, oversample_factor=oversample)
            val_std, test_std = 0.0, 0.0

        elapsed = time.time() - t0

        # Delta from baseline (k=0)
        if k == 0:
            baseline_val_ap = val_ap
            baseline_test_ap = test_ap
        delta_val = val_ap - baseline_val_ap

        # Format output
        val_str = f"{val_ap:.4f}" + (f"±{val_std:.4f}" if val_std > 0 else "")
        test_str = f"{test_ap:.4f}" + (f"±{test_std:.4f}" if test_std > 0 else "")
        os_str = f" {generated_proxies:>10d} |" if oversample > 1 else ""
        print(f"{k:4d} | {target_proxies:14d} |{os_str}"
              f" {val_str:>10s} | {val_loss:10.4f} | "
              f"{test_str:>10s} | {test_loss:10.4f} | "
              f"{delta_val:+10.4f} | {elapsed:7.1f}s", flush=True)

        row = {
            "k": k,
            "kept_proxies": target_proxies,
            "generated_proxies": generated_proxies,
            "oversample_factor": oversample,
            "val_ap": val_ap,
            "val_loss": val_loss,
            "test_ap": test_ap,
            "test_loss": test_loss,
            "delta_val_ap": delta_val,
            "delta_test_ap": test_ap - baseline_test_ap,
            "elapsed_s": elapsed,
        }
        if args.num_runs > 1 and k > 0:
            row["val_ap_std"] = val_std
            row["test_ap_std"] = test_std
            row["val_aps"] = val_aps
            row["test_aps"] = test_aps

        results.append(row)

        # --- Plateau detection ---
        if k > 0:
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                plateau_count = 0
            else:
                plateau_count += 1
                if plateau_count >= args.plateau_patience:
                    print(f"\nPlateau detected: val AP did not improve for "
                          f"{args.plateau_patience} consecutive k steps. Stopping.",
                          flush=True)
                    break
        else:
            best_val_ap = val_ap

    # --- Summary ---
    sep_len = 90 + (12 if oversample > 1 else 0)
    print("\n" + "=" * sep_len, flush=True)
    print("ABLATION SUMMARY", flush=True)
    print(f"  Backbone:  {args.backbone}", flush=True)
    print(f"  Generator: {args.generator}", flush=True)
    print(f"  Eval mode: {args.eval_model}", flush=True)
    print(f"  M (num_proxies): {args.num_proxies}", flush=True)
    print(f"  avg_N: {avg_n:.1f}", flush=True)
    if oversample > 1:
        print(f"  oversample_factor: {oversample}×  "
              f"(generate {oversample}× target, FPS-select most diverse)", flush=True)

    if len(results) > 1:
        best_row = max(results[1:], key=lambda r: r["val_ap"])
        print(f"\n  Best proxy config:  k={best_row['k']} "
              f"({best_row['kept_proxies']} proxies"
              f"{f' selected from {best_row["generated_proxies"]}' if oversample > 1 else ''})",
              flush=True)
        print(f"    val  AP: {best_row['val_ap']:.4f} "
              f"(Δ={best_row['delta_val_ap']:+.4f})", flush=True)
        print(f"    test AP: {best_row['test_ap']:.4f} "
              f"(Δ={best_row['delta_test_ap']:+.4f})", flush=True)

    baseline = results[0]
    print(f"\n  Baseline (k=0, no proxies):", flush=True)
    print(f"    val  AP: {baseline['val_ap']:.4f}", flush=True)
    print(f"    test AP: {baseline['test_ap']:.4f}", flush=True)

    # --- Save results ---
    if args.results_file is not None:
        out_path = args.results_file
    elif args.ckpt_dir is not None:
        out_path = os.path.join(args.ckpt_dir, "ablation_results.json")
    else:
        out_path = os.path.join(args.shared_dir, "ablation_results.json")

    summary = {
        "config": {
            "backbone": args.backbone,
            "generator": args.generator,
            "eval_model": args.eval_model,
            "num_proxies": args.num_proxies,
            "oversample_factor": oversample,
            "avg_N": avg_n,
            "max_k_evaluated": results[-1]["k"],
            "plateau_patience": args.plateau_patience,
            "num_runs": args.num_runs,
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    return summary


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    args = parse_args()
    run_ablation(args)
