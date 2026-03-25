"""
Phase 2: Validation Gate — Validate that optimized proxy embeddings can
improve a frozen GPS transformer by ≥5 AP points on training data.

Steps:
  1. Load best pretrained model from Phase 1. Freeze all parameters.
  2. Select a stratified subset of training graphs (500-1000).
  3. For each graph (processed as single-graph batches):
       - Run ProxyOptimizer with multiple restarts.
       - Record baseline vs optimized AP.
  4. Aggregate: mean AP improvement, distribution, per-class breakdown.
  5. Repeat for M ∈ {2, 4, 8, 16}.
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.phase1_config import Phase1Config
from data.peptides_func import get_peptides_func_loaders
from models.transformer import GPSModel
from models.proxy_optimizer import ProxyOptimizer
from evaluation.metrics import compute_macro_ap, compute_per_class_ap, compute_correct_classes
# compute_macro_ap is used for aggregated AP over the full subset (not per-graph)
from torch_geometric.loader import DataLoader
from utils.device import get_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_frozen_model(checkpoint_path, config, device):
    """Load a pretrained GPSModel and freeze all parameters."""
    model = GPSModel(config.model).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Freeze everything
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model


def get_param_snapshot(model):
    """Take a snapshot of model parameters for verification."""
    snapshot = {}
    for name, param in model.named_parameters():
        snapshot[name] = param.data.clone()
    return snapshot


def verify_frozen(snapshot_before, model):
    """Verify that no model parameters changed during proxy optimization."""
    for name, param in model.named_parameters():
        if not torch.equal(snapshot_before[name], param.data):
            raise RuntimeError(
                f"FROZEN MODEL VIOLATION: Parameter '{name}' changed during proxy optimization!"
            )
    return True


def stratified_sample(dataset, num_samples, num_classes=10, seed=42):
    """
    Select a stratified subset of graphs based on class distribution.

    Since Peptides-func is multi-label, we stratify by the most frequent class
    per sample to get approximately balanced representation.

    Args:
        dataset: List of PyG Data objects.
        num_samples: Target number of samples.
        num_classes: Number of classes.
        seed: Random seed.

    Returns:
        List of indices into the dataset.
    """
    rng = np.random.RandomState(seed)

    # Assign each graph to its "primary" class (the one with highest label, or first 1)
    class_to_indices = defaultdict(list)
    for i, data in enumerate(dataset):
        labels = data.y.numpy() if hasattr(data.y, 'numpy') else data.y
        if labels.ndim > 1:
            labels = labels.squeeze()
        active_classes = np.where(labels > 0.5)[0]
        if len(active_classes) > 0:
            primary = rng.choice(active_classes)
        else:
            primary = 0
        class_to_indices[primary].append(i)

    # Sample proportionally from each class
    selected = []
    samples_per_class = max(1, num_samples // num_classes)

    for cls in range(num_classes):
        indices = class_to_indices.get(cls, [])
        if len(indices) == 0:
            continue
        n_take = min(samples_per_class, len(indices))
        chosen = rng.choice(indices, size=n_take, replace=False).tolist()
        selected.extend(chosen)

    # Fill remaining if needed
    remaining = num_samples - len(selected)
    if remaining > 0:
        all_indices = list(range(len(dataset)))
        available = [i for i in all_indices if i not in set(selected)]
        extra = rng.choice(available, size=min(remaining, len(available)), replace=False).tolist()
        selected.extend(extra)

    rng.shuffle(selected)
    return selected[:num_samples]


def run_validation(config, checkpoint_path, device, M_values=(2, 4, 8, 16),
                   num_samples=500, num_diagnostic=50,
                   proxy_lr=1e-2, proxy_iterations=500, mmd_lambda=0.05,
                   num_restarts=5, gradient_clip=1.0, seed=42):
    """
    Run the full Phase 2 validation gate.

    Args:
        config: Phase1Config instance.
        checkpoint_path: Path to the best Phase 1 model checkpoint.
        device: torch device.
        M_values: Tuple of proxy counts to sweep.
        num_samples: Number of training graphs to evaluate.
        num_diagnostic: Number of graphs for detailed diagnostic logging.
        proxy_lr: Learning rate for proxy optimization.
        proxy_iterations: Number of optimization steps.
        mmd_lambda: MMD regularization weight.
        num_restarts: Random restarts per graph.
        gradient_clip: Gradient clipping norm.
        seed: Random seed.

    Returns:
        results: Dict with all results.
    """
    set_seed(seed)

    # Load data
    train_loader, val_loader, test_loader, dataset_info = get_peptides_func_loaders(config.data)

    # Get the raw processed dataset (list of Data objects) from the loader
    train_data = train_loader.dataset

    # Load frozen model
    print(f"Loading frozen model from {checkpoint_path}")
    model = load_frozen_model(checkpoint_path, config, device)
    param_snapshot = get_param_snapshot(model)

    # Stratified sample
    selected_indices = stratified_sample(train_data, num_samples, seed=seed)
    print(f"Selected {len(selected_indices)} training graphs (stratified)")

    # Mark which indices get diagnostic logging
    diagnostic_indices = set(selected_indices[:num_diagnostic])

    all_results = {}

    for M in M_values:
        print(f"\n{'='*60}")
        print(f"Running with M={M} proxies")
        print(f"{'='*60}")

        optimizer = ProxyOptimizer(
            model, M=M, lr=proxy_lr, num_iterations=proxy_iterations,
            mmd_lambda=mmd_lambda, gradient_clip=gradient_clip,
            num_restarts=num_restarts,
        )

        graph_results = []
        diagnostics = []
        # Collect raw predictions for aggregated AP computation
        all_baseline_probs = []
        all_optimized_probs = []
        all_labels = []
        start_time = time.time()

        for i, idx in enumerate(selected_indices):
            data = train_data[idx]
            # Create a single-graph batch using DataLoader with batch_size=1
            single_loader = DataLoader([data], batch_size=1, shuffle=False)
            batch = next(iter(single_loader))

            do_diagnostic = idx in diagnostic_indices

            result = optimizer.optimize(
                batch, device,
                log_trajectory=do_diagnostic,
                log_every=50,
            )

            # Collect raw predictions for aggregated AP
            all_baseline_probs.append(result['baseline_probs'])
            all_optimized_probs.append(result['optimized_probs'])
            all_labels.append(result['labels'])

            corrections = result['class_corrections']

            graph_results.append({
                'graph_idx': idx,
                'num_nodes': data.num_nodes,
                'num_edges': data.num_edges,
                'baseline_loss': result['baseline_loss'],
                'optimized_loss': result['optimized_loss'],
                'loss_improvement': result['loss_improvement'],
                'init_mmd': result['init_mmd'],
                'final_mmd': result['final_mmd'],
                'newly_correct': corrections['newly_correct'],
                'newly_wrong': corrections['newly_wrong'],
                'net_corrections': corrections['net_corrections'],
                'baseline_correct': corrections['baseline_correct'],
                'optimized_correct': corrections['optimized_correct'],
                'total_classes': corrections['total_classes'],
            })

            if do_diagnostic:
                diag = {
                    'graph_idx': idx,
                    'trajectory': result['trajectory'],
                    'proxy_norms': result['proxy_norms'].numpy().tolist(),
                    'proxy_cosine_sims': result['proxy_cosine_sims'].numpy().tolist(),
                    'init_mmd': result['init_mmd'],
                    'final_mmd': result['final_mmd'],
                }
                # Store attention pattern summary (not full tensor — too large)
                if result['last_attn_weights'] is not None:
                    attn = result['last_attn_weights']  # (1, H, N+M, N+M)
                    # Average over heads: (1, N+M, N+M)
                    avg_attn = attn.mean(dim=1).squeeze(0).numpy()
                    N = data.num_nodes
                    # Extract node→proxy attention: how much each node attends to proxies
                    # In the dense representation, proxies are at positions N_padded..N_padded+M
                    # But after to_dense_batch with a single graph, N_padded = N (no padding)
                    node_to_proxy = avg_attn[:N, N:N+M].tolist()  # (N, M)
                    proxy_to_node = avg_attn[N:N+M, :N].tolist()  # (M, N)
                    diag['node_to_proxy_attn'] = node_to_proxy
                    diag['proxy_to_node_attn'] = proxy_to_node

                diagnostics.append(diag)

            # Verify frozen
            if (i + 1) % 100 == 0:
                verify_frozen(param_snapshot, model)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                # Report meaningful per-graph metrics: loss improvement + class corrections
                mean_loss_imp = np.mean([r['loss_improvement'] for r in graph_results])
                mean_net_corr = np.mean([r['net_corrections'] for r in graph_results])
                # Also compute running aggregated AP
                running_bl_probs = np.concatenate(all_baseline_probs, axis=0)
                running_opt_probs = np.concatenate(all_optimized_probs, axis=0)
                running_labels = np.concatenate(all_labels, axis=0)
                running_bl_ap = compute_macro_ap(running_bl_probs, running_labels)
                running_opt_ap = compute_macro_ap(running_opt_probs, running_labels)
                print(f"  [{i+1}/{len(selected_indices)}] "
                      f"Loss imp: {mean_loss_imp:.4f} | "
                      f"Mean net corrections: {mean_net_corr:.2f} | "
                      f"Aggregated AP: {running_bl_ap:.4f} → {running_opt_ap:.4f} | "
                      f"Time: {elapsed:.1f}s")

        # Final frozen verification
        verify_frozen(param_snapshot, model)
        elapsed_total = time.time() - start_time

        # --- Aggregate predictions and compute AP over the full subset ---
        agg_baseline_probs = np.concatenate(all_baseline_probs, axis=0)
        agg_optimized_probs = np.concatenate(all_optimized_probs, axis=0)
        agg_labels = np.concatenate(all_labels, axis=0)

        aggregated_baseline_ap = compute_macro_ap(agg_baseline_probs, agg_labels)
        aggregated_optimized_ap = compute_macro_ap(agg_optimized_probs, agg_labels)
        aggregated_ap_improvement = aggregated_optimized_ap - aggregated_baseline_ap

        # Per-class AP breakdown (aggregated)
        from evaluation.metrics import compute_per_class_ap
        baseline_per_class = compute_per_class_ap(agg_baseline_probs, agg_labels)
        optimized_per_class = compute_per_class_ap(agg_optimized_probs, agg_labels)

        # Per-graph correction stats
        loss_improvements = [r['loss_improvement'] for r in graph_results]
        net_corrections = [r['net_corrections'] for r in graph_results]

        summary = {
            'M': M,
            'num_graphs': len(graph_results),
            # Aggregated AP (computed over full subset — the meaningful metric)
            'aggregated_baseline_ap': float(aggregated_baseline_ap),
            'aggregated_optimized_ap': float(aggregated_optimized_ap),
            'aggregated_ap_improvement': float(aggregated_ap_improvement),
            'baseline_per_class_ap': baseline_per_class.tolist(),
            'optimized_per_class_ap': optimized_per_class.tolist(),
            # Loss improvement stats (per-graph, meaningful for single samples)
            'mean_loss_improvement': float(np.mean(loss_improvements)),
            'std_loss_improvement': float(np.std(loss_improvements)),
            'fraction_loss_improved': float(np.mean([1 if imp > 0 else 0 for imp in loss_improvements])),
            # Class correction stats (per-graph, meaningful for single samples)
            'mean_net_corrections': float(np.mean(net_corrections)),
            'total_newly_correct': int(sum(r['newly_correct'] for r in graph_results)),
            'total_newly_wrong': int(sum(r['newly_wrong'] for r in graph_results)),
            'total_net_corrections': int(sum(r['net_corrections'] for r in graph_results)),
            'fraction_any_correction': float(np.mean([1 if r['net_corrections'] > 0 else 0 for r in graph_results])),
            # MMD stats
            'mean_init_mmd': float(np.mean([r['init_mmd'] for r in graph_results])),
            'mean_final_mmd': float(np.mean([r['final_mmd'] for r in graph_results])),
            'total_time_seconds': elapsed_total,
            # Histogram of loss improvements
            'loss_improvement_histogram': {
                'bins': np.histogram(loss_improvements, bins=20)[1].tolist(),
                'counts': np.histogram(loss_improvements, bins=20)[0].tolist(),
            },
        }

        print(f"\n--- M={M} Summary ---")
        print(f"  Aggregated Baseline AP:     {summary['aggregated_baseline_ap']:.4f}")
        print(f"  Aggregated Optimized AP:    {summary['aggregated_optimized_ap']:.4f}")
        print(f"  Aggregated AP Improvement:  {summary['aggregated_ap_improvement']:.4f}")
        print(f"  Mean loss improvement:      {summary['mean_loss_improvement']:.4f} "
              f"± {summary['std_loss_improvement']:.4f}")
        print(f"  Fraction loss improved:     {summary['fraction_loss_improved']:.2%}")
        print(f"  Total class corrections:    +{summary['total_newly_correct']} "
              f"-{summary['total_newly_wrong']} "
              f"(net {summary['total_net_corrections']:+d})")
        print(f"  Fraction with net fix:      {summary['fraction_any_correction']:.2%}")
        print(f"  Time: {elapsed_total:.1f}s")

        all_results[f'M={M}'] = {
            'summary': summary,
            'per_graph': graph_results,
            'diagnostics': diagnostics,
        }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Validation Gate")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt",
                        help="Path to Phase 1 best model checkpoint")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of training graphs to evaluate")
    parser.add_argument("--num_diagnostic", type=int, default=50,
                        help="Number of graphs for detailed diagnostics")
    parser.add_argument("--proxy_lr", type=float, default=1e-2)
    parser.add_argument("--proxy_iterations", type=int, default=500)
    parser.add_argument("--mmd_lambda", type=float, default=0.05)
    parser.add_argument("--num_restarts", type=int, default=5)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--M_values", type=int, nargs='+', default=[2, 4, 8, 16],
                        help="Proxy counts to sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: logs/phase2_<timestamp>.json)")
    args = parser.parse_args()

    config = Phase1Config()

    device = get_device(args.device)
    print(f"Using device: {device}")

    results = run_validation(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        M_values=tuple(args.M_values),
        num_samples=args.num_samples,
        num_diagnostic=args.num_diagnostic,
        proxy_lr=args.proxy_lr,
        proxy_iterations=args.proxy_iterations,
        mmd_lambda=args.mmd_lambda,
        num_restarts=args.num_restarts,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
    )

    # Save results
    output_path = args.output or os.path.join(
        "./logs", f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # --- Final Decision Table ---
    print("\n" + "=" * 85)
    print("PHASE 2 VALIDATION GATE — SUMMARY (Aggregated AP over full subset)")
    print("=" * 85)
    print(f"{'M':<6} {'Baseline AP':<14} {'Optimized AP':<14} "
          f"{'AP Improve':<12} {'Loss Imp':<12} {'Net Corrections':<16} {'% Loss Imp':<10}")
    print("-" * 85)
    for key, data in results.items():
        s = data['summary']
        print(f"{s['M']:<6} {s['aggregated_baseline_ap']:<14.4f} "
              f"{s['aggregated_optimized_ap']:<14.4f} "
              f"{s['aggregated_ap_improvement']:<12.4f} "
              f"{s['mean_loss_improvement']:<12.4f} "
              f"{s['total_net_corrections']:<+16d} "
              f"{s['fraction_loss_improved']:<10.1%}")
    print("=" * 85)

    # Gate check — now based on aggregated AP improvement
    best_M_key = max(results.keys(), key=lambda k: results[k]['summary']['aggregated_ap_improvement'])
    best_imp = results[best_M_key]['summary']['aggregated_ap_improvement']

    if best_imp >= 0.05:
        print(f"\n✓ GATE PASSED: Best aggregated AP improvement = {best_imp:.4f} (≥0.05) at {best_M_key}")
        print("  → Proceed to Phase 3 (simple proxy baselines)")
    elif best_imp >= 0.02:
        print(f"\n⚠ MARGINAL: Best aggregated AP improvement = {best_imp:.4f} (≥0.02 but <0.05)")
        print("  → Consider investigating further before proceeding")
    else:
        print(f"\n✗ GATE FAILED: Best aggregated AP improvement = {best_imp:.4f} (<0.02)")
        print("  → Proxy embeddings don't meaningfully help. Consider stopping.")


if __name__ == "__main__":
    main()
