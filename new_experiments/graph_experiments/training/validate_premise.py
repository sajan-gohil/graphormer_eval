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
from torch_geometric.loader import DataLoader


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

            graph_results.append({
                'graph_idx': idx,
                'num_nodes': data.num_nodes,
                'num_edges': data.num_edges,
                'baseline_loss': result['baseline_loss'],
                'optimized_loss': result['optimized_loss'],
                'baseline_ap': result['baseline_ap'],
                'optimized_ap': result['optimized_ap'],
                'ap_improvement': result['ap_improvement'],
                'init_mmd': result['init_mmd'],
                'final_mmd': result['final_mmd'],
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
                mean_imp = np.mean([r['ap_improvement'] for r in graph_results])
                print(f"  [{i+1}/{len(selected_indices)}] "
                      f"Mean AP improvement: {mean_imp:.4f} | "
                      f"Time: {elapsed:.1f}s")

        # Final frozen verification
        verify_frozen(param_snapshot, model)
        elapsed_total = time.time() - start_time

        # --- Aggregate results ---
        baseline_aps = [r['baseline_ap'] for r in graph_results]
        optimized_aps = [r['optimized_ap'] for r in graph_results]
        improvements = [r['ap_improvement'] for r in graph_results]

        summary = {
            'M': M,
            'num_graphs': len(graph_results),
            'mean_baseline_ap': float(np.mean(baseline_aps)),
            'mean_optimized_ap': float(np.mean(optimized_aps)),
            'mean_ap_improvement': float(np.mean(improvements)),
            'std_ap_improvement': float(np.std(improvements)),
            'median_ap_improvement': float(np.median(improvements)),
            'fraction_improved': float(np.mean([1 if imp > 0 else 0 for imp in improvements])),
            'fraction_improved_5pt': float(np.mean([1 if imp >= 0.05 else 0 for imp in improvements])),
            'max_improvement': float(np.max(improvements)),
            'min_improvement': float(np.min(improvements)),
            'mean_init_mmd': float(np.mean([r['init_mmd'] for r in graph_results])),
            'mean_final_mmd': float(np.mean([r['final_mmd'] for r in graph_results])),
            'total_time_seconds': elapsed_total,
            # Histogram bins for improvement distribution
            'improvement_histogram': {
                'bins': np.histogram(improvements, bins=20)[1].tolist(),
                'counts': np.histogram(improvements, bins=20)[0].tolist(),
            },
        }

        print(f"\n--- M={M} Summary ---")
        print(f"  Mean Baseline AP:     {summary['mean_baseline_ap']:.4f}")
        print(f"  Mean Optimized AP:    {summary['mean_optimized_ap']:.4f}")
        print(f"  Mean AP Improvement:  {summary['mean_ap_improvement']:.4f} "
              f"± {summary['std_ap_improvement']:.4f}")
        print(f"  Fraction improved:    {summary['fraction_improved']:.2%}")
        print(f"  Fraction ≥5pt gain:   {summary['fraction_improved_5pt']:.2%}")
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

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
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
    print("\n" + "=" * 70)
    print("PHASE 2 VALIDATION GATE — SUMMARY")
    print("=" * 70)
    print(f"{'M':<6} {'Baseline AP':<14} {'Optimized AP':<14} "
          f"{'Improvement':<14} {'% Improved':<12}")
    print("-" * 60)
    for key, data in results.items():
        s = data['summary']
        print(f"{s['M']:<6} {s['mean_baseline_ap']:<14.4f} "
              f"{s['mean_optimized_ap']:<14.4f} "
              f"{s['mean_ap_improvement']:<14.4f} "
              f"{s['fraction_improved']:<12.1%}")
    print("=" * 70)

    # Gate check
    best_M_key = max(results.keys(), key=lambda k: results[k]['summary']['mean_ap_improvement'])
    best_imp = results[best_M_key]['summary']['mean_ap_improvement']

    if best_imp >= 0.05:
        print(f"\n✓ GATE PASSED: Best improvement = {best_imp:.4f} (≥0.05) at {best_M_key}")
        print("  → Proceed to Phase 3 (simple proxy baselines)")
    elif best_imp >= 0.02:
        print(f"\n⚠ MARGINAL: Best improvement = {best_imp:.4f} (≥0.02 but <0.05)")
        print("  → Consider investigating further before proceeding")
    else:
        print(f"\n✗ GATE FAILED: Best improvement = {best_imp:.4f} (<0.02)")
        print("  → Proxy embeddings don't meaningfully help. Consider stopping.")


if __name__ == "__main__":
    main()
