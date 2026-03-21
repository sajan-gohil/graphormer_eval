"""
Phase 4: Ablation runner.

Runs three ablation sweeps on proxy optimization:
  4.1 Insertion layer:    insertion_point ∈ {0, 1, 2, "all"}
  4.2 Attention routing:  routing_mode ∈ {"full", "routed", "hybrid"}
  4.3 MMD sensitivity:    λ_MMD ∈ {0, 0.01, 0.05, 0.1, 0.5, 1.0}

Each ablation uses the Phase 1 frozen model and optimizes proxy embeddings
on a subset of training graphs, reporting AP improvement and diagnostics.

Usage:
    python training/run_ablations.py --checkpoint ./checkpoints/best_model.pt
    python training/run_ablations.py --ablation insertion   # run only one
    python training/run_ablations.py --ablation routing
    python training/run_ablations.py --ablation mmd
"""

import os
import sys
import json
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.phase1_config import Phase1Config
from data.peptides_func import get_peptides_func_loaders
from models.transformer import GPSModel
from models.ablation_forward import forward_with_proxies_ablation
from utils.mmd import mmd_squared
from evaluation.metrics import compute_macro_ap
from training.validate_premise import (
    load_frozen_model, get_param_snapshot, verify_frozen,
    stratified_sample, set_seed,
)
from torch_geometric.loader import DataLoader


class AblationProxyOptimizer:
    """
    Proxy optimizer with support for insertion_point and routing_mode ablations.
    Extends the Phase 2 ProxyOptimizer with ablation parameters.
    """

    def __init__(self, model, M=8, lr=1e-2, num_iterations=500,
                 mmd_lambda=0.05, gradient_clip=1.0, num_restarts=3,
                 insertion_point=0, routing_mode='full'):
        self.model = model
        self.M = M
        self.lr = lr
        self.num_iterations = num_iterations
        self.mmd_lambda = mmd_lambda
        self.gradient_clip = gradient_clip
        self.num_restarts = num_restarts
        self.d = model.hidden_dim
        self.insertion_point = insertion_point
        self.routing_mode = routing_mode

    def _init_proxies(self, batch, device):
        """Initialize proxy embeddings."""
        with torch.no_grad():
            h, batch_idx = self.model.get_initial_embeddings(batch)
            sigma = h.std().item()
            num_graphs = batch_idx.max().item() + 1

        if self.insertion_point == "all":
            num_layers = len(self.model.layers)
            B_param = torch.randn(num_layers, num_graphs, self.M, self.d, device=device) * sigma
        else:
            B_param = torch.randn(num_graphs, self.M, self.d, device=device) * sigma

        B_param = nn.Parameter(B_param)
        return B_param, h, batch_idx

    def optimize(self, batch, device):
        """Run proxy optimization and return results."""
        batch = batch.to(device)
        criterion = nn.BCEWithLogitsLoss()

        # Baseline
        self.model.eval()
        with torch.no_grad():
            baseline_logits = self.model(batch)
            baseline_loss = criterion(baseline_logits, batch.y.float()).item()
            baseline_probs = torch.sigmoid(baseline_logits).cpu().numpy()
            baseline_labels = batch.y.cpu().numpy()
        baseline_ap = compute_macro_ap(baseline_probs, baseline_labels)

        best_result = None
        best_loss = float('inf')

        for restart in range(self.num_restarts):
            B, h_init, batch_idx = self._init_proxies(batch, device)
            opt = torch.optim.Adam([B], lr=self.lr)

            with torch.no_grad():
                B_flat = B.reshape(-1, self.d) if self.insertion_point != "all" \
                    else B.reshape(-1, self.d)
                init_mmd = mmd_squared(B_flat, h_init.detach()).item()

            for step in range(self.num_iterations):
                opt.zero_grad()

                logits = forward_with_proxies_ablation(
                    self.model, batch, B,
                    insertion_point=self.insertion_point,
                    routing_mode=self.routing_mode
                )
                task_loss = criterion(logits, batch.y.float())

                if self.mmd_lambda > 0:
                    B_flat = B.reshape(-1, self.d)
                    mmd_loss = mmd_squared(B_flat, h_init.detach())
                    total_loss = task_loss + self.mmd_lambda * mmd_loss
                else:
                    mmd_loss = torch.tensor(0.0, device=device)
                    total_loss = task_loss

                total_loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_([B], self.gradient_clip)
                opt.step()

            # Final evaluation
            with torch.no_grad():
                logits = forward_with_proxies_ablation(
                    self.model, batch, B,
                    insertion_point=self.insertion_point,
                    routing_mode=self.routing_mode
                )
                final_loss = criterion(logits, batch.y.float()).item()
                final_probs = torch.sigmoid(logits).cpu().numpy()
                final_ap = compute_macro_ap(final_probs, baseline_labels)

                B_flat = B.reshape(-1, self.d)
                final_mmd = mmd_squared(B_flat, h_init.detach()).item()

                proxy_norms = B.detach().reshape(-1, self.M, self.d).norm(dim=-1).mean(dim=0)

            if final_loss < best_loss:
                best_loss = final_loss
                best_result = {
                    'baseline_ap': baseline_ap,
                    'optimized_ap': final_ap,
                    'ap_improvement': final_ap - baseline_ap,
                    'baseline_loss': baseline_loss,
                    'optimized_loss': final_loss,
                    'init_mmd': init_mmd,
                    'final_mmd': final_mmd,
                    'proxy_norms_mean': proxy_norms.mean().item(),
                    'proxy_norms_std': proxy_norms.std().item(),
                    'best_B': B.detach().cpu(),
                }

        return best_result


def run_ablation_sweep(model, train_data, selected_indices, device,
                       sweep_name, sweep_configs, base_config):
    """
    Run an ablation sweep over a set of configurations.

    Args:
        model: Frozen GPSModel.
        train_data: List of Data objects.
        selected_indices: Indices to evaluate on.
        device: torch device.
        sweep_name: Name of this sweep (for logging).
        sweep_configs: List of dicts, each overriding base_config keys.
        base_config: Dict of default optimizer kwargs.

    Returns:
        sweep_results: Dict mapping config label → aggregated results.
    """
    sweep_results = {}

    for cfg in sweep_configs:
        label = cfg.pop('label')
        optimizer_kwargs = {**base_config, **cfg}

        print(f"\n  [{sweep_name}] Config: {label}")
        print(f"    Params: {optimizer_kwargs}")

        opt = AblationProxyOptimizer(model, **optimizer_kwargs)

        graph_results = []
        start = time.time()

        for i, idx in enumerate(selected_indices):
            data = train_data[idx]
            single_loader = DataLoader([data], batch_size=1, shuffle=False)
            batch = next(iter(single_loader))

            result = opt.optimize(batch, device)
            graph_results.append(result)

            if (i + 1) % 25 == 0:
                mean_imp = np.mean([r['ap_improvement'] for r in graph_results])
                print(f"    [{i+1}/{len(selected_indices)}] Mean AP imp: {mean_imp:.4f}")

        elapsed = time.time() - start

        improvements = [r['ap_improvement'] for r in graph_results]
        summary = {
            'label': label,
            'config': {**base_config, **cfg, 'label': label},
            'mean_baseline_ap': float(np.mean([r['baseline_ap'] for r in graph_results])),
            'mean_optimized_ap': float(np.mean([r['optimized_ap'] for r in graph_results])),
            'mean_ap_improvement': float(np.mean(improvements)),
            'std_ap_improvement': float(np.std(improvements)),
            'fraction_improved': float(np.mean([1 if imp > 0 else 0 for imp in improvements])),
            'mean_final_mmd': float(np.mean([r['final_mmd'] for r in graph_results])),
            'mean_proxy_norms': float(np.mean([r['proxy_norms_mean'] for r in graph_results])),
            'time_seconds': elapsed,
        }

        print(f"    → Mean AP improvement: {summary['mean_ap_improvement']:.4f} "
              f"± {summary['std_ap_improvement']:.4f} "
              f"({summary['fraction_improved']:.0%} improved) "
              f"[{elapsed:.0f}s]")

        sweep_results[label] = summary

    return sweep_results


def run_phase4(config, checkpoint_path, device, num_samples=100, seed=42,
               ablations=None):
    """
    Run all Phase 4 ablations.

    Args:
        ablations: List of ablation names to run. None = run all.
                   Options: "insertion", "routing", "mmd"
    """
    set_seed(seed)

    if ablations is None:
        ablations = ["insertion", "routing", "mmd"]

    # Load data
    train_loader, val_loader, test_loader, dataset_info = get_peptides_func_loaders(config.data)
    train_data = train_loader.dataset

    # Load frozen model
    model = load_frozen_model(checkpoint_path, config, device)
    param_snapshot = get_param_snapshot(model)

    # Stratified sample
    selected_indices = stratified_sample(train_data, num_samples, seed=seed)
    print(f"Selected {len(selected_indices)} training graphs for ablation")

    # Base configuration (defaults from Phase 2)
    base_config = {
        'M': 8, 'lr': 1e-2, 'num_iterations': 300,
        'mmd_lambda': 0.05, 'gradient_clip': 1.0, 'num_restarts': 3,
        'insertion_point': 0, 'routing_mode': 'full',
    }

    all_results = {}

    # =================================================================
    # 4.1 Insertion layer ablation
    # =================================================================
    if "insertion" in ablations:
        print("\n" + "=" * 60)
        print("Ablation 4.1: Insertion Layer")
        print("=" * 60)

        sweep_configs = [
            {'label': 'insert_layer_0', 'insertion_point': 0},
            {'label': 'insert_layer_1', 'insertion_point': 1},
            {'label': 'insert_layer_2', 'insertion_point': 2},
            {'label': 'insert_all_layers', 'insertion_point': 'all'},
        ]

        results = run_ablation_sweep(
            model, train_data, selected_indices, device,
            "insertion", sweep_configs, base_config
        )
        all_results['insertion_layer'] = results
        verify_frozen(param_snapshot, model)

    # =================================================================
    # 4.2 Attention routing ablation
    # =================================================================
    if "routing" in ablations:
        print("\n" + "=" * 60)
        print("Ablation 4.2: Attention Routing")
        print("=" * 60)

        sweep_configs = [
            {'label': 'routing_full', 'routing_mode': 'full'},
            {'label': 'routing_routed', 'routing_mode': 'routed'},
            {'label': 'routing_hybrid', 'routing_mode': 'hybrid'},
        ]

        results = run_ablation_sweep(
            model, train_data, selected_indices, device,
            "routing", sweep_configs, base_config
        )
        all_results['attention_routing'] = results
        verify_frozen(param_snapshot, model)

    # =================================================================
    # 4.3 MMD sensitivity sweep
    # =================================================================
    if "mmd" in ablations:
        print("\n" + "=" * 60)
        print("Ablation 4.3: MMD Sensitivity")
        print("=" * 60)

        sweep_configs = [
            {'label': 'mmd_0.00', 'mmd_lambda': 0.0},
            {'label': 'mmd_0.01', 'mmd_lambda': 0.01},
            {'label': 'mmd_0.05', 'mmd_lambda': 0.05},
            {'label': 'mmd_0.10', 'mmd_lambda': 0.1},
            {'label': 'mmd_0.50', 'mmd_lambda': 0.5},
            {'label': 'mmd_1.00', 'mmd_lambda': 1.0},
        ]

        results = run_ablation_sweep(
            model, train_data, selected_indices, device,
            "mmd", sweep_configs, base_config
        )
        all_results['mmd_sensitivity'] = results
        verify_frozen(param_snapshot, model)

    return all_results


def print_summary_tables(all_results):
    """Print formatted summary tables for all ablations."""

    for ablation_name, results in all_results.items():
        print(f"\n{'='*70}")
        print(f"ABLATION: {ablation_name}")
        print(f"{'='*70}")
        print(f"{'Config':<25} {'Baseline AP':>12} {'Optimized AP':>13} "
              f"{'Improvement':>12} {'MMD':>8} {'% Imp':>7}")
        print("-" * 77)

        for label, summary in results.items():
            print(f"{label:<25} {summary['mean_baseline_ap']:>12.4f} "
                  f"{summary['mean_optimized_ap']:>13.4f} "
                  f"{summary['mean_ap_improvement']:>12.4f} "
                  f"{summary['mean_final_mmd']:>8.4f} "
                  f"{summary['fraction_improved']:>7.0%}")

    # Best config recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    for ablation_name, results in all_results.items():
        best_label = max(results.keys(), key=lambda k: results[k]['mean_ap_improvement'])
        best = results[best_label]
        print(f"  {ablation_name}: Best = {best_label} "
              f"(AP improvement = {best['mean_ap_improvement']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Ablation studies")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of training graphs for ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ablation", type=str, nargs='*', default=None,
                        choices=["insertion", "routing", "mmd"],
                        help="Which ablations to run (default: all)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = Phase1Config()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    results = run_phase4(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        num_samples=args.num_samples,
        seed=args.seed,
        ablations=args.ablation,
    )

    print_summary_tables(results)

    # Save
    output_path = args.output or os.path.join(
        "./logs", f"phase4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
