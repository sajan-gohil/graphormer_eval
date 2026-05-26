"""
Phase 3: Evaluate frozen-model proxy baselines.

No additional training — evaluates the frozen Phase 1 model with various
proxy generation strategies:
  - Mean proxies (global mean, k-means, random nodes, degree-weighted)
  - Random proxy embeddings (10 seeds, report mean ± std)

Uses forward_with_proxies from Phase 2 to inject generated proxies.

Usage:
    python training/eval_frozen_baselines.py --checkpoint ./checkpoints/best_model.pt
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.phase1_config import Phase1Config
from data.peptides_func import get_peptides_func_loaders
from models.transformer import GPSModel
from models.proxy_optimizer import forward_with_proxies
from models.mean_proxy import (
    generate_global_mean_proxies,
    generate_kmeans_proxies,
    generate_random_node_proxies,
    generate_degree_weighted_proxies,
)
from models.random_proxy import generate_random_proxies
from evaluation.metrics import compute_macro_ap, compute_per_class_ap


def load_frozen_model(checkpoint_path, config, device):
    model = GPSModel(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def evaluate_with_proxy_strategy(model, loader, device, strategy_fn, M, strategy_kwargs=None):
    """
    Evaluate a frozen model with a given proxy generation strategy.

    Args:
        model: Frozen GPSModel.
        loader: DataLoader.
        device: torch device.
        strategy_fn: Function(h_sparse, batch_idx, M, num_graphs, d, **kwargs) -> (B, M, d)
        M: Number of proxies.
        strategy_kwargs: Extra kwargs for strategy_fn.

    Returns:
        macro_ap, per_class_ap, avg_loss
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    if strategy_kwargs is None:
        strategy_kwargs = {}

    for batch in loader:
        batch = batch.to(device)

        # Get initial embeddings for proxy generation
        h_sparse, batch_idx = model.get_initial_embeddings(batch)
        num_graphs = batch_idx.max().item() + 1
        d = model.hidden_dim

        # Generate proxies
        proxies = strategy_fn(h_sparse, batch_idx, M, num_graphs, d, **strategy_kwargs)

        # Forward with proxies
        logits = forward_with_proxies(model, batch, proxies)
        loss = criterion(logits, batch.y.float())

        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        total_loss += loss.item()
        num_batches += 1

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    macro_ap = compute_macro_ap(y_pred, y_true)
    per_class = compute_per_class_ap(y_pred, y_true)
    avg_loss = total_loss / num_batches

    return macro_ap, per_class, avg_loss


@torch.no_grad()
def evaluate_with_degree_proxies(model, loader, device, M):
    """
    Special evaluation for degree-weighted proxies (needs edge_index in strategy call).
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        h_sparse, batch_idx = model.get_initial_embeddings(batch)
        num_graphs = batch_idx.max().item() + 1
        d = model.hidden_dim

        proxies = generate_degree_weighted_proxies(
            h_sparse, batch_idx, batch.edge_index, M, num_graphs, d
        )

        logits = forward_with_proxies(model, batch, proxies)
        loss = criterion(logits, batch.y.float())

        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        total_loss += loss.item()
        num_batches += 1

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    macro_ap = compute_macro_ap(y_pred, y_true)
    per_class = compute_per_class_ap(y_pred, y_true)
    avg_loss = total_loss / num_batches

    return macro_ap, per_class, avg_loss


@torch.no_grad()
def evaluate_baseline_no_proxy(model, loader, device):
    """Evaluate the base model without any proxies."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.float())
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        total_loss += loss.item()
        num_batches += 1

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    macro_ap = compute_macro_ap(y_pred, y_true)
    per_class = compute_per_class_ap(y_pred, y_true)
    avg_loss = total_loss / num_batches

    return macro_ap, per_class, avg_loss


def run_all_frozen_baselines(config, checkpoint_path, device, M=8, num_random_seeds=10):
    """
    Run all frozen-model proxy baselines and produce a results table.
    """
    # Load data
    train_loader, val_loader, test_loader, dataset_info = get_peptides_func_loaders(config.data)

    # Load frozen model
    print(f"Loading frozen model from {checkpoint_path}")
    model = load_frozen_model(checkpoint_path, config, device)

    results = {}

    # --- Baseline: no proxy ---
    print("\n[1/6] GPS baseline (no proxy)...")
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        ap, per_class, loss = evaluate_baseline_no_proxy(model, loader, device)
        results.setdefault("GPS_baseline", {})[split_name] = {
            "ap": ap, "per_class_ap": per_class.tolist(), "loss": loss
        }
        print(f"  {split_name}: AP={ap:.4f}")

    # --- Mean proxy strategies ---
    strategies = [
        ("global_mean", generate_global_mean_proxies, {}),
        ("kmeans", generate_kmeans_proxies, {}),
        ("random_nodes", generate_random_node_proxies, {}),
    ]

    for i, (name, fn, kwargs) in enumerate(strategies):
        print(f"\n[{i+2}/6] Mean proxy ({name}), M={M}...")
        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            ap, per_class, loss = evaluate_with_proxy_strategy(
                model, loader, device, fn, M, strategy_kwargs=kwargs
            )
            results.setdefault(f"mean_proxy_{name}", {})[split_name] = {
                "ap": ap, "per_class_ap": per_class.tolist(), "loss": loss
            }
            print(f"  {split_name}: AP={ap:.4f}")

    # --- Degree-weighted proxy ---
    print(f"\n[5/6] Mean proxy (degree_weighted), M={M}...")
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        ap, per_class, loss = evaluate_with_degree_proxies(model, loader, device, M)
        results.setdefault("mean_proxy_degree_weighted", {})[split_name] = {
            "ap": ap, "per_class_ap": per_class.tolist(), "loss": loss
        }
        print(f"  {split_name}: AP={ap:.4f}")

    # --- Random proxy embeddings (10 seeds) ---
    print(f"\n[6/6] Random proxies, M={M}, {num_random_seeds} seeds...")
    random_results_by_split = {"train": [], "val": [], "test": []}

    for seed in range(num_random_seeds):
        for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            ap, per_class, loss = evaluate_with_proxy_strategy(
                model, loader, device, generate_random_proxies, M,
                strategy_kwargs={"seed": seed * 1000 + 42}
            )
            random_results_by_split[split_name].append(ap)

        if (seed + 1) % 5 == 0:
            val_aps = random_results_by_split["val"]
            print(f"  Seed {seed+1}/{num_random_seeds}: "
                  f"Val AP = {np.mean(val_aps):.4f} ± {np.std(val_aps):.4f}")

    for split_name in ["train", "val", "test"]:
        aps = random_results_by_split[split_name]
        results.setdefault("random_proxies", {})[split_name] = {
            "ap_mean": float(np.mean(aps)),
            "ap_std": float(np.std(aps)),
            "ap_all_seeds": aps,
        }

    return results, dataset_info


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Evaluate frozen-model baselines")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--num_random_seeds", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = Phase1Config()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    results, dataset_info = run_all_frozen_baselines(
        config, args.checkpoint, device,
        M=args.M, num_random_seeds=args.num_random_seeds
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("PHASE 3 — FROZEN MODEL BASELINES (M={})".format(args.M))
    print("=" * 80)
    print(f"{'Method':<30} {'Train AP':>10} {'Val AP':>10} {'Test AP':>10}")
    print("-" * 60)

    for method, splits in results.items():
        if method == "random_proxies":
            train_str = f"{splits['train']['ap_mean']:.4f}±{splits['train']['ap_std']:.3f}"
            val_str = f"{splits['val']['ap_mean']:.4f}±{splits['val']['ap_std']:.3f}"
            test_str = f"{splits['test']['ap_mean']:.4f}±{splits['test']['ap_std']:.3f}"
            print(f"{'random_proxies (10 seeds)':<30} {train_str:>10} {val_str:>10} {test_str:>10}")
        else:
            print(f"{method:<30} {splits['train']['ap']:>10.4f} "
                  f"{splits['val']['ap']:>10.4f} {splits['test']['ap']:>10.4f}")

    print("=" * 80)

    # Save
    output_path = args.output or os.path.join(
        "./logs", f"phase3_frozen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({"results": results, "M": args.M, "dataset_info": dataset_info},
                  f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
