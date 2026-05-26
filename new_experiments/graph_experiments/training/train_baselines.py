"""
Phase 3: Training script for trainable proxy baselines.

Trains VN-Fixed, VN-Aggregated, and K-Fixed-VN models from scratch using
the same hyperparameters as Phase 1. Each model has a forward(batch) -> logits
interface, so the training loop is generic.

Usage:
    python training/train_baselines.py --model vn_fixed
    python training/train_baselines.py --model vn_aggregated
    python training/train_baselines.py --model kvn --M 8
"""

import os
import sys
import time
import json
import random
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.phase1_config import Phase1Config
from data.peptides_func import get_peptides_func_loaders
from models.virtual_node import GPSModelVN
from models.k_virtual_nodes import GPSModelKVN
from models.inducing_points import GPSModelISAB
from evaluation.metrics import compute_macro_ap, compute_per_class_ap


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        base_lr = optimizer.defaults['lr']
        target = min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine_decay
        return target
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_model(model_name, config, M=8):
    """Build the specified model variant."""
    if model_name == "vn_fixed":
        return GPSModelVN(config.model, vn_mode="fixed")
    elif model_name == "vn_aggregated":
        return GPSModelVN(config.model, vn_mode="aggregated")
    elif model_name == "kvn":
        return GPSModelKVN(config.model, M=M)
    elif model_name == "set_transformer_ip":
        return GPSModelISAB(config.model, M=M, shared_inducing=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, loader, optimizer, criterion, device, gradient_clip=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = criterion(logits, batch.y.float())
        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    macro_ap = compute_macro_ap(y_pred, y_true)
    per_class = compute_per_class_ap(y_pred, y_true)
    return avg_loss, macro_ap, per_class


def train(model_name, config, M=8, device=None):
    """Train a baseline model from scratch."""
    set_seed(config.training.seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Data
    train_loader, val_loader, test_loader, dataset_info = get_peptides_func_loaders(config.data)

    # Model
    model = build_model(model_name, config, M).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_label = f"{model_name}_M{M}" if model_name in ("kvn", "set_transformer_ip") else model_name
    print(f"\nTraining: {model_label}")
    print(f"Parameters: {num_params:,}")

    # Optimizer & scheduler (same as Phase 1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=config.training.warmup_epochs,
        total_epochs=config.training.max_epochs,
        min_lr=config.training.min_lr
    )
    criterion = nn.BCEWithLogitsLoss()

    # Log file
    log_file = os.path.join(
        config.log_dir,
        f"phase3_{model_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    history = {
        "model": model_label,
        "config": {
            "data": vars(config.data),
            "model": vars(config.model),
            "training": vars(config.training),
        },
        "M": M,
        "num_params": num_params,
        "dataset_info": dataset_info,
        "epochs": [],
    }

    # Training loop
    best_val_ap = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    ckpt_path = os.path.join(config.checkpoint_dir, f"best_{model_label}.pt")

    print(f"Max epochs: {config.training.max_epochs}, Patience: {config.training.patience}")
    print("-" * 80)

    for epoch in range(1, config.training.max_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip=config.training.gradient_clip
        )
        _, train_ap, _ = evaluate(model, train_loader, criterion, device)
        val_loss, val_ap, val_per_class = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ap": train_ap,
            "val_loss": val_loss,
            "val_ap": val_ap,
            "lr": current_lr,
            "epoch_time": epoch_time,
        }
        if epoch % config.training.log_per_class_every == 0:
            epoch_log["val_per_class_ap"] = val_per_class.tolist()
        history["epochs"].append(epoch_log)

        print(f"Epoch {epoch:3d}/{config.training.max_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train AP: {train_ap:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AP: {val_ap:.4f} | "
              f"LR: {current_lr:.2e} | {epoch_time:.1f}s")

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_ap": val_ap,
                "train_ap": train_ap,
                "model_name": model_label,
                "M": M,
            }, ckpt_path)
            print(f"  → New best! Val AP: {val_ap:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                print(f"\nEarly stopping at epoch {epoch}. Best: {best_val_ap:.4f} @ epoch {best_epoch}")
                break

    total_time = time.time() - start_time

    # Load best and evaluate test
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_ap, test_per_class = evaluate(model, test_loader, criterion, device)
    _, train_ap_final, _ = evaluate(model, train_loader, criterion, device)
    _, val_ap_final, _ = evaluate(model, val_loader, criterion, device)

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {model_label}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Value':>10}")
    print(f"{'-'*30}")
    print(f"{'Train AP':<20} {train_ap_final:>10.4f}")
    print(f"{'Val AP':<20} {val_ap_final:>10.4f}")
    print(f"{'Test AP':<20} {test_ap:>10.4f}")
    print(f"{'Best Epoch':<20} {best_epoch:>10d}")
    print(f"{'Training Time':<20} {total_time/3600:>9.2f}h")
    print(f"{'Params':<20} {num_params:>10,}")
    print(f"{'='*60}")

    history["final_results"] = {
        "train_ap": train_ap_final,
        "val_ap": val_ap_final,
        "test_ap": test_ap,
        "test_per_class_ap": test_per_class.tolist(),
        "best_epoch": best_epoch,
        "total_time_seconds": total_time,
        "num_params": num_params,
    }

    with open(log_file, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"Log saved to: {log_file}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Train proxy baselines")
    parser.add_argument("--model", type=str, required=True,
                        choices=["vn_fixed", "vn_aggregated", "kvn", "set_transformer_ip"],
                        help="Model variant to train")
    parser.add_argument("--M", type=int, default=8,
                        help="Number of virtual nodes (only for kvn)")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    config = Phase1Config()

    # Apply overrides
    if args.hidden_dim is not None: config.model.hidden_dim = args.hidden_dim
    if args.num_layers is not None: config.model.num_layers = args.num_layers
    if args.batch_size is not None: config.data.batch_size = args.batch_size
    if args.lr is not None: config.training.lr = args.lr
    if args.max_epochs is not None: config.training.max_epochs = args.max_epochs
    if args.patience is not None: config.training.patience = args.patience
    if args.seed is not None: config.training.seed = args.seed

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train(args.model, config, M=args.M, device=device)


if __name__ == "__main__":
    main()

