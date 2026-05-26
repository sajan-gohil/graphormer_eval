"""
Phase 1 Training Loop: Train GPS on Peptides-func with early stopping.
"""

import os
import sys
import time
import json
import random
import math
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.phase1_config import Phase1Config
from data.peptides_func import get_peptides_func_loaders
from models.transformer import GPSModel
from evaluation.metrics import compute_macro_ap, compute_per_class_ap, collect_predictions


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """Cosine annealing LR scheduler with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        # Scale so that minimum LR is min_lr
        base_lr = optimizer.defaults['lr']
        target = min_lr / base_lr + (1.0 - min_lr / base_lr) * cosine_decay
        return target
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, criterion, device, gradient_clip=1.0):
    """Train for one epoch. Returns average loss."""
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
    """Evaluate on a loader. Returns average loss, macro AP, per-class AP."""
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


def train(config: Phase1Config = None):
    """Main training function for Phase 1."""
    if config is None:
        config = Phase1Config()

    # Setup
    set_seed(config.training.seed)

    if config.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.training.device)
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Data
    train_loader, val_loader, test_loader, dataset_info = get_peptides_func_loaders(config.data)

    # Model
    model = GPSModel(config.model).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer & scheduler
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

    # Logging
    log_file = os.path.join(config.log_dir, f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    history = {
        "config": {
            "data": vars(config.data),
            "model": vars(config.model),
            "training": vars(config.training),
        },
        "dataset_info": dataset_info,
        "num_params": num_params,
        "epochs": [],
    }

    # Training loop with early stopping
    best_val_ap = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    print(f"\nStarting training for up to {config.training.max_epochs} epochs...")
    print(f"Early stopping patience: {config.training.patience}")
    print("-" * 80)

    for epoch in range(1, config.training.max_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip=config.training.gradient_clip
        )

        # Evaluate
        train_eval_loss, train_ap, train_per_class = evaluate(model, train_loader, criterion, device)
        val_loss, val_ap, val_per_class = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start

        # Log
        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ap": train_ap,
            "val_loss": val_loss,
            "val_ap": val_ap,
            "lr": current_lr,
            "epoch_time": epoch_time,
        }

        # Per-class AP logging
        if epoch % config.training.log_per_class_every == 0:
            epoch_log["val_per_class_ap"] = val_per_class.tolist()
            epoch_log["train_per_class_ap"] = train_per_class.tolist()

        history["epochs"].append(epoch_log)

        # Print progress
        print(f"Epoch {epoch:3d}/{config.training.max_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train AP: {train_ap:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AP: {val_ap:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        # Check for improvement
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_ap": val_ap,
                "train_ap": train_ap,
                "config": {
                    "data": vars(config.data),
                    "model": vars(config.model),
                    "training": vars(config.training),
                },
            }, checkpoint_path)
            print(f"  → New best model saved! Val AP: {val_ap:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best Val AP: {best_val_ap:.4f} at epoch {best_epoch}.")
                break

    total_time = time.time() - start_time

    # Load best model and evaluate on test set
    print("\n" + "=" * 80)
    print("Loading best model for test evaluation...")
    checkpoint = torch.load(
        os.path.join(config.checkpoint_dir, "best_model.pt"),
        map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_ap, test_per_class = evaluate(model, test_loader, criterion, device)
    train_loss_final, train_ap_final, _ = evaluate(model, train_loader, criterion, device)
    val_loss_final, val_ap_final, _ = evaluate(model, val_loader, criterion, device)

    # Final summary
    print("\n" + "=" * 80)
    print("PHASE 1 RESULTS — Base GPS Transformer")
    print("=" * 80)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 30)
    print(f"{'Train AP':<20} {train_ap_final:>10.4f}")
    print(f"{'Val AP':<20} {val_ap_final:>10.4f}")
    print(f"{'Test AP':<20} {test_ap:>10.4f}")
    print(f"{'Best Epoch':<20} {best_epoch:>10d}")
    print(f"{'Training Time':<20} {total_time/3600:>9.2f}h")
    print(f"{'Num Parameters':<20} {num_params:>10,}")
    print("=" * 80)

    print(f"\nPer-class Test AP:")
    for i, ap in enumerate(test_per_class):
        print(f"  Class {i}: {ap:.4f}")

    # Save final results
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
    print(f"\nTraining log saved to: {log_file}")

    return model, history


if __name__ == "__main__":
    config = Phase1Config()

    # Parse simple command-line overrides
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1: Train GPS Transformer")
    parser.add_argument("--hidden_dim", type=int, default=config.model.hidden_dim)
    parser.add_argument("--num_layers", type=int, default=config.model.num_layers)
    parser.add_argument("--num_heads", type=int, default=config.model.num_heads)
    parser.add_argument("--dropout", type=float, default=config.model.dropout)
    parser.add_argument("--batch_size", type=int, default=config.data.batch_size)
    parser.add_argument("--lr", type=float, default=config.training.lr)
    parser.add_argument("--max_epochs", type=int, default=config.training.max_epochs)
    parser.add_argument("--patience", type=int, default=config.training.patience)
    parser.add_argument("--seed", type=int, default=config.training.seed)
    parser.add_argument("--device", type=str, default=config.training.device)
    args = parser.parse_args()

    config.model.hidden_dim = args.hidden_dim
    config.model.num_layers = args.num_layers
    config.model.num_heads = args.num_heads
    config.model.dropout = args.dropout
    config.data.batch_size = args.batch_size
    config.training.lr = args.lr
    config.training.max_epochs = args.max_epochs
    config.training.patience = args.patience
    config.training.seed = args.seed
    config.training.device = args.device

    train(config)
