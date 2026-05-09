"""
Training script for Weighted GRED (model_weighted_gred.py).

Same training scaffold as train_khop_concat.py — what differs is the model:
each layer here does attention-weighted per-hop aggregation followed by an
LRU scan over the hop axis (vs. concat+MLP in Idea C).

Datasets:
    Peptides-func, Peptides-struct, PascalVOC-SP

Example:
    python train_weighted_gred.py --dataset Peptides-func --num_layers 4
    python train_weighted_gred.py --dataset Peptides-struct --num_layers 4
    python train_weighted_gred.py --dataset PascalVOC-SP --max_hops 12 \
        --hidden_dim 96 --state_dim 96 --batch_size 16
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
import torch
torch.set_float32_matmul_precision("high")

from data import get_loaders
from metrics import build_task
from model_weighted_gred import WeightedGREDModel
from optim_utils import build_grouped_optimizer_and_scheduler


def build_parser():
    p = argparse.ArgumentParser(description="Train Weighted GRED")

    # Dataset
    p.add_argument("--dataset", type=str, default="Peptides-func",
                   choices=["Peptides-func", "Peptides-struct", "PascalVOC-SP"])
    p.add_argument("--max_hops", type=int, default=40)
    p.add_argument("--use_lap_pe", action="store_true", default=False)
    p.add_argument("--lap_pe_dim", type=int, default=8)

    # Model
    p.add_argument("--hidden_dim", type=int, default=96)
    p.add_argument("--state_dim", type=int, default=96)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--expand", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--share_layers", action="store_true", default=False,
                   help="Reuse a single WeightedGRED layer for all depths.")
    p.add_argument("--graph_pool", type=str, default="sum",
                   choices=["sum", "mean"])
    p.add_argument("--no_residual_self", action="store_true")
    p.add_argument("--r_min", type=float, default=0.0)
    p.add_argument("--r_max", type=float, default=1.0)
    p.add_argument("--max_phase", type=float, default=6.28)
    p.add_argument("--gred_act", type=str, default="full-glu",
                   choices=["full-glu", "half-glu"])

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dist_mask_workers", type=int, default=8)
    p.add_argument("--recurrent_lr_factor", type=float, default=1.0,
                   help="LR multiplier for LRU recurrent params "
                        "(B_re, B_im, nu_log, theta_log, gamma_log).")

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints_weighted_gred")
    p.add_argument("--seed", type=int, default=0)
    return p


def parse_args():
    args = build_parser().parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def _move_batch_to_device(batch, device):
    pyg_batch, dist_masks, node_masks = batch
    return (
        pyg_batch.to(device),
        dist_masks.to(device),
        node_masks.to(device),
    )


def run_epoch(model, loader, task, device, optimizer=None, scheduler=None,
              grad_clip=1.0):
    is_train = optimizer is not None
    model.train(is_train)

    losses, preds_acc, labels_acc = [], [], []
    for batch in loader:
        pyg_batch, dist_masks, node_masks = _move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, _ = model(pyg_batch, dist_masks, node_masks)
            loss = task.loss(logits, pyg_batch.y)

        if is_train:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        losses.append(loss.item())
        preds_acc.append(task.predict(logits))
        labels_acc.append(task.labels_to_numpy(pyg_batch.y))

    y_pred = np.concatenate(preds_acc, axis=0)
    y_true = np.concatenate(labels_acc, axis=0)
    metric = task.compute_metric(y_pred, y_true)
    return float(np.mean(losses)), float(metric)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    task = build_task(args.dataset)

    print(f"[{args.dataset}] task={task.task_type} level={task.level} "
          f"metric={task.metric_name} (higher_is_better={task.higher_is_better})",
          flush=True)

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_dist_masks=True,
        max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe,
        lap_pe_dim=args.lap_pe_dim,
        dataset_name=args.dataset,
    )

    model = WeightedGREDModel(
        hidden_dim=args.hidden_dim,
        state_dim=args.state_dim,
        num_layers=args.num_layers,
        max_hops=args.max_hops,
        num_heads=args.num_heads,
        share_layers=args.share_layers,
        expand=args.expand,
        dropout=args.dropout,
        r_min=args.r_min,
        r_max=args.r_max,
        max_phase=args.max_phase,
        act=args.gred_act,
        output_dim=task.output_dim,
        graph_pool=args.graph_pool,
        task_level=task.level,
        dataset_name=args.dataset,
        lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
        residual_self=not args.no_residual_self,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_params/1e6:.3f}M", flush=True)

    total_steps = max(args.max_epochs * len(train_loader), 1)
    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=list(model.named_parameters()),
        lr_max=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        recurrent_lr_factor=args.recurrent_lr_factor,
    )

    best_val = -float("inf") if task.higher_is_better else float("inf")
    best_test = None
    best_epoch = -1
    epochs_since_improve = 0

    for epoch in range(args.max_epochs):
        t0 = time.time()
        tr_loss, tr_metric = run_epoch(
            model, train_loader, task, args.device,
            optimizer=optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
        )
        va_loss, va_metric = run_epoch(model, val_loader, task, args.device)
        te_loss, te_metric = run_epoch(model, test_loader, task, args.device)
        dt = time.time() - t0

        improved = (
            va_metric > best_val if task.higher_is_better else va_metric < best_val
        )
        if improved:
            best_val = va_metric
            best_test = te_metric
            best_epoch = epoch
            epochs_since_improve = 0
            ckpt_path = os.path.join(args.save_dir, f"best_{args.dataset}.pt")
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                 "val_metric": va_metric, "test_metric": te_metric},
                ckpt_path,
            )
        else:
            epochs_since_improve += 1

        ml = task.metric_label
        print(
            f"epoch {epoch:03d} | {dt:5.1f}s | "
            f"train loss {tr_loss:.4f} {ml} {tr_metric:.4f} | "
            f"val loss {va_loss:.4f} {ml} {va_metric:.4f} | "
            f"test loss {te_loss:.4f} {ml} {te_metric:.4f} | "
            f"best val {best_val:.4f} (epoch {best_epoch}, test {best_test})",
            flush=True,
        )

        if epochs_since_improve >= args.patience:
            print(f"early stopping at epoch {epoch} "
                  f"(no val improvement in {args.patience} epochs)", flush=True)
            break

    print(f"BEST: val {ml} {best_val:.4f} | test {ml} {best_test:.4f} "
          f"(epoch {best_epoch})", flush=True)


if __name__ == "__main__":
    main()
