"""
Training script for the MoE hop-masked transformer model.

Each attention head *learns* which hop-distance masks to attend through
via a soft gating network (Mixture of Experts), rather than receiving a
static deterministic assignment.  An auxiliary gate-regularisation loss
(load-balancing + entropy) is added to the task loss to prevent gate
collapse.

Datasets: Peptides-func, Peptides-struct, PascalVOC-SP.

Examples:
    # Dense gating (full softmax over K hops per head).
    python train_moe_hop_transformer.py --dataset Peptides-func \\
        --max_hops 40 --num_heads 8 --num_layers 4

    # Sparse gating (each head picks top-5 hops).
    python train_moe_hop_transformer.py --dataset Peptides-func \\
        --max_hops 40 --num_heads 8 --top_k 5

    # Higher gate regularisation.
    python train_moe_hop_transformer.py --dataset Peptides-func \\
        --balance_coeff 0.05 --entropy_coeff 0.05

    # PascalVOC-SP — smaller batch and shorter hop horizon.
    python train_moe_hop_transformer.py --dataset PascalVOC-SP \\
        --max_hops 12 --num_heads 4 --num_layers 4 --batch_size 16
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
from model_moe_hop_transformer import MoEHopTransformerModel
from optim_utils import build_grouped_optimizer_and_scheduler


def build_parser():
    p = argparse.ArgumentParser(description="Train MoE hop-masked transformer")

    # Dataset
    p.add_argument("--dataset", type=str, default="Peptides-func",
                   choices=["Peptides-func", "Peptides-struct", "PascalVOC-SP"])
    p.add_argument("--max_hops", type=int, default=40,
                   help="K — total hop levels in the precomputed dist_masks. "
                        "Use ~12 for PascalVOC-SP.")
    p.add_argument("--use_lap_pe", action="store_true", default=False)
    p.add_argument("--lap_pe_dim", type=int, default=8)

    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8,
                   help="Total heads. Must divide hidden_dim.")
    p.add_argument("--ffn_ratio", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--graph_pool", type=str, default="sum",
                   choices=["sum", "mean"])

    # MoE gating
    p.add_argument("--top_k", type=int, default=0,
                   help="Sparse gating: each head keeps only top-k hops. "
                        "0 = dense (full softmax over all K hops).")
    p.add_argument("--gate_noise", type=float, default=0.1,
                   help="Gaussian noise std added to gate logits during "
                        "training to encourage exploration.")
    p.add_argument("--balance_coeff", type=float, default=0.01,
                   help="Weight for Switch-style load-balancing aux loss.")
    p.add_argument("--entropy_coeff", type=float, default=0.01,
                   help="Weight for entropy regularisation aux loss "
                        "(encourages diffuse gate distributions).")

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--reduce_lr_patience", type=int, default=10,
                   help="Patience for ReduceLROnPlateau (epochs).")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dist_mask_workers", type=int, default=8)

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints_moe_hop")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a checkpoint .pt file to resume training from.")
    return p


def parse_args():
    args, unknown = build_parser().parse_known_args()
    print(args.__dict__)
    print("Unknown args = ", unknown)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def _move_batch_to_device(batch, device):
    pyg_batch, dist_masks, node_masks = batch
    return pyg_batch.to(device), dist_masks.to(device), node_masks.to(device)


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
            logits, _, aux_loss = model(pyg_batch, dist_masks, node_masks)
            task_loss = task.loss(logits, pyg_batch.y)
            loss = task_loss + aux_loss

        if is_train:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        losses.append(task_loss.item())
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

    model = MoEHopTransformerModel(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ffn_ratio=args.ffn_ratio,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_hops=args.max_hops,
        top_k=args.top_k,
        gate_noise=args.gate_noise,
        balance_coeff=args.balance_coeff,
        entropy_coeff=args.entropy_coeff,
        output_dim=task.output_dim,
        graph_pool=args.graph_pool,
        task_level=task.level,
        dataset_name=args.dataset,
        lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
    ).to(args.device)

    print(f"MoE gating config: top_k={args.top_k}, gate_noise={args.gate_noise}, "
          f"balance_coeff={args.balance_coeff}, entropy_coeff={args.entropy_coeff}",
          flush=True)

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
    )

    plateau_mode = "max" if task.higher_is_better else "min"
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=plateau_mode,
        factor=0.5,
        patience=args.reduce_lr_patience,
        min_lr=args.lr_min,
    )
    print(f"ReduceLROnPlateau: mode={plateau_mode}, patience={args.reduce_lr_patience}, "
          f"factor=0.5, min_lr={args.lr_min}", flush=True)

    best_val = -float("inf") if task.higher_is_better else float("inf")
    best_test = None
    best_epoch = -1
    epochs_since_improve = 0
    start_epoch = 0

    # ------------------------------------------------------------------ #
    # Resume from checkpoint                                               #
    # ------------------------------------------------------------------ #
    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=args.device)

        # Model weights (required)
        model.load_state_dict(ckpt["model"])
        print("  ✓ model weights loaded", flush=True)

        # Optimizer state
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("  ✓ optimizer state loaded", flush=True)

        # Per-step cosine scheduler — fast-forward to the saved epoch
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("  ✓ step-scheduler state loaded", flush=True)
        elif "epoch" in ckpt:
            completed_steps = ckpt["epoch"] * len(train_loader)
            for _ in range(completed_steps):
                scheduler.step()
            print(f"  ✓ step-scheduler fast-forwarded ({completed_steps} steps)",
                  flush=True)

        # ReduceLROnPlateau scheduler
        if "plateau_scheduler" in ckpt:
            plateau_scheduler.load_state_dict(ckpt["plateau_scheduler"])
            print("  ✓ plateau-scheduler state loaded", flush=True)

        # Training bookkeeping
        if "best_val" in ckpt:
            best_val = ckpt["best_val"]
        if "best_test" in ckpt:
            best_test = ckpt["best_test"]
        if "best_epoch" in ckpt:
            best_epoch = ckpt["best_epoch"]
        if "epochs_since_improve" in ckpt:
            epochs_since_improve = ckpt["epochs_since_improve"]
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

        print(f"  Resuming from epoch {start_epoch} "
              f"(best_val={best_val:.4f} at epoch {best_epoch})", flush=True)

    for epoch in range(start_epoch, args.max_epochs):
        t0 = time.time()
        tr_loss, tr_metric = run_epoch(
            model, train_loader, task, args.device,
            optimizer=optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
        )
        va_loss, va_metric = run_epoch(model, val_loader, task, args.device)
        plateau_scheduler.step(va_metric)
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
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "plateau_scheduler": plateau_scheduler.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val": best_val,
                    "best_test": best_test,
                    "best_epoch": best_epoch,
                    "epochs_since_improve": epochs_since_improve,
                    "val_metric": va_metric,
                    "test_metric": te_metric,
                },
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
