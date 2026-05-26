"""
Training script for Idea A — cross-attention router only (no global SA).

Stack of cross-attention router blocks (proxy generation + N->M->N routing
per block) without any global self-attention. End-to-end: no frozen
transformer, no pretraining.

Datasets:
    Peptides-func, Peptides-struct, PascalVOC-SP

Example:
    python train_cross_attn_only.py --dataset Peptides-func \\
        --generator score_based --num_proxies 32 --num_layers 4

    python train_cross_attn_only.py --dataset PascalVOC-SP \\
        --generator gnn_pooling --num_proxies 64 --num_layers 4 \\
        --batch_size 16
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
from model_cross_attn_only import CrossAttnOnlyModel
from optim_utils import build_grouped_optimizer_and_scheduler


def build_parser():
    p = argparse.ArgumentParser(description="Train cross-attn-only model (Idea A)")

    # Dataset
    p.add_argument("--dataset", type=str, default="Peptides-func",
                   choices=["Peptides-func", "Peptides-struct", "PascalVOC-SP"])
    p.add_argument("--use_lap_pe", action="store_true", default=False)
    p.add_argument("--lap_pe_dim", type=int, default=8)

    # Model
    p.add_argument("--hidden_dim", type=int, default=96)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_cross_layers", type=int, default=2,
                   help="N->M->N internal iterations per router block.")
    p.add_argument("--no_proxy_self_attn", action="store_true",
                   help="Disable the M-by-M proxy self-attention refine step.")
    p.add_argument("--share_blocks", action="store_true", default=False)
    p.add_argument("--graph_pool", type=str, default="sum",
                   choices=["sum", "mean"])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--aux_loss_weight", type=float, default=0.0,
                   help="Multiplier for total generator aux loss "
                        "(e.g. mincut/orthogonality regularization).")
    p.add_argument("--aux_loss_decay", type=float, default=1.0,
                   help="Geometric decay applied across stacked blocks.")
    p.add_argument("--final_sa_layers", type=int, default=0,
                   help="Optional global self-attention layers AFTER the "
                        "cross-attn router stack and BEFORE the head. "
                        "0 (default) = off.")
    p.add_argument("--final_sa_heads", type=int, default=0,
                   help="Heads in the final SA stack. 0 = use --num_heads.")

    # Generator
    p.add_argument("--generator", type=str, default="score_based",
                   choices=["score_based", "gnn_pooling", "pma", "graph_coarsening"])
    p.add_argument("--num_proxies", type=int, default=32)
    p.add_argument("--score_hidden", type=int, default=128)
    p.add_argument("--score_layers", type=int, default=1)
    p.add_argument("--score_heads", type=int, default=4)

    # GNN-pooling generator
    p.add_argument("--gnn_layers", type=int, default=3)
    p.add_argument("--gnn_type", type=str, default="GINE",
                   choices=["GCN", "GIN", "GINE", "GAT"])
    p.add_argument("--pool_types", type=str, nargs="+", default=["mean"])
    p.add_argument("--decode_hidden", type=int, default=64)
    p.add_argument("--decode_layers", type=int, default=3)
    p.add_argument("--idx_emb_dim", type=int, default=64)
    p.add_argument("--decode_mode", type=str, default="shared",
                   choices=["shared", "grouped"])

    # PMA / coarsening
    p.add_argument("--pma_query_mode", type=str, default="farthest_point",
                   choices=["farthest_point", "soft_kmeans"])
    p.add_argument("--coarsen_gnn_type", type=str, default="GIN",
                   choices=["GIN", "GCN"])
    p.add_argument("--coarsen_reg_weight", type=float, default=0.0)

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

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints_cross_attn_only")
    p.add_argument("--seed", type=int, default=0)
    return p


def parse_args():
    args, unknown = build_parser().parse_known_args()
    print(args.__dict__)
    print("Unknown = ", unknown)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def run_epoch(model, loader, task, device, optimizer=None, scheduler=None,
              grad_clip=1.0, aux_loss_weight=0.0):
    is_train = optimizer is not None
    model.train(is_train)

    losses, preds_acc, labels_acc = [], [], []
    aux_losses = []
    for batch in loader:
        batch = batch.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, _ = model(batch)
            main_loss = task.loss(logits, batch.y)
            aux = model.last_aux_loss
            if isinstance(aux, torch.Tensor) and aux_loss_weight > 0:
                loss = main_loss + aux_loss_weight * aux
                aux_losses.append(float(aux.item()))
            else:
                loss = main_loss

        if is_train:
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        losses.append(main_loss.item())
        preds_acc.append(task.predict(logits))
        labels_acc.append(task.labels_to_numpy(batch.y))

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

    # Cross-attn-only does not require dist_masks — use the standard PyG loader.
    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_dist_masks=False,
        use_lap_pe=args.use_lap_pe,
        lap_pe_dim=args.lap_pe_dim,
        dataset_name=args.dataset,
    )

    model = CrossAttnOnlyModel(
        hidden_dim=args.hidden_dim,
        num_proxies=args.num_proxies,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_cross_layers=args.num_cross_layers,
        use_proxy_self_attn=not args.no_proxy_self_attn,
        generator_name=args.generator,
        share_blocks=args.share_blocks,
        output_dim=task.output_dim,
        dropout=args.dropout,
        graph_pool=args.graph_pool,
        task_level=task.level,
        dataset_name=args.dataset,
        lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
        aux_loss_decay=args.aux_loss_decay,
        final_sa_layers=args.final_sa_layers,
        final_sa_heads=(args.final_sa_heads or None),
        gnn_layers=args.gnn_layers,
        gnn_type=args.gnn_type,
        pool_types=args.pool_types,
        decode_hidden=args.decode_hidden,
        decode_layers=args.decode_layers,
        idx_emb_dim=args.idx_emb_dim,
        decode_mode=args.decode_mode,
        pma_query_mode=args.pma_query_mode,
        coarsen_gnn_type=args.coarsen_gnn_type,
        coarsen_reg_weight=args.coarsen_reg_weight,
        score_hidden=args.score_hidden,
        score_layers=args.score_layers,
        score_heads=args.score_heads,
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
            aux_loss_weight=args.aux_loss_weight,
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
