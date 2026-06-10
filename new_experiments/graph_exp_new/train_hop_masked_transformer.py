"""
Training script for the hop-masked transformer model.

Each attention head is constrained to attend within a chosen hop range,
preserving per-hop identity in a transformer-native way. This addresses
the empirical observation that approaches collapsing the K hop axis
(LRU, proxy-bottleneck routing, GNN-pooling) underperform the simple
concat baseline.

Datasets: Peptides-func, Peptides-struct, PascalVOC-SP.

Examples:
    # Default: contiguous partition of hops [1..K-1] across heads.
    python train_hop_masked_transformer.py --dataset Peptides-func \\
        --max_hops 40 --num_heads 8 --num_layers 4 --hop_mode contiguous

    # Window mode: each head sees ~3 adjacent hops, centres spread evenly.
    python train_hop_masked_transformer.py --dataset Peptides-func \\
        --max_hops 40 --num_heads 8 --hop_mode window --hop_window 1

    # Mix structured heads with unrestricted "global" heads.
    python train_hop_masked_transformer.py --dataset Peptides-func \\
        --max_hops 40 --num_heads 8 --num_global_heads 2

    # PascalVOC-SP — smaller batch and shorter hop horizon.
    python train_hop_masked_transformer.py --dataset PascalVOC-SP \\
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
from model_hop_masked_transformer_temp import HopMaskedTransformerModel
from optim_utils import build_grouped_optimizer_and_scheduler


def build_parser():
    p = argparse.ArgumentParser(description="Train hop-masked transformer")

    # Dataset
    p.add_argument("--dataset", type=str, default="Peptides-func",
                   choices=["Peptides-func", "Peptides-struct", "PascalVOC-SP"])
    p.add_argument("--max_hops", type=int, default=40,
                   help="K — total hop levels in the precomputed dist_masks. "
                        "Use ~12 for PascalVOC-SP.")
    p.add_argument("--use_lap_pe", action="store_true", default=False)
    p.add_argument("--lap_pe_dim", type=int, default=8)
    p.add_argument("--mask_type", type=str, default="shortest_path",
                   choices=["shortest_path", "adj_power"],
                   help="How hop masks are built. 'shortest_path' (default) "
                        "uses the precomputed distance shells. 'adj_power' "
                        "rebuilds masks from powers of the adjacency: slot k = "
                        "(A^k > 0), reconstructing A from the distance-1 shell. "
                        "Pair with --adj_self_loops to use (A+I)^k.")
    p.add_argument("--adj_self_loops", action="store_true", default=False,
                   help="Only with --mask_type adj_power: use (A+I)^k instead "
                        "of A^k. (A+I)^k means 'reachable in <= k steps' "
                        "(monotone, no parity striping on near-bipartite "
                        "graphs); A^k is exact-length-k walks.")

    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8,
                   help="Total heads. Must divide hidden_dim.")
    p.add_argument("--ffn_ratio", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--graph_pool", type=str, default="sum",
                   choices=["sum", "mean", "attention"],
                   help="Graph-level readout. 'attention' uses a learnable "
                        "query attending over nodes (Set-Transformer PMA).")
    p.add_argument("--norm_type", type=str, default="layer",
                   choices=["layer", "rms", "graph"],
                   help="Normalization in transformer layers. 'layer' "
                        "(default) = LayerNorm; 'rms' = RMSNorm; 'graph' = "
                        "masked GraphNorm (per-graph statistics).")
    p.add_argument("--v_head_dim", type=int, default=None,
                   help="Value head dim. Default None = equals QK head dim "
                        "(hidden_dim // num_heads, original behaviour). Set "
                        "larger to decouple value capacity from QK "
                        "(asymmetric attention).")
    p.add_argument("--block_diag_out", action="store_true", default=False,
                   help="Use block-diagonal out_proj in MHA (no cross-head "
                        "mixing inside attention).")
    p.add_argument("--dynamic_cross_hop", action="store_true", default=False,
                   help="Insert a dynamic cross-hop attention sublayer "
                        "between MHA and FFN. Best paired with "
                        "--block_diag_out.")

    # Hop-to-head assignment
    p.add_argument("--hop_mode", type=str, default="contiguous",
                   choices=["contiguous", "window", "single", "interleaved"],
                   help="How to assign hops to heads. contiguous = partition "
                        "[1..K-1] into num_heads chunks; window = evenly-spaced "
                        "centres with hop_window half-width; single = exactly "
                        "one hop per head (requires num_heads == K-1); "
                        "interleaved = each head covers every hop_window-th hop "
                        "starting from an evenly-spaced centre k.")
    p.add_argument("--hop_window", type=int, default=1,
                   help="Half-window for 'window' mode (head covers "
                        "[c-w, c+w]).")
    p.add_argument("--num_global_heads", type=int, default=0,
                   help="Last N heads are unrestricted (free global "
                        "attention) rather than hop-masked.")

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
    p.add_argument("--save_dir", type=str, default="checkpoints_hop_masked")
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

    model = HopMaskedTransformerModel(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ffn_ratio=args.ffn_ratio,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_hops=args.max_hops,
        hop_mode=args.hop_mode,
        hop_window=args.hop_window,
        num_global_heads=args.num_global_heads,
        output_dim=task.output_dim,
        graph_pool=args.graph_pool,
        task_level=task.level,
        dataset_name=args.dataset,
        lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
        block_diag_out=args.block_diag_out,
        dynamic_cross_hop=args.dynamic_cross_hop,
        norm_type=args.norm_type,
        v_head_dim=args.v_head_dim,
        mask_type=args.mask_type,
        adj_self_loops=args.adj_self_loops,
    ).to(args.device)

    # Print the head -> hop-set assignment so it's logged for reproducibility.
    print("Head -> hop set assignment:", flush=True)
    for h, s in enumerate(model.head_hop_sets):
        print(f"  head {h}: {'GLOBAL (no hop mask)' if s is None else s}", flush=True)

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
        # verbose=True,
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
            # Fall back: replay the correct number of steps
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

