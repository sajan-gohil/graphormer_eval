# train_hop_masked_light.py
"""
Training script for the lightweight hop-masked transformer model.

Features:
- Hop-masked attention with configurable modes
- Multi-hop attention (H × K views)
- Adjacency-power walk-count blending
- Dynamic cross-hop mixer
- Edge features and attention bias
- Post-transformer GATv2 layers
- Grouped optimizer + cosine warmup scheduler
- ReduceLROnPlateau for adaptive learning
- Checkpoint save/resume and early stopping
"""

from __future__ import annotations

import argparse
import os
import time
import numpy as np
import torch

from data import DATASET_CHOICES, get_loaders
from metrics import build_task, compute_pos_weight, compute_class_weights
from model_hop_masked_light import HopMaskedTransformerModel
from optim_utils import build_grouped_optimizer_and_scheduler

os.environ["PYTHON_HASH_SEED"] = "42"
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _gather_train_labels(loader, num_classes):
    """Collect the (N, C) multi-label target matrix from a training loader."""
    ds = loader.dataset
    base = getattr(ds, "pyg_dataset", ds)
    rows = []
    for i in range(len(base)):
        g = base[i]
        y = g.y
        y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        rows.append(y.reshape(-1)[:num_classes])
    return np.stack(rows, axis=0)


def _gather_node_labels(loader, num_classes):
    """Collect all per-node class indices from a training loader."""
    ds = loader.dataset
    base = getattr(ds, "pyg_dataset", ds)
    rows = []
    for i in range(len(base)):
        g = base[i]
        y = g.y
        y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        rows.append(y.reshape(-1))
    labels = np.concatenate(rows, axis=0)
    return labels[labels >= 0]


def build_pos_or_class_weight(args, dataset_info, train_loader, device):
    """Compute loss weights when --use_pos_weight is set."""
    if not args.use_pos_weight:
        return None
    ttype = dataset_info["task_type"]
    C = dataset_info["output_dim"]
    if ttype == "multi_label":
        w = compute_pos_weight(_gather_train_labels(train_loader, C), C)
        print(f"pos_weight: {np.round(w, 3)}", flush=True)
    elif ttype == "multiclass":
        w = compute_class_weights(_gather_node_labels(train_loader, C), C)
        print(f"class_weight: {np.round(w, 3)}", flush=True)
    else:
        return None
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def build_parser():
    p = argparse.ArgumentParser(description="Train lightweight hop-masked transformer")

    # Dataset
    p.add_argument("--dataset", type=str, default="Peptides-func",
                   choices=DATASET_CHOICES)
    p.add_argument("--max_hops", type=int, default=40,
                   help="K — total hop levels in the precomputed dist_masks.")

    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--ffn_ratio", type=float, default=4.0)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--graph_pool", type=str, default="sum",
                   choices=["sum", "mean", "attention"])
    p.add_argument("--norm_type", type=str, default="layer",
                   choices=["layer", "rms", "graph"])
    p.add_argument("--block_diag_out", action="store_true", default=False)
    p.add_argument("--dynamic_cross_hop", action="store_true", default=False)

    # Attention features
    p.add_argument("--blend_adj_power", action="store_true", default=False)
    p.add_argument("--use_edge_bias", action="store_true", default=False)
    p.add_argument("--use_edge_features", action="store_true", default=False)

    # Hop-to-head assignment
    p.add_argument("--hop_mode", type=str, default="contiguous",
                   choices=["contiguous", "window", "single", "interleaved",
                            "alternating", "file"])
    p.add_argument("--hop_window", type=int, default=1)
    p.add_argument("--num_global_heads", type=int, default=0)
    p.add_argument("--hop_file", type=str, default=None)

    # Multi-hop attention
    p.add_argument("--multihop_attn", action="store_true", default=False)
    p.add_argument("--multihop_readout", type=str, default="sum",
                   choices=["sum", "mean"])
    p.add_argument("--multihop_no_global", action="store_true", default=False)

    # Mask type
    p.add_argument("--mask_type", type=str, default="shortest_path",
                   choices=["shortest_path", "adj_power"])
    p.add_argument("--adj_self_loops", action="store_true", default=False)

    # Post-transformer GATv2
    p.add_argument("--num_post_gat_layers", type=int, default=0)
    p.add_argument("--num_gat_heads", type=int, default=4)

    # Loss
    p.add_argument("--use_pos_weight", action="store_true", default=False)

    # Path-aware embeddings
    p.add_argument("--use_path_embeddings", action="store_true", default=False,
                   help="Enable path-aware embeddings for K/V conditioning in attention.")

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--reduce_lr_patience", type=int, default=10)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dist_mask_workers", type=int, default=8)

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints_hop_masked_light")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=str, default=None)
    return p


def parse_args():
    args, unknown = build_parser().parse_known_args()
    print(args.__dict__)
    print("Unknown args = ", unknown)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def _move_batch_to_device(batch, device):
    # batch can be (pyg_batch, dist_masks, node_masks) or
    #             (pyg_batch, dist_masks, node_masks, path_data)
    if len(batch) == 4:
        pyg_batch, dist_masks, node_masks, path_data = batch
        return pyg_batch.to(device), dist_masks.to(device), node_masks.to(device), path_data
    else:
        pyg_batch, dist_masks, node_masks = batch
        return pyg_batch.to(device), dist_masks.to(device), node_masks.to(device), None


def run_epoch(model, loader, task, device, optimizer=None, scheduler=None,
              grad_clip=1.0):
    """Run one epoch. Returns (loss, metric)."""
    is_train = optimizer is not None
    model.train(is_train)

    losses, preds_acc, labels_acc = [], [], []
    for batch in loader:
        pyg_batch, dist_masks, node_masks, path_data = _move_batch_to_device(batch, device)
        # Attach path_data to batch if present
        if path_data is not None:
            pyg_batch.path_data = path_data
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits, _, aux_loss, _ = model(pyg_batch, dist_masks, node_masks)
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
    train_loader, val_loader, test_loader, _, _, _, dataset_info = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_dist_masks=True,
        max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=False,
        lap_pe_dim=0,
        dataset_name=args.dataset,
        return_info=True,
        seed=args.seed,
        use_path_embeddings=args.use_path_embeddings,
    )
    dataset_name = dataset_info["name"]

    pos_weight = build_pos_or_class_weight(args, dataset_info, train_loader, args.device)

    task = build_task(dataset_name, dataset_info=dataset_info, pos_weight=pos_weight,
                      focal_gamma=0.0, label_smoothing=0.0)
    task.loss_fn = task.loss_fn.to(args.device)

    print(f"[{dataset_name}] task={task.task_type} level={task.level} "
          f"metric={task.metric_name}",
          flush=True)

    model = HopMaskedTransformerModel(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        ffn_ratio=args.ffn_ratio,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_hops=args.max_hops,
        hop_mode=args.hop_mode,
        hop_window=args.hop_window,
        hop_file=args.hop_file,
        num_global_heads=args.num_global_heads,
        output_dim=task.output_dim,
        graph_pool=args.graph_pool,
        task_level=task.level,
        dataset_name=dataset_name,
        node_feat_dim=dataset_info.get("node_feat_dim"),
        block_diag_out=args.block_diag_out,
        dynamic_cross_hop=args.dynamic_cross_hop,
        norm_type=args.norm_type,
        mask_type=args.mask_type,
        adj_self_loops=args.adj_self_loops,
        num_post_gat_layers=args.num_post_gat_layers,
        num_gat_heads=args.num_gat_heads,
        use_edge_features=args.use_edge_features,
        edge_feat_dim=dataset_info.get("edge_feat_dim"),
        blend_adj_power=args.blend_adj_power,
        use_edge_bias=args.use_edge_bias,
        multihop_attn=args.multihop_attn,
        multihop_readout=args.multihop_readout,
        multihop_include_global=not args.multihop_no_global,
        use_path_embeddings=args.use_path_embeddings,
    ).to(args.device)

    if args.multihop_attn:
        print(f"Multi-hop attention: H×K views, readout={args.multihop_readout}",
              flush=True)
    else:
        print("Head -> hop set assignment:", flush=True)
        for h, s in enumerate(model.head_hop_sets):
            print(f"  head {h}: {'GLOBAL' if s is None else s}", flush=True)

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

    best_val = -float("inf") if task.higher_is_better else float("inf")
    best_test = None
    best_epoch = -1
    epochs_since_improve = 0
    start_epoch = 0

    # Resume from checkpoint
    if args.checkpoint is not None:
        print(f"Loading checkpoint from: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=args.device)

        model.load_state_dict(ckpt["model"])
        print("  ✓ model weights loaded", flush=True)

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("  ✓ optimizer state loaded", flush=True)

        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("  ✓ step-scheduler state loaded", flush=True)
        elif "epoch" in ckpt:
            completed_steps = ckpt["epoch"] * len(train_loader)
            for _ in range(completed_steps):
                scheduler.step()
            print(f"  ✓ step-scheduler fast-forwarded ({completed_steps} steps)",
                  flush=True)

        if "plateau_scheduler" in ckpt:
            plateau_scheduler.load_state_dict(ckpt["plateau_scheduler"])
            print("  ✓ plateau-scheduler state loaded", flush=True)

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
            ckpt_path = os.path.join(args.save_dir, f"best_{dataset_name}.pt")
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

