"""
Training script for the hop-masked transformer model.

Each attention head is constrained to attend within a chosen hop range,
preserving per-hop identity in a transformer-native way. This addresses
the empirical observation that approaches collapsing the K hop axis
(LRU, proxy-bottleneck routing, GNN-pooling) underperform the simple
concat baseline.

Datasets: Peptides-func, Peptides-struct, PascalVOC-SP, MNIST, CIFAR10,
PATTERN, CLUSTER, ZINC12k.

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

from data import DATASET_CHOICES, get_loaders
from metrics import build_task, compute_pos_weight
from model_hop_masked_transformer import HopMaskedTransformerModel
from optim_utils import build_grouped_optimizer_and_scheduler

os.environ["PYTHON_HASH_SEED"] = "42"
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def _gather_train_labels(loader, num_classes):
    """Collect the (N, C) multi-label target matrix from a training loader."""
    ds = loader.dataset
    base = getattr(ds, "pyg_dataset", ds)   # unwrap DistMaskDataset
    rows = []
    for i in range(len(base)):
        g = base[i]
        y = g.y
        y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        rows.append(y.reshape(-1)[:num_classes])
    return np.stack(rows, axis=0)


def build_parser():
    p = argparse.ArgumentParser(description="Train hop-masked transformer")

    # Dataset
    p.add_argument("--dataset", type=str, default="Peptides-func",
                   choices=DATASET_CHOICES)
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
    p.add_argument("--cross_hop_hop_embedding", action="store_true", default=False,
                   help="Add a learnable per-head (per-hop-band) embedding "
                        "inside the DynamicCrossHopMixer so it is no longer "
                        "permutation-invariant over hop slabs. Only has an "
                        "effect when --dynamic_cross_hop is set.")
    p.add_argument("--use_edge_features", action="store_true", default=False,
                   help="Incorporate edge/bond features: a BondEncoder embeds "
                        "edge_attr, aggregated into node features in the node "
                        "encoder, and (if --num_post_gat_layers>0) fed to the "
                        "GATv2 layers via edge_dim.")
    p.add_argument("--use_pos_weight", action="store_true", default=False,
                   help="Use per-class pos_weight = sqrt(N/(C*n_k)) in the BCE "
                        "loss for multi-label tasks (computed from the training "
                        "set). No effect on non-multi-label tasks.")
    p.add_argument("--use_virtual_node", action="store_true", default=False,
                   help="Prepend a learnable virtual-node embedding that "
                        "participates in every attention head.  For graph-"
                        "level tasks its final embedding is used as the "
                        "graph representation (replaces pooling).")
    p.add_argument("--num_post_gat_layers", type=int, default=0,
                   help="Number of GATv2Conv layers applied after the "
                        "transformer stack, before the task head.  "
                        "0 = disabled (default).")
    p.add_argument("--num_gat_heads", type=int, default=4,
                   help="Number of attention heads in each post-transformer "
                        "GATv2 layer.  Must divide hidden_dim.")

    # Hop-to-head assignment
    p.add_argument("--hop_mode", type=str, default="contiguous",
                   choices=["contiguous", "window", "single", "interleaved",
                            "alternating"],
                   help="How to assign hops to heads. contiguous = partition "
                        "[1..K-1] into num_heads chunks; window = evenly-spaced "
                        "centres with hop_window half-width; single = exactly "
                        "one hop per head (requires num_heads == K-1); "
                        "interleaved = each head covers every hop_window-th hop "
                        "starting from an evenly-spaced centre k; "
                        "alternating = even-indexed layers get even hops, "
                        "odd-indexed layers get odd hops.")
    p.add_argument("--hop_window", type=int, default=1,
                   help="Half-window for 'window' mode (head covers "
                        "[c-w, c+w]).")
    p.add_argument("--num_global_heads", type=int, default=0,
                   help="Last N heads are unrestricted (free global "
                        "attention) rather than hop-masked.")

    # MoE gating (optional, replaces deterministic hop assignment)
    p.add_argument("--use_moe_gating", action="store_true", default=False,
                   help="Replace deterministic hop-to-head assignment with a "
                        "learned MoE gating network. Each head dynamically "
                        "selects which hop masks to attend through.")
    p.add_argument("--top_k", type=int, default=0,
                   help="Sparse gating: each head keeps only top-k hops. "
                        "0 = dense (full softmax over all K hops). "
                        "Only used when --use_moe_gating is set.")
    p.add_argument("--gate_noise", type=float, default=0.1,
                   help="Gaussian noise std added to gate logits during "
                        "training to encourage exploration. "
                        "Only used when --use_moe_gating is set.")
    p.add_argument("--balance_coeff", type=float, default=0.01,
                   help="Weight for Switch-style load-balancing aux loss. "
                        "Only used when --use_moe_gating is set.")
    p.add_argument("--entropy_coeff", type=float, default=0.01,
                   help="Weight for entropy regularisation aux loss "
                        "(encourages diffuse gate distributions). "
                        "Only used when --use_moe_gating is set.")

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
    train_loader, val_loader, test_loader, _, _, _, dataset_info = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_dist_masks=True,
        max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe,
        lap_pe_dim=args.lap_pe_dim,
        dataset_name=args.dataset,
        return_info=True,
    )
    dataset_name = dataset_info["name"]

    # Optional per-class pos_weight = sqrt(N / (C * n_k)) for multi-label BCE.
    pos_weight = None
    if args.use_pos_weight:
        if dataset_info["task_type"] != "multi_label":
            print(f"--use_pos_weight ignored: task_type="
                  f"{dataset_info['task_type']} is not multi_label.", flush=True)
        else:
            num_classes = dataset_info["output_dim"]
            labels = _gather_train_labels(train_loader, num_classes)
            pw = compute_pos_weight(labels, num_classes)
            pos_weight = torch.as_tensor(pw, dtype=torch.float32, device=args.device)
            print(f"pos_weight (sqrt(N/(C*n_k))): {np.round(pw, 3)}", flush=True)

    task = build_task(dataset_name, dataset_info=dataset_info, pos_weight=pos_weight)
    # Move the loss module (and its pos_weight buffer) onto the device.
    task.loss_fn = task.loss_fn.to(args.device)

    print(f"[{dataset_name}] task={task.task_type} level={task.level} "
          f"metric={task.metric_name} (higher_is_better={task.higher_is_better})",
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
        num_global_heads=args.num_global_heads,
        output_dim=task.output_dim,
        graph_pool=args.graph_pool,
        task_level=task.level,
        dataset_name=dataset_name,
        node_feat_dim=dataset_info.get("node_feat_dim"),
        lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
        block_diag_out=args.block_diag_out,
        dynamic_cross_hop=args.dynamic_cross_hop,
        norm_type=args.norm_type,
        v_head_dim=args.v_head_dim,
        mask_type=args.mask_type,
        adj_self_loops=args.adj_self_loops,
        use_moe_gating=args.use_moe_gating,
        top_k=args.top_k,
        gate_noise=args.gate_noise,
        balance_coeff=args.balance_coeff,
        entropy_coeff=args.entropy_coeff,
        use_virtual_node=args.use_virtual_node,
        num_post_gat_layers=args.num_post_gat_layers,
        num_gat_heads=args.num_gat_heads,
        cross_hop_hop_embedding=args.cross_hop_hop_embedding,
        use_edge_features=args.use_edge_features,
        edge_feat_dim=dataset_info.get("edge_feat_dim"),
    ).to(args.device)

    # Print the head -> hop-set assignment so it's logged for reproducibility.
    if args.use_moe_gating:
        print(f"MoE gating config: top_k={args.top_k}, gate_noise={args.gate_noise}, "
              f"balance_coeff={args.balance_coeff}, entropy_coeff={args.entropy_coeff}",
              flush=True)
    else:
        print("Head -> hop set assignment:", flush=True)
        for h, s in enumerate(model.head_hop_sets):
            print(f"  head {h}: {'GLOBAL (no hop mask)' if s is None else s}", flush=True)
    if args.use_virtual_node:
        print("Virtual node: ENABLED (learnable embedding, visible to all heads)",
              flush=True)
    if args.hop_mode == "alternating":
        print("Alternating hop mode:", flush=True)
        for i, hop_sets in enumerate(model.per_layer_hop_sets):
            label = "even" if i % 2 == 0 else "odd"
            print(f"  layer {i} ({label}):", flush=True)
            for h, s in enumerate(hop_sets):
                print(f"    head {h}: {'GLOBAL' if s is None else s}", flush=True)
    if args.num_post_gat_layers > 0:
        print(f"Post-transformer GATv2: {args.num_post_gat_layers} layer(s), "
              f"{args.num_gat_heads} heads", flush=True)
    if args.dynamic_cross_hop and args.cross_hop_hop_embedding:
        print("Cross-hop mixer: per-head hop embedding ENABLED", flush=True)
    if args.use_edge_features:
        print("Edge features: ENABLED (BondEncoder in node encoder"
              + (" + GATv2" if args.num_post_gat_layers > 0 else "") + ")", flush=True)

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


