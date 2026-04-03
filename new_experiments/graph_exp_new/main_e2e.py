"""
Pipeline B — End-to-End training: generator + transformer jointly from scratch.
No proxy optimization targets. Task loss only (+ optional MMD regularization).

Usage:
    python main_e2e.py --generator score_based --num_proxies 32 --max_epochs 500
    python main_e2e.py --config configs/e2e.yaml --generator gnn_pooling
"""

import argparse
import os
import pickle
import time
import yaml
import numpy as np
import torch
import torch.nn as nn

from data import get_loaders
from models import GraphTransformer
from generators import ScoreBasedGenerator, GNNPoolingGenerator
from metrics import compute_macro_ap
from mmd import mmd_squared


# ================================================================
# CONFIG: argparse + yaml override
# ================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Pipeline B: End-to-End Proxy Training")
    p.add_argument("--config", type=str, default=None, help="Path to yaml config (CLI overrides yaml)")

    # Generator
    p.add_argument("--generator", type=str, default="score_based",
                   choices=["score_based", "gnn_pooling"],
                   help="Generator architecture (flow_matching not supported in e2e)")
    p.add_argument("--num_proxies", type=int, default=64)

    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.3)

    # Generator-specific
    p.add_argument("--gen_hidden_dim", type=int, default=128)
    p.add_argument("--gen_num_layers", type=int, default=3)
    p.add_argument("--gen_num_heads", type=int, default=8)
    p.add_argument("--gen_dropout", type=float, default=0.2)
    # GNN-specific
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--gnn_type", type=str, default="GINE",
                   choices=["GCN", "GIN", "GINE", "GAT"])
    p.add_argument("--pool_types", type=str, nargs="+", default=["mean"])
    p.add_argument("--decode_hidden", type=int, default=256)
    p.add_argument("--decode_layers", type=int, default=3)
    p.add_argument("--idx_emb_dim", type=int, default=128)
    p.add_argument("--decode_mode", type=str, default="shared",
                   choices=["shared", "grouped"])

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)

    # E2E-specific
    p.add_argument("--mmd_lambda", type=float, default=0.01,
                   help="Weight for MMD regularization (0 to disable)")
    p.add_argument("--proxy_warmup_epochs", type=int, default=0,
                   help="Epochs to train transformer without proxies before activating generator")
    p.add_argument("--readout_scope", type=str, default="nodes_only",
                   choices=["nodes_only", "all_tokens"],
                   help="Pool over N original nodes or all N+M tokens")

    # Paths
    p.add_argument("--save_dir", type=str, default="checkpoints_e2e2")
    p.add_argument("--device", type=str, default=None)

    return p


def parse_args():
    parser = build_parser()
    # First pass: check if --config is provided
    preliminary, _ = parser.parse_known_args()
    if preliminary.config is not None:
        with open(preliminary.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        # Set yaml values as defaults, CLI will override
        parser.set_defaults(**yaml_cfg)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


# ================================================================
# MODEL BUILDING
# ================================================================

def build_generator(args):
    if args.generator == "score_based":
        return ScoreBasedGenerator(
            num_proxies=args.num_proxies,
            input_dim=args.hidden_dim,
            hidden_dim=args.gen_hidden_dim,
            num_layers=args.gen_num_layers,
            num_heads=args.gen_num_heads,
            dropout=args.gen_dropout,
        )
    elif args.generator == "gnn_pooling":
        return GNNPoolingGenerator(
            num_proxies=args.num_proxies,
            input_dim=args.hidden_dim,
            gnn_layers=args.gnn_layers,
            gnn_type=args.gnn_type,
            pool_types=tuple(args.pool_types),
            decode_hidden=args.decode_hidden,
            decode_layers=args.decode_layers,
            idx_emb_dim=args.idx_emb_dim,
            dropout=args.gen_dropout,
            decode_mode=args.decode_mode,
        )
    else:
        raise ValueError(f"Unknown generator: {args.generator}")


# ================================================================
# FORWARD PASS
# ================================================================

def forward_e2e(model, generator, batch, args, use_proxies=True):
    """
    Compose: encode -> generate proxies -> transformer with proxies -> classify.

    Returns:
        logits: (B, output_dim)
        mmd_loss: scalar tensor (0 if mmd_lambda == 0 or no proxies)
        node_emb: (total_N, d)
    """
    if not use_proxies:
        logits, node_emb = model(batch, readout_scope=args.readout_scope)
        return logits, torch.tensor(0.0, device=batch.x.device), node_emb

    # Step 1: Encode nodes
    dense_x, dense_mask = model.encode_dense(batch)

    # Step 2: Generate proxies
    if args.generator == "gnn_pooling":
        # GNN generator needs flat node embeddings + graph structure
        flat_node_emb = model.encode_nodes(batch)
        proxy_emb, _ = generator(
            flat_node_emb, mask=None,
            edge_index=batch.edge_index,
            batch_vec=batch.batch,
            edge_attr=batch.edge_attr,
        )
    else:
        # Score-based generator takes dense (B, N, d)
        proxy_emb, _ = generator(dense_x, dense_mask)

    # Step 3: Compute MMD loss (proxies vs node embeddings)
    mmd_loss = torch.tensor(0.0, device=batch.x.device)
    if args.mmd_lambda > 0:
        B = dense_x.shape[0]
        mmd_losses = []
        for i in range(B):
            nodes_i = dense_x[i][dense_mask[i]]  # (N_i, d)
            proxies_i = proxy_emb[i]  # (M, d)
            mmd_losses.append(mmd_squared(proxies_i, nodes_i))
        mmd_loss = torch.stack(mmd_losses).mean()

    # Step 4: Transformer forward with proxies (skip re-encoding via precomputed_dense)
    logits, node_emb = model(
        batch,
        proxy_embeddings=proxy_emb,
        precomputed_dense=(dense_x, dense_mask),
        readout_scope=args.readout_scope,
    )

    return logits, mmd_loss, node_emb


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate(model, generator, loader, device, args, use_proxies=True):
    """Evaluate on a loader. Returns (macro_AP, mean_task_loss, mean_mmd_loss)."""
    model.eval()
    generator.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    all_preds, all_labels = [], []
    task_losses, mmd_losses = [], []

    for batch in loader:
        batch = batch.to(device)
        logits, mmd_loss, _ = forward_e2e(model, generator, batch, args,
                                          use_proxies=use_proxies)
        task_loss = loss_fn(logits, batch.y)
        task_losses.append(task_loss.item())
        mmd_losses.append(mmd_loss.item())
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    ap = compute_macro_ap(y_pred, y_true)
    return ap, float(np.mean(task_losses)), float(np.mean(mmd_losses))


# ================================================================
# TRAINING LOOP
# ================================================================

def run_e2e(args):
    print(f"Pipeline B — End-to-End Training", flush=True)
    print(f"  Generator: {args.generator}", flush=True)
    print(f"  Proxies: {args.num_proxies}, Warmup: {args.proxy_warmup_epochs} epochs", flush=True)
    print(f"  Readout: {args.readout_scope}, MMD lambda: {args.mmd_lambda}", flush=True)
    print(f"  Device: {args.device}", flush=True)

    # Data
    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Model
    model = GraphTransformer(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(args.device)

    generator = build_generator(args).to(args.device)

    total_params = sum(p.numel() for p in model.parameters()) + \
                   sum(p.numel() for p in generator.parameters())
    print(f"  Total parameters: {total_params:,}", flush=True)
    print(f"    Transformer: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"    Generator:   {sum(p.numel() for p in generator.parameters()):,}", flush=True)

    # Single optimizer for everything
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(generator.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(args.save_dir, exist_ok=True)

    # Tracking
    best_val_ap = 0.0
    best_epoch = -1
    patience_counter = 0
    diagnostics = []  # (epoch, train_loss, val_AP) for correlation analysis

    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()
        use_proxies = epoch > args.proxy_warmup_epochs

        # --- Train ---
        model.train()
        generator.train()
        train_task_losses, train_mmd_losses = [], []
        all_train_preds, all_train_labels = [], []

        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()

            logits, mmd_loss, _ = forward_e2e(model, generator, batch, args,
                                              use_proxies=use_proxies)
            task_loss = loss_fn(logits, batch.y)
            total_loss = task_loss + args.mmd_lambda * mmd_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(generator.parameters()),
                args.grad_clip,
            )
            optimizer.step()

            train_task_losses.append(task_loss.item())
            train_mmd_losses.append(mmd_loss.item())
            all_train_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_train_labels.append(batch.y.cpu().numpy())

        train_pred = np.concatenate(all_train_preds, axis=0)
        train_true = np.concatenate(all_train_labels, axis=0)
        train_ap = compute_macro_ap(train_pred, train_true)
        train_task = float(np.mean(train_task_losses))
        train_mmd = float(np.mean(train_mmd_losses))

        # --- Val ---
        val_ap, val_loss, val_mmd = evaluate(
            model, generator, val_loader, args.device, args,
            use_proxies=use_proxies,
        )

        # --- Test ---
        test_ap, test_loss, test_mmd = evaluate(
            model, generator, test_loader, args.device, args,
            use_proxies=use_proxies,
        )

        elapsed = time.time() - epoch_start
        proxies_str = "ON" if use_proxies else "OFF (warmup)"

        # Memory usage
        mem_str = ""
        if args.device.startswith("cuda"):
            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            mem_str = f" mem={mem_mb:.0f}MB"
            torch.cuda.reset_peak_memory_stats()

        # Diagnostic collection
        diagnostics.append({
            "epoch": epoch,
            "train_loss": train_task,
            "train_mmd": train_mmd,
            "train_ap": train_ap,
            "val_loss": val_loss,
            "val_ap": val_ap,
            "test_ap": test_ap,
            "elapsed": elapsed,
        })

        print(
            f"Epoch {epoch:3d}/{args.max_epochs} [{elapsed:.1f}s{mem_str}] proxies={proxies_str} | "
            f"train_loss={train_task:.4f} mmd={train_mmd:.4f} train_AP={train_ap:.4f} | "
            f"val_loss={val_loss:.4f} val_AP={val_ap:.4f} | "
            f"test_loss={test_loss:.4f} test_AP={test_ap:.4f}",
            flush=True,
        )

        # --- Early stopping on val AP ---
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "generator_state": generator.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_ap": val_ap,
                "test_ap": test_ap,
                "args": vars(args),
            }
            save_path = os.path.join(args.save_dir, "best_e2e.pt")
            torch.save(ckpt, save_path)
            print(f"  -> New best val AP={val_ap:.4f} (test AP={test_ap:.4f}), saved to {save_path}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}). "
                      f"Best val AP={best_val_ap:.4f} at epoch {best_epoch}.", flush=True)
                break

    # Save diagnostics
    diag_path = os.path.join(args.save_dir, "e2e_diagnostics.pkl")
    with open(diag_path, "wb") as f:
        pickle.dump(diagnostics, f)

    # Print correlation summary
    if len(diagnostics) > 5:
        losses = [d["train_loss"] for d in diagnostics]
        aps = [d["val_ap"] for d in diagnostics]
        corr = float(np.corrcoef(losses, aps)[0, 1])
        print(f"Diagnostic: train_loss vs val_AP correlation = {corr:.4f}", flush=True)

    print(f"\nTraining complete. Best val AP={best_val_ap:.4f} at epoch {best_epoch}.", flush=True)
    print(f"Diagnostics saved to {diag_path}", flush=True)
    return os.path.join(args.save_dir, "best_e2e.pt")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    args = parse_args()
    run_e2e(args)
