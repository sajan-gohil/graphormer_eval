"""
Pipeline A — Staged training with proxy optimization targets.

Stage 1: Pretrain GraphTransformer → freeze
Stage 2: Per-graph proxy optimization → save target embeddings
Stage 3: Train a generator to predict targets from node embeddings
Stage 4: End-to-end finetune (generator + transformer)

Usage:
    python main_staged.py --stage all --generator score_based
    python main_staged.py --stage 2 --model_path checkpoints_staged/stage1_best.pt
    python main_staged.py --config configs/staged.yaml --stage 3
"""

import argparse
import os
import pickle
import time
from types import SimpleNamespace
import yaml
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import get_loaders, ProxyTargetDataset, collate_with_proxies
from models import GraphTransformer
from generators import (
    ScoreBasedGenerator, FlowMatchingGenerator, GNNPoolingGenerator,
    PMAGenerator, GraphCoarseningGenerator, CrossAttentionRouter,
    MultiPointProxyWrapper,
)
from metrics import compute_macro_ap
from mmd import mmd_squared, cross_sample_moment_loss, prior_moment_loss
from optim_utils import (
    build_grouped_optimizer_and_scheduler,
    build_warmup_cosine_scheduler,
)


# ================================================================
# CONFIG
# ================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Pipeline A: Staged Proxy Training")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--stage", type=str, default="all",
                   choices=["1", "2", "3", "4", "all"])
    p.add_argument("--generator", type=str, default="score_based",
                   choices=["flow_matching", "score_based", "gnn_pooling",
                            "pma", "graph_coarsening"])

    # Paths (for resuming individual stages)
    p.add_argument("--model_path", type=str, default=None,
                   help="Pretrained transformer checkpoint (skip stage 1)")
    p.add_argument("--proxy_pairs_path", type=str, default=None,
                   help="Proxy pairs pickle (skip stage 2)")
    p.add_argument("--generator_path", type=str, default=None,
                   help="Trained generator checkpoint (skip stage 3)")

    # Model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)

    # Laplacian positional encoding
    p.add_argument("--use_lap_pe", action=argparse.BooleanOptionalAction, default=True,
                   help="Add Laplacian eigenvector positional encodings to node features")
    p.add_argument("--lap_pe_dim", type=int, default=8,
                   help="Number of Laplacian eigenvectors for positional encoding")

    # Stage 1
    p.add_argument("--s1_lr", type=float, default=1e-3)
    p.add_argument("--s1_weight_decay", type=float, default=3e-4)
    p.add_argument("--s1_max_epochs", type=int, default=200)
    p.add_argument("--s1_patience", type=int, default=50)
    p.add_argument("--s1_grad_clip", type=float, default=1.0)

    # Stage 2
    p.add_argument("--num_proxies", type=int, default=4)
    p.add_argument("--s2_proxy_lr", type=float, default=5e-2)
    p.add_argument("--s2_num_steps", type=int, default=75)
    p.add_argument("--s2_mmd_lambda", type=float, default=0.01)
    p.add_argument("--s2_cross_moment_lambda", type=float, default=0.5,
                   help="Weight for intra-batch cross-sample moment matching")
    p.add_argument("--s2_prior_moment_lambda", type=float, default=1,
                   help="Weight for fixed-prior moment regularization")
    p.add_argument("--s2_prior_target_var", type=float, default=-1.0,
                   help="Target variance for prior loss; <=0 estimates from train embeddings")
    p.add_argument("--s2_num_restarts", type=int, default=50)
    p.add_argument("--s2_grad_clip", type=float, default=1.0)
    p.add_argument("--s2_loss_threshold", type=float, default=0.01,
                   help="Save proxy if opt_loss < threshold")

    # Stage 3 — Generator training
    p.add_argument("--gen_hidden_dim", type=int, default=256)
    p.add_argument("--gen_num_layers", type=int, default=4)
    p.add_argument("--gen_num_heads", type=int, default=8)
    p.add_argument("--gen_dropout", type=float, default=0.2)
    p.add_argument("--s3_lr", type=float, default=5e-4)
    p.add_argument("--s3_weight_decay", type=float, default=1e-4)
    p.add_argument("--s3_max_epochs", type=int, default=200)
    p.add_argument("--s3_patience", type=int, default=50)
    p.add_argument("--s3_eval_every", type=int, default=1)
    p.add_argument("--s3_grad_clip", type=float, default=1.0)
    p.add_argument("--target_noise_std", type=float, default=0.02)
    # Flow matching specific
    p.add_argument("--denoiser_dim", type=int, default=256)
    p.add_argument("--denoiser_layers", type=int, default=4)
    p.add_argument("--denoiser_heads", type=int, default=8)
    p.add_argument("--euler_steps", type=int, default=1)
    # GNN specific
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--gnn_type", type=str, default="GINE",
                   choices=["GCN", "GIN", "GINE", "GAT"])
    p.add_argument("--pool_types", type=str, nargs="+", default=["mean", "max"])
    p.add_argument("--decode_hidden", type=int, default=256)
    p.add_argument("--decode_layers", type=int, default=4)
    p.add_argument("--idx_emb_dim", type=int, default=64)
    p.add_argument("--decode_mode", type=str, default="shared",
                   choices=["shared", "grouped"])
    # PMA specif
    p.add_argument("--pma_query_mode", type=str, default="farthest_point",
                   choices=["farthest_point", "soft_kmeans"])
    # Graph coarsening specific
    p.add_argument("--coarsen_gnn_type", type=str, default="GIN",
                   choices=["GIN", "GCN"])
    p.add_argument("--coarsen_reg_weight", type=float, default=0.1)
    p.add_argument("--coarsen_reg_type", type=str, default="mincut")

    # Cross-attention routing (N→M→N)
    p.add_argument("--use_cross_attn_routing", action="store_true", default=False,
                   help="Use N→M→N cross-attention routing instead of N+M concat")
    p.add_argument("--num_cross_layers", type=int, default=2,
                   help="Number of N→M→N routing layers (only with --use_cross_attn_routing)")
    p.add_argument("--cross_attn_proxy_self_attn", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Include M×M self-attention within cross-attention routing")

    # Multi-point proxy insertion
    p.add_argument("--proxy_insertion_layers", type=str, default="-1",
                   help="Comma-separated layer indices for multi-point proxy insertion. "
                        "-1 = no insertion (default). 0 = before layer 0 (like current). "
                        "E.g., '0,2,4' for three insertion points.")
    p.add_argument("--separate_proxy_generators", action="store_true", default=False,
                   help="Use independent generator per insertion point (Option B).")
    p.add_argument("--separate_proxy_routers", action="store_true", default=False,
                   help="Use independent router per insertion point.")
    p.add_argument("--proxy_aux_loss_decay", type=float, default=1.0,
                   help="Geometric decay factor for multi-point aux losses.")

    # Stage 4 — Finetune
    p.add_argument("--s4_lr_transformer", type=float, default=1e-5)
    p.add_argument("--s4_lr_generator", type=float, default=1e-4)
    p.add_argument("--s4_max_epochs", type=int, default=200)
    p.add_argument("--s4_patience", type=int, default=50)
    p.add_argument("--s4_grad_clip", type=float, default=1.0)
    p.add_argument("--s4_weight_decay", type=float, default=1e-4)

    # Common
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr_min", type=float, default=1e-7,
                   help="Minimum LR floor for warmup-cosine schedule")
    p.add_argument("--warmup_ratio", type=float, default=0.05,
                   help="Warmup fraction of total optimization steps")
    p.add_argument("--recurrent_lr_factor", type=float, default=1.0,
                   help="LR multiplier for recurrent GRED parameters")
    p.add_argument("--save_dir", type=str, default="checkpoints_staged")
    p.add_argument("--device", type=str, default=None)

    return p


def parse_args():
    parser = build_parser()
    preliminary, _ = parser.parse_known_args()
    if preliminary.config is not None:
        with open(preliminary.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        parser.set_defaults(**yaml_cfg)
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# Save a zip of all code files in save dir for reproducibility
def save_code_snapshot(save_dir):
    import zipfile
    code_files = [f for f in os.listdir(".") if f.endswith(".py")]
    zip_path = os.path.join(save_dir, "code_snapshot.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in code_files:
            zf.write(f)
    print(f"Saved code snapshot to {zip_path}", flush=True)

# ================================================================
# HELPERS
# ================================================================

def _build_cross_attn_router(args):
    """Build CrossAttentionRouter if --use_cross_attn_routing is set."""
    if getattr(args, 'use_cross_attn_routing', False):
        return CrossAttentionRouter(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_cross_layers=args.num_cross_layers,
            dropout=args.dropout,
            use_proxy_self_attn=args.cross_attn_proxy_self_attn,
        )
    return None


def _parse_insertion_layers(s):
    """Parse '--proxy_insertion_layers' string into a sorted list or None."""
    layers = [int(x.strip()) for x in s.split(",")]
    if layers == [-1]:
        return None
    return sorted(layers)


def _build_multi_point_proxy(args, generator, router):
    """Build MultiPointProxyWrapper if multi-point is enabled."""
    layers = _parse_insertion_layers(args.proxy_insertion_layers)
    if layers is None:
        return None
    return MultiPointProxyWrapper(
        generator=generator,
        router=router,
        insertion_layers=layers,
        separate_generators=args.separate_proxy_generators,
        separate_routers=args.separate_proxy_routers,
        aux_loss_decay=args.proxy_aux_loss_decay,
    )


def _build_graph_transformer(args):
    """Build GraphTransformer with optional cross-attention router."""
    return GraphTransformer(
        num_layers=args.num_layers, num_heads=args.num_heads,
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        dropout=args.dropout,
        lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
        cross_attn_router=_build_cross_attn_router(args),
    )


def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)


def _unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)


def _per_sample_bce(logits, labels):
    """Per-sample BCE loss averaged over classes."""
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)


def _count_correct(logits, labels):
    """Per-sample count of correctly predicted classes (threshold 0.5)."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == labels).sum(dim=1)


def _dense_mask_to_batch_vec(dense_mask):
    counts = dense_mask.sum(dim=1).to(torch.long)
    return torch.arange(dense_mask.size(0), device=dense_mask.device).repeat_interleave(counts)


@torch.no_grad()
def _estimate_empirical_proxy_prior_var(model, loader, device):
    """
    Estimate a scalar target variance from encoder node embeddings on train set.
    """
    was_training = model.training
    model.eval()

    total_elems = 0
    sum_x = torch.zeros((), device=device)
    sum_x2 = torch.zeros((), device=device)

    for batch in loader:
        batch = batch.to(device)
        dense_x, dense_mask = model.encode_dense(batch)
        nodes = dense_x[dense_mask]
        if nodes.numel() == 0:
            continue
        sum_x += nodes.sum()
        sum_x2 += nodes.pow(2).sum()
        total_elems += nodes.numel()

    if was_training:
        model.train()

    if total_elems == 0:
        return 1.0

    mean = sum_x / total_elems
    var = (sum_x2 / total_elems) - mean.pow(2)
    return float(torch.clamp(var, min=1e-6).item())


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
    elif args.generator == "flow_matching":
        return FlowMatchingGenerator(
            num_proxies=args.num_proxies,
            node_dim=args.hidden_dim,
            denoiser_dim=args.denoiser_dim,
            denoiser_layers=args.denoiser_layers,
            denoiser_heads=args.denoiser_heads,
            dropout=args.gen_dropout,
            euler_steps=args.euler_steps,
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
    elif args.generator == "pma":
        return PMAGenerator(
            num_proxies=args.num_proxies,
            input_dim=args.hidden_dim,
            num_heads=args.gen_num_heads,
            num_layers=args.gen_num_layers,
            dropout=args.gen_dropout,
            query_mode=args.pma_query_mode,
        )
    elif args.generator == "graph_coarsening":
        return GraphCoarseningGenerator(
            num_proxies=args.num_proxies,
            input_dim=args.hidden_dim,
            gnn_layers=args.gen_num_layers,
            gnn_type=args.coarsen_gnn_type,
            dropout=args.gen_dropout,
            reg_type=args.coarsen_reg_type,
            reg_weight=args.coarsen_reg_weight,
            num_refine_layers=1,
            num_heads=args.gen_num_heads,
        )
    else:
        raise ValueError(f"Unknown generator: {args.generator}")


def generate_proxies(model, generator, batch, args):
    """Generate proxies for a batch. Returns (proxy_emb (B,M,d), dense_x, dense_mask)."""
    dense_x, dense_mask = model.encode_dense(batch)
    if args.generator in ("gnn_pooling", "graph_coarsening"):
        flat_emb = model.encode_nodes(batch)
        proxy_emb = generator.generate(
            flat_emb, mask=None,
            edge_index=batch.edge_index,
            batch_vec=batch.batch,
            edge_attr=batch.edge_attr,
        )
    elif args.generator == "flow_matching":
        proxy_emb = generator.generate(dense_x, dense_mask)
    else:
        proxy_emb = generator.generate(dense_x, dense_mask)
    return proxy_emb, dense_x, dense_mask


@torch.no_grad()
def downstream_eval(model, generator, loader, device, args):
    """Evaluate generated proxies through frozen transformer. Returns (AP, loss)."""
    model.eval()
    generator.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_preds, all_labels, losses = [], [], []

    for batch in loader:
        batch = batch.to(device)

        # Multi-point proxy path: model handles generation internally
        if getattr(model, 'multi_point_proxy', None) is not None:
            logits, _ = model(batch)
        else:
            # Single-point proxy path: generate proxies externally
            proxy_emb, dense_x, dense_mask = generate_proxies(model, generator, batch, args)
            logits, _ = model(batch, proxy_embeddings=proxy_emb,
                              precomputed_dense=(dense_x, dense_mask))

        loss = loss_fn(logits, batch.y)
        losses.append(loss.item())
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    ap = compute_macro_ap(y_pred, y_true)
    return ap, float(np.mean(losses))


# ================================================================
# STAGE 1 — PRETRAIN TRANSFORMER
# ================================================================

def run_stage1(args):
    print("\n" + "=" * 60, flush=True)
    print("STAGE 1: Pretrain Graph Transformer", flush=True)
    print("=" * 60, flush=True)

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    model = _build_graph_transformer(args).to(args.device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    total_steps = max(1, len(train_loader) * args.s1_max_epochs)
    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=[(f"model.{name}", param) for name, param in model.named_parameters()],
        lr_max=args.s1_lr,
        lr_min=args.lr_min,
        weight_decay=args.s1_weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        recurrent_lr_factor=args.recurrent_lr_factor,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_ap = 0.0
    best_epoch = -1
    patience_counter = 0
    save_path = os.path.join(args.save_dir, "stage1_best.pt")

    for epoch in range(1, args.s1_max_epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_losses = []
        all_preds, all_labels = [], []

        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.s1_grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        train_ap = compute_macro_ap(
            np.concatenate(all_preds), np.concatenate(all_labels))
        train_loss = float(np.mean(train_losses))

        # Val
        model.eval()
        val_preds, val_labels, val_losses = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(args.device)
                logits, _ = model(batch)
                val_losses.append(loss_fn(logits, batch.y).item())
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(batch.y.cpu().numpy())
        val_ap = compute_macro_ap(
            np.concatenate(val_preds), np.concatenate(val_labels))
        val_loss = float(np.mean(val_losses))

        elapsed = time.time() - epoch_start
        mem_str = ""
        if args.device.startswith("cuda"):
            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            mem_str = f" mem={mem_mb:.0f}MB"
            torch.cuda.reset_peak_memory_stats()
        log_line = (f"Epoch {epoch:3d}/{args.s1_max_epochs} [{elapsed:.1f}s{mem_str}] | "
                    f"train_loss={train_loss:.4f} train_AP={train_ap:.4f} | "
                    f"val_loss={val_loss:.4f} val_AP={val_ap:.4f}")

        # Test every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            test_preds, test_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(args.device)
                    logits, _ = model(batch)
                    test_preds.append(torch.sigmoid(logits).cpu().numpy())
                    test_labels.append(batch.y.cpu().numpy())
            test_ap = compute_macro_ap(
                np.concatenate(test_preds), np.concatenate(test_labels))
            log_line += f" | test_AP={test_ap:.4f}"

        print(log_line, flush=True)

        # Early stopping
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch, "val_ap": val_ap,
                "args": vars(args),
            }, save_path)
            print(f"  -> New best val AP={val_ap:.4f}, saved", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.s1_patience:
                print(f"Early stopping at epoch {epoch}. "
                      f"Best val AP={best_val_ap:.4f} at epoch {best_epoch}.",
                      flush=True)
                break

    print(f"Stage 1 done. Best val AP={best_val_ap:.4f} at epoch {best_epoch}.",
          flush=True)
    return save_path


# ================================================================
# STAGE 2 — OPTIMIZE PROXY EMBEDDINGS
# ================================================================

def _optimize_batch(model, batch, dense_x, dense_mask, args,
                    init_proxy=None):
    """
    Optimize proxy embeddings for a batch of graphs.
    Returns per-graph: best_proxy, base_loss, best_loss, base_correct,
                       best_correct, best_mmd, best_cross, best_prior.
    """
    B = batch.y.size(0)
    device = batch.y.device

    if init_proxy is None:
        init_proxy = torch.randn(B, args.num_proxies, args.hidden_dim, device=device) * 0.02
    proxy = nn.Parameter(init_proxy.detach().clone())
    opt = torch.optim.Adam([proxy], lr=args.s2_proxy_lr)

    # Base predictions (no proxies)
    with torch.no_grad():
        base_logits, _ = model(batch, precomputed_dense=(dense_x, dense_mask))
        base_loss = _per_sample_bce(base_logits, batch.y)
        base_correct = _count_correct(base_logits, batch.y)

    best_loss = base_loss.clone()
    best_proxy = proxy.detach().clone()
    best_correct = base_correct.clone()
    best_mmd = torch.full((B,), float("inf"), device=device)
    best_cross = torch.full((B,), float("inf"), device=device)
    best_prior = torch.full((B,), float("inf"), device=device)

    for step in range(args.s2_num_steps):
        opt.zero_grad()
        logits, _ = model(batch, proxy_embeddings=proxy,
                          precomputed_dense=(dense_x, dense_mask))
        task_loss = _per_sample_bce(logits, batch.y)

        # Per-graph MMD regularization
        mmd_losses = []
        for i in range(B):
            nodes_i = dense_x[i][dense_mask[i]]
            mmd_losses.append(mmd_squared(proxy[i], nodes_i))
        mmd_batch = torch.stack(mmd_losses)

        # Intra-batch and fixed-prior moment regularization
        cross_batch = cross_sample_moment_loss(proxy)
        prior_batch = prior_moment_loss(
            proxy, target_variance=args.s2_prior_target_var)

        total = (
            task_loss
            + args.s2_mmd_lambda * mmd_batch
            + args.s2_cross_moment_lambda * cross_batch
            + args.s2_prior_moment_lambda * prior_batch
        ).mean()
        total.backward()
        nn.utils.clip_grad_norm_([proxy], args.s2_grad_clip)
        opt.step()

        with torch.no_grad():
            improved = (task_loss < best_loss) & (mmd_batch < best_mmd)
            if (improved).any():
                best_loss[improved] = task_loss[improved]
                best_proxy[improved] = proxy.detach()[improved]
                best_mmd[improved] = mmd_batch.detach()[improved]
                best_cross[improved] = cross_batch.detach()[improved]
                best_prior[improved] = prior_batch.detach()[improved]
                cur_correct = _count_correct(logits, batch.y)
                best_correct[improved] = cur_correct[improved]

    return (
        best_proxy,
        base_loss,
        best_loss,
        base_correct,
        best_correct,
        best_mmd,
        best_cross,
        best_prior,
    )


def run_stage2(args, model_path):
    print("\n" + "=" * 60, flush=True)
    print("STAGE 2: Per-Graph Proxy Optimization", flush=True)
    print(f"  Restarts: {args.s2_num_restarts}, Steps: {args.s2_num_steps}, "
        f"MMD lambda: {args.s2_mmd_lambda}", flush=True)
    print(f"  Cross moment lambda: {args.s2_cross_moment_lambda}, "
        f"Prior moment lambda: {args.s2_prior_moment_lambda}", flush=True)
    print("=" * 60, flush=True)

    train_loader, _, _, train_ds, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    # Load and freeze transformer
    model = _build_graph_transformer(args).to(args.device)
    ckpt = torch.load(model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    _freeze(model)
    model.eval()

    if args.s2_prior_target_var <= 0:
        args.s2_prior_target_var = _estimate_empirical_proxy_prior_var(
            model, train_loader, args.device)
        print(f"  Estimated prior target variance from train embeddings: "
              f"{args.s2_prior_target_var:.6f}", flush=True)
    else:
        print(f"  Using provided prior target variance: "
              f"{args.s2_prior_target_var:.6f}", flush=True)

    proxy_pairs = []
    # Track which samples ever got a valid pair (across restarts)
    sample_improved = set()
    # For AP logging every 1000 samples
    ap_preds, ap_labels = [], []
    samples_since_ap_log = 0

    for restart in range(args.s2_num_restarts):
        processed = 0
        print(f"\n--- Restart {restart + 1}/{args.s2_num_restarts} ---", flush=True)

        for batch in train_loader:
            batch = batch.to(args.device)
            B = batch.y.size(0)

            with torch.no_grad():
                dense_x, dense_mask = model.encode_dense(batch)

            (best_proxy, base_loss, best_loss,
             base_correct, best_correct, best_mmd,
             best_cross, best_prior) = _optimize_batch(
                model, batch, dense_x, dense_mask, args)

            # Extended optimization for non-improved graphs
            not_improved_mask = torch.ones(B, dtype=torch.bool, device=args.device)
            for j in range(B):
                sid = processed + j
                bl = float(base_loss[j])
                ol = float(best_loss[j])
                bc = int(base_correct[j])
                oc = int(best_correct[j])
                if ol < args.s2_loss_threshold or oc > bc:
                    not_improved_mask[j] = False

            # If any graphs didn't meet criteria, run another round of optimization
            n_not_improved = not_improved_mask.sum().item()
            if n_not_improved > 0:
                # Continue from the best proxy found so far, but only for the subset
                # that still needs improvement.
                subset_mask = not_improved_mask
                subset_dense_x = dense_x[subset_mask]
                subset_dense_mask = dense_mask[subset_mask]
                subset_batch = SimpleNamespace(
                    batch=_dense_mask_to_batch_vec(subset_dense_mask),
                    y=batch.y[subset_mask],
                )

                (subset_best_proxy, _, subset_best_loss, _,
                 subset_best_correct, subset_best_mmd,
                 subset_best_cross, subset_best_prior) = _optimize_batch(
                    model,
                    subset_batch,
                    subset_dense_x,
                    subset_dense_mask,
                    args,
                    init_proxy=best_proxy[subset_mask],
                )

                best_proxy[subset_mask] = subset_best_proxy
                best_loss[subset_mask] = subset_best_loss
                best_correct[subset_mask] = subset_best_correct
                best_mmd[subset_mask] = subset_best_mmd
                best_cross[subset_mask] = subset_best_cross
                best_prior[subset_mask] = subset_best_prior

            # Collect predictions for AP logging (using best proxies)
            with torch.no_grad():
                best_logits, _ = model(batch, proxy_embeddings=best_proxy,
                                       precomputed_dense=(dense_x, dense_mask))
                ap_preds.append(torch.sigmoid(best_logits).cpu().numpy())
                ap_labels.append(batch.y.cpu().numpy())

            # Save per-graph results
            for j in range(B):
                sid = processed + j
                bl = float(base_loss[j])
                ol = float(best_loss[j])
                bc = int(base_correct[j])
                oc = int(best_correct[j])

                is_significant = ol < args.s2_loss_threshold or oc > bc
                # For non-improved: save best variant anyway (lowest loss)
                if is_significant or (sid not in sample_improved and ol < bl):
                    if is_significant:
                        sample_improved.add(sid)
                    proxy_pairs.append({
                        "encoder_emb": dense_x[j].cpu(),
                        "mask": dense_mask[j].cpu(),
                        "proxy_emb": best_proxy[j].cpu(),
                        "base_loss": bl,
                        "opt_loss": ol,
                        "mmd_loss": float(best_mmd[j]),
                        "cross_moment_loss": float(best_cross[j]),
                        "prior_moment_loss": float(best_prior[j]),
                        "base_correct": bc,
                        "best_correct": oc,
                        "sample_idx": sid,
                        "restart": restart,
                    })

            processed += B
            samples_since_ap_log += B

            # Log AP every ~1000 samples
            if samples_since_ap_log >= 1000:
                y_pred_chunk = np.concatenate(ap_preds, axis=0)
                y_true_chunk = np.concatenate(ap_labels, axis=0)
                chunk_ap = compute_macro_ap(y_pred_chunk, y_true_chunk)
                print(f"  [restart {restart+1} | {processed} samples] "
                      f"AP (w/ proxies, last {len(y_pred_chunk)} samples)={chunk_ap:.4f} | "
                      f"Pairs saved: {len(proxy_pairs)} | "
                      f"Unique improved: {len(sample_improved)}/{len(train_ds)}",
                      flush=True)
                ap_preds, ap_labels = [], []
                samples_since_ap_log = 0

        save_path = os.path.join(args.save_dir, "proxy_pairs_temp.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(proxy_pairs, f)

    # --- MMD-based filtering ---
    print("\nFiltering by MMD outliers...", flush=True)
    mmd_values = [p["mmd_loss"] for p in proxy_pairs]
    if len(mmd_values) > 0:
        finite_mmd_values = [v for v in mmd_values if np.isfinite(v)]
        if len(finite_mmd_values) == 0:
            print("  All MMD values were NaN or infinite; skipping outlier filtering.",
                flush=True)
        else:
            mmd_mean = float(np.mean(finite_mmd_values))
            mmd_std = float(np.std(finite_mmd_values))
            threshold = mmd_mean + 3 * mmd_std
            before = len(proxy_pairs)
            proxy_pairs = [
                p for p in proxy_pairs
                if np.isfinite(p["mmd_loss"]) and p["mmd_loss"] <= threshold
            ]
            after = len(proxy_pairs)
            print(f"  MMD mean={mmd_mean:.6f} std={mmd_std:.6f} threshold={threshold:.6f}",
                flush=True)
            print(f"  Filtered: {before} -> {after} pairs ({before - after} removed)",
                flush=True)

    # Save
    save_path = os.path.join(args.save_dir, "proxy_pairs.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(proxy_pairs, f)

    unique_samples = len(set(p["sample_idx"] for p in proxy_pairs))
    print(f"\nStage 2 done. Saved {len(proxy_pairs)} pairs "
          f"({unique_samples} unique graphs) -> {save_path}", flush=True)
    return save_path


# ================================================================
# STAGE 3 — TRAIN GENERATOR
# ================================================================

def run_stage3(args, model_path, proxy_pairs_path):
    print("\n" + "=" * 60, flush=True)
    print(f"STAGE 3: Train Generator ({args.generator})", flush=True)
    print("=" * 60, flush=True)

    _, val_loader, test_loader, train_ds, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    # Load and freeze transformer (without multi-point initially)
    model = _build_graph_transformer(args).to(args.device)
    ckpt = torch.load(model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    _freeze(model)
    model.eval()

    # Generator
    generator = build_generator(args).to(args.device)

    # Build and attach multi-point proxy wrapper if enabled
    cross_attn_router = _build_cross_attn_router(args)
    multi_point_proxy = _build_multi_point_proxy(args, generator, cross_attn_router)
    if multi_point_proxy is not None:
        model.multi_point_proxy = multi_point_proxy

    # Proxy target dataset
    proxy_train_ds = ProxyTargetDataset(train_ds, proxy_pairs_path)
    proxy_train_loader = DataLoader(
        proxy_train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_with_proxies, num_workers=args.num_workers,
    )
    print(f"  Training on {len(proxy_train_ds)} proxy-paired graphs", flush=True)

    gen_params = sum(p.numel() for p in generator.parameters())
    print(f"  Generator parameters: {gen_params:,}", flush=True)
    if args.proxy_insertion_layers != "-1":
        print(f"  Multi-point proxy: insertion_layers={args.proxy_insertion_layers}, "
              f"separate_gens={args.separate_proxy_generators}, "
              f"separate_routers={args.separate_proxy_routers}, "
              f"aux_decay={args.proxy_aux_loss_decay}", flush=True)

    total_steps = max(1, len(proxy_train_loader) * args.s3_max_epochs)

    # When multi-point is active, generator is inside model's wrapper
    # We freeze model, then unfreeze only the wrapper's parameters
    if multi_point_proxy is not None:
        # Model is frozen, unfreeze only the wrapper
        for p in model.multi_point_proxy.parameters():
            p.requires_grad_(True)
        optimizer, scheduler = build_grouped_optimizer_and_scheduler(
            named_parameters=[(f"wrapper.{name}", param)
                            for name, param in model.multi_point_proxy.named_parameters()],
            lr_max=args.s3_lr,
            lr_min=args.lr_min,
            weight_decay=args.s3_weight_decay,
            total_steps=total_steps,
            warmup_ratio=args.warmup_ratio,
            recurrent_lr_factor=1.0,
        )
    else:
        optimizer, scheduler = build_grouped_optimizer_and_scheduler(
            named_parameters=[(f"generator.{name}", param) for name, param in generator.named_parameters()],
            lr_max=args.s3_lr,
            lr_min=args.lr_min,
            weight_decay=args.s3_weight_decay,
            total_steps=total_steps,
            warmup_ratio=args.warmup_ratio,
            recurrent_lr_factor=1.0,
        )

    best_val_ap = 0.0
    best_epoch = -1
    patience_counter = 0
    diagnostics = []  # (epoch, gen_loss, val_AP) for correlation analysis
    save_path = os.path.join(args.save_dir, "stage3_generator.pt")

    for epoch in range(1, args.s3_max_epochs + 1):
        epoch_start = time.time()

        # --- Train generator ---
        generator.train()
        train_losses = []

        for pyg_batch, encoder_embs, emb_masks, target_proxies in proxy_train_loader:
            pyg_batch = pyg_batch.to(args.device)
            encoder_embs = encoder_embs.to(args.device)
            emb_masks = emb_masks.to(args.device)
            target_proxies = target_proxies.to(args.device)

            # Optional target noise
            if args.target_noise_std > 0:
                targets = target_proxies + torch.randn_like(target_proxies) * args.target_noise_std
            else:
                targets = target_proxies

            optimizer.zero_grad()

            # Get the active generator (from wrapper if multi-point, else standalone)
            active_gen = model.multi_point_proxy.generators[0] if multi_point_proxy is not None else generator

            # Generate proxies with targets (supervised training)
            if args.generator in ("gnn_pooling", "graph_coarsening"):
                # GNN-based generators need flat embeddings + graph structure
                with torch.no_grad():
                    flat_emb = model.encode_nodes(pyg_batch)
                proxy_emb, aux_loss = active_gen(
                    flat_emb, mask=None, targets=targets,
                    edge_index=pyg_batch.edge_index,
                    batch_vec=pyg_batch.batch,
                    edge_attr=pyg_batch.edge_attr,
                )
            else:
                # Dense-interface generators (score_based, flow_matching, pma)
                proxy_emb, aux_loss = active_gen(encoder_embs, emb_masks, targets=targets)

            if aux_loss is not None:
                # Reconstruction / regularization loss (MMD, CFM, ortho)
                train_loss = aux_loss
            else:
                # PMA has no reconstruction loss — fall back to downstream task loss
                logits, _ = model(pyg_batch, proxy_embeddings=proxy_emb,
                                  precomputed_dense=(encoder_embs, emb_masks))
                train_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, pyg_batch.y)

            train_loss.backward()
            # Clip gradients for both multi-point and single-point cases
            all_params = (list(model.multi_point_proxy.parameters()) if multi_point_proxy is not None
                         else list(generator.parameters()))
            nn.utils.clip_grad_norm_(all_params, args.s3_grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(train_loss.item())

        mean_train_loss = float(np.mean(train_losses))

        # --- Downstream evaluation ---
        if epoch % args.s3_eval_every == 0:
            val_ap, val_loss = downstream_eval(
                model, generator, val_loader, args.device, args)
            test_ap, test_loss = downstream_eval(
                model, generator, test_loader, args.device, args)

            elapsed = time.time() - epoch_start
            mem_str = ""
            if args.device.startswith("cuda"):
                mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                mem_str = f" mem={mem_mb:.0f}MB"
                torch.cuda.reset_peak_memory_stats()

            diagnostics.append({
                "epoch": epoch,
                "gen_train_loss": mean_train_loss,
                "downstream_val_ap": val_ap,
                "downstream_test_ap": test_ap,
                "elapsed": elapsed,
            })

            print(
                f"Epoch {epoch:3d}/{args.s3_max_epochs} [{elapsed:.1f}s{mem_str}] | "
                f"gen_loss={mean_train_loss:.4f} | "
                f"val_AP={val_ap:.4f} val_loss={val_loss:.4f} | "
                f"test_AP={test_ap:.4f}",
                flush=True,
            )

            # Early stopping on downstream val AP
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "generator_state": generator.state_dict(),
                    "epoch": epoch, "val_ap": val_ap, "test_ap": test_ap,
                    "args": vars(args),
                }, save_path)
                print(f"  -> New best downstream val AP={val_ap:.4f}", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= args.s3_patience:
                    print(f"Early stopping at epoch {epoch}. "
                          f"Best val AP={best_val_ap:.4f} at epoch {best_epoch}.",
                          flush=True)
                    break
        else:
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch:3d}/{args.s3_max_epochs} [{elapsed:.1f}s] | "
                  f"gen_loss={mean_train_loss:.4f}", flush=True)

    # Save diagnostics
    diag_path = os.path.join(args.save_dir, "stage3_diagnostics.pkl")
    with open(diag_path, "wb") as f:
        pickle.dump(diagnostics, f)

    # Correlation summary
    if len(diagnostics) > 5:
        losses = [d["gen_train_loss"] for d in diagnostics]
        aps = [d["downstream_val_ap"] for d in diagnostics]
        corr = float(np.corrcoef(losses, aps)[0, 1])
        print(f"Diagnostic: gen_loss vs downstream_val_AP correlation = {corr:.4f}",
              flush=True)

    print(f"Stage 3 done. Best downstream val AP={best_val_ap:.4f} at epoch {best_epoch}.",
          flush=True)
    print(f"Diagnostics saved to {diag_path}", flush=True)
    return save_path


# ================================================================
# STAGE 4 — END-TO-END FINETUNE
# ================================================================

def run_stage4(args, model_path, generator_path):
    print("\n" + "=" * 60, flush=True)
    print("STAGE 4: End-to-End Finetune", flush=True)
    print(f"  LR transformer={args.s4_lr_transformer}, "
          f"LR generator={args.s4_lr_generator}", flush=True)
    print("=" * 60, flush=True)

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    # Load transformer from stage 1
    model = _build_graph_transformer(args).to(args.device)
    model_ckpt = torch.load(model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(model_ckpt["model_state"])

    # Load generator from stage 3
    generator = build_generator(args).to(args.device)
    gen_ckpt = torch.load(generator_path, map_location=args.device, weights_only=True)
    generator.load_state_dict(gen_ckpt["generator_state"])

    # Build and attach multi-point proxy wrapper if enabled
    cross_attn_router = _build_cross_attn_router(args)
    multi_point_proxy = _build_multi_point_proxy(args, generator, cross_attn_router)
    if multi_point_proxy is not None:
        model.multi_point_proxy = multi_point_proxy
        print(f"  Multi-point proxy: insertion_layers={args.proxy_insertion_layers}, "
              f"separate_gens={args.separate_proxy_generators}, "
              f"separate_routers={args.separate_proxy_routers}, "
              f"aux_decay={args.proxy_aux_loss_decay}", flush=True)

    # Unfreeze everything
    _unfreeze(model)
    _unfreeze(generator)

    # Differential LR groups
    # When multi-point is active, generator is inside model, so don't add generator separately
    param_groups = [
        {"params": model.encoder.parameters(), "lr": args.s4_lr_transformer},
        {"params": model.layers.parameters(), "lr": args.s4_lr_transformer},
        {"params": model.head.parameters(), "lr": args.s4_lr_transformer},
    ]
    if getattr(model, 'multi_point_proxy', None) is None:
        param_groups.append({"params": generator.parameters(), "lr": args.s4_lr_generator})
    else:
        # Multi-point wrapper is inside model, so its params are already covered above
        # But we add it explicitly to ensure it gets the generator LR
        param_groups.append({"params": model.multi_point_proxy.parameters(), "lr": args.s4_lr_generator})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.s4_weight_decay)
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=max(1, len(train_loader) * args.s4_max_epochs),
        lr_min=args.lr_min,
        warmup_ratio=args.warmup_ratio,
    )

    loss_fn = nn.BCEWithLogitsLoss()

    best_val_ap = 0.0
    best_epoch = -1
    patience_counter = 0
    save_path = os.path.join(args.save_dir, "stage4_best.pt")

    for epoch in range(1, args.s4_max_epochs + 1):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        generator.train()
        train_losses = []
        all_preds, all_labels = [], []

        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()

            # Multi-point proxy path: model handles generation internally
            if getattr(model, 'multi_point_proxy', None) is not None:
                logits, _ = model(batch)
                loss = loss_fn(logits, batch.y)
                # Optionally add auxiliary loss from multi-point proxy
                aux_loss = getattr(model, '_last_mp_aux_loss', None)
                if aux_loss is not None:
                    if not isinstance(aux_loss, torch.Tensor):
                        aux_loss = torch.tensor(aux_loss, device=batch.x.device)
                    # aux_loss is already weighted by decay factors; add it
                    loss = loss + aux_loss
            else:
                # Single-point proxy path: generate proxies externally
                # Encode once
                dense_x, dense_mask = model.encode_dense(batch)

                # Generate proxies (differentiable)
                if args.generator in ("gnn_pooling", "graph_coarsening"):
                    flat_emb = model.encode_nodes(batch)
                    proxy_emb, _ = generator(
                        flat_emb, mask=None,
                        edge_index=batch.edge_index,
                        batch_vec=batch.batch,
                        edge_attr=batch.edge_attr,
                    )
                elif args.generator == "flow_matching":
                    # Single Euler step, fully differentiable
                    proxy_emb = generator.forward_differentiable(
                        dense_x, dense_mask, euler_steps=args.euler_steps)
                else:
                    proxy_emb, _ = generator(dense_x, dense_mask)

                # Forward through transformer with proxies
                logits, _ = model(batch, proxy_embeddings=proxy_emb,
                                  precomputed_dense=(dense_x, dense_mask))

                # Task loss only — no proxy matching loss
                loss = loss_fn(logits, batch.y)

            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(generator.parameters()),
                args.s4_grad_clip,
            )
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        train_ap = compute_macro_ap(
            np.concatenate(all_preds), np.concatenate(all_labels))
        train_loss = float(np.mean(train_losses))

        # --- Val ---
        val_ap, val_loss = downstream_eval(
            model, generator, val_loader, args.device, args)

        # --- Test ---
        test_ap, test_loss = downstream_eval(
            model, generator, test_loader, args.device, args)

        elapsed = time.time() - epoch_start
        mem_str = ""
        if args.device.startswith("cuda"):
            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            mem_str = f" mem={mem_mb:.0f}MB"
            torch.cuda.reset_peak_memory_stats()
        print(
            f"Epoch {epoch:3d}/{args.s4_max_epochs} [{elapsed:.1f}s{mem_str}] | "
            f"train_loss={train_loss:.4f} train_AP={train_ap:.4f} | "
            f"val_loss={val_loss:.4f} val_AP={val_ap:.4f} | "
            f"test_AP={test_ap:.4f}",
            flush=True,
        )

        # Early stopping on val AP
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "generator_state": generator.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "val_ap": val_ap, "test_ap": test_ap,
                "args": vars(args),
            }, save_path)
            print(f"  -> New best val AP={val_ap:.4f} (test={test_ap:.4f})",
                  flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.s4_patience:
                print(f"Early stopping at epoch {epoch}. "
                      f"Best val AP={best_val_ap:.4f} at epoch {best_epoch}.",
                      flush=True)
                break

    print(f"Stage 4 done. Best val AP={best_val_ap:.4f} at epoch {best_epoch}.",
          flush=True)
    return save_path


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    save_code_snapshot(args.save_dir)

    stages = [args.stage] if args.stage != "all" else ["1", "2", "3", "4"]
    model_path = args.model_path
    proxy_pairs_path = args.proxy_pairs_path
    generator_path = args.generator_path

    for stage in stages:
        if stage == "1":
            model_path = run_stage1(args)

        elif stage == "2":
            if model_path is None:
                model_path = os.path.join(args.save_dir, "stage1_best.pt")
            assert os.path.exists(model_path), \
                f"Stage 2 requires pretrained model at {model_path}"
            proxy_pairs_path = run_stage2(args, model_path)

        elif stage == "3":
            if model_path is None:
                model_path = os.path.join(args.save_dir, "stage1_best.pt")
            if proxy_pairs_path is None:
                proxy_pairs_path = os.path.join(args.save_dir, "proxy_pairs.pkl")
            assert os.path.exists(model_path), \
                f"Stage 3 requires pretrained model at {model_path}"
            assert os.path.exists(proxy_pairs_path), \
                f"Stage 3 requires proxy pairs at {proxy_pairs_path}"
            generator_path = run_stage3(args, model_path, proxy_pairs_path)

        elif stage == "4":
            if model_path is None:
                model_path = os.path.join(args.save_dir, "stage1_best.pt")
            if generator_path is None:
                generator_path = os.path.join(args.save_dir, "stage3_generator.pt")
            assert os.path.exists(model_path), \
                f"Stage 4 requires pretrained model at {model_path}"
            assert os.path.exists(generator_path), \
                f"Stage 4 requires trained generator at {generator_path}"
            run_stage4(args, model_path, generator_path)

    print("\nAll requested stages complete.", flush=True)
