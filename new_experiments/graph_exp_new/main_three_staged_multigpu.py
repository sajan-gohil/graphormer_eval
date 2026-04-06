"""
Three-Staged Pipeline  generator trained directly on task loss.

Stage 1: Pretrain GraphTransformer  freeze (identical to main_staged.py)
Stage 2: Train generator on task loss through frozen transformer (NEW)
Stage 3: End-to-end fine-tune with Phase A (frozen transformer, reduced LR)
         + Phase B (unfreeze transformer at 0.1x gen LR) + proxy dropout

Usage:
    python main_three_staged_multigpu.py --stage all --generator score_based
    python main_three_staged_multigpu.py --stage 2 --model_path checkpoints_three_staged/stage1_best.pt
    python main_three_staged_multigpu.py --stage 3 --model_path ... --generator_path ...
    torchrun --standalone --nproc_per_node=2 main_three_staged_multigpu.py --stage all --generator score_based
"""

import argparse
import os
import pickle
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from data import get_loaders
from models import GraphTransformer
from generators import (
    ScoreBasedGenerator, GNNPoolingGenerator, PMAGenerator, GraphCoarseningGenerator,
)
from metrics import compute_macro_ap


# ================================================================
# CONFIG
# ================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Three-Staged Pipeline: task-loss generator training")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--stage", type=str, default="all",
                   choices=["1", "2", "3", "all"])
    p.add_argument("--generator", type=str, default="score_based",
                   choices=["score_based", "pma", "graph_coarsening", "gnn_pooling"])

    # Paths (for resuming individual stages)
    p.add_argument("--model_path", type=str, default=None,
                   help="Pretrained transformer checkpoint (skip stage 1)")
    p.add_argument("--generator_path", type=str, default=None,
                   help="Trained generator checkpoint (skip stage 2)")

    # Transformer model
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num_proxies", type=int, default=32)

    # Stage 1  Pretrain transformer
    p.add_argument("--s1_lr", type=float, default=1e-3)
    p.add_argument("--s1_weight_decay", type=float, default=3e-4)
    p.add_argument("--s1_max_epochs", type=int, default=500)
    p.add_argument("--s1_patience", type=int, default=5)
    p.add_argument("--s1_grad_clip", type=float, default=1.0)

    # Stage 2 - Train generator on task loss
    p.add_argument("--s2_lr", type=float, default=5e-4)
    p.add_argument("--s2_weight_decay", type=float, default=1e-4)
    p.add_argument("--s2_max_epochs", type=int, default=500)
    p.add_argument("--s2_patience", type=int, default=30)
    p.add_argument("--s2_eval_every", type=int, default=1)
    p.add_argument("--s2_grad_clip", type=float, default=1.0)

    # Stage 3 - End-to-end finetune
    p.add_argument("--s3_phase_a_epochs", type=int, default=20,
                   help="Epochs to keep transformer frozen before Phase B")
    p.add_argument("--s3_lr_gen", type=float, default=None,
                   help="Generator LR for stage 3 (default: 0.1 * s2_lr)")
    p.add_argument("--s3_lr_transformer", type=float, default=None,
                   help="Transformer LR for Phase B (default: 0.1 * s3_lr_gen)")
    p.add_argument("--s3_proxy_dropout", type=float, default=0.3,
                   help="Fraction of batches that train without proxies")
    p.add_argument("--s3_max_epochs", type=int, default=200)
    p.add_argument("--s3_patience", type=int, default=20)
    p.add_argument("--s3_grad_clip", type=float, default=1.0)
    p.add_argument("--s3_weight_decay", type=float, default=1e-4)

    # Generator architecture
    p.add_argument("--gen_hidden_dim", type=int, default=128)
    p.add_argument("--gen_num_layers", type=int, default=2)
    p.add_argument("--gen_num_heads", type=int, default=4)
    p.add_argument("--gen_dropout", type=float, default=0.2)
    # PMA specific
    p.add_argument("--pma_query_mode", type=str, default="farthest_point",
                   choices=["farthest_point", "soft_kmeans"])
    # Graph coarsening specific
    p.add_argument("--coarsen_gnn_type", type=str, default="GIN",
                   choices=["GIN", "GCN"])
    p.add_argument("--coarsen_reg_weight", type=float, default=0.1)
    p.add_argument("--coarsen_reg_type", type=str, default="mincut")
    # GNN pooling specific (kept for ablation)
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--gnn_type", type=str, default="GINE",
                   choices=["GCN", "GIN", "GINE", "GAT"])
    p.add_argument("--pool_types", type=str, nargs="+", default=["mean", "max", "std"])
    p.add_argument("--decode_hidden", type=int, default=256)
    p.add_argument("--decode_layers", type=int, default=3)
    p.add_argument("--idx_emb_dim", type=int, default=32)
    p.add_argument("--decode_mode", type=str, default="shared",
                   choices=["shared", "grouped"])

    # Common
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_dir", type=str, default="checkpoints_three_staged")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ddp", action="store_true",
                   help="Enable DistributedDataParallel (use torchrun for multi-GPU)")

    return p


def parse_args():
    parser = build_parser()
    preliminary, _ = parser.parse_known_args()
    if preliminary.config is not None:
        with open(preliminary.config, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        parser.set_defaults(**yaml_cfg)
    args = parser.parse_args()

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.world_size = env_world_size
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = args.world_size > 1

    if args.ddp and args.world_size == 1:
        raise RuntimeError(
            "--ddp was set but WORLD_SIZE=1. Launch with torchrun, e.g. "
            "torchrun --standalone --nproc_per_node=2 main_three_staged_multigpu.py ..."
        )

    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA GPUs. Launch with torchrun on a CUDA machine.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"
    else:
        if args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Derive Stage 3 LRs from Stage 2 LR if not set
    if args.s3_lr_gen is None:
        args.s3_lr_gen = args.s2_lr * 0.1
    if args.s3_lr_transformer is None:
        args.s3_lr_transformer = args.s3_lr_gen * 0.1
    return args


def save_code_snapshot(save_dir):
    import zipfile
    code_files = [f for f in os.listdir(".") if f.endswith(".py")]
    zip_path = os.path.join(save_dir, "code_snapshot.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in code_files:
            zf.write(f)
    print(f"Saved code snapshot to {zip_path}", flush=True)


def cleanup_distributed(args):
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


# ================================================================
# HELPERS
# ================================================================

def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)


def _unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)


def _uses_flat_interface(generator_name):
    """True for generators that need flat (total_N, d) embeddings + edge_index."""
    return generator_name in ("graph_coarsening", "gnn_pooling")


def _is_main_process(args):
    return (not args.distributed) or (args.rank == 0)


def _print0(args, msg):
    if _is_main_process(args):
        print(msg, flush=True)


def _unwrap(module):
    return module.module if isinstance(module, DDP) else module


def _all_gather_np(arr, args):
    if not args.distributed:
        return arr
    gathered = [None for _ in range(args.world_size)]
    dist.all_gather_object(gathered, arr)
    return np.concatenate(gathered, axis=0)


def _reduce_mean(value, args):
    if not args.distributed:
        return float(value)
    t = torch.tensor([float(value)], device=args.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / args.world_size).item()


def _reduce_loss_mean(local_sum, local_count, args):
    if not args.distributed:
        return float(local_sum / max(local_count, 1))
    t = torch.tensor([float(local_sum), float(local_count)], device=args.device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    denom = max(t[1].item(), 1.0)
    return (t[0].item() / denom)


def _build_loaders(args):
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    if not args.distributed:
        return train_loader, val_loader, test_loader, None, None, None

    train_sampler = DistributedSampler(
        train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False
    )
    test_sampler = DistributedSampler(
        test_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False
    )

    pin_mem = str(args.device).startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    return train_loader, val_loader, test_loader, train_sampler, val_sampler, test_sampler


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


def _generate_proxies(model, generator, batch, dense_x, dense_mask, args):
    """
    Generate proxy embeddings for a batch.
    dense_x and dense_mask must be precomputed (possibly inside no_grad).
    Returns proxy_embeddings (B, M, d).
    """
    if _uses_flat_interface(args.generator):
        flat_emb = dense_x[dense_mask]  # reconstruct flat from dense
        proxies, aux_loss = generator(
            flat_emb, mask=None,
            edge_index=batch.edge_index,
            batch_vec=batch.batch,
            edge_attr=getattr(batch, "edge_attr", None),
        )
    else:
        proxies, aux_loss = generator(dense_x, dense_mask)
    return proxies, aux_loss


@torch.no_grad()
def downstream_eval(model, generator, loader, device, args):
    """Evaluate generated proxies through frozen transformer. Returns (AP, loss)."""
    model.eval()
    generator.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_preds, all_labels = [], []
    loss_sum = 0.0
    loss_count = 0
    model_base = _unwrap(model)

    for batch in loader:
        batch = batch.to(device)
        dense_x, dense_mask = model_base.encode_dense(batch)
        proxies, _ = _generate_proxies(model, generator, batch, dense_x, dense_mask, args)
        logits, _ = model(batch, proxy_embeddings=proxies,
                          precomputed_dense=(dense_x, dense_mask))
        loss_sum += loss_fn(logits, batch.y).item()
        loss_count += 1
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    pred_np = np.concatenate(all_preds, axis=0)
    label_np = np.concatenate(all_labels, axis=0)
    pred_np = _all_gather_np(pred_np, args)
    label_np = _all_gather_np(label_np, args)

    ap = compute_macro_ap(pred_np, label_np)
    loss = _reduce_loss_mean(loss_sum, loss_count, args)
    return ap, loss


@torch.no_grad()
def mean_proxy_eval(model, generator, loader, device, args):
    """
    Sanity check: replace each proxy with the mean over the M dimension,
    then evaluate AP. If AP doesn't drop vs normal eval, proxies are inert.
    """
    model.eval()
    generator.eval()
    all_preds, all_labels = [], []
    model_base = _unwrap(model)

    for batch in loader:
        batch = batch.to(device)
        dense_x, dense_mask = model_base.encode_dense(batch)
        proxies, _ = _generate_proxies(model, generator, batch, dense_x, dense_mask, args)
        # Replace proxies with their mean (collapse to virtual-node equivalent)
        mean_p = proxies.mean(dim=1, keepdim=True).expand_as(proxies)
        logits, _ = model(batch, proxy_embeddings=mean_p,
                          precomputed_dense=(dense_x, dense_mask))
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    pred_np = np.concatenate(all_preds, axis=0)
    label_np = np.concatenate(all_labels, axis=0)
    pred_np = _all_gather_np(pred_np, args)
    label_np = _all_gather_np(label_np, args)
    return compute_macro_ap(pred_np, label_np)


# ================================================================
# STAGE 1 - PRETRAIN TRANSFORMER (identical to main_staged.py)
# ================================================================

def run_stage1(args):
    _print0(args, "\n" + "=" * 60)
    _print0(args, "STAGE 1: Pretrain Graph Transformer")
    _print0(args, "=" * 60)

    train_loader, val_loader, test_loader, train_sampler, _, _ = _build_loaders(args)

    model = GraphTransformer(
        num_layers=args.num_layers, num_heads=args.num_heads,
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(args.device)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    _print0(args, f"  Parameters: {sum(p.numel() for p in _unwrap(model).parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.s1_lr,
                                  weight_decay=args.s1_weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_ap = 0.0
    best_epoch = -1
    patience_counter = 0
    save_path = os.path.join(args.save_dir, "stage1_best.pt")

    for epoch in range(1, args.s1_max_epochs + 1):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = 0.0
        train_loss_count = 0
        all_preds, all_labels = [], []

        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.s1_grad_clip)
            optimizer.step()
            train_loss_sum += loss.item()
            train_loss_count += 1
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        train_preds = _all_gather_np(np.concatenate(all_preds), args)
        train_labels = _all_gather_np(np.concatenate(all_labels), args)
        train_ap = compute_macro_ap(train_preds, train_labels)
        train_loss = _reduce_loss_mean(train_loss_sum, train_loss_count, args)

        model.eval()
        val_preds, val_labels = [], []
        val_loss_sum = 0.0
        val_loss_count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(args.device)
                logits, _ = model(batch)
                val_loss_sum += loss_fn(logits, batch.y).item()
                val_loss_count += 1
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(batch.y.cpu().numpy())
        val_preds = _all_gather_np(np.concatenate(val_preds), args)
        val_labels = _all_gather_np(np.concatenate(val_labels), args)
        val_ap = compute_macro_ap(val_preds, val_labels)
        val_loss = _reduce_loss_mean(val_loss_sum, val_loss_count, args)

        elapsed = time.time() - epoch_start
        mem_str = ""
        if str(args.device).startswith("cuda"):
            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            mem_str = f" mem={mem_mb:.0f}MB"
            torch.cuda.reset_peak_memory_stats()
        log_line = (f"Epoch {epoch:3d}/{args.s1_max_epochs} [{elapsed:.1f}s{mem_str}] | "
                    f"train_loss={train_loss:.4f} train_AP={train_ap:.4f} | "
                    f"val_loss={val_loss:.4f} val_AP={val_ap:.4f}")

        if epoch % 10 == 0:
            model.eval()
            test_preds, test_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(args.device)
                    logits, _ = model(batch)
                    test_preds.append(torch.sigmoid(logits).cpu().numpy())
                    test_labels.append(batch.y.cpu().numpy())
            test_preds = _all_gather_np(np.concatenate(test_preds), args)
            test_labels = _all_gather_np(np.concatenate(test_labels), args)
            test_ap = compute_macro_ap(test_preds, test_labels)
            log_line += f" | test_AP={test_ap:.4f}"

        _print0(args, log_line)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            patience_counter = 0
            if _is_main_process(args):
                torch.save({
                    "model_state": _unwrap(model).state_dict(),
                    "epoch": epoch, "val_ap": val_ap,
                    "args": vars(args),
                }, save_path)
            _print0(args, f"  -> New best val AP={val_ap:.4f}, saved")
        else:
            patience_counter += 1
            if patience_counter >= args.s1_patience:
                _print0(args, f"Early stopping at epoch {epoch}. "
                           f"Best val AP={best_val_ap:.4f} at epoch {best_epoch}.")
                break

    _print0(args, f"Stage 1 done. Best val AP={best_val_ap:.4f} at epoch {best_epoch}.")
    return save_path


# ================================================================
# STAGE 2 - TRAIN GENERATOR ON TASK LOSS (NEW)
# ================================================================

def _proxy_cosine_sim(proxies):
    """Mean pairwise cosine similarity among M proxies (off-diagonal). Scalar."""
    B, M, d = proxies.shape
    if M < 2:
        return 0.0
    with torch.no_grad():
        pn = F.normalize(proxies.detach(), dim=-1)         # (B, M, d)
        cos = torch.bmm(pn, pn.transpose(1, 2))            # (B, M, M)
        mask_diag = ~torch.eye(M, dtype=torch.bool, device=proxies.device)
        return cos[:, mask_diag].mean().item()


def _attention_entropy(generator, dense_x, dense_mask):
    """
    Per-epoch attention weight entropy for ScoreBasedGenerator.
    Higher entropy = more uniform attention (virtual-node averaging).
    Lower entropy = peaked attention (selective).
    """
    with torch.no_grad():
        S = generator.score_mlp(dense_x)                   # (B, N, M)
        S = S.masked_fill(~dense_mask.unsqueeze(-1), float("-inf"))
        A = F.softmax(S, dim=1)                             # (B, N, M)
        A = torch.nan_to_num(A, nan=0.0)
        # Entropy per proxy: -sum_n A_n * log(A_n + eps)
        entropy = -(A * (A + 1e-8).log()).sum(dim=1).mean()
        return entropy.item()


def run_stage2(args, model_path):
    _print0(args, "\n" + "=" * 60)
    _print0(args, f"STAGE 2: Train Generator ({args.generator}) on Task Loss")
    _print0(args, f"  LR={args.s2_lr}, patience={args.s2_patience}, max_epochs={args.s2_max_epochs}")
    _print0(args, "=" * 60)

    train_loader, val_loader, test_loader, train_sampler, _, _ = _build_loaders(args)

    # Load and freeze transformer
    model = GraphTransformer(
        num_layers=args.num_layers, num_heads=args.num_heads,
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(args.device)
    ckpt = torch.load(model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    _freeze(model)
    model.eval()

    generator = build_generator(args).to(args.device)
    if args.distributed:
        generator = DDP(generator, device_ids=[args.local_rank], output_device=args.local_rank)

    _print0(args, f"  Generator parameters: {sum(p.numel() for p in _unwrap(generator).parameters()):,}")

    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.s2_lr,
                                  weight_decay=args.s2_weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_ap = 0.0
    best_epoch = -1
    patience_counter = 0
    diagnostics = []
    save_path = os.path.join(args.save_dir, "stage2_generator.pt")

    for epoch in range(1, args.s2_max_epochs + 1):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        generator.train()
        train_loss_sum = 0.0
        train_loss_count = 0
        epoch_cos_sims = []
        epoch_grad_norms = []
        last_dense_x = last_dense_mask = None  # for entropy diagnostic
        model_base = _unwrap(model)

        for batch in train_loader:
            batch = batch.to(args.device)

            # Encode with frozen transformer (no grad through encoder)
            with torch.no_grad():
                dense_x, dense_mask = model_base.encode_dense(batch)

            optimizer.zero_grad()

            proxies, aux_loss = _generate_proxies(
                model, generator, batch, dense_x, dense_mask, args)

            # Forward through frozen transformer with generated proxies
            logits, _ = model(batch, proxy_embeddings=proxies,
                              precomputed_dense=(dense_x, dense_mask))

            loss = loss_fn(logits, batch.y)
            # Add aux_loss if any (e.g., orthogonality regularization for graph_coarsening)
            if aux_loss is not None:
                loss = loss + aux_loss

            loss.backward()

            # Gradient norm diagnostic (before clipping)
            grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in generator.parameters() if p.grad is not None
            ) ** 0.5
            epoch_grad_norms.append(grad_norm)

            nn.utils.clip_grad_norm_(generator.parameters(), args.s2_grad_clip)
            optimizer.step()

            train_loss_sum += loss.item()
            train_loss_count += 1
            epoch_cos_sims.append(_proxy_cosine_sim(proxies))

            # Save tensors for attention entropy (score_based only, last batch)
            if args.generator == "score_based":
                last_dense_x = dense_x.detach()
                last_dense_mask = dense_mask.detach()

        mean_train_loss = _reduce_loss_mean(train_loss_sum, train_loss_count, args)
        mean_cos_sim = _reduce_mean(float(np.mean(epoch_cos_sims)), args)
        mean_grad_norm = _reduce_mean(float(np.mean(epoch_grad_norms)), args)

        if args.generator == "score_based" and last_dense_x is not None:
            attn_entropy = _attention_entropy(_unwrap(generator), last_dense_x, last_dense_mask)
        else:
            attn_entropy = float("nan")
        if not np.isnan(attn_entropy):
            attn_entropy = _reduce_mean(attn_entropy, args)

        # Downstream evaluation
        if epoch % args.s2_eval_every == 0:
            val_ap, val_loss = downstream_eval(model, generator, val_loader, args.device, args)
            test_ap, _ = downstream_eval(model, generator, test_loader, args.device, args)

            # Sanity check: mean-proxy AP
            mean_ap = mean_proxy_eval(model, generator, val_loader, args.device, args)
            proxy_delta = val_ap - mean_ap  # positive = proxies add value

            elapsed = time.time() - epoch_start
            mem_str = ""
            if args.device.startswith("cuda"):
                mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                mem_str = f" mem={mem_mb:.0f}MB"
                torch.cuda.reset_peak_memory_stats()

            _print0(args, (
                f"Epoch {epoch:3d}/{args.s2_max_epochs} [{elapsed:.1f}s{mem_str}] | "
                f"task_loss={mean_train_loss:.4f} | "
                f"val_AP={val_ap:.4f} mean_AP={mean_ap:.4f} delta={proxy_delta:+.4f} | "
                f"test_AP={test_ap:.4f} | "
                f"cos_sim={mean_cos_sim:.4f} grad_norm={mean_grad_norm:.4f} "
                f"attn_entr={attn_entropy:.4f}"
            ))

            if _is_main_process(args):
                diagnostics.append({
                    "epoch": epoch,
                    "task_loss": mean_train_loss,
                    "val_ap": val_ap,
                    "test_ap": test_ap,
                    "mean_proxy_val_ap": mean_ap,
                    "proxy_delta": proxy_delta,
                    "cos_sim": mean_cos_sim,
                    "grad_norm": mean_grad_norm,
                    "attn_entropy": attn_entropy,
                })

            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_epoch = epoch
                patience_counter = 0
                if _is_main_process(args):
                    torch.save({
                        "generator_state": _unwrap(generator).state_dict(),
                        "epoch": epoch, "val_ap": val_ap, "test_ap": test_ap,
                        "args": vars(args),
                    }, save_path)
                _print0(args, f"  -> New best val AP={val_ap:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= args.s2_patience:
                    _print0(args, f"Early stopping at epoch {epoch}. "
                               f"Best val AP={best_val_ap:.4f} at epoch {best_epoch}.")
                    break
        else:
            elapsed = time.time() - epoch_start
            _print0(args, f"Epoch {epoch:3d}/{args.s2_max_epochs} [{elapsed:.1f}s] | "
                       f"task_loss={mean_train_loss:.4f} | "
                       f"cos_sim={mean_cos_sim:.4f} grad_norm={mean_grad_norm:.4f}")

    diag_path = os.path.join(args.save_dir, "stage2_diagnostics.pkl")
    if _is_main_process(args):
        with open(diag_path, "wb") as f:
            pickle.dump(diagnostics, f)
    _print0(args, f"Stage 2 done. Best val AP={best_val_ap:.4f} at epoch {best_epoch}.")
    _print0(args, f"Diagnostics saved to {diag_path}")
    return save_path


# ================================================================
# STAGE 3 - END-TO-END FINETUNE (Phase A + Phase B + proxy dropout)
# ================================================================

def run_stage3(args, model_path, generator_path):
    _print0(args, "\n" + "=" * 60)
    _print0(args, "STAGE 3: End-to-End Finetune")
    _print0(args, f"  Phase A: {args.s3_phase_a_epochs} epochs (frozen transformer, "
                f"gen_lr={args.s3_lr_gen:.2e})")
    _print0(args, f"  Phase B: unfreeze transformer at lr={args.s3_lr_transformer:.2e}, "
                f"gen_lr={args.s3_lr_gen:.2e}")
    _print0(args, f"  Proxy dropout: {args.s3_proxy_dropout}")
    _print0(args, "=" * 60)

    train_loader, val_loader, test_loader, train_sampler, _, _ = _build_loaders(args)

    # Load transformer
    model = GraphTransformer(
        num_layers=args.num_layers, num_heads=args.num_heads,
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(args.device)
    model_ckpt = torch.load(model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(model_ckpt["model_state"])

    # Load generator
    generator = build_generator(args).to(args.device)
    gen_ckpt = torch.load(generator_path, map_location=args.device, weights_only=True)
    generator.load_state_dict(gen_ckpt["generator_state"])

    if args.distributed:
        generator = DDP(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    loss_fn = nn.BCEWithLogitsLoss()

    # Phase A optimizer: generator only, transformer frozen
    _freeze(model)
    opt_a = torch.optim.AdamW(generator.parameters(), lr=args.s3_lr_gen,
                               weight_decay=args.s3_weight_decay)

    # Phase B optimizer: built once Phase B starts
    opt_b = None

    best_val_ap = 0.0
    best_epoch = -1
    patience_counter = 0
    phase = "A"
    save_path = os.path.join(args.save_dir, "stage3_best.pt")

    for epoch in range(1, args.s3_max_epochs + 1):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Phase transition
        if epoch > args.s3_phase_a_epochs and phase == "A":
            phase = "B"
            _unfreeze(model)
            if args.distributed:
                model = DDP(
                    model,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank,
                    find_unused_parameters=True,
                )
            model_base = _unwrap(model)
            opt_b = torch.optim.AdamW([
                {"params": model_base.encoder.parameters(), "lr": args.s3_lr_transformer},
                {"params": model_base.layers.parameters(), "lr": args.s3_lr_transformer},
                {"params": model_base.head.parameters(), "lr": args.s3_lr_transformer},
                {"params": generator.parameters(), "lr": args.s3_lr_gen},
            ], weight_decay=args.s3_weight_decay)
            _print0(args, f"  [Epoch {epoch}] Switching to Phase B - transformer unfrozen.")

        optimizer = opt_a if phase == "A" else opt_b

        model.train()
        generator.train()
        train_loss_sum = 0.0
        train_loss_count = 0
        all_preds, all_labels = [], []
        model_base = _unwrap(model)

        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()

            dense_x, dense_mask = model_base.encode_dense(batch)

            # Proxy dropout: randomly skip proxies for a fraction of batches
            use_proxy = torch.rand(1).item() > args.s3_proxy_dropout

            if use_proxy:
                proxies, aux_loss = _generate_proxies(
                    model, generator, batch, dense_x, dense_mask, args)
                logits, _ = model(batch, proxy_embeddings=proxies,
                                  precomputed_dense=(dense_x, dense_mask))
                loss = loss_fn(logits, batch.y)
                if aux_loss is not None:
                    loss = loss + aux_loss
            else:
                logits, _ = model(batch, precomputed_dense=(dense_x, dense_mask))
                loss = loss_fn(logits, batch.y)

            loss.backward()

            params_to_clip = list(generator.parameters())
            if phase == "B":
                params_to_clip += list(model.parameters())
            nn.utils.clip_grad_norm_(params_to_clip, args.s3_grad_clip)
            optimizer.step()

            train_loss_sum += loss.item()
            train_loss_count += 1
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        train_preds = _all_gather_np(np.concatenate(all_preds), args)
        train_labels = _all_gather_np(np.concatenate(all_labels), args)
        train_ap = compute_macro_ap(train_preds, train_labels)
        train_loss = _reduce_loss_mean(train_loss_sum, train_loss_count, args)

        val_ap, val_loss = downstream_eval(model, generator, val_loader, args.device, args)
        test_ap, _ = downstream_eval(model, generator, test_loader, args.device, args)

        elapsed = time.time() - epoch_start
        mem_str = ""
        if args.device.startswith("cuda"):
            mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            mem_str = f" mem={mem_mb:.0f}MB"
            torch.cuda.reset_peak_memory_stats()
        _print0(args, (
            f"Epoch {epoch:3d}/{args.s3_max_epochs} [{phase}][{elapsed:.1f}s{mem_str}] | "
            f"train_loss={train_loss:.4f} train_AP={train_ap:.4f} | "
            f"val_loss={val_loss:.4f} val_AP={val_ap:.4f} | "
            f"test_AP={test_ap:.4f}"
        ))

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            patience_counter = 0
            if _is_main_process(args):
                torch.save({
                    "model_state": _unwrap(model).state_dict(),
                    "generator_state": _unwrap(generator).state_dict(),
                    "epoch": epoch, "val_ap": val_ap, "test_ap": test_ap,
                    "args": vars(args),
                }, save_path)
            _print0(args, f"  -> New best val AP={val_ap:.4f} (test={test_ap:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.s3_patience:
                _print0(args, f"Early stopping at epoch {epoch}. "
                           f"Best val AP={best_val_ap:.4f} at epoch {best_epoch}.")
                break

    _print0(args, f"Stage 3 done. Best val AP={best_val_ap:.4f} at epoch {best_epoch}.")
    return save_path


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    args = parse_args()
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        if _is_main_process(args):
            save_code_snapshot(args.save_dir)
        if args.distributed:
            dist.barrier()

        stages = [args.stage] if args.stage != "all" else ["1", "2", "3"]
        model_path = args.model_path
        generator_path = args.generator_path

        for stage in stages:
            if stage == "1":
                model_path = run_stage1(args)
                if args.distributed:
                    dist.barrier()

            elif stage == "2":
                if model_path is None:
                    model_path = os.path.join(args.save_dir, "stage1_best.pt")
                assert os.path.exists(model_path), \
                    f"Stage 2 requires pretrained model at {model_path}"
                generator_path = run_stage2(args, model_path)
                if args.distributed:
                    dist.barrier()

            elif stage == "3":
                if model_path is None:
                    model_path = os.path.join(args.save_dir, "stage1_best.pt")
                if generator_path is None:
                    generator_path = os.path.join(args.save_dir, "stage2_generator.pt")
                assert os.path.exists(model_path), \
                    f"Stage 3 requires pretrained model at {model_path}"
                assert os.path.exists(generator_path), \
                    f"Stage 3 requires trained generator at {generator_path}"
                run_stage3(args, model_path, generator_path)
                if args.distributed:
                    dist.barrier()

        _print0(args, "\nAll requested stages complete.")
    finally:
        cleanup_distributed(args)
