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
torch.set_float32_matmul_precision('high')
import torch.nn as nn

from data import get_loaders
from models import GraphTransformer, GREDEncoder, GREDHybridTransformer
from generators import (
    ScoreBasedGenerator, GNNPoolingGenerator, PMAGenerator, GraphCoarseningGenerator,
    CrossAttentionRouter, MultiPointProxyWrapper,
)
from metrics import compute_macro_ap
from mmd import mmd_squared
from optim_utils import build_grouped_optimizer_and_scheduler


# ================================================================
# CONFIG: argparse + yaml override
# ================================================================

def build_parser():
    p = argparse.ArgumentParser(description="Pipeline B: End-to-End Proxy Training")
    p.add_argument("--config", type=str, default=None, help="Path to yaml config (CLI overrides yaml)")

    # Backbone
    p.add_argument("--backbone", type=str, default="vanilla_gt",
                   choices=["vanilla_gt", "gred", "hybrid"],
                   help="Backbone architecture: vanilla_gt (transformer only), "
                        "gred (GRED distance filtering only), "
                        "hybrid (GRED encoder + transformer layers for proxy integration)")

    # Generator
    p.add_argument("--generator", type=str, default="score_based",
                   choices=["score_based", "gnn_pooling", "pma", "graph_coarsening"],
                   help="Generator architecture (flow_matching not supported in e2e)")
    p.add_argument("--num_proxies", type=int, default=64)

    # Model (shared)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.3)

    # Laplacian positional encoding
    p.add_argument("--use_lap_pe", action=argparse.BooleanOptionalAction, default=True,
                   help="Add Laplacian eigenvector positional encodings to node features")
    p.add_argument("--lap_pe_dim", type=int, default=32,
                   help="Number of Laplacian eigenvectors for positional encoding")

    # GRED-specific
    p.add_argument("--state_dim", type=int, default=88,
                   help="LRU complex state dimension (GRED/hybrid only)")
    p.add_argument("--num_gred_layers", type=int, default=8,
                   help="Number of GRED layers (GRED/hybrid only)")
    p.add_argument("--num_transformer_layers", type=int, default=2,
                   help="Number of transformer layers for proxy integration (hybrid only)")
    p.add_argument("--gred_expand", type=int, default=1,
                   help="FFN expansion factor for GRED DeepSets MLP")
    p.add_argument("--r_min", type=float, default=0.0,
                   help="Min eigenvalue magnitude for LRU init")
    p.add_argument("--r_max", type=float, default=1.0,
                   help="Max eigenvalue magnitude for LRU init")
    p.add_argument("--max_phase", type=float, default=6.28,
                   help="Max eigenvalue phase for LRU init")
    p.add_argument("--gred_act", type=str, default="full-glu",
                   choices=["full-glu", "half-glu"],
                   help="GLU activation variant for GRED")
    p.add_argument("--max_hops", type=int, default=40,
                   help="Maximum number of hop levels for distance masks")
    p.add_argument("--dist_mask_workers", type=int, default=8,
                   help="Number of workers for distance mask computation")

    # Generator-specific
    p.add_argument("--gen_hidden_dim", type=int, default=64)
    p.add_argument("--gen_num_layers", type=int, default=3)
    p.add_argument("--gen_num_heads", type=int, default=8)
    p.add_argument("--gen_dropout", type=float, default=0.2)
    # GNN-specific
    p.add_argument("--gnn_layers", type=int, default=3)
    p.add_argument("--gnn_type", type=str, default="GINE",
                   choices=["GCN", "GIN", "GINE", "GAT"])
    p.add_argument("--pool_types", type=str, nargs="+", default=["mean"])
    p.add_argument("--decode_hidden", type=int, default=64)
    p.add_argument("--decode_layers", type=int, default=3)
    p.add_argument("--idx_emb_dim", type=int, default=64)
    p.add_argument("--decode_mode", type=str, default="shared",
                   choices=["shared", "grouped"])
    # PMA-specific
    p.add_argument("--pma_query_mode", type=str, default="farthest_point",
                   choices=["farthest_point", "soft_kmeans"])
    # Graph coarsening-specific
    p.add_argument("--coarsen_gnn_type", type=str, default="GIN",
                   choices=["GIN", "GCN"])
    p.add_argument("--coarsen_reg_weight", type=float, default=0.1)
    p.add_argument("--coarsen_reg_type", type=str, default="mincut")

    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr_min", type=float, default=1e-7,
                   help="Minimum LR floor for warmup-cosine schedule")
    p.add_argument("--warmup_ratio", type=float, default=0.05,
                   help="Warmup fraction of total optimization steps")
    p.add_argument("--recurrent_lr_factor", type=float, default=1.0,
                   help="LR multiplier for recurrent GRED parameters")

    # E2E-specific
    p.add_argument("--mmd_lambda", type=float, default=0.01,
                   help="Weight for MMD regularization (0 to disable)")
    p.add_argument("--proxy_warmup_epochs", type=int, default=0,
                   help="Epochs to train transformer without proxies before activating generator")
    p.add_argument("--readout_scope", type=str, default="nodes_only",
                   choices=["nodes_only", "all_tokens"],
                   help="Pool over N original nodes or all N+M tokens")

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
        return None  # disabled
    return sorted(layers)


def _build_multi_point_proxy(args, generator, router):
    """Build MultiPointProxyWrapper if multi-point is enabled."""
    layers = _parse_insertion_layers(args.proxy_insertion_layers)
    if layers is None:
        return None
    return MultiPointProxyWrapper(
        generator=generator,
        router=router,  # None when not using cross-attn routing
        insertion_layers=layers,
        separate_generators=args.separate_proxy_generators,
        separate_routers=args.separate_proxy_routers,
        aux_loss_decay=args.proxy_aux_loss_decay,
    )


def build_model(args):
    """Build backbone model based on --backbone arg."""
    cross_attn_router = _build_cross_attn_router(args)
    generator = build_generator(args)
    mp_wrapper = _build_multi_point_proxy(args, generator, cross_attn_router)

    # When multi-point is active, the wrapper owns the router; the model gets None.
    model_router = cross_attn_router if mp_wrapper is None else None

    if args.backbone == "vanilla_gt":
        model = GraphTransformer(
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            dropout=args.dropout,
            lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
            cross_attn_router=model_router,
            multi_point_proxy=mp_wrapper,
        )
    elif args.backbone == "gred":
        model = GREDEncoder(
            hidden_dim=args.hidden_dim,
            state_dim=args.state_dim,
            num_layers=args.num_gred_layers,
            expand=args.gred_expand,
            r_min=args.r_min,
            r_max=args.r_max,
            max_phase=args.max_phase,
            dropout=args.dropout,
            act=args.gred_act,
            output_dim=args.output_dim,
            lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
        )
    elif args.backbone == "hybrid":
        model = GREDHybridTransformer(
            hidden_dim=args.hidden_dim,
            state_dim=args.state_dim,
            num_gred_layers=args.num_gred_layers,
            num_transformer_layers=args.num_transformer_layers,
            num_heads=args.num_heads,
            expand=args.gred_expand,
            r_min=args.r_min,
            r_max=args.r_max,
            max_phase=args.max_phase,
            dropout=args.dropout,
            act=args.gred_act,
            output_dim=args.output_dim,
            lap_pe_dim=args.lap_pe_dim if args.use_lap_pe else 0,
            cross_attn_router=model_router,
            multi_point_proxy=mp_wrapper,
        )
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

    return model, generator


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


# ================================================================
# FORWARD PASS
# ================================================================

def forward_e2e(model, generator, batch, args, use_proxies=True,
                dist_masks=None, node_masks=None):
    """
    Compose: encode -> generate proxies -> model with proxies -> classify.

    Supports vanilla_gt, gred, and hybrid backbones.

    Args:
        model: backbone model (GraphTransformer, GREDEncoder, or GREDHybridTransformer).
        generator: proxy generator module.
        batch: PyG Batch object.
        args: parsed CLI args.
        use_proxies: whether to generate and use proxies.
        dist_masks: (B, K, max_N, max_N) distance masks (required for gred/hybrid).
        node_masks: (B, max_N) boolean node masks (required for gred/hybrid).

    Returns:
        logits: (B, output_dim)
        mmd_loss: scalar tensor (0 if mmd_lambda == 0 or no proxies)
        aux_loss: scalar tensor from generator regularization (e.g. graph coarsening)
        node_emb: (total_N, d)
    """
    is_gred = args.backbone in ("gred", "hybrid")

    # --- No-proxy path ---
    if not use_proxies:
        if args.backbone == "vanilla_gt":
            logits, node_emb = model(batch, readout_scope=args.readout_scope)
        elif args.backbone == "gred":
            logits, node_emb = model(batch, dist_masks, node_masks)
        elif args.backbone == "hybrid":
            logits, node_emb = model(batch, dist_masks, node_masks,
                                     readout_scope=args.readout_scope)
        _zero = torch.tensor(0.0, device=batch.x.device)
        return logits, _zero, _zero, node_emb

    # --- Multi-point proxy path ---
    # When active the model generates + routes proxies internally at each
    # insertion point.  We skip external generation / MMD entirely.
    if getattr(model, 'multi_point_proxy', None) is not None:
        _zero = torch.tensor(0.0, device=batch.x.device)
        if args.backbone == "vanilla_gt":
            logits, node_emb = model(batch, readout_scope=args.readout_scope)
        elif args.backbone == "hybrid":
            logits, node_emb = model(batch, dist_masks, node_masks,
                                     readout_scope=args.readout_scope)
        else:
            raise ValueError("GRED backbone does not support proxy integration. Use 'hybrid'.")
        aux_loss = model._last_mp_aux_loss
        if not isinstance(aux_loss, torch.Tensor):
            aux_loss = torch.tensor(aux_loss, device=batch.x.device)
        return logits, _zero, aux_loss, node_emb

    # --- Single-point proxy path (existing behaviour) ---
    # Step 1: Encode nodes (dense)
    dense_x, dense_mask = model.encode_dense(batch)

    # For hybrid: run GRED encoding to produce topology-aware embeddings
    # Generators receive GRED-encoded features, not raw encoder features
    if args.backbone == "hybrid":
        gred_h = model.encode_gred(dense_x, dist_masks, node_masks)
        gen_input = gred_h
        gen_mask = dense_mask
    else:
        gen_input = dense_x
        gen_mask = dense_mask

    # Step 2: Generate proxies
    if args.generator in ("gnn_pooling", "graph_coarsening"):
        flat_node_emb = gen_input[gen_mask]  # flatten from dense
        proxy_emb, aux_loss = generator(
            flat_node_emb, mask=None,
            edge_index=batch.edge_index,
            batch_vec=batch.batch,
            edge_attr=getattr(batch, "edge_attr", None),
        )
    else:
        proxy_emb, aux_loss = generator(gen_input, gen_mask)

    # Step 3: Compute MMD loss
    mmd_loss = torch.tensor(0.0, device=batch.x.device)
    if args.mmd_lambda > 0:
        B = gen_input.shape[0]
        mmd_losses = []
        for i in range(B):
            nodes_i = gen_input[i][gen_mask[i]]
            proxies_i = proxy_emb[i]
            mmd_losses.append(mmd_squared(proxies_i, nodes_i))
        mmd_loss = torch.stack(mmd_losses).mean()

    # Step 4: Forward through model with proxies
    if args.backbone == "vanilla_gt":
        logits, node_emb = model(
            batch, proxy_embeddings=proxy_emb,
            precomputed_dense=(dense_x, dense_mask),
            readout_scope=args.readout_scope,
        )
    elif args.backbone == "gred":
        # Standalone GRED doesn't support proxies — shouldn't reach here
        raise ValueError("GRED backbone does not support proxy integration. Use 'hybrid'.")
    elif args.backbone == "hybrid":
        logits, node_emb = model(
            batch, dist_masks, node_masks,
            proxy_embeddings=proxy_emb,
            precomputed_dense=(dense_x, dense_mask),
            precomputed_gred=gred_h,
            readout_scope=args.readout_scope,
        )

    # aux_loss from generator (e.g. graph_coarsening regularization)
    if aux_loss is None:
        aux_loss = torch.tensor(0.0, device=batch.x.device)

    return logits, mmd_loss, aux_loss, node_emb


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate(model, generator, loader, device, args, use_proxies=True):
    """Evaluate on a loader. Returns (macro_AP, mean_task_loss, mean_mmd_loss)."""
    model.eval()
    generator.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    is_gred = args.backbone in ("gred", "hybrid")

    all_preds, all_labels = [], []
    task_losses, mmd_losses = [], []

    for batch_data in loader:
        if is_gred:
            batch, dist_masks_batch, node_masks_batch = batch_data
            batch = batch.to(device)
            dist_masks_batch = dist_masks_batch.to(device)
            node_masks_batch = node_masks_batch.to(device)
        else:
            batch = batch_data.to(device)
            dist_masks_batch = None
            node_masks_batch = None

        logits, mmd_loss, aux_loss, _ = forward_e2e(
            model, generator, batch, args,
            use_proxies=use_proxies,
            dist_masks=dist_masks_batch,
            node_masks=node_masks_batch,
        )
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
    is_gred = args.backbone in ("gred", "hybrid")

    print(f"Pipeline B — End-to-End Training", flush=True)
    print(f"  Backbone: {args.backbone}", flush=True)
    print(f"  Generator: {args.generator}", flush=True)
    print(f"  Proxies: {args.num_proxies}, Warmup: {args.proxy_warmup_epochs} epochs", flush=True)
    print(f"  Readout: {args.readout_scope}, MMD lambda: {args.mmd_lambda}", flush=True)
    if args.use_cross_attn_routing:
        print(f"  Cross-attn routing: layers={args.num_cross_layers}, "
              f"proxy_self_attn={args.cross_attn_proxy_self_attn}", flush=True)
    if args.proxy_insertion_layers != "-1":
        print(f"  Multi-point proxy: insertion_layers={args.proxy_insertion_layers}, "
              f"separate_gens={args.separate_proxy_generators}, "
              f"separate_routers={args.separate_proxy_routers}, "
              f"aux_decay={args.proxy_aux_loss_decay}", flush=True)
    if args.use_lap_pe:
        print(f"  Laplacian PE: dim={args.lap_pe_dim}", flush=True)
    if is_gred:
        print(f"  GRED: layers={args.num_gred_layers}, state_dim={args.state_dim}, "
              f"max_hops={args.max_hops}, act={args.gred_act}", flush=True)
        if args.backbone == "hybrid":
            print(f"  Hybrid: transformer_layers={args.num_transformer_layers}", flush=True)
    print(f"  Device: {args.device}", flush=True)

    # Validate backbone+generator compatibility
    if args.backbone == "gred" and args.num_proxies > 0:
        print("  WARNING: GRED backbone ignores proxies. Use --backbone hybrid for proxy support.",
              flush=True)

    # Data
    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    # Model
    model, generator = build_model(args)
    model = model.to(args.device)
    generator = generator.to(args.device)

    # When multi-point is active the generator lives inside the model's
    # MultiPointProxyWrapper, so model.parameters() already covers it.
    mp_active = getattr(model, 'multi_point_proxy', None) is not None
    if mp_active:
        total_params = sum(p.numel() for p in model.parameters())
        gen_params = sum(p.numel() for p in generator.parameters())
        print(f"  Total parameters: {total_params:,}", flush=True)
        print(f"    Model (incl. multi-point gen+router): {total_params:,}", flush=True)
        print(f"    Generator (inside wrapper):           {gen_params:,}", flush=True)
    else:
        total_params = sum(p.numel() for p in model.parameters()) + \
                       sum(p.numel() for p in generator.parameters())
        print(f"  Total parameters: {total_params:,}", flush=True)
        print(f"    Model:     {sum(p.numel() for p in model.parameters()):,}", flush=True)
        print(f"    Generator: {sum(p.numel() for p in generator.parameters()):,}", flush=True)

    # Official-style grouped optimization + warmup-cosine schedule.
    train_steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, train_steps_per_epoch * args.max_epochs)
    if mp_active:
        # Generator is already inside model — only list model params.
        named_params = [
            (f"model.{name}", param) for name, param in model.named_parameters()
        ]
    else:
        named_params = [
            (f"model.{name}", param) for name, param in model.named_parameters()
        ] + [
            (f"generator.{name}", param) for name, param in generator.named_parameters()
        ]
    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=named_params,
        lr_max=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        recurrent_lr_factor=args.recurrent_lr_factor,
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
        use_proxies = (epoch > args.proxy_warmup_epochs and args.num_proxies>0 and args.backbone!="gred")

        # --- Train ---
        model.train()
        generator.train()
        train_task_losses, train_mmd_losses = [], []
        all_train_preds, all_train_labels = [], []

        for batch_data in train_loader:
            if is_gred:
                batch, dist_masks_batch, node_masks_batch = batch_data
                batch = batch.to(args.device)
                dist_masks_batch = dist_masks_batch.to(args.device)
                node_masks_batch = node_masks_batch.to(args.device)
            else:
                batch = batch_data.to(args.device)
                dist_masks_batch = None
                node_masks_batch = None

            optimizer.zero_grad()

            logits, mmd_loss, aux_loss, _ = forward_e2e(
                model, generator, batch, args,
                use_proxies=use_proxies,
                dist_masks=dist_masks_batch,
                node_masks=node_masks_batch,
            )
            task_loss = loss_fn(logits, batch.y)
            total_loss = task_loss + args.mmd_lambda * mmd_loss + aux_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(generator.parameters()),
                args.grad_clip,
            )
            optimizer.step()
            scheduler.step()

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
                "scheduler_state": scheduler.state_dict(),
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

