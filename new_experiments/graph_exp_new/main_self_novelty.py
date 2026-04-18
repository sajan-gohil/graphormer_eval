"""
Self-Novelty Proxy Pipeline — main_self_novelty.py

Proxy generator trained with a self-referential novelty signal: each batch
runs the same current model twice (with proxies, without proxies) and
penalises proxies that fail to shift the output distribution or the node
representations away from the no-proxy baseline. Task loss keeps the shift
task-useful; no teacher matching, no reconstruction targets.

Stages:
    1. Pretrain backbone on full N-node graphs (standard BCE).
    2. Train generator with the frozen backbone using
       task_loss + alpha * output_penalty + alpha_node * node_penalty.
    3. Joint finetune backbone + generator with the same loss.

Usage:
    python main_self_novelty.py --stage all --backbone hybrid \\
        --generator graph_coarsening
    python main_self_novelty.py --stage 2 \\
        --model_path checkpoints_self_novelty/stage1_best.pt
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
import torch.nn.functional as F

import warnings
warnings.filterwarnings(
    "ignore",
    message="k >= N for N \\* N square matrix",
    category=RuntimeWarning,
)

from data import get_loaders
from models import GraphTransformer, GREDEncoder, GREDHybridTransformer
from generators import (
    ScoreBasedGenerator, FlowMatchingGenerator, GNNPoolingGenerator,
    PMAGenerator, GraphCoarseningGenerator, CrossAttentionRouter,
    MultiPointProxyWrapper,
)
from metrics import compute_macro_ap
from losses import novelty_loss, inter_proxy_cosine_stats
from optim_utils import (
    build_grouped_optimizer_and_scheduler,
    build_warmup_cosine_scheduler,
)


# ================================================================
# CONFIG
# ================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Self-Novelty Proxy Pipeline: task_loss + self-novelty penalties"
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--stage", type=str, default="all",
                   choices=["1", "2", "3", "all"])
    p.add_argument("--generator", type=str, default="score_based",
                   choices=["score_based", "pma", "graph_coarsening",
                            "gnn_pooling", "flow_matching"])
    p.add_argument("--backbone", type=str, default="vanilla_gt",
                   choices=["vanilla_gt", "gred", "hybrid"])

    # Resume paths
    p.add_argument("--model_path", type=str, default=None,
                   help="Stage 1 backbone checkpoint (also accepts Phase 1 / "
                        "Stage 1 checkpoints from other pipelines if architecture matches)")
    p.add_argument("--generator_path", type=str, default=None,
                   help="Stage 2 generator checkpoint")

    # Backbone architecture
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_proxies", type=int, default=32)

    # Laplacian PE
    p.add_argument("--use_lap_pe", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lap_pe_dim", type=int, default=8)

    # GRED-specific
    p.add_argument("--state_dim", type=int, default=88)
    p.add_argument("--num_gred_layers", type=int, default=8)
    p.add_argument("--num_transformer_layers", type=int, default=2)
    p.add_argument("--gred_expand", type=int, default=1)
    p.add_argument("--r_min", type=float, default=0.0)
    p.add_argument("--r_max", type=float, default=1.0)
    p.add_argument("--max_phase", type=float, default=6.28)
    p.add_argument("--gred_act", type=str, default="full-glu",
                   choices=["full-glu", "half-glu"])
    p.add_argument("--max_hops", type=int, default=40)
    p.add_argument("--dist_mask_workers", type=int, default=8)

    # Stage 1
    p.add_argument("--s1_lr", type=float, default=1e-3)
    p.add_argument("--s1_weight_decay", type=float, default=3e-4)
    p.add_argument("--s1_max_epochs", type=int, default=300)
    p.add_argument("--s1_patience", type=int, default=50)
    p.add_argument("--s1_grad_clip", type=float, default=1.0)

    # Stage 2 — generator-only training
    p.add_argument("--s2_lr", type=float, default=1e-3)
    p.add_argument("--s2_weight_decay", type=float, default=3e-4)
    p.add_argument("--s2_max_epochs", type=int, default=500)
    p.add_argument("--s2_patience", type=int, default=50)
    p.add_argument("--s2_eval_every", type=int, default=1)
    p.add_argument("--s2_grad_clip", type=float, default=1.0)

    # Stage 3 — joint finetune (no proxy dropout, no Phase A/B split)
    p.add_argument("--s3_lr_gen", type=float, default=5e-4)
    p.add_argument("--s3_lr_transformer", type=float, default=5e-4,
                   help="Backbone LR. Default same as generator LR (no differential).")
    p.add_argument("--s3_max_epochs", type=int, default=500)
    p.add_argument("--s3_patience", type=int, default=50)
    p.add_argument("--s3_grad_clip", type=float, default=1.0)
    p.add_argument("--s3_weight_decay", type=float, default=1e-4)

    # Novelty loss hyperparameters
    p.add_argument("--novelty_temperature", type=float, default=1.0,
                   help="Sigmoid temperature for output-level novelty. 1.0 = unscaled.")
    p.add_argument("--novelty_alpha", type=float, default=1.0,
                   help="Weight for output_penalty = 1 - output_novelty.")
    p.add_argument("--novelty_alpha_node", type=float, default=1.0,
                   help="Weight for node_penalty = 2 - node_novelty. "
                        "Set 0 to disable node-level novelty.")

    # Generator architecture
    p.add_argument("--gen_hidden_dim", type=int, default=256)
    p.add_argument("--gen_num_layers", type=int, default=4)
    p.add_argument("--gen_num_heads", type=int, default=8)
    p.add_argument("--gen_dropout", type=float, default=0.2)
    # PMA
    p.add_argument("--pma_query_mode", type=str, default="farthest_point",
                   choices=["farthest_point", "soft_kmeans"])
    # Graph coarsening
    p.add_argument("--coarsen_gnn_type", type=str, default="GIN",
                   choices=["GIN", "GCN"])
    p.add_argument("--coarsen_reg_weight", type=float, default=0.1)
    p.add_argument("--coarsen_reg_type", type=str, default="mincut")
    # GNN pooling
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--gnn_type", type=str, default="GINE",
                   choices=["GCN", "GIN", "GINE", "GAT"])
    p.add_argument("--pool_types", type=str, nargs="+", default=["max"])
    p.add_argument("--decode_hidden", type=int, default=128)
    p.add_argument("--decode_layers", type=int, default=3)
    p.add_argument("--idx_emb_dim", type=int, default=128)
    p.add_argument("--decode_mode", type=str, default="shared",
                   choices=["shared", "grouped"])
    # Flow matching
    p.add_argument("--denoiser_dim", type=int, default=256)
    p.add_argument("--denoiser_layers", type=int, default=4)
    p.add_argument("--denoiser_heads", type=int, default=8)
    p.add_argument("--euler_steps", type=int, default=1)

    # Cross-attention routing (N→M→N)
    p.add_argument("--use_cross_attn_routing", action="store_true", default=False)
    p.add_argument("--num_cross_layers", type=int, default=2)
    p.add_argument("--cross_attn_proxy_self_attn",
                   action=argparse.BooleanOptionalAction, default=True)

    # Multi-point proxy insertion
    p.add_argument("--proxy_insertion_layers", type=str, default="-1",
                   help="Comma-separated layer indices for multi-point insertion; "
                        "-1 disables.")
    p.add_argument("--separate_proxy_generators", action="store_true", default=False)
    p.add_argument("--separate_proxy_routers", action="store_true", default=False)
    p.add_argument("--proxy_aux_loss_decay", type=float, default=1.0)

    # Common
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr_min", type=float, default=1e-7)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--recurrent_lr_factor", type=float, default=1.0)
    p.add_argument("--save_dir", type=str, default="checkpoints_self_novelty")
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
    print(args.__dict__)
    return args


def save_code_snapshot(save_dir):
    import zipfile
    code_files = [f for f in os.listdir(".") if f.endswith(".py")]
    zip_path = os.path.join(save_dir, "code_snapshot.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in code_files:
            zf.write(f)
    print(f"Saved code snapshot to {zip_path}", flush=True)


# ================================================================
# HELPERS (shared with existing pipelines' conventions)
# ================================================================

def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)


def _unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)


def _uses_flat_interface(generator_name):
    return generator_name in ("graph_coarsening", "gnn_pooling")


def _build_cross_attn_router(args):
    if getattr(args, "use_cross_attn_routing", False):
        return CrossAttentionRouter(
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_cross_layers=args.num_cross_layers,
            dropout=args.dropout,
            use_proxy_self_attn=args.cross_attn_proxy_self_attn,
        )
    return None


def _parse_insertion_layers(s):
    layers = [int(x.strip()) for x in s.split(",")]
    if layers == [-1]:
        return None
    return sorted(layers)


def _build_multi_point_proxy(args, generator, router):
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


def build_model(args):
    lap_pe_dim = args.lap_pe_dim if args.use_lap_pe else 0
    cross_attn_router = _build_cross_attn_router(args)
    if args.backbone == "vanilla_gt":
        return GraphTransformer(
            num_layers=args.num_layers, num_heads=args.num_heads,
            hidden_dim=args.hidden_dim, output_dim=args.output_dim,
            dropout=args.dropout, lap_pe_dim=lap_pe_dim,
            cross_attn_router=cross_attn_router,
        )
    elif args.backbone == "gred":
        return GREDEncoder(
            hidden_dim=args.hidden_dim, state_dim=args.state_dim,
            num_layers=args.num_gred_layers, expand=args.gred_expand,
            r_min=args.r_min, r_max=args.r_max, max_phase=args.max_phase,
            dropout=args.dropout, act=args.gred_act, output_dim=args.output_dim,
            lap_pe_dim=lap_pe_dim,
        )
    elif args.backbone == "hybrid":
        return GREDHybridTransformer(
            hidden_dim=args.hidden_dim, state_dim=args.state_dim,
            num_gred_layers=args.num_gred_layers,
            num_transformer_layers=args.num_transformer_layers,
            num_heads=args.num_heads, expand=args.gred_expand,
            r_min=args.r_min, r_max=args.r_max, max_phase=args.max_phase,
            dropout=args.dropout, act=args.gred_act, output_dim=args.output_dim,
            lap_pe_dim=lap_pe_dim,
            cross_attn_router=cross_attn_router,
        )
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")


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
    else:
        raise ValueError(f"Unknown generator: {args.generator}")


def _generate_proxies(generator, batch, dense_x, dense_mask, args, gred_h=None):
    """Route the generator call based on its interface (flat vs dense)."""
    gen_input = gred_h if gred_h is not None else dense_x
    gen_mask = dense_mask

    if _uses_flat_interface(args.generator):
        flat_emb = gen_input[gen_mask]
        proxies, aux = generator(
            flat_emb, mask=None,
            edge_index=batch.edge_index,
            batch_vec=batch.batch,
            edge_attr=getattr(batch, "edge_attr", None),
        )
    else:
        proxies, aux = generator(gen_input, gen_mask)
    return proxies, aux


# ================================================================
# WITH/WITHOUT PROXY FORWARDS
# ================================================================

def _forward_with_proxies(model, batch, dist_masks, node_masks,
                          proxies, dense_x, dense_mask, gred_h, args):
    """Run the "with proxies" forward. Returns (logits, node_emb_flat, aux_loss).

    For multi-point proxy configs, the wrapper handles injection internally
    (``proxies`` is ignored) and ``aux_loss`` comes from the wrapper's
    ``_last_mp_aux_loss``. For single-point, ``proxies`` is concatenated / routed.
    """
    mp_mode = (getattr(model, "multi_point_proxy", None) is not None)

    if mp_mode:
        if args.backbone == "vanilla_gt":
            logits, node_emb = model(
                batch, precomputed_dense=(dense_x, dense_mask)
            )
        elif args.backbone == "hybrid":
            logits, node_emb = model(
                batch, dist_masks, node_masks,
                precomputed_dense=(dense_x, dense_mask),
                precomputed_gred=gred_h,
            )
        elif args.backbone == "gred":
            logits, node_emb = model(batch, dist_masks, node_masks)
        aux = getattr(model, "_last_mp_aux_loss", None)
        if aux is not None and not isinstance(aux, torch.Tensor):
            aux = torch.tensor(float(aux), device=dense_x.device)
    else:
        if args.backbone == "vanilla_gt":
            logits, node_emb = model(
                batch, proxy_embeddings=proxies,
                precomputed_dense=(dense_x, dense_mask),
            )
        elif args.backbone == "hybrid":
            logits, node_emb = model(
                batch, dist_masks, node_masks,
                proxy_embeddings=proxies,
                precomputed_dense=(dense_x, dense_mask),
                precomputed_gred=gred_h,
            )
        elif args.backbone == "gred":
            # Standalone GRED has no proxy path; fall through.
            logits, node_emb = model(batch, dist_masks, node_masks)
        aux = None
    return logits, node_emb, aux


def _forward_without_proxies(model, batch, dist_masks, node_masks,
                             dense_x, dense_mask, gred_h, args):
    """Run the "without proxies" forward. Returns (logits, node_emb_flat)."""
    if args.backbone == "vanilla_gt":
        logits, node_emb = model(
            batch,
            precomputed_dense=(dense_x, dense_mask),
            disable_proxy_injection=True,
        )
    elif args.backbone == "hybrid":
        logits, node_emb = model(
            batch, dist_masks, node_masks,
            precomputed_dense=(dense_x, dense_mask),
            precomputed_gred=gred_h,
            disable_proxy_injection=True,
        )
    elif args.backbone == "gred":
        logits, node_emb = model(batch, dist_masks, node_masks)
    return logits, node_emb


def _node_emb_to_dense(node_emb_flat, dense_mask):
    """Scatter flat (total_N, d) back into dense (B, max_N, d) aligned with mask."""
    B, max_N = dense_mask.shape
    d = node_emb_flat.shape[-1]
    dense = node_emb_flat.new_zeros(B, max_N, d)
    dense[dense_mask] = node_emb_flat
    return dense


# ================================================================
# DOWNSTREAM EVAL (AP with current proxies through current model)
# ================================================================

@torch.no_grad()
def downstream_eval(model, generator, loader, device, args):
    model.eval()
    if generator is not None:
        generator.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_preds, all_labels, losses = [], [], []
    is_gred = args.backbone in ("gred", "hybrid")
    mp_mode = getattr(model, "multi_point_proxy", None) is not None

    for batch_data in loader:
        if is_gred:
            batch, dist_masks, node_masks = batch_data
            batch = batch.to(device)
            dist_masks = dist_masks.to(device)
            node_masks = node_masks.to(device)
        else:
            batch = batch_data.to(device)
            dist_masks = None
            node_masks = None

        dense_x, dense_mask = model.encode_dense(batch)
        gred_h = None
        if args.backbone == "hybrid":
            gred_h = model.encode_gred(dense_x, dist_masks, node_masks)

        if mp_mode:
            proxies = None
        else:
            proxies, _ = _generate_proxies(
                generator, batch, dense_x, dense_mask, args, gred_h=gred_h
            )

        logits, _, _ = _forward_with_proxies(
            model, batch, dist_masks, node_masks,
            proxies, dense_x, dense_mask, gred_h, args,
        )

        losses.append(loss_fn(logits, batch.y).item())
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    ap = compute_macro_ap(
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_labels, axis=0),
    )
    return ap, float(np.mean(losses))


# ================================================================
# STAGE 1 — Pretrain backbone
# ================================================================

def run_stage1(args):
    print("\n" + "=" * 60, flush=True)
    print(f"STAGE 1: Pretrain backbone ({args.backbone})", flush=True)
    print("=" * 60, flush=True)
    is_gred = args.backbone in ("gred", "hybrid")

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    model = build_model(args).to(args.device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    total_steps = max(1, len(train_loader) * args.s1_max_epochs)
    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=[(f"model.{n}", p) for n, p in model.named_parameters()],
        lr_max=args.s1_lr, lr_min=args.lr_min,
        weight_decay=args.s1_weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        recurrent_lr_factor=args.recurrent_lr_factor,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_ap, best_epoch, patience = 0.0, -1, 0
    save_path = os.path.join(args.save_dir, "stage1_best.pt")

    for epoch in range(1, args.s1_max_epochs + 1):
        t0 = time.time()
        model.train()
        train_losses, preds, labels = [], [], []
        for batch_data in train_loader:
            if is_gred:
                batch, dist_masks, node_masks = batch_data
                batch = batch.to(args.device)
                dist_masks = dist_masks.to(args.device)
                node_masks = node_masks.to(args.device)
            else:
                batch = batch_data.to(args.device)
                dist_masks = node_masks = None

            optimizer.zero_grad()
            if args.backbone == "vanilla_gt":
                logits, _ = model(batch)
            else:
                logits, _ = model(batch, dist_masks, node_masks)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.s1_grad_clip)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(batch.y.cpu().numpy())

        train_ap = compute_macro_ap(np.concatenate(preds), np.concatenate(labels))
        train_loss = float(np.mean(train_losses))

        model.eval()
        val_preds, val_labels, val_losses = [], [], []
        with torch.no_grad():
            for batch_data in val_loader:
                if is_gred:
                    batch, dist_masks, node_masks = batch_data
                    batch = batch.to(args.device)
                    dist_masks = dist_masks.to(args.device)
                    node_masks = node_masks.to(args.device)
                else:
                    batch = batch_data.to(args.device)
                    dist_masks = node_masks = None
                if args.backbone == "vanilla_gt":
                    logits, _ = model(batch)
                else:
                    logits, _ = model(batch, dist_masks, node_masks)
                val_losses.append(loss_fn(logits, batch.y).item())
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(batch.y.cpu().numpy())
        val_ap = compute_macro_ap(np.concatenate(val_preds), np.concatenate(val_labels))
        val_loss = float(np.mean(val_losses))

        elapsed = time.time() - t0
        mem = ""
        if args.device.startswith("cuda"):
            mem = f" mem={torch.cuda.max_memory_allocated() / 1024 / 1024:.0f}MB"
            torch.cuda.reset_peak_memory_stats()
        print(
            f"Epoch {epoch:3d}/{args.s1_max_epochs} [{elapsed:.1f}s{mem}] | "
            f"train_loss={train_loss:.4f} train_AP={train_ap:.4f} | "
            f"val_loss={val_loss:.4f} val_AP={val_ap:.4f}",
            flush=True,
        )

        if val_ap > best_val_ap:
            best_val_ap, best_epoch, patience = val_ap, epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch, "val_ap": val_ap,
                "args": vars(args),
            }, save_path)
            print(f"  -> New best val AP={val_ap:.4f}, saved", flush=True)
        else:
            patience += 1
            if patience >= args.s1_patience:
                print(f"Early stopping at epoch {epoch}. Best val AP={best_val_ap:.4f} at epoch {best_epoch}.", flush=True)
                break

    print(f"Stage 1 done. Best val AP={best_val_ap:.4f} at epoch {best_epoch}.", flush=True)
    return save_path


# ================================================================
# STAGE 2 — Generator training with self-novelty (frozen backbone)
# ================================================================

def run_stage2(args, model_path):
    print("\n" + "=" * 60, flush=True)
    print(f"STAGE 2: Train generator ({args.generator}) with self-novelty", flush=True)
    print(f"  Backbone: {args.backbone} (frozen)", flush=True)
    print(f"  alpha={args.novelty_alpha} alpha_node={args.novelty_alpha_node} "
          f"T={args.novelty_temperature}", flush=True)
    if _parse_insertion_layers(args.proxy_insertion_layers) is not None:
        print(f"  Multi-point insertion layers: {_parse_insertion_layers(args.proxy_insertion_layers)}", flush=True)
    print("=" * 60, flush=True)
    is_gred = args.backbone in ("gred", "hybrid")

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    # Load and freeze backbone
    model = build_model(args).to(args.device)
    ckpt = torch.load(model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    _freeze(model)
    model.eval()

    generator = build_generator(args).to(args.device)
    print(f"  Generator parameters: {sum(p.numel() for p in generator.parameters()):,}",
          flush=True)

    # Attach multi-point wrapper if configured
    multi_point = _build_multi_point_proxy(args, generator, router=None)
    if multi_point is not None:
        model.multi_point_proxy = multi_point
        print("  Multi-point proxy wrapper attached", flush=True)

    total_steps = max(1, len(train_loader) * args.s2_max_epochs)
    if multi_point is not None:
        named_params = [(f"multi_point_proxy.{n}", p)
                        for n, p in model.multi_point_proxy.named_parameters()]
    else:
        named_params = [(f"generator.{n}", p) for n, p in generator.named_parameters()]

    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=named_params,
        lr_max=args.s2_lr, lr_min=args.lr_min,
        weight_decay=args.s2_weight_decay,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        recurrent_lr_factor=1.0,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_ap, best_epoch, patience = 0.0, -1, 0
    diagnostics = []
    save_path = os.path.join(args.save_dir, "stage2_generator.pt")

    for epoch in range(1, args.s2_max_epochs + 1):
        t0 = time.time()
        generator.train()

        train_task_losses = []
        train_total_losses = []
        train_output_novelty = []
        train_node_novelty = []
        train_output_penalty = []
        train_node_penalty = []
        inter_proxy_mean = []
        inter_proxy_std = []

        for batch_data in train_loader:
            if is_gred:
                batch, dist_masks, node_masks = batch_data
                batch = batch.to(args.device)
                dist_masks = dist_masks.to(args.device)
                node_masks = node_masks.to(args.device)
            else:
                batch = batch_data.to(args.device)
                dist_masks = node_masks = None

            # Encode once with frozen backbone; reuse across both forwards
            with torch.no_grad():
                dense_x, dense_mask = model.encode_dense(batch)
                gred_h = None
                if args.backbone == "hybrid":
                    gred_h = model.encode_gred(dense_x, dist_masks, node_masks)

            optimizer.zero_grad()

            # Generate proxies (gradient flows through generator only)
            if multi_point is None:
                proxies, aux_loss = _generate_proxies(
                    generator, batch, dense_x, dense_mask, args, gred_h=gred_h
                )
            else:
                # Wrapper generates proxies internally during the with-proxies forward.
                proxies, aux_loss = None, None

            # ── With proxies (gradient)
            logits_with, node_emb_with_flat, mp_aux = _forward_with_proxies(
                model, batch, dist_masks, node_masks,
                proxies, dense_x, dense_mask, gred_h, args,
            )

            # ── Without proxies (no gradient needed)
            with torch.no_grad():
                logits_without, node_emb_without_flat = _forward_without_proxies(
                    model, batch, dist_masks, node_masks,
                    dense_x, dense_mask, gred_h, args,
                )

            # Task loss on the with-proxies forward
            task = loss_fn(logits_with, batch.y)

            # Node-level novelty over original-N positions only.
            # Both node_emb_* come back flat (total_N, d) aligned with dense_mask.
            node_emb_with = node_emb_with_flat if args.novelty_alpha_node > 0 else None
            node_emb_without = node_emb_without_flat if args.novelty_alpha_node > 0 else None

            total, metrics = novelty_loss(
                task_loss=task,
                logits_with=logits_with,
                logits_without=logits_without.detach(),
                node_emb_with=node_emb_with,
                node_emb_without=(node_emb_without.detach()
                                  if node_emb_without is not None else None),
                mask=None,                         # flat inputs
                alpha=args.novelty_alpha,
                alpha_node=args.novelty_alpha_node,
                temperature=args.novelty_temperature,
            )

            # Aux loss from generator (e.g. graph_coarsening orthogonality)
            if aux_loss is not None:
                total = total + aux_loss
            if mp_aux is not None:
                total = total + mp_aux

            total.backward()

            # Clip only the parameters we are actually updating
            if multi_point is None:
                params_to_clip = list(generator.parameters())
            else:
                params_to_clip = list(model.multi_point_proxy.parameters())
            nn.utils.clip_grad_norm_(params_to_clip, args.s2_grad_clip)
            optimizer.step()
            scheduler.step()

            train_task_losses.append(metrics["task_loss"].item())
            train_total_losses.append(total.item())
            train_output_novelty.append(metrics["output_novelty"].item())
            train_output_penalty.append(metrics["output_penalty"].item())
            if "node_novelty" in metrics:
                train_node_novelty.append(metrics["node_novelty"].item())
                train_node_penalty.append(metrics["node_penalty"].item())

            if proxies is not None:
                mean_s, std_s = inter_proxy_cosine_stats(proxies)
                inter_proxy_mean.append(mean_s.item())
                inter_proxy_std.append(std_s.item())

        mean_task = float(np.mean(train_task_losses))
        mean_total = float(np.mean(train_total_losses))
        mean_out_nov = float(np.mean(train_output_novelty))
        mean_out_pen = float(np.mean(train_output_penalty))
        mean_node_nov = float(np.mean(train_node_novelty)) if train_node_novelty else float("nan")
        mean_node_pen = float(np.mean(train_node_penalty)) if train_node_penalty else float("nan")
        mean_ip = float(np.mean(inter_proxy_mean)) if inter_proxy_mean else float("nan")
        std_ip = float(np.mean(inter_proxy_std)) if inter_proxy_std else float("nan")

        if epoch % args.s2_eval_every == 0:
            val_ap, val_loss = downstream_eval(model, generator, val_loader, args.device, args)
            test_ap, _ = downstream_eval(model, generator, test_loader, args.device, args)

            elapsed = time.time() - t0
            mem = ""
            if args.device.startswith("cuda"):
                mem = f" mem={torch.cuda.max_memory_allocated() / 1024 / 1024:.0f}MB"
                torch.cuda.reset_peak_memory_stats()

            print(
                f"Epoch {epoch:3d}/{args.s2_max_epochs} [{elapsed:.1f}s{mem}] | "
                f"total={mean_total:.4f} task={mean_task:.4f} | "
                f"out_nov={mean_out_nov:.4f} out_pen={mean_out_pen:.4f} | "
                f"node_nov={mean_node_nov:.4f} node_pen={mean_node_pen:.4f} | "
                f"ip_mean={mean_ip:.4f} ip_std={std_ip:.4f} | "
                f"val_AP={val_ap:.4f} test_AP={test_ap:.4f}",
                flush=True,
            )

            diagnostics.append({
                "epoch": epoch,
                "total_loss": mean_total, "task_loss": mean_task,
                "output_novelty": mean_out_nov, "output_penalty": mean_out_pen,
                "node_novelty": mean_node_nov, "node_penalty": mean_node_pen,
                "inter_proxy_cos_mean": mean_ip, "inter_proxy_cos_std": std_ip,
                "val_ap": val_ap, "test_ap": test_ap,
            })

            if val_ap > best_val_ap:
                best_val_ap, best_epoch, patience = val_ap, epoch, 0
                torch.save({
                    "generator_state": generator.state_dict(),
                    "epoch": epoch, "val_ap": val_ap, "test_ap": test_ap,
                    "args": vars(args),
                }, save_path)
                print(f"  -> New best val AP={val_ap:.4f}", flush=True)
            else:
                patience += 1
                if patience >= args.s2_patience:
                    print(f"Early stopping at epoch {epoch}. Best val AP={best_val_ap:.4f} at epoch {best_epoch}.", flush=True)
                    break
        else:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{args.s2_max_epochs} [{elapsed:.1f}s] | "
                  f"total={mean_total:.4f} task={mean_task:.4f} "
                  f"out_nov={mean_out_nov:.4f} node_nov={mean_node_nov:.4f}",
                  flush=True)

    diag_path = os.path.join(args.save_dir, "stage2_diagnostics.pkl")
    with open(diag_path, "wb") as f:
        pickle.dump(diagnostics, f)
    print(f"Stage 2 done. Best val AP={best_val_ap:.4f} at epoch {best_epoch}. "
          f"Diagnostics saved to {diag_path}", flush=True)
    return save_path


# ================================================================
# STAGE 3 — Joint finetune (backbone + generator, no proxy dropout)
# ================================================================

def run_stage3(args, model_path, generator_path):
    print("\n" + "=" * 60, flush=True)
    print(f"STAGE 3: Joint finetune (backbone + generator)", flush=True)
    print(f"  Backbone: {args.backbone}", flush=True)
    print(f"  lr_gen={args.s3_lr_gen} lr_backbone={args.s3_lr_transformer}", flush=True)
    print(f"  alpha={args.novelty_alpha} alpha_node={args.novelty_alpha_node} "
          f"T={args.novelty_temperature}", flush=True)
    if _parse_insertion_layers(args.proxy_insertion_layers) is not None:
        print(f"  Multi-point insertion layers: {_parse_insertion_layers(args.proxy_insertion_layers)}", flush=True)
    print("=" * 60, flush=True)
    is_gred = args.backbone in ("gred", "hybrid")

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    model = build_model(args).to(args.device)
    m_ckpt = torch.load(model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(m_ckpt["model_state"])

    generator = build_generator(args).to(args.device)
    g_ckpt = torch.load(generator_path, map_location=args.device, weights_only=True)
    generator.load_state_dict(g_ckpt["generator_state"])

    multi_point = _build_multi_point_proxy(args, generator, router=None)
    if multi_point is not None:
        model.multi_point_proxy = multi_point
        print("  Multi-point proxy wrapper attached", flush=True)

    _unfreeze(model)

    # Build param groups. Keep generator and backbone in separate groups so
    # their LRs are independent. Multi-point proxy wrapper lives inside the
    # model, so its params are covered by the backbone group when enabled.
    if multi_point is None:
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad],
             "lr": args.s3_lr_transformer},
            {"params": generator.parameters(), "lr": args.s3_lr_gen},
        ]
    else:
        # Split backbone params from wrapper params: wrapper on its own LR.
        wrapper_params = list(model.multi_point_proxy.parameters())
        wrapper_ids = {id(p) for p in wrapper_params}
        backbone_params = [p for p in model.parameters()
                           if p.requires_grad and id(p) not in wrapper_ids]
        param_groups = [
            {"params": backbone_params, "lr": args.s3_lr_transformer},
            {"params": wrapper_params, "lr": args.s3_lr_gen},
        ]

    total_steps = max(1, len(train_loader) * args.s3_max_epochs)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.s3_weight_decay)
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        lr_min=args.lr_min,
        warmup_ratio=args.warmup_ratio,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_ap, best_epoch, patience = 0.0, -1, 0
    diagnostics = []
    save_path = os.path.join(args.save_dir, "stage3_best.pt")

    for epoch in range(1, args.s3_max_epochs + 1):
        t0 = time.time()
        model.train()
        generator.train()

        train_task_losses, train_total_losses = [], []
        train_output_novelty, train_output_penalty = [], []
        train_node_novelty, train_node_penalty = [], []
        inter_proxy_mean, inter_proxy_std = [], []
        all_preds, all_labels = [], []

        for batch_data in train_loader:
            if is_gred:
                batch, dist_masks, node_masks = batch_data
                batch = batch.to(args.device)
                dist_masks = dist_masks.to(args.device)
                node_masks = node_masks.to(args.device)
            else:
                batch = batch_data.to(args.device)
                dist_masks = node_masks = None

            # With grad through model: encode once, reuse for both forwards
            dense_x, dense_mask = model.encode_dense(batch)
            gred_h = None
            if args.backbone == "hybrid":
                gred_h = model.encode_gred(dense_x, dist_masks, node_masks)

            optimizer.zero_grad()

            if multi_point is None:
                proxies, aux_loss = _generate_proxies(
                    generator, batch, dense_x, dense_mask, args, gred_h=gred_h
                )
            else:
                proxies, aux_loss = None, None

            # With-proxies (gradient flows end-to-end)
            logits_with, node_emb_with_flat, mp_aux = _forward_with_proxies(
                model, batch, dist_masks, node_masks,
                proxies, dense_x, dense_mask, gred_h, args,
            )

            # Without-proxies (no grad; used only as novelty reference)
            with torch.no_grad():
                logits_without, node_emb_without_flat = _forward_without_proxies(
                    model, batch, dist_masks, node_masks,
                    dense_x.detach(), dense_mask,
                    gred_h.detach() if gred_h is not None else None,
                    args,
                )

            task = loss_fn(logits_with, batch.y)
            node_emb_with = node_emb_with_flat if args.novelty_alpha_node > 0 else None
            node_emb_without = node_emb_without_flat if args.novelty_alpha_node > 0 else None

            total, metrics = novelty_loss(
                task_loss=task,
                logits_with=logits_with,
                logits_without=logits_without.detach(),
                node_emb_with=node_emb_with,
                node_emb_without=(node_emb_without.detach()
                                  if node_emb_without is not None else None),
                mask=None,
                alpha=args.novelty_alpha,
                alpha_node=args.novelty_alpha_node,
                temperature=args.novelty_temperature,
            )
            if aux_loss is not None:
                total = total + aux_loss
            if mp_aux is not None:
                total = total + mp_aux

            total.backward()

            # Dedupe params for clip_grad_norm: when multi_point is attached,
            # the generator is inside model.multi_point_proxy already, so
            # model.parameters() covers it. Otherwise generator is separate.
            if multi_point is None:
                params_to_clip = list(generator.parameters()) + list(model.parameters())
            else:
                params_to_clip = list(model.parameters())
            nn.utils.clip_grad_norm_(params_to_clip, args.s3_grad_clip)
            optimizer.step()
            scheduler.step()

            train_task_losses.append(metrics["task_loss"].item())
            train_total_losses.append(total.item())
            train_output_novelty.append(metrics["output_novelty"].item())
            train_output_penalty.append(metrics["output_penalty"].item())
            if "node_novelty" in metrics:
                train_node_novelty.append(metrics["node_novelty"].item())
                train_node_penalty.append(metrics["node_penalty"].item())
            if proxies is not None:
                m_s, s_s = inter_proxy_cosine_stats(proxies)
                inter_proxy_mean.append(m_s.item())
                inter_proxy_std.append(s_s.item())

            all_preds.append(torch.sigmoid(logits_with).detach().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        train_ap = compute_macro_ap(np.concatenate(all_preds), np.concatenate(all_labels))
        mean_task = float(np.mean(train_task_losses))
        mean_total = float(np.mean(train_total_losses))
        mean_out_nov = float(np.mean(train_output_novelty))
        mean_out_pen = float(np.mean(train_output_penalty))
        mean_node_nov = float(np.mean(train_node_novelty)) if train_node_novelty else float("nan")
        mean_node_pen = float(np.mean(train_node_penalty)) if train_node_penalty else float("nan")
        mean_ip = float(np.mean(inter_proxy_mean)) if inter_proxy_mean else float("nan")
        std_ip = float(np.mean(inter_proxy_std)) if inter_proxy_std else float("nan")

        val_ap, val_loss = downstream_eval(model, generator, val_loader, args.device, args)
        test_ap, _ = downstream_eval(model, generator, test_loader, args.device, args)

        elapsed = time.time() - t0
        mem = ""
        if args.device.startswith("cuda"):
            mem = f" mem={torch.cuda.max_memory_allocated() / 1024 / 1024:.0f}MB"
            torch.cuda.reset_peak_memory_stats()
        print(
            f"Epoch {epoch:3d}/{args.s3_max_epochs} [{elapsed:.1f}s{mem}] | "
            f"total={mean_total:.4f} task={mean_task:.4f} train_AP={train_ap:.4f} | "
            f"out_nov={mean_out_nov:.4f} out_pen={mean_out_pen:.4f} | "
            f"node_nov={mean_node_nov:.4f} node_pen={mean_node_pen:.4f} | "
            f"ip_mean={mean_ip:.4f} ip_std={std_ip:.4f} | "
            f"val_AP={val_ap:.4f} test_AP={test_ap:.4f}",
            flush=True,
        )

        diagnostics.append({
            "epoch": epoch,
            "total_loss": mean_total, "task_loss": mean_task, "train_ap": train_ap,
            "output_novelty": mean_out_nov, "output_penalty": mean_out_pen,
            "node_novelty": mean_node_nov, "node_penalty": mean_node_pen,
            "inter_proxy_cos_mean": mean_ip, "inter_proxy_cos_std": std_ip,
            "val_ap": val_ap, "test_ap": test_ap,
        })

        if val_ap > best_val_ap:
            best_val_ap, best_epoch, patience = val_ap, epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "generator_state": generator.state_dict(),
                "epoch": epoch, "val_ap": val_ap, "test_ap": test_ap,
                "args": vars(args),
            }, save_path)
            print(f"  -> New best val AP={val_ap:.4f} (test={test_ap:.4f})", flush=True)
        else:
            patience += 1
            if patience >= args.s3_patience:
                print(f"Early stopping at epoch {epoch}. Best val AP={best_val_ap:.4f} at epoch {best_epoch}.", flush=True)
                break

    diag_path = os.path.join(args.save_dir, "stage3_diagnostics.pkl")
    with open(diag_path, "wb") as f:
        pickle.dump(diagnostics, f)
    print(f"Stage 3 done. Best val AP={best_val_ap:.4f} at epoch {best_epoch}. "
          f"Diagnostics saved to {diag_path}", flush=True)
    return save_path


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    save_code_snapshot(args.save_dir)

    stages = [args.stage] if args.stage != "all" else ["1", "2", "3"]
    model_path = args.model_path
    generator_path = args.generator_path

    if "1" in stages:
        model_path = run_stage1(args)

    if "2" in stages:
        if model_path is None:
            raise ValueError("Stage 2 requires --model_path (or run stage 1 first).")
        generator_path = run_stage2(args, model_path)

    if "3" in stages:
        if model_path is None or generator_path is None:
            raise ValueError("Stage 3 requires both --model_path and --generator_path "
                             "(or run stages 1 and 2 first).")
        run_stage3(args, model_path, generator_path)

