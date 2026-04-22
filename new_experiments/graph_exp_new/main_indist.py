"""
In-Distribution Proxy Learning (IDPL) Pipeline — main_indist.py

5-Phase pipeline for training proxy generators that produce in-distribution
node-like embeddings for graph transformers.

Phase 1: Train full transformer on all N nodes (baseline)
Phase 2: Train transformer on (N-M) node subsets (frozen encoder from Phase 1)
Phase 3: Train proxy generator to reconstruct missing M nodes
         (reconstruction + task loss through frozen Phase 2 transformer)
Phase 4: Augmented evaluation — N nodes + 2M generated proxies (no node dropping)
Phase 5: End-to-end joint fine-tuning from epoch 1 (no separate warmup)

Usage:
    python main_indist.py --phase all --generator score_based --backbone vanilla_gt
    python main_indist.py --phase 2 --model_path checkpoints_indist/phase1_best.pt
    python main_indist.py --phase 4 --model_path checkpoints_indist/phase1_best.pt \\
                                    --generator_path checkpoints_indist/phase3_generator.pt
"""

import argparse
import os
import pickle
import time
import yaml
import numpy as np
import torch
from torch_geometric.utils import subgraph
# torch.set_float32_matmul_precision('high')
import torch.nn as nn

from data import get_loaders
from models import GraphTransformer, GREDEncoder, GREDHybridTransformer
from generators import (
    ScoreBasedGenerator, GNNPoolingGenerator, PMAGenerator, GraphCoarseningGenerator,
    CrossAttentionRouter, MultiPointProxyWrapper,
)
from metrics import compute_macro_ap
from optim_utils import (
    build_grouped_optimizer_and_scheduler,
    build_warmup_cosine_scheduler,
)
from losses import novelty_loss, inter_proxy_cosine_stats, proxy_diversity_loss


# ================================================================
# CONFIG
# ================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="In-Distribution Proxy Learning (IDPL) Pipeline")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--phase", type=str, default="all",
                   choices=["1", "2", "3", "4", "5", "all"])
    p.add_argument("--generator", type=str, default="score_based",
                   choices=["score_based", "pma", "graph_coarsening", "gnn_pooling"])

    # Backbone
    p.add_argument("--backbone", type=str, default="vanilla_gt",
                   choices=["vanilla_gt", "gred", "hybrid"])

    # Paths
    p.add_argument("--model_path", type=str, default=None,
                   help="Phase 1 model checkpoint")
    p.add_argument("--phase2_model_path", type=str, default=None,
                   help="Phase 2 partial-graph model checkpoint")
    p.add_argument("--generator_path", type=str, default=None,
                   help="Phase 3 generator checkpoint")

    # Model architecture
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_proxies", type=int, default=64,
                   help="M: number of nodes to drop (Phase 2) / proxies to generate")
    p.add_argument("--proxy_multiplier", type=int, default=2,
                   help="Number of sets of M proxies to generate (e.g. 1 or 2)")

    # Laplacian PE
    p.add_argument("--use_lap_pe", action="store_true", default=False)
    p.add_argument("--lap_pe_dim", type=int, default=8)

    # GRED-specific
    p.add_argument("--state_dim", type=int, default=88)
    p.add_argument("--num_gred_layers", type=int, default=8)
    p.add_argument("--num_transformer_layers", type=int, default=2)
    p.add_argument("--gred_expand", type=int, default=1)
    p.add_argument("--r_min", type=float, default=0.0)
    p.add_argument("--r_max", type=float, default=1.0)
    p.add_argument("--max_phase_lru", type=float, default=6.28,
                   help="LRU max phase (renamed to avoid conflict with --phase)")
    p.add_argument("--gred_act", type=str, default="full-glu",
                   choices=["full-glu", "half-glu"])
    p.add_argument("--max_hops", type=int, default=40)
    p.add_argument("--dist_mask_workers", type=int, default=8)

    # Phase 1 — Pretrain full transformer
    p.add_argument("--p1_lr", type=float, default=1e-4)
    p.add_argument("--p1_weight_decay", type=float, default=3e-4)
    p.add_argument("--p1_max_epochs", type=int, default=500)
    p.add_argument("--p1_patience", type=int, default=50)
    p.add_argument("--p1_grad_clip", type=float, default=1.0)

    # Phase 2 — Train partial-graph transformer
    p.add_argument("--p2_lr", type=float, default=1e-4)
    p.add_argument("--p2_weight_decay", type=float, default=3e-4)
    p.add_argument("--p2_max_epochs", type=int, default=500)
    p.add_argument("--p2_patience", type=int, default=50)
    p.add_argument("--p2_grad_clip", type=float, default=1.0)

    # Phase 3 — Train proxy generator
    p.add_argument("--p3_lr", type=float, default=1e-4)
    p.add_argument("--p3_weight_decay", type=float, default=3e-4)
    p.add_argument("--p3_max_epochs", type=int, default=500)
    p.add_argument("--p3_patience", type=int, default=50)
    p.add_argument("--p3_grad_clip", type=float, default=1.0)
    p.add_argument("--p3_eval_every", type=int, default=1)
    p.add_argument("--p3_recon_weight", type=float, default=0.05,
                   help="Initial weight for reconstruction loss")
    p.add_argument("--p3_recon_anneal_to", type=float, default=0.1,
                   help="Final weight for reconstruction loss after annealing")
    p.add_argument("--p3_recon_anneal_epochs", type=int, default=100,
                   help="Number of epochs over which to anneal recon weight")
    p.add_argument("--p3_no_task_loss", action="store_true", default=False,
                   help="If set, skip task loss in Phase 3 and train on reconstruction only")

    # Phase 5 — End-to-end fine-tuning
    p.add_argument(
        "--p5_phase_a_epochs", type=int, default=0,
        help="Deprecated: ignored (Phase 5 starts directly with joint training)")
    p.add_argument("--p5_lr_gen", type=float, default=1e-4/5,
                   help="Generator LR for Phase 5 (default: 0.1 * p3_lr)")
    p.add_argument("--p5_lr_model", type=float, default=1e-5,
                   help="Model LR for Phase 5 (default: same as p5_lr_gen)")
    p.add_argument("--p5_proxy_dropout", type=float, default=0.1)
    p.add_argument("--p5_max_epochs", type=int, default=500)
    p.add_argument("--p5_patience", type=int, default=50)
    p.add_argument("--p5_grad_clip", type=float, default=1.0)
    p.add_argument("--p5_weight_decay", type=float, default=5e-5)

    # Novelty loss hyperparameters
    p.add_argument("--novelty_temperature", type=float, default=1.0,
                   help="Sigmoid temperature for output-level novelty.")
    p.add_argument("--novelty_alpha", type=float, default=0.0,
                   help="Weight for output_penalty. 0 disables novelty loss.")
    p.add_argument("--novelty_alpha_node", type=float, default=0.0,
                   help="Weight for node_penalty. 0 disables node-level novelty.")

    # Proxy diversity loss
    p.add_argument("--diversity_weight", type=float, default=0.0,
                   help="Weight for proxy diversity loss. 0 disables.")

    # Generator architecture
    p.add_argument("--gen_hidden_dim", type=int, default=256)
    p.add_argument("--gen_num_layers", type=int, default=6)
    p.add_argument("--gen_num_heads", type=int, default=8)
    p.add_argument("--gen_dropout", type=float, default=0.2)
    # PMA specific
    p.add_argument("--pma_query_mode", type=str, default="farthest_point",
                   choices=["farthest_point", "soft_kmeans"])
    # Graph coarsening specific
    p.add_argument("--coarsen_gnn_type", type=str, default="GIN",
                   choices=["GIN", "GCN"])
    p.add_argument("--coarsen_reg_weight", type=float, default=0.1)
    p.add_argument("--coarsen_reg_type", type=str, default="mincut")
    # GNN pooling specific
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--gnn_type", type=str, default="GINE",
                   choices=["GCN", "GIN", "GINE", "GAT"])
    p.add_argument("--pool_types", type=str, nargs="+", default=["max"])
    p.add_argument("--decode_hidden", type=int, default=128)
    p.add_argument("--decode_layers", type=int, default=3)
    p.add_argument("--idx_emb_dim", type=int, default=128)
    p.add_argument("--decode_mode", type=str, default="shared",
                   choices=["shared", "grouped"])

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

    # Common
    p.add_argument("--readout_scope", type=str, default="all_tokens")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr_min", type=float, default=1e-7,
                   help="Minimum LR floor for warmup-cosine schedule")
    p.add_argument("--warmup_ratio", type=float, default=0.05,
                   help="Warmup fraction of total optimization steps")
    p.add_argument("--recurrent_lr_factor", type=float, default=1.0,
                   help="LR multiplier for recurrent GRED parameters")
    p.add_argument("--save_dir", type=str, default="checkpoints_indist")
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
    if args.p5_lr_gen is None:
        args.p5_lr_gen = args.p3_lr * 0.1
    if args.p5_lr_model is None:
        args.p5_lr_model = args.p5_lr_gen
    print("Args = ", args.__dict__)
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


def build_model(args, include_proxy_modules=True):
    """Build backbone model based on --backbone arg.

    Args:
        include_proxy_modules: If False, the cross_attn_router is NOT attached
            to the model. Use False for Phases 1 and 2 where no proxies are
            ever generated — this ensures proxy parameters receive zero gradients
            and are not part of the optimizer in those phases.
    """
    lap_pe_dim = args.lap_pe_dim if args.use_lap_pe else 0
    # Only build the cross-attention router when we actually need proxies (Phase 3+).
    cross_attn_router = _build_cross_attn_router(args) if include_proxy_modules else None
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
            r_min=args.r_min, r_max=args.r_max,
            max_phase=args.max_phase_lru,
            dropout=args.dropout, act=args.gred_act,
            output_dim=args.output_dim, lap_pe_dim=lap_pe_dim,
        )
    elif args.backbone == "hybrid":
        return GREDHybridTransformer(
            hidden_dim=args.hidden_dim, state_dim=args.state_dim,
            num_gred_layers=args.num_gred_layers,
            num_transformer_layers=args.num_transformer_layers,
            num_heads=args.num_heads, expand=args.gred_expand,
            r_min=args.r_min, r_max=args.r_max,
            max_phase=args.max_phase_lru,
            dropout=args.dropout, act=args.gred_act,
            output_dim=args.output_dim, lap_pe_dim=lap_pe_dim,
            cross_attn_router=cross_attn_router,
        )
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")


def build_generator(args):
    if args.generator == "score_based":
        return ScoreBasedGenerator(
            num_proxies=args.num_proxies, input_dim=args.hidden_dim,
            hidden_dim=args.gen_hidden_dim, num_layers=args.gen_num_layers,
            num_heads=args.gen_num_heads, dropout=args.gen_dropout,
        )
    elif args.generator == "pma":
        return PMAGenerator(
            num_proxies=args.num_proxies, input_dim=args.hidden_dim,
            num_heads=args.gen_num_heads, num_layers=args.gen_num_layers,
            dropout=args.gen_dropout, query_mode=args.pma_query_mode,
        )
    elif args.generator == "graph_coarsening":
        return GraphCoarseningGenerator(
            num_proxies=args.num_proxies, input_dim=args.hidden_dim,
            gnn_layers=args.gen_num_layers, gnn_type=args.coarsen_gnn_type,
            dropout=args.gen_dropout, reg_type=args.coarsen_reg_type,
            reg_weight=args.coarsen_reg_weight,
            num_refine_layers=1, num_heads=args.gen_num_heads,
        )
    elif args.generator == "gnn_pooling":
        return GNNPoolingGenerator(
            num_proxies=args.num_proxies, input_dim=args.hidden_dim,
            gnn_layers=args.gnn_layers, gnn_type=args.gnn_type,
            pool_types=tuple(args.pool_types), decode_hidden=args.decode_hidden,
            decode_layers=args.decode_layers, idx_emb_dim=args.idx_emb_dim,
            dropout=args.gen_dropout, decode_mode=args.decode_mode,
        )
    else:
        raise ValueError(f"Unknown generator: {args.generator}")


# ================================================================
# NODE SUBSAMPLING UTILITIES
# ================================================================

def subsample_nodes(dense_x, dense_mask, num_drop):
    """
    For each graph in the batch, randomly drop `num_drop` nodes from the valid
    set. Returns the subsampled embeddings/mask and the held-out node embeddings.

    Args:
        dense_x: (B, max_N, d) node embeddings
        dense_mask: (B, max_N) boolean mask (True = real node)
        num_drop: int, number of nodes to drop (M)

    Returns:
        sub_x: (B, max_N, d) — dropped node positions zeroed out
        sub_mask: (B, max_N) — dropped node positions set to False
        dropped_embs: (B, M, d) — embeddings of the dropped nodes (zero-padded
                      if graph has fewer than M valid nodes)
        valid_drop_mask: (B,) — True if graph had enough nodes to drop M
    """
    B, max_N, d = dense_x.shape
    device = dense_x.device

    sub_x = dense_x.clone()
    sub_mask = dense_mask.clone()
    dropped_embs = torch.zeros(B, num_drop, d, device=device)
    valid_drop_mask = torch.zeros(B, dtype=torch.bool, device=device)

    for b in range(B):
        valid_idx = dense_mask[b].nonzero(as_tuple=True)[0]  # indices of real nodes
        n_valid = valid_idx.shape[0]

        if n_valid <= num_drop:
            # Too few nodes to drop M — skip subsampling for this graph
            # Keep all nodes, mark as invalid for reconstruction loss
            continue

        # Randomly select num_drop nodes to remove
        perm = torch.randperm(n_valid, device=device)[:num_drop]
        drop_idx = valid_idx[perm]

        # Store dropped embeddings
        dropped_embs[b] = dense_x[b, drop_idx]

        # Zero out and mask the dropped nodes
        sub_x[b, drop_idx] = 0.0
        sub_mask[b, drop_idx] = False
        valid_drop_mask[b] = True

    return sub_x, sub_mask, dropped_embs, valid_drop_mask


def _build_flat_generator_inputs(gen_input, gen_mask, batch):
    """
    Build flat-interface generator inputs aligned to a subsampled dense mask.

    Returns:
        flat_emb: kept node embeddings in flat order
        sub_edge_index: relabeled edges among kept nodes
        flat_batch_vec: per-node graph ids for kept nodes
        sub_edge_attr: edge attributes aligned with sub_edge_index (or None)
    """
    # Flat embeddings in dense-to-flat order.
    flat_emb = gen_input[gen_mask]

    # Map dense mask back to original flat PyG node indexing.
    num_graphs = gen_mask.size(0)
    pre_subsample_total_nodes = batch.batch.numel()
    keep_mask_flat = torch.zeros(pre_subsample_total_nodes, dtype=torch.bool, device=gen_mask.device)

    # Iterate per graph because dense masks are ragged (different node counts per graph).
    # This preserves exact dense-to-flat alignment for each graph slice.
    for graph_idx in range(num_graphs):
        graph_nodes = (batch.batch == graph_idx).nonzero(as_tuple=True)[0]
        num_nodes_in_graph = graph_nodes.numel()
        if num_nodes_in_graph == 0:
            continue
        keep_local = gen_mask[graph_idx, :num_nodes_in_graph]
        keep_mask_flat[graph_nodes[keep_local]] = True

    flat_batch_vec = batch.batch[keep_mask_flat]
    edge_attr = getattr(batch, "edge_attr", None)
    sub_edge_index, sub_edge_attr = subgraph(
        keep_mask_flat,
        batch.edge_index,
        edge_attr=edge_attr,
        relabel_nodes=True,
        num_nodes=pre_subsample_total_nodes,
    )

    if flat_batch_vec.numel() != flat_emb.size(0):
        raise RuntimeError(
            f"Flat interface mismatch: flat_emb has {flat_emb.size(0)} nodes "
            f"but flat_batch_vec has {flat_batch_vec.numel()} nodes. "
            "Check dense-to-flat keep-mask alignment for the subsampled batch."
        )

    return flat_emb, sub_edge_index, flat_batch_vec, sub_edge_attr


def subsample_dist_masks(dist_masks, node_masks, sub_mask):
    """
    Recompute distance masks for the subsampled node set.
    Simply zero out rows/columns corresponding to dropped nodes.

    Args:
        dist_masks: (B, K, max_N, max_N) original distance masks
        node_masks: (B, max_N) original node masks
        sub_mask: (B, max_N) subsampled node masks

    Returns:
        sub_dist_masks: (B, K, max_N, max_N) masked distance masks
        sub_node_masks: (B, max_N) same as sub_mask (for clarity)
    """
    # Mask out dropped nodes in distance matrices
    # sub_mask: (B, max_N) → expand to (B, 1, max_N, 1) and (B, 1, 1, max_N)
    row_mask = sub_mask.unsqueeze(1).unsqueeze(3).float()   # (B, 1, max_N, 1)
    col_mask = sub_mask.unsqueeze(1).unsqueeze(2).float()   # (B, 1, 1, max_N)
    sub_dist_masks = dist_masks * row_mask * col_mask
    return sub_dist_masks, sub_mask


# ================================================================
# DIFFERENTIABLE DISTRIBUTION RECONSTRUCTION LOSS
# ================================================================

def distribution_reconstruction_loss(generated, targets, valid_mask):
    """
    Differentiable set-to-set reconstruction loss using symmetric Chamfer distance.
    This avoids the non-differentiable assignment step in Hungarian matching.

    Args:
        generated: (B, M, d) generated proxy embeddings
        targets: (B, M, d) target node embeddings
        valid_mask: (B,) boolean — True if this graph had a valid drop

    Returns:
        loss: scalar reconstruction loss (averaged over valid graphs)
    """
    if not torch.any(valid_mask):
        return generated.sum() * 0.0

    losses = []
    for b in range(generated.shape[0]):
        if not valid_mask[b]:
            continue

        gen_b = F.normalize(generated[b], dim=-1)   # (M, d)
        tgt_b = F.normalize(targets[b], dim=-1)     # (M, d)
        dist2 = torch.cdist(gen_b, tgt_b, p=2).pow(2)  # (M, M)

        # Symmetric Chamfer: generated->target and target->generated.
        loss_b = 0.5 * (
            dist2.min(dim=1).values.mean() +
            dist2.min(dim=0).values.mean()
        )
        losses.append(loss_b)

    if len(losses) == 0:
        return generated.sum() * 0.0
    return torch.stack(losses).mean()


def hungarian_reconstruction_loss(generated, targets, valid_mask):
    """Backward-compatible alias kept for old call sites/checkpoints."""
    return distribution_reconstruction_loss(generated, targets, valid_mask)


# ================================================================
# PROXY GENERATION HELPER
# ================================================================

def _generate_proxies(model, generator, batch, dense_x, dense_mask, args,
                      gred_h=None, num_proxy_sets=1):
    """Generate proxy embeddings for a batch. Handles flat vs dense interface."""
    gen_input = gred_h if gred_h is not None else dense_x
    gen_mask = dense_mask
    num_proxy_sets = max(int(num_proxy_sets), 1)

    proxy_list = []
    aux_terms = []

    for _ in range(num_proxy_sets):
        if _uses_flat_interface(args.generator):
            flat_emb, sub_edge_index, flat_batch_vec, sub_edge_attr = _build_flat_generator_inputs(
                gen_input, gen_mask, batch
            )
            proxies, aux_loss = generator(
                flat_emb, mask=None,
                edge_index=sub_edge_index, batch_vec=flat_batch_vec,
                edge_attr=sub_edge_attr,
            )
        else:
            proxies, aux_loss = generator(gen_input, gen_mask)

        proxy_list.append(proxies)
        if aux_loss is not None:
            aux_terms.append(aux_loss)

    proxies = proxy_list[0] if len(proxy_list) == 1 else torch.cat(proxy_list, dim=1)
    aux_loss = torch.stack(aux_terms).mean() if len(aux_terms) > 0 else None
    return proxies, aux_loss


# ================================================================
# EVALUATION HELPERS
# ================================================================

@torch.no_grad()
def evaluate_model_only(model, loader, device, args):
    """Evaluate model without any proxies. Returns (AP, loss)."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_preds, all_labels, losses = [], [], []
    is_gred = args.backbone in ("gred", "hybrid")

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

        if args.backbone == "vanilla_gt":
            logits, _ = model(batch)
        else:
            logits, _ = model(batch, dist_masks_batch, node_masks_batch)

        losses.append(loss_fn(logits, batch.y).item())
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    ap = compute_macro_ap(
        np.concatenate(all_preds), np.concatenate(all_labels))
    return ap, float(np.mean(losses))


@torch.no_grad()
def evaluate_with_proxies(model, generator, loader, device, args,
                          proxy_multiplier=1):
    """Evaluate model with generated proxies from full nodes (no node dropping)."""
    model.eval()
    generator.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_preds, all_labels, losses = [], [], []
    is_gred = args.backbone in ("gred", "hybrid")

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

        # --- Multi-point proxy path ---
        if getattr(model, 'multi_point_proxy', None) is not None:
            with torch.no_grad():
                if args.backbone == "vanilla_gt":
                    logits, _ = model(batch, readout_scope=args.readout_scope)
                elif args.backbone == "hybrid":
                    logits, _ = model(batch, dist_masks_batch, node_masks_batch,
                                      readout_scope=args.readout_scope)
                else:
                    raise ValueError("GRED backbone does not support proxy integration. Use 'hybrid'.")
        else:
            # --- Single-point proxy path (existing behaviour) ---
            with torch.no_grad():
                dense_x, dense_mask = model.encode_dense(batch)

                gred_h = None
                if args.backbone == "hybrid":
                    gred_h = model.encode_gred(dense_x, dist_masks_batch, node_masks_batch)

                proxies, _ = _generate_proxies(
                    model, generator, batch, dense_x, dense_mask, args,
                    gred_h=gred_h, num_proxy_sets=proxy_multiplier)

                if args.backbone == "vanilla_gt":
                    logits, _ = model(batch, proxy_embeddings=proxies,
                                      precomputed_dense=(dense_x, dense_mask))
                elif args.backbone == "hybrid":
                    logits, _ = model(batch, dist_masks_batch, node_masks_batch,
                                      proxy_embeddings=proxies,
                                      precomputed_dense=(dense_x, dense_mask),
                                      precomputed_gred=gred_h)
                elif args.backbone == "gred":
                    # Standalone GRED doesn't support proxies
                    logits, _ = model(batch, dist_masks_batch, node_masks_batch)

        losses.append(loss_fn(logits, batch.y).item())
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    ap = compute_macro_ap(
        np.concatenate(all_preds), np.concatenate(all_labels))
    return ap, float(np.mean(losses))


# ================================================================
# PHASE 1 — PRETRAIN FULL TRANSFORMER
# ================================================================

def run_phase1(args):
    """Train the full transformer on all N nodes (baseline)."""
    print("\n" + "=" * 60, flush=True)
    print("PHASE 1: Pretrain Full Transformer (Baseline)", flush=True)
    print(f"  Backbone: {args.backbone}", flush=True)
    print("=" * 60, flush=True)
    is_gred = args.backbone in ("gred", "hybrid")

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    # Phase 1: no proxies or cross-attention router needed.
    model = build_model(args, include_proxy_modules=False).to(args.device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=[(f"model.{name}", param) for name, param in model.named_parameters()],
        lr_max=args.p1_lr,
        lr_min=args.lr_min,
        weight_decay=args.p1_weight_decay,
        total_steps=max(1, len(train_loader) * args.p1_max_epochs),
        warmup_ratio=args.warmup_ratio,
        recurrent_lr_factor=args.recurrent_lr_factor,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_ap = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    save_path = os.path.join(args.save_dir, "phase1_best.pt")

    for epoch in range(1, args.p1_max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses, all_preds, all_labels = [], [], []

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
            if args.backbone == "vanilla_gt":
                logits, _ = model(batch)
            else:
                logits, _ = model(batch, dist_masks_batch, node_masks_batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.p1_grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        train_ap = compute_macro_ap(
            np.concatenate(all_preds), np.concatenate(all_labels))
        train_loss = float(np.mean(train_losses))

        val_ap, val_loss = evaluate_model_only(model, val_loader, args.device, args)

        elapsed = time.time() - epoch_start
        mem_str = _mem_str(args)
        log = (f"Epoch {epoch:3d}/{args.p1_max_epochs} [{elapsed:.1f}s{mem_str}] | "
               f"train_loss={train_loss:.4f} train_AP={train_ap:.4f} | "
               f"val_loss={val_loss:.4f} val_AP={val_ap:.4f}")

        if epoch % 10 == 0:
            test_ap, _ = evaluate_model_only(model, test_loader, args.device, args)
            log += f" | test_AP={test_ap:.4f}"

        print(log, flush=True)

        if val_loss < best_val_loss:
            best_val_ap = val_ap
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch, "val_ap": val_ap,
                "args": vars(args),
            }, save_path)
            print(f"  -> New best val loss={val_loss:.4f}, saved", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.p1_patience:
                print(f"Early stopping at epoch {epoch}. "
                      f"Best val loss={best_val_loss:.4f} at epoch {best_epoch}.",
                      flush=True)
                break

    print(f"Phase 1 done. Best val loss={best_val_loss:.4f} at epoch {best_epoch}.",
          flush=True)
    return save_path


# ================================================================
# PHASE 2 — TRAIN PARTIAL-GRAPH TRANSFORMER
# ================================================================

def run_phase2(args, model_path):
    """
    Train transformer layers on (N-M) node subsets.
    The node encoder is frozen from Phase 1.
    """
    print("\n" + "=" * 60, flush=True)
    print("PHASE 2: Train Partial-Graph Transformer (N-M nodes)", flush=True)
    print(f"  Backbone: {args.backbone}", flush=True)
    print(f"  Dropping M={args.num_proxies} nodes per graph", flush=True)
    print("=" * 60, flush=True)
    is_gred = args.backbone in ("gred", "hybrid")

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    # Load Phase 1 model
    # Phase 2: no proxies or cross-attention router needed.
    model = build_model(args, include_proxy_modules=False).to(args.device)
    ckpt = torch.load(model_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    # Freeze the encoder — keep embedding space consistent with Phase 1
    _freeze(model.encoder)
    print("  Encoder frozen. Training transformer layers + head only.", flush=True)

    # Collect trainable parameters (everything except encoder)
    named_trainable = [
        (f"model.{name}", param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    trainable_params = [param for _, param in named_trainable]

    print(f"  Trainable params: {sum(p.numel() for p in trainable_params):,}",
          flush=True)

    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=named_trainable,
        lr_max=args.p2_lr,
        lr_min=args.lr_min,
        weight_decay=args.p2_weight_decay,
        total_steps=max(1, len(train_loader) * args.p2_max_epochs),
        warmup_ratio=args.warmup_ratio,
        recurrent_lr_factor=args.recurrent_lr_factor,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_ap = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    M = args.num_proxies
    save_path = os.path.join(args.save_dir, "phase2_best.pt")

    for epoch in range(1, args.p2_max_epochs + 1):
        epoch_start = time.time()
        model.train()
        # Keep encoder in eval mode (frozen)
        model.encoder.eval()
        train_losses, all_preds, all_labels = [], [], []

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

            # Encode all nodes with frozen encoder
            with torch.no_grad():
                dense_x, dense_mask = model.encode_dense(batch)

            # Subsample: drop M nodes per graph
            sub_x, sub_mask, _, _ = subsample_nodes(dense_x, dense_mask, M)

            # Forward through trainable layers with subsampled nodes
            if args.backbone == "vanilla_gt":
                # Pass subsampled dense directly through transformer layers
                B, max_N, d = sub_x.shape
                aug_mask = sub_mask
                h = sub_x
                for layer in model.layers:
                    h = layer(h, aug_mask)
                # Readout
                node_emb_masked = h[sub_mask]
                batch_vec = torch.arange(
                    B, device=h.device
                ).unsqueeze(1).expand(B, max_N)[sub_mask]
                from torch_geometric.nn import global_add_pool
                pooled = global_add_pool(node_emb_masked, batch_vec)
                logits = model.head(pooled)

            elif args.backbone == "gred":
                # Subsample distance masks
                sub_dm, sub_nm = subsample_dist_masks(
                    dist_masks_batch, node_masks_batch, sub_mask)
                logits, _ = model(
                    batch, sub_dm, sub_nm,
                    precomputed_dense=(sub_x, sub_mask))

            elif args.backbone == "hybrid":
                sub_dm, sub_nm = subsample_dist_masks(
                    dist_masks_batch, node_masks_batch, sub_mask)
                logits, _ = model(
                    batch, sub_dm, sub_nm,
                    precomputed_dense=(sub_x, sub_mask))

            loss = loss_fn(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, args.p2_grad_clip)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        train_ap = compute_macro_ap(
            np.concatenate(all_preds), np.concatenate(all_labels))
        train_loss = float(np.mean(train_losses))

        # Validate on partial graphs too (consistent with training)
        val_ap, val_loss = _evaluate_partial(
            model, val_loader, args.device, args, M)

        elapsed = time.time() - epoch_start
        mem_str = _mem_str(args)
        log = (f"Epoch {epoch:3d}/{args.p2_max_epochs} [{elapsed:.1f}s{mem_str}] | "
               f"train_loss={train_loss:.4f} train_AP={train_ap:.4f} | "
               f"val_loss={val_loss:.4f} val_AP={val_ap:.4f}")

        if epoch % 10 == 0:
            test_ap, _ = _evaluate_partial(
                model, test_loader, args.device, args, M)
            # Also evaluate on full graph for comparison
            full_val_ap, _ = evaluate_model_only(
                model, val_loader, args.device, args)
            log += f" | test_AP={test_ap:.4f} full_val_AP={full_val_ap:.4f}"

        print(log, flush=True)

        if val_loss < best_val_loss:
            best_val_ap = val_ap
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch, "val_ap": val_ap,
                "args": vars(args),
            }, save_path)
            print(f"  -> New best val loss={val_loss:.4f}, saved", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.p2_patience:
                print(f"Early stopping at epoch {epoch}. "
                      f"Best val loss={best_val_loss:.4f} at epoch {best_epoch}.",
                      flush=True)
                break

    print(f"Phase 2 done. Best val loss={best_val_loss:.4f} at epoch {best_epoch}.",
          flush=True)
    return save_path


@torch.no_grad()
def _evaluate_partial(model, loader, device, args, num_drop):
    """Evaluate model on partial (N-M) node subsets."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_preds, all_labels, losses = [], [], []
    is_gred = args.backbone in ("gred", "hybrid")

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

        dense_x, dense_mask = model.encode_dense(batch)
        sub_x, sub_mask, _, _ = subsample_nodes(dense_x, dense_mask, num_drop)

        if args.backbone == "vanilla_gt":
            B, max_N, d = sub_x.shape
            h = sub_x
            for layer in model.layers:
                h = layer(h, sub_mask)
            node_emb_masked = h[sub_mask]
            batch_vec = torch.arange(
                B, device=h.device
            ).unsqueeze(1).expand(B, max_N)[sub_mask]
            from torch_geometric.nn import global_add_pool
            pooled = global_add_pool(node_emb_masked, batch_vec)
            logits = model.head(pooled)
        elif args.backbone == "gred":
            sub_dm, sub_nm = subsample_dist_masks(
                dist_masks_batch, node_masks_batch, sub_mask)
            logits, _ = model(
                batch, sub_dm, sub_nm,
                precomputed_dense=(sub_x, sub_mask))
        elif args.backbone == "hybrid":
            sub_dm, sub_nm = subsample_dist_masks(
                dist_masks_batch, node_masks_batch, sub_mask)
            logits, _ = model(
                batch, sub_dm, sub_nm,
                precomputed_dense=(sub_x, sub_mask))

        losses.append(loss_fn(logits, batch.y).item())
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    ap = compute_macro_ap(
        np.concatenate(all_preds), np.concatenate(all_labels))
    return ap, float(np.mean(losses))


# ================================================================
# PHASE 3 — TRAIN PROXY GENERATOR (RECONSTRUCTION + TASK LOSS)
# ================================================================

def run_phase3(args, phase1_model_path, phase2_model_path):
    """
    Train proxy generator to reconstruct missing M nodes.
    Uses both differentiable distribution reconstruction loss and task loss
    through the frozen Phase 2 transformer.
    """
    print("\n" + "=" * 60, flush=True)
    print(f"PHASE 3: Train Proxy Generator ({args.generator})", flush=True)
    print(f"  Backbone: {args.backbone}", flush=True)
    print(f"  Recon weight: {args.p3_recon_weight} → {args.p3_recon_anneal_to} "
          f"over {args.p3_recon_anneal_epochs} epochs", flush=True)
    print("=" * 60, flush=True)
    is_gred = args.backbone in ("gred", "hybrid")

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    # Load Phase 1 model (frozen — for target embeddings).
    # The Phase 1 checkpoint was saved WITHOUT proxy modules, so we rebuild
    # without them here to match the state_dict exactly.
    phase1_model = build_model(args, include_proxy_modules=False).to(args.device)
    ckpt1 = torch.load(phase1_model_path, map_location=args.device, weights_only=True)
    phase1_model.load_state_dict(ckpt1["model_state"])
    _freeze(phase1_model)
    phase1_model.eval()

    # Load Phase 2 model (frozen — for task loss).
    # Phase 2 checkpoint also has no proxy modules; build to match.
    # The cross_attn_router IS included here because proxies generated by the
    # generator are fed into this model (single-point path in Phase 3 task loss).
    phase2_model = build_model(args, include_proxy_modules=True).to(args.device)
    ckpt2 = torch.load(phase2_model_path, map_location=args.device, weights_only=False)
    phase2_model.load_state_dict(ckpt2["model_state"], strict=False)
    _freeze(phase2_model)
    phase2_model.eval()

    # Generator (trainable)
    generator = build_generator(args).to(args.device)
    print(f"  Generator parameters: {sum(p.numel() for p in generator.parameters()):,}",
          flush=True)

    optimizer, scheduler = build_grouped_optimizer_and_scheduler(
        named_parameters=[(f"generator.{name}", param) for name, param in generator.named_parameters()],
        lr_max=args.p3_lr,
        lr_min=args.lr_min,
        weight_decay=args.p3_weight_decay,
        total_steps=max(1, len(train_loader) * args.p3_max_epochs),
        warmup_ratio=args.warmup_ratio,
        recurrent_lr_factor=1.0,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    M = args.num_proxies

    best_val_ap = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    diagnostics = []
    save_path = os.path.join(args.save_dir, "phase3_generator.pt")

    for epoch in range(1, args.p3_max_epochs + 1):
        epoch_start = time.time()
        generator.train()

        # Anneal reconstruction weight
        if epoch <= args.p3_recon_anneal_epochs:
            frac = (epoch - 1) / max(args.p3_recon_anneal_epochs - 1, 1)
            recon_w = args.p3_recon_weight + frac * (
                args.p3_recon_anneal_to - args.p3_recon_weight)
        else:
            recon_w = args.p3_recon_anneal_to

        train_losses, recon_losses, task_losses = [], [], []

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

            # Phase 1 encoder: get full-graph embeddings (for targets)
            with torch.no_grad():
                full_dense_x, full_dense_mask = phase1_model.encode_dense(batch)

            # Subsample nodes: drop M
            sub_x, sub_mask, dropped_embs, valid_drop = subsample_nodes(
                full_dense_x, full_dense_mask, M)

            optimizer.zero_grad()

            # Generate proxies from (N-M) node subset
            if _uses_flat_interface(args.generator):
                flat_emb, sub_edge_index, flat_batch_vec, sub_edge_attr = _build_flat_generator_inputs(
                    sub_x, sub_mask, batch
                )
                proxies, aux_loss = generator(
                    flat_emb, mask=None,
                    edge_index=sub_edge_index, batch_vec=flat_batch_vec,
                    edge_attr=sub_edge_attr,
                )
            else:
                proxies, aux_loss = generator(sub_x, sub_mask)

            # Loss 1: Differentiable reconstruction on dropped-node distribution.
            l_recon = distribution_reconstruction_loss(
                proxies, dropped_embs, valid_drop)

            # Loss 2: Task loss through Phase 2 transformer (optional)
            if args.p3_no_task_loss:
                l_task = torch.tensor(0.0, device=args.device)
                logits = None
                node_emb_with = None
            else:
                # Feed (N-M) subset + generated proxies through Phase 2 model
                if args.backbone == "vanilla_gt":
                    logits, node_emb_with = phase2_model(
                        batch, proxy_embeddings=proxies,
                        precomputed_dense=(sub_x, sub_mask))
                elif args.backbone == "hybrid":
                    sub_dm, sub_nm = subsample_dist_masks(
                        dist_masks_batch, node_masks_batch, sub_mask)
                    gred_h = phase2_model.encode_gred(sub_x, sub_dm, sub_nm)
                    logits, node_emb_with = phase2_model(
                        batch, sub_dm, sub_nm,
                        proxy_embeddings=proxies,
                        precomputed_dense=(sub_x, sub_mask),
                        precomputed_gred=gred_h)
                elif args.backbone == "gred":
                    sub_dm, sub_nm = subsample_dist_masks(
                        dist_masks_batch, node_masks_batch, sub_mask)
                    logits, node_emb_with = phase2_model(
                        batch, sub_dm, sub_nm,
                        precomputed_dense=(sub_x, sub_mask))
                l_task = loss_fn(logits, batch.y)

            # Novelty loss (when enabled)
            use_novelty = (args.novelty_alpha > 0 or args.novelty_alpha_node > 0) and not args.p3_no_task_loss
            nov_metrics = {}
            if use_novelty:
                # logits and node_emb_with from with-proxies forward above
                # Need without-proxies forward through phase2_model
                with torch.no_grad():
                    if args.backbone == "vanilla_gt":
                        logits_without, node_emb_without = phase2_model(
                            batch, precomputed_dense=(sub_x, sub_mask))
                    elif args.backbone == "hybrid":
                        logits_without, node_emb_without = phase2_model(
                            batch, sub_dm, sub_nm,
                            precomputed_dense=(sub_x, sub_mask),
                            precomputed_gred=gred_h)
                    elif args.backbone == "gred":
                        logits_without, node_emb_without = phase2_model(
                            batch, sub_dm, sub_nm,
                            precomputed_dense=(sub_x, sub_mask))

                node_w = node_emb_with if args.novelty_alpha_node > 0 else None
                node_wo = node_emb_without if args.novelty_alpha_node > 0 else None
                # novelty_loss returns (total_loss, metrics_dict)
                nov_total, nov_metrics = novelty_loss(
                    l_task, logits, logits_without.detach(),
                    node_w, node_wo.detach() if node_wo is not None else None,
                    mask=None, alpha=args.novelty_alpha,
                    alpha_node=args.novelty_alpha_node,
                    temperature=args.novelty_temperature)
                # nov_total already includes l_task, so replace l_task with novelty-augmented version
                l_task = nov_total

            # Proxy diversity loss
            diversity_loss = torch.tensor(0.0, device=args.device)
            if args.diversity_weight > 0:
                diversity_loss = proxy_diversity_loss(proxies)

            # Combined loss
            loss = recon_w * l_recon
            if not args.p3_no_task_loss:
                loss = loss + l_task
            if args.diversity_weight > 0:
                loss = loss + args.diversity_weight * diversity_loss
            if aux_loss is not None:
                loss = loss + aux_loss

            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), args.p3_grad_clip)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            recon_losses.append(l_recon.item())
            task_losses.append(l_task.item())

        mean_loss = float(np.mean(train_losses))
        mean_recon = float(np.mean(recon_losses))
        mean_task = float(np.mean(task_losses))

        # Compute inter-proxy cosine stats for logging (only when proxies exist)
        cosine_mean = 0.0
        cosine_std = 0.0
        if args.diversity_weight > 0 and len(train_loader) > 0:
            # Recompute on a sample batch for stats
            try:
                with torch.no_grad():
                    for sample_batch_data in train_loader:
                        if is_gred:
                            sample_batch = sample_batch_data[0].to(args.device)
                        else:
                            sample_batch = sample_batch_data.to(args.device)
                        full_dense_x, full_dense_mask = phase1_model.encode_dense(sample_batch)
                        sub_x_s, sub_mask_s, _, _ = subsample_nodes(
                            full_dense_x, full_dense_mask, M)
                        if _uses_flat_interface(args.generator):
                            flat_emb, sub_edge_index, flat_batch_vec, sub_edge_attr = _build_flat_generator_inputs(
                                sub_x_s, sub_mask_s, sample_batch
                            )
                            proxies_s, _ = generator(
                                flat_emb, mask=None,
                                edge_index=sub_edge_index, batch_vec=flat_batch_vec,
                                edge_attr=sub_edge_attr,
                            )
                        else:
                            proxies_s, _ = generator(sub_x_s, sub_mask_s)
                        cosine_mean, cosine_std = inter_proxy_cosine_stats(proxies_s)
                        break
            except Exception:
                pass

        if epoch % args.p3_eval_every == 0:
            # Evaluate: use (N-M) + generated proxies through Phase 2 model
            val_ap, val_loss = _evaluate_phase3(
                phase1_model, phase2_model, generator, val_loader,
                args.device, args, M)
            test_ap, _ = _evaluate_phase3(
                phase1_model, phase2_model, generator, test_loader,
                args.device, args, M)

            elapsed = time.time() - epoch_start
            mem_str = _mem_str(args)
            log_str = (
                f"Epoch {epoch:3d}/{args.p3_max_epochs} [{elapsed:.1f}s{mem_str}] | "
                f"loss={mean_loss:.4f} task={mean_task:.4f} recon={mean_recon:.4f} "
                f"(w={recon_w:.3f})"
            )
            if args.diversity_weight > 0:
                log_str += f" | cosine_mean={cosine_mean:.4f} cosine_std={cosine_std:.4f}"
            log_str += f" | val_AP={val_ap:.4f} test_AP={test_ap:.4f}"
            print(log_str, flush=True)

            diagnostics.append({
                "epoch": epoch, "loss": mean_loss, "task_loss": mean_task,
                "recon_loss": mean_recon, "recon_weight": recon_w,
                "val_ap": val_ap, "test_ap": test_ap,
            })

            if val_loss < best_val_loss:
                best_val_ap = val_ap
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "generator_state": generator.state_dict(),
                    "epoch": epoch, "val_ap": val_ap, "test_ap": test_ap,
                    "args": vars(args),
                }, save_path)
                print(f"  -> New best val loss={val_loss:.4f}", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= args.p3_patience:
                    print(f"Early stopping at epoch {epoch}. "
                          f"Best val loss={best_val_loss:.4f} at epoch {best_epoch}.",
                          flush=True)
                    break
        else:
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch:3d}/{args.p3_max_epochs} [{elapsed:.1f}s] | "
                  f"loss={mean_loss:.4f} task={mean_task:.4f} recon={mean_recon:.4f}",
                  flush=True)

    diag_path = os.path.join(args.save_dir, "phase3_diagnostics.pkl")
    with open(diag_path, "wb") as f:
        pickle.dump(diagnostics, f)
    print(f"Phase 3 done. Best val loss={best_val_loss:.4f} at epoch {best_epoch}.",
          flush=True)
    return save_path


@torch.no_grad()
def _evaluate_phase3(phase1_model, phase2_model, generator, loader, device,
                     args, num_drop):
    """
    Phase 3 evaluation: subsample (N-M) nodes, generate M proxies,
    pass through Phase 2 model.
    """
    phase1_model.eval()
    phase2_model.eval()
    generator.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_preds, all_labels, losses = [], [], []
    is_gred = args.backbone in ("gred", "hybrid")

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

        full_dense_x, full_dense_mask = phase1_model.encode_dense(batch)
        sub_x, sub_mask, _, _ = subsample_nodes(
            full_dense_x, full_dense_mask, num_drop)

        if _uses_flat_interface(args.generator):
            flat_emb, sub_edge_index, flat_batch_vec, sub_edge_attr = _build_flat_generator_inputs(
                sub_x, sub_mask, batch
            )
            proxies, _ = generator(
                flat_emb, mask=None,
                edge_index=sub_edge_index, batch_vec=flat_batch_vec,
                edge_attr=sub_edge_attr,
            )
        else:
            proxies, _ = generator(sub_x, sub_mask)

        if args.backbone == "vanilla_gt":
            logits, _ = phase2_model(
                batch, proxy_embeddings=proxies,
                precomputed_dense=(sub_x, sub_mask))
        elif args.backbone == "hybrid":
            sub_dm, sub_nm = subsample_dist_masks(
                dist_masks_batch, node_masks_batch, sub_mask)
            gred_h = phase2_model.encode_gred(sub_x, sub_dm, sub_nm)
            logits, _ = phase2_model(
                batch, sub_dm, sub_nm,
                proxy_embeddings=proxies,
                precomputed_dense=(sub_x, sub_mask),
                precomputed_gred=gred_h)
        elif args.backbone == "gred":
            sub_dm, sub_nm = subsample_dist_masks(
                dist_masks_batch, node_masks_batch, sub_mask)
            logits, _ = phase2_model(
                batch, sub_dm, sub_nm,
                precomputed_dense=(sub_x, sub_mask))

        losses.append(loss_fn(logits, batch.y).item())
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    ap = compute_macro_ap(
        np.concatenate(all_preds), np.concatenate(all_labels))
    return ap, float(np.mean(losses))


# ================================================================
# PHASE 4 — AUGMENTED EVALUATION
# ================================================================

def run_phase4(args, phase1_model_path, generator_path):
    """
    Augmented evaluation: N original nodes + generated proxies
    through the Phase 1 transformer. No training — evaluation only.
    """
    print("\n" + "=" * 60, flush=True)
    print("PHASE 4: Augmented Evaluation (N + generated nodes)", flush=True)
    print(f"  Backbone: {args.backbone}", flush=True)
    print("=" * 60, flush=True)
    is_gred = args.backbone in ("gred", "hybrid")
    proxy_multiplier = args.proxy_multiplier
    num_aug_proxies = proxy_multiplier * args.num_proxies

    _, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    # Load Phase 1 model. Built with proxy modules so cross_attn_router /
    # multi_point_proxy slots exist; use strict=False since Phase 1 checkpoint
    # has no router keys (it was trained without them).
    model = build_model(args).to(args.device)
    ckpt = torch.load(phase1_model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=False)
    _freeze(model)
    model.eval()

    # Load generator
    generator = build_generator(args).to(args.device)
    gen_ckpt = torch.load(generator_path, map_location=args.device, weights_only=True)
    generator.load_state_dict(gen_ckpt["generator_state"])
    _freeze(generator)
    generator.eval()

    # Attach multi-point proxy wrapper if enabled
    cross_attn_router = _build_cross_attn_router(args)
    multi_point_proxy = _build_multi_point_proxy(args, generator, cross_attn_router)
    if multi_point_proxy is not None:
        model.multi_point_proxy = multi_point_proxy
        print(f"  Multi-point proxy: insertion_layers={args.proxy_insertion_layers}, "
              f"separate_generators={args.separate_proxy_generators}, "
              f"separate_routers={args.separate_proxy_routers}", flush=True)

    # Baseline: Phase 1 model without proxies
    print("\nBaseline (Phase 1, no proxies):", flush=True)
    val_ap_base, val_loss_base = evaluate_model_only(
        model, val_loader, args.device, args)
    test_ap_base, test_loss_base = evaluate_model_only(
        model, test_loader, args.device, args)
    print(f"  Val  AP={val_ap_base:.4f} loss={val_loss_base:.4f}", flush=True)
    print(f"  Test AP={test_ap_base:.4f} loss={test_loss_base:.4f}", flush=True)

    # Augmented: N + 2M generated proxies, no node dropping.
    print(f"\nAugmented (N + {num_aug_proxies} generated proxies = 2M):",
          flush=True)
    val_ap_aug, val_loss_aug = evaluate_with_proxies(
        model, generator, val_loader, args.device, args,
        proxy_multiplier=proxy_multiplier)
    test_ap_aug, test_loss_aug = evaluate_with_proxies(
        model, generator, test_loader, args.device, args,
        proxy_multiplier=proxy_multiplier)
    print(f"  Val  AP={val_ap_aug:.4f} loss={val_loss_aug:.4f}", flush=True)
    print(f"  Test AP={test_ap_aug:.4f} loss={test_loss_aug:.4f}", flush=True)

    # Summary
    print("\n" + "-" * 40, flush=True)
    print("Phase 4 Summary:", flush=True)
    print(f"  Val  AP: {val_ap_base:.4f} → {val_ap_aug:.4f} "
          f"(delta={val_ap_aug - val_ap_base:+.4f})", flush=True)
    print(f"  Test AP: {test_ap_base:.4f} → {test_ap_aug:.4f} "
          f"(delta={test_ap_aug - test_ap_base:+.4f})", flush=True)

    # Save results
    results = {
        "val_ap_baseline": val_ap_base,
        "val_ap_augmented": val_ap_aug,
        "test_ap_baseline": test_ap_base,
        "test_ap_augmented": test_ap_aug,
        "proxy_multiplier": proxy_multiplier,
        "num_generated_proxies": num_aug_proxies,
        "val_delta": val_ap_aug - val_ap_base,
        "test_delta": test_ap_aug - test_ap_base,
    }
    results_path = os.path.join(args.save_dir, "phase4_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_path}", flush=True)

    return results


# ================================================================
# PHASE 5 — END-TO-END FINE-TUNING
# ================================================================

def run_phase5(args, phase1_model_path, generator_path):
    """
    End-to-end fine-tuning: jointly train transformer + generator from epoch 1.
    Uses the same learning rate for both model and generator.
    """
    print("\n" + "=" * 60, flush=True)
    print("PHASE 5: End-to-End Fine-tuning", flush=True)
    print(f"  Backbone: {args.backbone}", flush=True)
    joint_lr = args.p5_lr_gen
    if not np.isclose(args.p5_lr_model, joint_lr):
      print(f"  Overriding model LR {args.p5_lr_model:.2e} -> {joint_lr:.2e} "
          f"to match generator LR for joint training.", flush=True)
    args.p5_lr_model = joint_lr
    print(f"  Joint LR (model + generator): {joint_lr:.2e}", flush=True)
    print(f"  Proxy dropout: {args.p5_proxy_dropout}", flush=True)
    print("=" * 60, flush=True)
    is_gred = args.backbone in ("gred", "hybrid")

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=args.max_hops,
        dist_mask_workers=args.dist_mask_workers,
        use_lap_pe=args.use_lap_pe, lap_pe_dim=args.lap_pe_dim,
    )

    # Load Phase 1 model. Built with proxy modules (cross_attn_router) so they
    # can be fine-tuned in Phase 5; strict=False skips missing router keys.
    model = build_model(args).to(args.device)
    ckpt = torch.load(phase1_model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=False)

    # Load Phase 3 generator
    generator = build_generator(args).to(args.device)
    gen_ckpt = torch.load(generator_path, map_location=args.device, weights_only=True)
    generator.load_state_dict(gen_ckpt["generator_state"])

    # Attach multi-point proxy wrapper if enabled
    cross_attn_router = _build_cross_attn_router(args)
    multi_point_proxy = _build_multi_point_proxy(args, generator, cross_attn_router)
    if multi_point_proxy is not None:
        model.multi_point_proxy = multi_point_proxy
        print(f"  Multi-point proxy: insertion_layers={args.proxy_insertion_layers}, "
              f"separate_generators={args.separate_proxy_generators}, "
              f"separate_routers={args.separate_proxy_routers}", flush=True)

    loss_fn = nn.BCEWithLogitsLoss()

    # Phase A: freeze model, train generator only
    _freeze(model)
    # When multi-point is active, generator is inside model, so unfreeze it
    if getattr(model, 'multi_point_proxy', None) is not None:
        _unfreeze(model.multi_point_proxy)

    phase_a_epochs = min(args.p5_phase_a_epochs, args.p5_max_epochs)
    phase_a_total_steps = max(1, len(train_loader) * max(phase_a_epochs, 1))
    phase_b_total_steps = max(1, len(train_loader) * max(args.p5_max_epochs - phase_a_epochs, 1))

    # When multi-point is active, generator params are inside model, don't add separately
    if getattr(model, 'multi_point_proxy', None) is not None:
        gen_named_params = [(f"multi_point_proxy.{name}", param)
                            for name, param in model.multi_point_proxy.named_parameters()]
    else:
        gen_named_params = [(f"generator.{name}", param) for name, param in generator.named_parameters()]

    opt_a, sched_a = build_grouped_optimizer_and_scheduler(
        named_parameters=gen_named_params,
        lr_max=args.p5_lr_gen,
        lr_min=args.lr_min,
        weight_decay=args.p5_weight_decay,
        total_steps=phase_a_total_steps,
        warmup_ratio=args.warmup_ratio,
        recurrent_lr_factor=1.0,
    )
    opt_b = None
    sched_b = None

    best_val_ap = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    current_phase = "A"
    save_path = os.path.join(args.save_dir, "phase5_best.pt")

    for epoch in range(1, args.p5_max_epochs + 1):
        epoch_start = time.time()

        # Phase transition A → B
        if epoch > args.p5_phase_a_epochs and current_phase == "A":
            current_phase = "B"
            _unfreeze(model)

            if args.backbone == "vanilla_gt":
                model_params = [
                    {"params": model.encoder.parameters(),
                     "lr": args.p5_lr_model},
                    {"params": model.layers.parameters(),
                     "lr": args.p5_lr_model},
                    {"params": model.head.parameters(),
                     "lr": args.p5_lr_model},
                ]
            elif args.backbone == "gred":
                model_params = [
                    {"params": model.encoder.parameters(),
                     "lr": args.p5_lr_model},
                    {"params": model.layers.parameters(),
                     "lr": args.p5_lr_model},
                    {"params": model.head.parameters(),
                     "lr": args.p5_lr_model},
                ]
            elif args.backbone == "hybrid":
                model_params = [
                    {"params": model.encoder.parameters(),
                     "lr": args.p5_lr_model},
                    {"params": model.gred_layers.parameters(),
                     "lr": args.p5_lr_model},
                    {"params": model.transformer_layers.parameters(),
                     "lr": args.p5_lr_model},
                    {"params": model.head.parameters(),
                     "lr": args.p5_lr_model},
                ]

            # When multi-point is active, generator is inside model
            if getattr(model, 'multi_point_proxy', None) is not None:
                opt_b = torch.optim.AdamW(model_params,
                                          weight_decay=args.p5_weight_decay)
            else:
                opt_b = torch.optim.AdamW(
                    model_params + [
                        {"params": generator.parameters(), "lr": args.p5_lr_gen}
                    ],
                    weight_decay=args.p5_weight_decay)
            sched_b = build_warmup_cosine_scheduler(
                optimizer=opt_b,
                total_steps=phase_b_total_steps,
                lr_min=args.lr_min,
                warmup_ratio=args.warmup_ratio,
            )
            print(f"  [Epoch {epoch}] Switching to Phase B — model unfrozen.",
                  flush=True)

        optimizer = opt_a if current_phase == "A" else opt_b
        scheduler = sched_a if current_phase == "A" else sched_b

        model.train()
        generator.train()
        train_losses, all_preds, all_labels = [], [], []

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

            # --- Multi-point proxy path ---
            if getattr(model, 'multi_point_proxy', None) is not None:
                # Model generates + routes proxies internally
                if args.backbone == "vanilla_gt":
                    logits, node_emb_with = model(batch, readout_scope=args.readout_scope)
                elif args.backbone == "hybrid":
                    logits, node_emb_with = model(batch, dist_masks_batch, node_masks_batch,
                                      readout_scope=args.readout_scope)
                else:
                    raise ValueError("GRED backbone does not support proxy integration. Use 'hybrid'.")

                # Novelty loss (when enabled)
                use_novelty_mp = args.novelty_alpha > 0 or args.novelty_alpha_node > 0
                if use_novelty_mp:
                    with torch.no_grad():
                        if args.backbone == "vanilla_gt":
                            logits_without, node_emb_without = model(
                                batch, readout_scope=args.readout_scope, disable_proxy_injection=True)
                        elif args.backbone == "hybrid":
                            logits_without, node_emb_without = model(
                                batch, dist_masks_batch, node_masks_batch,
                                readout_scope=args.readout_scope, disable_proxy_injection=True)

                    node_w = node_emb_with if args.novelty_alpha_node > 0 else None
                    node_wo = node_emb_without if args.novelty_alpha_node > 0 else None
                    base_task_loss = loss_fn(logits, batch.y)
                    nov_total, _ = novelty_loss(
                        base_task_loss, logits, logits_without.detach(),
                        node_w, node_wo.detach() if node_wo is not None else None,
                        mask=None, alpha=args.novelty_alpha,
                        alpha_node=args.novelty_alpha_node,
                        temperature=args.novelty_temperature)
                    loss = nov_total
                else:
                    loss = loss_fn(logits, batch.y)

                # Proxy diversity loss (proxies from multi-point wrapper)
                if args.diversity_weight > 0 and hasattr(model.multi_point_proxy, '_last_proxies'):
                    loss = loss + args.diversity_weight * proxy_diversity_loss(
                        model.multi_point_proxy._last_proxies)

                aux_loss = model._last_mp_aux_loss
                if aux_loss is not None:
                    if not isinstance(aux_loss, torch.Tensor):
                        aux_loss = torch.tensor(aux_loss, device=batch.x.device)
                    loss = loss + aux_loss
                params_to_clip = list(model.parameters())
            else:
                # --- Single-point proxy path (existing behaviour) ---
                dense_x, dense_mask = model.encode_dense(batch)

                gred_h = None
                if args.backbone == "hybrid":
                    gred_h = model.encode_gred(
                        dense_x, dist_masks_batch, node_masks_batch)

                # Proxy dropout during joint training.
                # In Phase A the model is fully frozen — skipping proxies yields
                # a loss with no grad_fn and breaks backward.  Only apply dropout
                # in Phase B when the model is also being trained.
                use_proxy = (current_phase == "A") or (torch.rand(1).item() > args.p5_proxy_dropout)

                if use_proxy:
                    proxies, aux_loss = _generate_proxies(
                        model, generator, batch, dense_x, dense_mask, args,
                        gred_h=gred_h, num_proxy_sets=args.proxy_multiplier)

                    if args.backbone == "vanilla_gt":
                        logits, node_emb_with = model(
                            batch, proxy_embeddings=proxies,
                            precomputed_dense=(dense_x, dense_mask))
                    elif args.backbone == "hybrid":
                        logits, node_emb_with = model(
                            batch, dist_masks_batch, node_masks_batch,
                            proxy_embeddings=proxies,
                            precomputed_dense=(dense_x, dense_mask),
                            precomputed_gred=gred_h)
                    elif args.backbone == "gred":
                        logits, node_emb_with = model(
                            batch, dist_masks_batch, node_masks_batch)

                    # Novelty loss (when enabled)
                    use_novelty_p5 = args.novelty_alpha > 0 or args.novelty_alpha_node > 0
                    if use_novelty_p5:
                        with torch.no_grad():
                            if args.backbone == "vanilla_gt":
                                logits_without, node_emb_without = model(
                                    batch, precomputed_dense=(dense_x, dense_mask))
                            elif args.backbone == "hybrid":
                                logits_without, node_emb_without = model(
                                    batch, dist_masks_batch, node_masks_batch,
                                    precomputed_dense=(dense_x, dense_mask),
                                    precomputed_gred=gred_h)
                            elif args.backbone == "gred":
                                logits_without, node_emb_without = model(
                                    batch, dist_masks_batch, node_masks_batch,
                                    precomputed_dense=(dense_x, dense_mask))

                        node_w = node_emb_with if args.novelty_alpha_node > 0 else None
                        node_wo = node_emb_without if args.novelty_alpha_node > 0 else None
                        base_task_loss = loss_fn(logits, batch.y)
                        nov_total, _ = novelty_loss(
                            base_task_loss, logits, logits_without.detach(),
                            node_w, node_wo.detach() if node_wo is not None else None,
                            mask=None, alpha=args.novelty_alpha,
                            alpha_node=args.novelty_alpha_node,
                            temperature=args.novelty_temperature)
                        loss = nov_total
                    else:
                        loss = loss_fn(logits, batch.y)

                    # Proxy diversity loss
                    if args.diversity_weight > 0:
                        loss = loss + args.diversity_weight * proxy_diversity_loss(proxies)

                    if aux_loss is not None:
                        loss = loss + aux_loss
                else:
                    # Proxy dropout: skip proxies and novelty/diversity losses
                    if args.backbone == "vanilla_gt":
                        logits, _ = model(
                            batch, precomputed_dense=(dense_x, dense_mask))
                    elif args.backbone == "hybrid":
                        logits, _ = model(
                            batch, dist_masks_batch, node_masks_batch,
                            precomputed_dense=(dense_x, dense_mask),
                            precomputed_gred=gred_h)
                    elif args.backbone == "gred":
                        logits, _ = model(
                            batch, dist_masks_batch, node_masks_batch,
                            precomputed_dense=(dense_x, dense_mask))
                    loss = loss_fn(logits, batch.y)

                params_to_clip = list(generator.parameters()) + list(model.parameters())

            loss.backward()

            nn.utils.clip_grad_norm_(params_to_clip, args.p5_grad_clip)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        train_ap = compute_macro_ap(
            np.concatenate(all_preds), np.concatenate(all_labels))
        train_loss = float(np.mean(train_losses))

        val_ap, val_loss = evaluate_with_proxies(
            model, generator, val_loader, args.device, args,
            proxy_multiplier=args.proxy_multiplier)
        test_ap, _ = evaluate_with_proxies(
            model, generator, test_loader, args.device, args,
            proxy_multiplier=args.proxy_multiplier)

        elapsed = time.time() - epoch_start
        mem_str = _mem_str(args)
        print(
            f"Epoch {epoch:3d}/{args.p5_max_epochs} "
            f"[joint][{elapsed:.1f}s{mem_str}] | "
            f"train_loss={train_loss:.4f} train_AP={train_ap:.4f} | "
            f"val_loss={val_loss:.4f} val_AP={val_ap:.4f} | "
            f"test_AP={test_ap:.4f}",
            flush=True)

        if val_loss < best_val_loss:
            best_val_ap = val_ap
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "generator_state": generator.state_dict(),
                "epoch": epoch, "val_ap": val_ap, "test_ap": test_ap,
                "args": vars(args),
            }, save_path)
            print(f"  -> New best val loss={val_loss:.4f} (test={test_ap:.4f})",
                  flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.p5_patience:
                print(f"Early stopping at epoch {epoch}. "
                      f"Best val loss={best_val_loss:.4f} at epoch {best_epoch}.",
                      flush=True)
                break

    print(f"Phase 5 done. Best val loss={best_val_loss:.4f} at epoch {best_epoch}.",
          flush=True)
    return save_path


# ================================================================
# UTILITY
# ================================================================

def _mem_str(args):
    """GPU memory usage string for logging."""
    if args.device.startswith("cuda"):
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
        return f" mem={mem_mb:.0f}MB"
    return ""


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    save_code_snapshot(args.save_dir)

    phases = [args.phase] if args.phase != "all" else ["1", "2", "3", "4", "5"]
    model_path = args.model_path
    phase2_model_path = args.phase2_model_path
    generator_path = args.generator_path

    for phase in phases:
        if phase == "1":
            model_path = run_phase1(args)

        elif phase == "2":
            if model_path is None:
                model_path = os.path.join(args.save_dir, "phase1_best.pt")
            assert os.path.exists(model_path), \
                f"Phase 2 requires Phase 1 model at {model_path}"
            phase2_model_path = run_phase2(args, model_path)

        elif phase == "3":
            if model_path is None:
                model_path = os.path.join(args.save_dir, "phase1_best.pt")
            if phase2_model_path is None:
                phase2_model_path = os.path.join(args.save_dir, "phase2_best.pt")
            assert os.path.exists(model_path), \
                f"Phase 3 requires Phase 1 model at {model_path}"
            assert os.path.exists(phase2_model_path), \
                f"Phase 3 requires Phase 2 model at {phase2_model_path}"
            generator_path = run_phase3(args, model_path, phase2_model_path)

        elif phase == "4":
            if model_path is None:
                model_path = os.path.join(args.save_dir, "phase1_best.pt")
            if generator_path is None:
                generator_path = os.path.join(args.save_dir, "phase3_generator.pt")
            assert os.path.exists(model_path), \
                f"Phase 4 requires Phase 1 model at {model_path}"
            assert os.path.exists(generator_path), \
                f"Phase 4 requires generator at {generator_path}"
            run_phase4(args, model_path, generator_path)

        elif phase == "5":
            if model_path is None:
                model_path = os.path.join(args.save_dir, "phase1_best.pt")
            if generator_path is None:
                generator_path = os.path.join(args.save_dir, "phase3_generator.pt")
            assert os.path.exists(model_path), \
                f"Phase 5 requires Phase 1 model at {model_path}"
            assert os.path.exists(generator_path), \
                f"Phase 5 requires generator at {generator_path}"
            run_phase5(args, model_path, generator_path)

    print("\nAll requested phases complete.", flush=True)
