"""
Experiment: Attention Distillation from Proxy-Augmented Transformer

Two-phase approach:
  Phase 1 (extract — run once, slow):
    - Load pretrained transformer + randomly initialized cross-attention routing layers
    - For each training graph, optimize proxy embeddings through cross-attention routing
    - Extract N×N self-attention matrices from each transformer layer
    - Save attention targets to disk (one file per graph)

  Phase 2 (train — run many epochs, fast):
    - Load saved attention targets
    - Train vanilla transformer (no proxies, no cross-attention) with:
      loss = task_BCE + λ * KL(saved_attn_targets, current_attn)

Cross-attention routing per layer (Phase 1 only):
  N→M: proxies aggregate from nodes (Q=proxy, K=V=nodes)
  M→M: proxy self-attention
  M→N: nodes aggregate from proxies (Q=nodes, K=V=proxies)
  Result: enriched node embeddings → different N×N self-attention patterns

Usage:
    # Pretrain from scratch (skip if you already have a checkpoint)
    python exp_attn_distill.py pretrain --hidden_dim 128 --num_layers 3

    # Phase 1: Extract attention targets (run once)
    python exp_attn_distill.py extract --model_path checkpoints_attn_distill/pretrain_best.pt

    # Phase 2: Train with distillation (run many times with different hyperparams)
    python exp_attn_distill.py train --model_path checkpoints_attn_distill/pretrain_best.pt
    python exp_attn_distill.py train --model_path checkpoints_attn_distill/pretrain_best.pt --distill_weight 0.5

    # All three phases in sequence
    python exp_attn_distill.py pretrain --hidden_dim 128 --num_layers 3
    python exp_attn_distill.py extract --model_path checkpoints_attn_distill/pretrain_best.pt
    python exp_attn_distill.py train --model_path checkpoints_attn_distill/pretrain_best.pt
"""

import argparse
import os
import pickle
import time
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
from torch.utils.data import Dataset, DataLoader

from data import get_loaders, _compute_dist_mask_single
from models import NodeEncoder, GraphTransformer, GREDEncoder, GREDHybridTransformer
from metrics import compute_macro_ap
from mmd import mmd_squared


# ================================================================
# CROSS-ATTENTION ROUTING LAYER (N→M→N)
# ================================================================

class CrossAttentionRoutingLayer(nn.Module):
    """
    Single N→M→N cross-attention routing layer.

    Step 1 (N→M): Proxies attend to nodes — proxies gather info from graph.
    Step 2 (M→M): Proxy self-attention — proxies exchange info among themselves.
    Step 3 (M→N): Nodes attend to proxies — nodes absorb proxy-aggregated info.
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1, use_proxy_self_attn=True):
        super().__init__()
        self.use_proxy_self_attn = use_proxy_self_attn

        # N→M: proxies attend to nodes
        self.norm_n2m_q = nn.LayerNorm(hidden_dim)
        self.norm_n2m_kv = nn.LayerNorm(hidden_dim)
        self.n2m_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # M→M: proxy self-attention (optional)
        if use_proxy_self_attn:
            self.norm_mm = nn.LayerNorm(hidden_dim)
            self.mm_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm_mm_ff = nn.LayerNorm(hidden_dim)
            self.mm_ff = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim),
            )

        # M→N: nodes attend to proxies
        self.norm_m2n_q = nn.LayerNorm(hidden_dim)
        self.norm_m2n_kv = nn.LayerNorm(hidden_dim)
        self.m2n_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # FFN after M→N
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, nodes, proxies, node_mask):
        node_key_pad = ~node_mask

        # Step 1: N→M
        q = self.norm_n2m_q(proxies)
        kv = self.norm_n2m_kv(nodes)
        proxy_update, _ = self.n2m_attn(q, kv, kv, key_padding_mask=node_key_pad)
        proxies = proxies + proxy_update

        # Step 2: M→M
        if self.use_proxy_self_attn:
            normed = self.norm_mm(proxies)
            mm_out, _ = self.mm_attn(normed, normed, normed)
            proxies = proxies + mm_out
            proxies = proxies + self.mm_ff(self.norm_mm_ff(proxies))

        # Step 3: M→N
        q = self.norm_m2n_q(nodes)
        kv = self.norm_m2n_kv(proxies)
        node_update, _ = self.m2n_attn(q, kv, kv)
        nodes = nodes + node_update

        nodes = nodes + self.ff(self.norm_ff(nodes))
        return nodes, proxies


# ================================================================
# PARAMETER-FREE PROXY ROUTING (Form B)
# ================================================================

def parameter_free_proxy_routing(nodes, proxies, node_mask):
    """
    Parameter-free Form B routing used in Phase 1 only.
    Nodes attend to proxies (no W_Q / W_K / W_V / FFN), receive a residual update.
    Proxies live in the same space as nodes, which makes MMD-to-nodes a meaningful
    regularizer and shrinks the gaming surface vs. learnable cross-attention.

    Args:
        nodes:     (B, N, d)
        proxies:   (B, M, d)   — the optimization variable
        node_mask: (B, N) bool — True for real nodes
    Returns:
        updated nodes: (B, N, d)
    """
    d = nodes.size(-1)
    scores = torch.matmul(nodes, proxies.transpose(-2, -1)) / (d ** 0.5)  # (B, N, M)
    attn = F.softmax(scores, dim=-1)                                       # (B, N, M)
    update = torch.matmul(attn, proxies)                                   # (B, N, d)
    update = update * node_mask.unsqueeze(-1).float()                      # zero pad rows
    return nodes + update


# ================================================================
# TRANSFORMER LAYER — returns attention weights
# ================================================================

class TransformerLayerWithAttn(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=8, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wout = nn.Linear(hidden_dim, hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.res_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        B, N, d = x.shape
        normed = self.norm1(x)

        q = self.wq(normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            key_pad = (~mask).unsqueeze(1).unsqueeze(2)
            query_pad = (~mask).unsqueeze(1).unsqueeze(-1)
            attn = attn.masked_fill(key_pad, float("-inf"))
            attn = attn.masked_fill(query_pad, float("-inf"))

        attn_w = F.softmax(attn, dim=-1)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        attn_w_clean = attn_w.detach().clone()

        attn_w_dropped = self.attn_drop(attn_w)
        out = (attn_w_dropped @ v).transpose(1, 2).reshape(B, N, d)
        x = x + self.res_drop(self.wout(out))
        x = x + self.res_drop(self.ff(self.norm2(x)))
        return x, attn_w_clean


# ================================================================
# GRAPH TRANSFORMER WITH CROSS-ATTENTION PROXY ROUTING
# ================================================================

class GraphTransformerWithCrossAttn(nn.Module):
    def __init__(self, num_layers=5, num_heads=8, hidden_dim=64,
                 output_dim=10, dropout=0.3, num_cross_layers=None,
                 use_proxy_self_attn=True, use_parameter_free_proxy=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # When True, the cross_attn_layers ModuleList is left in place (so checkpoints
        # and other files that import this class still work) but is bypassed in the
        # forward pass in favor of parameter_free_proxy_routing.
        self.use_parameter_free_proxy = use_parameter_free_proxy
        self.encoder = NodeEncoder(hidden_dim)

        self.layers = nn.ModuleList([
            TransformerLayerWithAttn(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        n_cross = num_cross_layers if num_cross_layers is not None else num_layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionRoutingLayer(hidden_dim, num_heads, dropout, use_proxy_self_attn)
            for _ in range(n_cross)
        ])
        self.cross_layer_mapping = list(range(num_layers - n_cross, num_layers))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
        )

    def encode_nodes(self, batch):
        return self.encoder(batch.x, batch.edge_index, batch.edge_attr)

    def encode_dense(self, batch):
        h = self.encode_nodes(batch)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, proxy_embeddings=None, precomputed_dense=None,
                return_attention=False):
        if precomputed_dense is not None:
            dense_x, dense_mask = precomputed_dense
        else:
            dense_x, dense_mask = self.encode_dense(batch)

        B, max_N, d = dense_x.shape
        all_attn = []
        proxies = proxy_embeddings

        for layer_idx, layer in enumerate(self.layers):
            if proxies is not None and layer_idx in self.cross_layer_mapping:
                if self.use_parameter_free_proxy:
                    # Form B: bypass learnable cross-attn, modify dense_x via
                    # node→proxy attention with no learnable parameters.
                    dense_x = parameter_free_proxy_routing(
                        dense_x, proxies, dense_mask)
                else:
                    cross_idx = self.cross_layer_mapping.index(layer_idx)
                    dense_x, proxies = self.cross_attn_layers[cross_idx](
                        dense_x, proxies, dense_mask)

            dense_x, attn_w = layer(dense_x, dense_mask)
            if return_attention:
                all_attn.append(attn_w)

        node_emb_masked = dense_x[dense_mask]
        pooled = global_mean_pool(node_emb_masked, batch.batch)
        logits = self.head(pooled)

        if return_attention:
            return logits, node_emb_masked, all_attn
        return logits, node_emb_masked


# ================================================================
# MODEL FACTORY
# ================================================================

def build_model(args):
    """Build backbone model based on --backbone arg."""
    lap_pe_dim = args.lap_pe_dim if getattr(args, 'use_lap_pe', False) else 0
    if getattr(args, 'backbone', 'vanilla_gt') == "vanilla_gt":
        return GraphTransformerWithCrossAttn(
            num_layers=args.num_layers, num_heads=args.num_heads,
            hidden_dim=args.hidden_dim, output_dim=args.output_dim,
            dropout=args.dropout,
            use_parameter_free_proxy=getattr(args, 'use_parameter_free_proxy', True),
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
        )
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")


def _build_dist_and_node_masks_for_batch(pyg_batch, max_hops, device):
    """Build padded distance masks and node masks for a PyG batch."""
    graphs = pyg_batch.to_data_list()
    B = len(graphs)
    max_N = max(g.x.size(0) for g in graphs)

    dist_masks = torch.zeros(B, max_hops, max_N, max_N, device=device)
    node_masks = torch.zeros(B, max_N, dtype=torch.bool, device=device)

    for i, g in enumerate(graphs):
        n = g.x.size(0)
        adj = np.zeros((n, n), dtype=np.float32)
        edge_index = g.edge_index.detach().cpu().numpy()
        adj[edge_index[0], edge_index[1]] = 1.0

        dm = _compute_dist_mask_single(adj, max_hops=max_hops)
        k_use = min(dm.shape[0], max_hops)
        dist_masks[i, :k_use, :n, :n] = torch.from_numpy(dm[:k_use].astype(np.float32)).to(device)
        node_masks[i, :n] = True

    return dist_masks, node_masks


# ================================================================
# PROXY OPTIMIZATION
# ================================================================

def _per_sample_bce(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)


@torch.no_grad()
def optimize_proxies_for_batch(model, batch, dense_x, dense_mask, args):
    B = batch.y.size(0)
    device = batch.y.device
    mmd_lambda = getattr(args, 'proxy_mmd_lambda', 0.0)
    rel_threshold = getattr(args, 'extract_rel_threshold', 0.1)
    jitter_scale = getattr(args, 'proxy_init_jitter', 0.5)

    # Simple init: pick a random node embedding per (graph, proxy_slot) and add
    # large jitter. The jitter breaks symmetry so the M proxies for one graph
    # follow distinct optimization trajectories instead of collapsing together.
    # Proxies are throwaway after extraction, so we don't need anything fancier.
    _, N_pad, d = dense_x.shape
    M = args.num_proxies
    idx = torch.randint(0, N_pad, (B, M), device=device)
    batch_arange = torch.arange(B, device=device).view(B, 1).expand(B, M)
    proxy = dense_x[batch_arange, idx].clone()              # (B, M, d) on-manifold
    proxy = proxy + jitter_scale * torch.randn_like(proxy)  # large jitter
    proxy = nn.Parameter(proxy)
    opt = torch.optim.Adam([proxy], lr=args.proxy_lr)
    proxy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.proxy_opt_steps, eta_min=args.proxy_lr * 0.01)

    logits_base, _, _ = model(batch, precomputed_dense=(dense_x, dense_mask),
                               return_attention=True)
    base_loss = _per_sample_bce(logits_base, batch.y)
    best_loss = base_loss.clone()
    best_proxy = proxy.detach().clone()

    # Per-sample relative target: opt_loss[i] < rel_threshold * base_loss[i].
    # Exit early only when ALL samples have crossed it (no per-sample freezing
    # — we want to keep pushing opt_loss as low as possible, ideally toward 0).
    target = rel_threshold * base_loss
    steps_taken = 0

    with torch.enable_grad():
        for step in range(args.proxy_opt_steps):
            if (best_loss < target).all():
                break

            opt.zero_grad()
            logits, _, _ = model(batch, proxy_embeddings=proxy,
                                  precomputed_dense=(dense_x, dense_mask),
                                  return_attention=True)
            task_loss = _per_sample_bce(logits, batch.y)

            if mmd_lambda > 0:
                mmd_losses = []
                for i in range(B):
                    nodes_i = dense_x[i][dense_mask[i]]
                    mmd_losses.append(mmd_squared(proxy[i], nodes_i))
                mmd_batch = torch.stack(mmd_losses)
                total = (task_loss + mmd_lambda * mmd_batch).mean()
            else:
                total = task_loss.mean()

            total.backward()
            nn.utils.clip_grad_norm_([proxy], 1.0)
            opt.step()
            proxy_scheduler.step()
            steps_taken = step + 1

            with torch.no_grad():
                improved = task_loss < best_loss
                if improved.any():
                    best_loss[improved] = task_loss[improved]
                    best_proxy[improved] = proxy.detach()[improved]

    num_converged = int((best_loss < target).sum().item())
    return best_proxy, base_loss, best_loss, steps_taken, num_converged


# ================================================================
# PRETRAIN — Train vanilla transformer from scratch
# ================================================================

def run_pretrain(args):
    backbone = getattr(args, 'backbone', 'vanilla_gt')
    is_gred = backbone in ("gred", "hybrid")

    print("\n" + "=" * 60, flush=True)
    print("PRETRAIN: Train Model from scratch", flush=True)
    print(f"  backbone={backbone}", flush=True)
    print(f"  hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, "
          f"num_heads={args.num_heads}", flush=True)
    print(f"  s1_lr={args.s1_lr}, s1_weight_decay={args.s1_weight_decay}, "
          f"dropout={args.dropout}", flush=True)
    print(f"  s1_max_epochs={args.s1_max_epochs}, s1_patience={args.s1_patience}",
          flush=True)
    print("=" * 60, flush=True)

    use_lap_pe = getattr(args, 'use_lap_pe', False)
    lap_pe_dim = getattr(args, 'lap_pe_dim', 0)
    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=getattr(args, 'max_hops', 40),
        dist_mask_workers=getattr(args, 'dist_mask_workers', 8),
        use_lap_pe=use_lap_pe, lap_pe_dim=lap_pe_dim,
    )

    if backbone == "vanilla_gt":
        model = GraphTransformerWithCrossAttn(
            num_layers=args.num_layers, num_heads=args.num_heads,
            hidden_dim=args.hidden_dim, output_dim=args.output_dim,
            dropout=args.dropout,
            use_parameter_free_proxy=getattr(args, 'use_parameter_free_proxy', True),
        ).to(args.device)
    else:
        model = build_model(args).to(args.device)

    # Only count encoder + self-attention + head params (not cross-attn for vanilla_gt)
    trainable_params = []
    for name, param in model.named_parameters():
        if "cross_attn_layers" not in name:
            param.requires_grad_(True)
            trainable_params.append(param)
        else:
            param.requires_grad_(False)

    print(f"  Parameters: {sum(p.numel() for p in trainable_params):,}", flush=True)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.s1_lr,
                                   weight_decay=args.s1_weight_decay)
    total_steps = max(1, len(train_loader) * args.s1_max_epochs)
    # Warmup + cosine decay
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(args.lr_min / args.s1_lr, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.BCEWithLogitsLoss()
    best_val_ap = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    save_path = os.path.join(args.save_dir, "pretrain_best.pt")

    def _unpack_batch(batch_data):
        if is_gred:
            batch, dist_masks_batch, node_masks_batch = batch_data
            batch = batch.to(args.device)
            dist_masks_batch = dist_masks_batch.to(args.device)
            node_masks_batch = node_masks_batch.to(args.device)
        else:
            batch = batch_data.to(args.device)
            dist_masks_batch = None
            node_masks_batch = None
        return batch, dist_masks_batch, node_masks_batch

    def _forward_no_proxy(batch, dist_masks_batch, node_masks_batch):
        if backbone == "vanilla_gt":
            return model(batch, return_attention=False)
        else:
            return model(batch, dist_masks_batch, node_masks_batch)

    for epoch in range(1, args.s1_max_epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_losses = []
        all_preds, all_labels = [], []

        for batch_data in train_loader:
            batch, dist_masks_batch, node_masks_batch = _unpack_batch(batch_data)
            optimizer.zero_grad()
            logits, _ = _forward_no_proxy(batch, dist_masks_batch, node_masks_batch)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, args.s1_grad_clip)
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
            for batch_data in val_loader:
                batch, dist_masks_batch, node_masks_batch = _unpack_batch(batch_data)
                logits, _ = _forward_no_proxy(batch, dist_masks_batch, node_masks_batch)
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
                for batch_data in test_loader:
                    batch, dist_masks_batch, node_masks_batch = _unpack_batch(batch_data)
                    logits, _ = _forward_no_proxy(batch, dist_masks_batch, node_masks_batch)
                    test_preds.append(torch.sigmoid(logits).cpu().numpy())
                    test_labels.append(batch.y.cpu().numpy())
            test_ap = compute_macro_ap(
                np.concatenate(test_preds), np.concatenate(test_labels))
            log_line += f" | test_AP={test_ap:.4f}"

        print(log_line, flush=True)

        # Early stopping
        improved_val_loss = val_loss < best_val_loss
        improved_val_ap = val_ap > best_val_ap
        if improved_val_loss or improved_val_ap:
            if improved_val_loss:
                best_val_loss = val_loss
            if improved_val_ap:
                best_val_ap = val_ap
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch, "val_ap": val_ap,
                "args": vars(args),
            }, save_path)
            print(f"  -> New best: val_loss={best_val_loss:.4f} "
                  f"val_AP={best_val_ap:.4f}, saved", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.s1_patience:
                print(f"Early stopping at epoch {epoch}. "
                      f"Best val_loss={best_val_loss:.4f} val_AP={best_val_ap:.4f} "
                      f"at epoch {best_epoch}.", flush=True)
                break

    print(f"Pretrain done. Best val_loss={best_val_loss:.4f} "
          f"val_AP={best_val_ap:.4f} at epoch {best_epoch}.", flush=True)
    return save_path


# ================================================================
# PHASE 1: EXTRACT ATTENTION TARGETS
# ================================================================

def _extract_attention_for_loader(teacher, loader, args, distill_layer_indices,
                                    save_path, split_name="train"):
    """Run proxy optimization + attention extraction over a single loader.

    Loader MUST be non-shuffled so ``sample_idx == dataset index``. Saves a
    pickle to ``save_path`` and returns it.
    """
    print(f"\n--- Extracting {split_name} split -> {save_path} ---", flush=True)
    all_targets = []
    graphs_processed = 0
    total_base_loss, total_opt_loss = 0.0, 0.0
    total_converged = 0
    total_steps_taken = 0
    total_mmd = 0.0
    all_ratios = []  # opt_loss / base_loss per sample, for histogram
    num_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(args.device)
        B = batch.y.size(0)

        with torch.no_grad():
            dense_x, dense_mask = teacher.encode_dense(batch)

        # Optimize proxies (runs until opt_loss < rel_threshold * base_loss for
        # all samples, or max steps reached)
        best_proxies, base_loss, opt_loss, steps_taken, num_converged = \
            optimize_proxies_for_batch(teacher, batch, dense_x, dense_mask, args)

        total_base_loss += base_loss.sum().item()
        total_converged += num_converged
        total_steps_taken += steps_taken
        total_opt_loss += opt_loss.sum().item()

        # Per-sample ratio for diagnostic histogram
        with torch.no_grad():
            ratio_b = (opt_loss / base_loss.clamp_min(1e-12)).cpu().tolist()
            all_ratios.extend(ratio_b)

        # Log MMD
        if args.proxy_mmd_lambda > 0:
            with torch.no_grad():
                for i in range(B):
                    nodes_i = dense_x[i][dense_mask[i]]
                    total_mmd += mmd_squared(best_proxies[i], nodes_i).item()

        # Extract teacher attention with optimized proxies
        with torch.no_grad():
            _, _, teacher_attns = teacher(
                batch, proxy_embeddings=best_proxies,
                precomputed_dense=(dense_x, dense_mask),
                return_attention=True)

        # Save per-graph attention targets (only distilled layers, only real nodes)
        for i in range(B):
            n_i = int(dense_mask[i].sum().item())
            graph_attns = []
            for layer_idx in distill_layer_indices:
                # Extract attention for real nodes only: (H, n_i, n_i)
                attn_i = teacher_attns[layer_idx][i, :, :n_i, :n_i].cpu()
                graph_attns.append(attn_i)

            all_targets.append({
                "sample_idx": graphs_processed + i,
                "num_nodes": n_i,
                "attn_targets": graph_attns,  # list of (H, n_i, n_i) tensors
                "base_loss": float(base_loss[i]),
                "opt_loss": float(opt_loss[i]),
            })

        graphs_processed += B

        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            avg_base = total_base_loss / graphs_processed
            avg_opt = total_opt_loss / graphs_processed
            avg_mmd = total_mmd / graphs_processed if args.proxy_mmd_lambda > 0 else 0
            ratio = avg_opt / max(avg_base, 1e-12)
            conv_rate = total_converged / graphs_processed
            avg_steps = total_steps_taken / (batch_idx + 1)
            print(f"  [{batch_idx+1}/{num_batches}] {graphs_processed} graphs | "
                  f"base_loss={avg_base:.4f} opt_loss={avg_opt:.4f} "
                  f"ratio={ratio:.4f} mmd={avg_mmd:.6f} "
                  f"avg_steps={avg_steps:.1f} conv={conv_rate:.2%}",
                  flush=True)

    # Save
    with open(save_path, "wb") as f:
        pickle.dump({
            "targets": all_targets,
            "distill_layer_indices": distill_layer_indices,
            "num_heads": args.num_heads,
            "split": split_name,
            "args": vars(args),
        }, f)

    avg_base = total_base_loss / graphs_processed
    avg_opt = total_opt_loss / graphs_processed
    overall_ratio = avg_opt / max(avg_base, 1e-12)
    avg_steps = total_steps_taken / max(num_batches, 1)
    conv_rate = total_converged / graphs_processed
    print(f"  {split_name}: {graphs_processed} graphs saved to {save_path}")
    print(f"  Avg base_loss={avg_base:.4f}  opt_loss={avg_opt:.4f}  "
          f"ratio={overall_ratio:.4f}")
    print(f"  Avg proxy steps/batch={avg_steps:.1f}  "
          f"converged={conv_rate:.2%} (target: opt < "
          f"{getattr(args, 'extract_rel_threshold', 0.1)} * base)")

    # opt/base ratio histogram
    if all_ratios:
        ratios_np = np.array(all_ratios)
        bins = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, float('inf')]
        hist, _ = np.histogram(ratios_np, bins=bins)
        print(f"  opt/base ratio histogram (per-sample, {len(ratios_np)} samples):")
        for lo, hi, count in zip(bins[:-1], bins[1:], hist):
            pct = count / len(ratios_np)
            hi_str = "inf" if hi == float('inf') else f"{hi:.2f}"
            print(f"    [{lo:.2f}, {hi_str}): {count:6d}  ({pct:6.2%})")
        print(f"  ratio stats: mean={ratios_np.mean():.4f}  "
              f"median={np.median(ratios_np):.4f}  "
              f"p90={np.percentile(ratios_np, 90):.4f}")
    return save_path, all_targets


def run_extraction(args):
    backbone = getattr(args, 'backbone', 'vanilla_gt')
    if backbone != "vanilla_gt":
        print(f"  NOTE: backbone={backbone} but extraction always uses "
              f"GraphTransformerWithCrossAttn (vanilla GT) as teacher for "
              f"attention target extraction.", flush=True)

    extract_splits = getattr(args, 'extract_splits', 'train')
    print("=" * 60)
    print("Phase 1: Extract Attention Targets")
    print(f"  model_path={args.model_path}")
    print(f"  proxy_opt_steps={args.proxy_opt_steps} (max)")
    print(f"  extract_rel_threshold={args.extract_rel_threshold} "
          f"(target: opt < rel * base)")
    print(f"  proxy_init_jitter={args.proxy_init_jitter}")
    print(f"  num_proxies={args.num_proxies}")
    print(f"  proxy_mmd_lambda={args.proxy_mmd_lambda}")
    print(f"  num_cross_layers={args.num_cross_layers}")
    print(f"  use_proxy_self_attn={args.use_proxy_self_attn}")
    print(f"  use_parameter_free_proxy="
          f"{getattr(args, 'use_parameter_free_proxy', True)}")
    print(f"  distill_layers={args.distill_layers}")
    print(f"  extract_splits={extract_splits}")
    print("=" * 60)

    use_lap_pe = getattr(args, 'use_lap_pe', False)
    lap_pe_dim = getattr(args, 'lap_pe_dim', 0)
    _, _, _, train_ds, val_ds, test_ds = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_lap_pe=use_lap_pe, lap_pe_dim=lap_pe_dim,
    )
    # Use non-shuffled loaders so sample_idx == dataset index
    from torch_geometric.loader import DataLoader as PyGDataLoader
    train_loader = PyGDataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers)
    val_loader_extract = PyGDataLoader(val_ds, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers)
    test_loader_extract = PyGDataLoader(test_ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)

    if args.phase != "pretrain":
        ckpt = torch.load(args.model_path, map_location=args.device, weights_only=True)

    teacher = GraphTransformerWithCrossAttn(
        num_layers=args.num_layers, num_heads=args.num_heads,
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        dropout=args.dropout, num_cross_layers=args.num_cross_layers,
        use_proxy_self_attn=args.use_proxy_self_attn,
        use_parameter_free_proxy=getattr(args, 'use_parameter_free_proxy', True),
    ).to(args.device)

    pretrained_state = ckpt["model_state"]
    model_state = teacher.state_dict()
    loaded_keys = []
    for k, v in pretrained_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded_keys.append(k)
    teacher.load_state_dict(model_state)
    print(f"  Loaded {len(loaded_keys)} pretrained keys, "
          f"cross-attn layers randomly initialized")

    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # Parse distill layers
    if args.distill_layers == "all":
        distill_layer_indices = list(range(args.num_layers))
    else:
        distill_layer_indices = [int(x) for x in args.distill_layers.split(",")]
    print(f"  Extracting attention from layers: {distill_layer_indices}")

    # ----- Run extraction for each requested split -----
    splits_to_run = []
    if extract_splits in ("train", "all"):
        splits_to_run.append(("train", train_loader, train_ds,
                              os.path.join(args.save_dir, "attn_targets.pkl")))
    if extract_splits in ("val", "all"):
        splits_to_run.append(("val", val_loader_extract, val_ds,
                              os.path.join(args.save_dir, "attn_targets_val.pkl")))
    if extract_splits in ("test", "all"):
        splits_to_run.append(("test", test_loader_extract, test_ds,
                              os.path.join(args.save_dir, "attn_targets_test.pkl")))
    if not splits_to_run:
        raise ValueError(f"--extract_splits={extract_splits} matched no splits "
                         f"(expected one of: train, val, test, all)")

    save_paths = {}
    train_targets_for_sanity = None
    for split_name, loader, _ds, split_save_path in splits_to_run:
        sp, all_t = _extract_attention_for_loader(
            teacher, loader, args, distill_layer_indices,
            split_save_path, split_name=split_name)
        save_paths[split_name] = sp
        if split_name == "train":
            train_targets_for_sanity = all_t

    # Quick sanity check (train split only): teacher vs vanilla attention
    if train_targets_for_sanity is not None:
        print("\nSanity check (train): attention difference (teacher vs vanilla)...")
        attn_diffs = []
        graph_counter = 0
        check_loader = PyGDataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch_data in check_loader:
                batch_data = batch_data.to(args.device)
                _, _, vanilla_attns = teacher(batch_data, return_attention=True)
                B_check = batch_data.y.size(0)
                dense_x_c, dense_mask_c = teacher.encode_dense(batch_data)

                for i in range(B_check):
                    if graph_counter >= len(train_targets_for_sanity):
                        break
                    n_vanilla = int(dense_mask_c[i].sum().item())
                    for layer_idx in distill_layer_indices:
                        v_attn = vanilla_attns[layer_idx][i, :, :n_vanilla, :n_vanilla].cpu()
                        t_attn = train_targets_for_sanity[graph_counter]["attn_targets"][
                            distill_layer_indices.index(layer_idx)]
                        n_min = min(v_attn.shape[-1], t_attn.shape[-1])
                        diff = (t_attn[:, :n_min, :n_min] - v_attn[:, :n_min, :n_min]).abs().mean().item()
                        attn_diffs.append(diff)
                    graph_counter += 1

                    if len(attn_diffs) >= 500:
                        break
                if len(attn_diffs) >= 500:
                    break

        if attn_diffs:
            mean_diff = float(np.mean(attn_diffs))
            print(f"  Mean |teacher_attn - vanilla_attn|: {mean_diff:.6f}")
            if mean_diff < 1e-5:
                print("  WARNING: Attention difference is near zero! "
                      "Cross-attention routing may not be changing attention patterns.")
            else:
                print(f"  Good: attention patterns are meaningfully different.")

    # Backwards-compat return: the path of the train file (or first available)
    return save_paths.get("train", next(iter(save_paths.values())))


# ================================================================
# ATTENTION TARGET DATASET (for Phase 2)
# ================================================================

def reduce_attn_to_weights(attn_targets):
    """Reduce a per-graph saved attention list to a single (n, n) weight matrix.

    Reduction: take the LAST distilled layer, then max over heads. Result is
    used to weight GRED's per-hop aggregation as
        agg[v, k] = sum_{u : d(v,u)=k} w[v, u] * h_u

    Args:
        attn_targets: list of (H, n, n) tensors, one per distilled layer.
            Comes straight out of AttnTargetDataset / the saved pickle.
    Returns:
        (n, n) tensor, w[v, u] = max_h attn_targets[-1][h, v, u].
    """
    last = attn_targets[-1]            # (H, n, n)
    return last.max(dim=0).values      # (n, n)


def load_attn_weights(path):
    """Load saved attention targets and reduce each graph to a (n, n) weight.

    Returns a dict ``{sample_idx -> (n, n) tensor}``. The reduction is
    last-layer max-over-heads (see ``reduce_attn_to_weights``). Pre-computing
    this once at load time avoids redoing the max on every batch.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return {t["sample_idx"]: reduce_attn_to_weights(t["attn_targets"])
            for t in data["targets"]}


class DistMaskWithAttnWeights(Dataset):
    """Wraps a DistMaskDataset so each item also carries an attention weight
    matrix derived from saved teacher attention.

    Used for GRED/hybrid training when ``--use_attn_weighting`` is on. Only
    includes graphs that have attention targets (i.e. were processed in the
    extract phase). The attn weight matrix is broadcast over hops in
    ``GREDLayer.forward``.
    """
    def __init__(self, dist_mask_ds, attn_weight_dict):
        self.dist_mask_ds = dist_mask_ds
        self.attn_weight_dict = attn_weight_dict
        self.valid_indices = sorted(self.attn_weight_dict.keys())

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        sample_idx = self.valid_indices[idx]
        graph, dist_mask_array = self.dist_mask_ds[sample_idx]
        return graph, dist_mask_array, self.attn_weight_dict[sample_idx]


def collate_gred_with_attn_weights(batch, max_hops=40):
    """Collate (graph, dist_mask_array, attn_weight) triples into batched tensors.

    Reuses the existing dist-mask padding (``collate_with_dist_masks``), then
    pads the per-graph attention weights to (B, max_N, max_N).

    Returns:
        pyg_batch, dist_masks, node_masks, padded_w
    """
    from data import collate_with_dist_masks
    graphs, dm_list, w_list = zip(*batch)
    base_batch = list(zip(graphs, dm_list))
    pyg_batch, dist_masks, node_masks = collate_with_dist_masks(
        base_batch, max_hops=max_hops)

    B = len(graphs)
    max_N = node_masks.shape[1]
    padded_w = torch.zeros(B, max_N, max_N)
    for i, w in enumerate(w_list):
        n = w.shape[0]
        padded_w[i, :n, :n] = w

    return pyg_batch, dist_masks, node_masks, padded_w


class AttnTargetDataset(Dataset):
    """Wraps original PyG dataset with pre-extracted attention targets.

    Also computes a per-graph reduced (n, n) weight matrix used for
    attention-weighted GRED aggregation (see ``reduce_attn_to_weights``).
    """
    def __init__(self, pyg_dataset, targets_path):
        with open(targets_path, "rb") as f:
            data = pickle.load(f)

        self.targets = data["targets"]
        self.distill_layer_indices = data["distill_layer_indices"]
        self.pyg_dataset = pyg_dataset

        # Map sample_idx -> target entry
        self.idx_to_target = {t["sample_idx"]: t for t in self.targets}
        # Only include graphs that have targets
        self.valid_indices = sorted(self.idx_to_target.keys())
        print(f"  AttnTargetDataset: {len(self.valid_indices)} graphs with targets")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        sample_idx = self.valid_indices[idx]
        graph = self.pyg_dataset[sample_idx]
        target = self.idx_to_target[sample_idx]
        # attn_targets: list of (H, n, n) tensors
        # attn_weights: (n, n) reduced weight matrix for GRED aggregation
        # opt_loss: scalar — extraction proxy-optimized BCE loss for this sample.
        #           Used at training time to filter samples whose extraction
        #           didn't converge below distill_loss_threshold.
        attn_weights = reduce_attn_to_weights(target["attn_targets"])
        opt_loss = target.get("opt_loss", 0.0)
        return (graph, target["attn_targets"], target["num_nodes"],
                attn_weights, opt_loss)


def collate_attn_targets(batch):
    """
    Collate graphs + variable-size attention targets + reduced (n, n) weights
    + per-sample extraction opt_loss. Pads everything to max_N in the batch.

    Returns:
        pyg_batch:        PyG Batch
        padded_attns:     (B, num_layers, H, max_N, max_N) full saved attention
        attn_masks:       (B, max_N) bool — True for real nodes
        padded_w:         (B, max_N, max_N) reduced weight matrix per graph
        opt_loss_tensor:  (B,) extraction opt_loss per sample (for sample_mask)
    """
    import torch_geometric
    graphs, attn_list, num_nodes_list, weight_list, opt_losses = zip(*batch)

    pyg_batch = torch_geometric.data.Batch.from_data_list(list(graphs))
    B = len(graphs)
    max_N = max(num_nodes_list)
    num_layers = len(attn_list[0])
    H = attn_list[0][0].shape[0]

    # Pad attention targets: (B, num_layers, H, max_N, max_N)
    padded_attns = torch.zeros(B, num_layers, H, max_N, max_N)
    attn_masks = torch.zeros(B, max_N, dtype=torch.bool)
    padded_w = torch.zeros(B, max_N, max_N)
    opt_loss_tensor = torch.tensor(opt_losses, dtype=torch.float32)  # (B,)

    for i in range(B):
        n = num_nodes_list[i]
        attn_masks[i, :n] = True
        for l in range(num_layers):
            padded_attns[i, l, :, :n, :n] = attn_list[i][l]
        padded_w[i, :n, :n] = weight_list[i]

    return pyg_batch, padded_attns, attn_masks, padded_w, opt_loss_tensor


# ================================================================
# ATTENTION DISTILLATION LOSS
# ================================================================

def attention_distillation_loss(student_attns, teacher_attns_padded, mask,
                                temperature=1.0, sample_mask=None):
    """
    Distillation loss between student and pre-extracted teacher attention.

    Returns (kl_loss, mse_loss) so the caller can weight them independently.

    KL: row-wise KL divergence on the (optionally temperature-softened) attention
        distributions. Drives student modes to match teacher modes; gradient is
        weighted by t_k so it under-pressures low-attention entries.
    MSE: per-entry squared error on the raw softmax outputs (NO temperature, even
        when temperature != 1.0 — exact-match goal). Gives uniform pressure across
        all entries including the tails. Mask is applied to both query and key
        dimensions so padded positions don't contribute.

    Args:
        student_attns: list of (B, H, N, N) per distilled layer
        teacher_attns_padded: (B, num_distill_layers, H, N, N) padded teacher targets
        mask: (B, N) boolean
        temperature: softening temperature (KL only)
        sample_mask: (B,) boolean — if provided, only include distillation loss for
                     samples where sample_mask[i] is True (i.e. extraction opt_loss
                     was below threshold). Samples with False are zeroed out.
    """
    total_kl = 0.0
    total_mse = 0.0
    num_layers = len(student_attns)

    for l_idx in range(num_layers):
        s_attn = student_attns[l_idx]                # (B, H, N_s, N_s)
        t_attn = teacher_attns_padded[:, l_idx]      # (B, H, N_t, N_t)

        # Align spatial dimensions: student's N may differ from teacher's N
        N_s = s_attn.shape[-1]
        N_t = t_attn.shape[-1]
        if N_s != N_t:
            N = max(N_s, N_t)
            if N_s < N:
                s_attn = F.pad(s_attn, (0, N - N_s, 0, N - N_s))
            if N_t < N:
                t_attn = F.pad(t_attn, (0, N - N_t, 0, N - N_t))
            # Extend mask to match
            if mask.shape[-1] < N:
                mask = F.pad(mask, (0, N - mask.shape[-1]), value=False)

        B, H, N, _ = s_attn.shape

        # ----- KL branch (with optional temperature softening) -----
        if temperature != 1.0:
            s_log = torch.log(s_attn + 1e-10) / temperature
            t_log = torch.log(t_attn + 1e-10) / temperature

            pad_mask = (~mask).unsqueeze(1).unsqueeze(2)
            s_log = s_log.masked_fill(pad_mask, float("-inf"))
            t_log = t_log.masked_fill(pad_mask, float("-inf"))

            s_soft = F.softmax(s_log, dim=-1)
            t_soft = F.softmax(t_log, dim=-1)
            s_soft = torch.nan_to_num(s_soft, nan=0.0)
            t_soft = torch.nan_to_num(t_soft, nan=0.0)
        else:
            s_soft = s_attn
            t_soft = t_attn

        kl = t_soft * (torch.log(t_soft + 1e-10) - torch.log(s_soft + 1e-10))
        kl = kl.sum(dim=-1)  # (B, H, N)

        query_mask = mask.unsqueeze(1)  # (B, 1, N)
        kl = kl * query_mask.float()

        # Per-sample KL: sum over heads and query positions
        per_sample_kl = kl.sum(dim=(1, 2))  # (B,)
        per_sample_valid_q = query_mask.float().sum(dim=-1).squeeze(1) * H  # (B,)
        per_sample_kl_loss = per_sample_kl / (per_sample_valid_q + 1e-10)  # (B,)

        # ----- MSE branch (always on raw softmax — exact match) -----
        # Mask both query AND key dims so padded entries don't contribute on either side.
        sq_err = (s_attn - t_attn) ** 2  # (B, H, N, N)
        q_mask = mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, N, 1)
        k_mask = mask.unsqueeze(1).unsqueeze(2).float()   # (B, 1, 1, N)
        entry_mask = q_mask * k_mask                       # (B, 1, N, N)
        sq_err = sq_err * entry_mask
        per_sample_mse_sum = sq_err.sum(dim=(1, 2, 3))     # (B,)
        per_sample_valid_e = entry_mask.sum(dim=(1, 2, 3)) * H  # (B,) entries × heads
        per_sample_mse_loss = per_sample_mse_sum / (per_sample_valid_e + 1e-10)

        # ----- Sample-level masking + reduction -----
        if sample_mask is not None:
            sm = sample_mask.float()
            num_active = sm.sum().clamp_min(1.0)
            kl_layer = (per_sample_kl_loss * sm).sum() / num_active
            mse_layer = (per_sample_mse_loss * sm).sum() / num_active
        else:
            kl_layer = per_sample_kl_loss.mean()
            mse_layer = per_sample_mse_loss.mean()

        total_kl = total_kl + kl_layer
        total_mse = total_mse + mse_layer

    return total_kl / num_layers, total_mse / num_layers


# ================================================================
# PHASE 2: TRAIN WITH DISTILLATION
# ================================================================

def run_training(args):
    backbone = getattr(args, 'backbone', 'vanilla_gt')
    is_gred = backbone in ("gred", "hybrid")
    use_distillation = (backbone == "vanilla_gt")

    targets_path = os.path.join(args.save_dir, "attn_targets.pkl")

    if use_distillation:
        assert os.path.exists(targets_path), \
            f"Attention targets not found at {targets_path}. Run 'extract' phase first."
        # Load target metadata
        with open(targets_path, "rb") as f:
            target_data = pickle.load(f)
        distill_layer_indices = target_data["distill_layer_indices"]
    else:
        distill_layer_indices = []
        print(f"  NOTE: backbone={backbone} — no attention distillation "
              f"(GRED has no self-attention). Training with task loss only.",
              flush=True)

    use_attn_weighting = (backbone in ("gred", "hybrid")
                          and getattr(args, 'use_attn_weighting', False))

    print("=" * 60)
    print("Phase 2: Train with Attention Distillation")
    print(f"  backbone={backbone}")
    if use_distillation:
        print(f"  distill_weight={args.distill_weight}  (KL term)")
        print(f"  mse_weight={args.mse_weight}  (per-entry MSE term)")
        print(f"  temperature={args.temperature}")
        print(f"  distill_layers={distill_layer_indices}")
        print(f"  distill_loss_threshold={args.distill_loss_threshold}")
        print(f"  targets from: {targets_path}")
    else:
        print(f"  distill_weight=N/A (task-only)")
        print(f"  temperature=N/A")
        print(f"  distill_layers=N/A")
    print(f"  use_attn_weighting={use_attn_weighting}")
    print("=" * 60)

    # Standard loaders for val/test
    use_lap_pe = getattr(args, 'use_lap_pe', False)
    lap_pe_dim = getattr(args, 'lap_pe_dim', 0)
    max_hops = getattr(args, 'max_hops', 40)

    train_loader_gred_default, val_loader, test_loader, train_ds, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        use_dist_masks=is_gred, max_hops=max_hops,
        dist_mask_workers=getattr(args, 'dist_mask_workers', 8),
        use_lap_pe=use_lap_pe, lap_pe_dim=lap_pe_dim,
    )

    if use_distillation:
        # Distillation train loader (pairs graphs with attention targets)
        distill_ds = AttnTargetDataset(train_ds, targets_path)
        distill_loader = DataLoader(
            distill_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_attn_targets, num_workers=args.num_workers,
        )
    else:
        distill_loader = None
        if use_attn_weighting:
            # Wrap each split's DistMaskDataset with attn weights extracted from
            # the per-split saved attention pickles. Rebuild loaders with the
            # 4-tuple collate so each batch carries (B, max_N, max_N) weights.
            from torch.utils.data import DataLoader as TorchDataLoader

            train_targets_path = os.path.join(args.save_dir, "attn_targets.pkl")
            val_targets_path = os.path.join(args.save_dir, "attn_targets_val.pkl")
            test_targets_path = os.path.join(args.save_dir, "attn_targets_test.pkl")
            for pth, name in [(train_targets_path, "train"),
                              (val_targets_path, "val"),
                              (test_targets_path, "test")]:
                assert os.path.exists(pth), (
                    f"--use_attn_weighting requires extracted attention for the "
                    f"{name} split at {pth}. Run 'extract' phase with "
                    f"--extract_splits all first.")

            print(f"  Loading attn weights for all splits...")
            train_w = load_attn_weights(train_targets_path)
            val_w = load_attn_weights(val_targets_path)
            test_w = load_attn_weights(test_targets_path)
            print(f"    train: {len(train_w)} graphs, val: {len(val_w)}, "
                  f"test: {len(test_w)}")

            train_dist_ds = train_loader_gred_default.dataset
            val_dist_ds = val_loader.dataset
            test_dist_ds = test_loader.dataset

            train_wrapped = DistMaskWithAttnWeights(train_dist_ds, train_w)
            val_wrapped = DistMaskWithAttnWeights(val_dist_ds, val_w)
            test_wrapped = DistMaskWithAttnWeights(test_dist_ds, test_w)

            collate_w = partial(collate_gred_with_attn_weights, max_hops=max_hops)
            train_loader_gred = TorchDataLoader(
                train_wrapped, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, collate_fn=collate_w)
            val_loader = TorchDataLoader(
                val_wrapped, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=collate_w)
            test_loader = TorchDataLoader(
                test_wrapped, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=collate_w)
        else:
            train_loader_gred = train_loader_gred_default

    # Student model
    ckpt = torch.load(args.model_path, map_location=args.device, weights_only=True)

    if backbone == "vanilla_gt":
        student = GraphTransformerWithCrossAttn(
            num_layers=args.num_layers, num_heads=args.num_heads,
            hidden_dim=args.hidden_dim, output_dim=args.output_dim,
            dropout=args.dropout,
            use_parameter_free_proxy=getattr(args, 'use_parameter_free_proxy', True),
        ).to(args.device)
    else:
        student = build_model(args).to(args.device)

    pretrained_state = ckpt["model_state"]
    model_state_s = student.state_dict()
    loaded_keys = []
    for k, v in pretrained_state.items():
        if k in model_state_s and model_state_s[k].shape == v.shape:
            model_state_s[k] = v
            loaded_keys.append(k)
    student.load_state_dict(model_state_s)
    print(f"  Loaded {len(loaded_keys)}/{len(model_state_s)} keys from checkpoint")

    # Baseline
    print("\nBaseline (pretrained, no distillation):")
    baseline_val_ap = evaluate(student, val_loader, args)
    baseline_test_ap = evaluate(student, test_loader, args)
    print(f"  val_AP={baseline_val_ap:.4f}  test_AP={baseline_test_ap:.4f}")

    # Only train encoder + self-attention + head
    trainable_params = []
    for name, param in student.named_parameters():
        if "cross_attn_layers" not in name:
            param.requires_grad_(True)
            trainable_params.append(param)
        else:
            param.requires_grad_(False)
    print(f"  Student trainable params: {sum(p.numel() for p in trainable_params):,}")

    active_loader = distill_loader if use_distillation else train_loader_gred
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                   weight_decay=args.weight_decay)
    total_steps = max(1, len(active_loader) * args.max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7)

    loss_fn = nn.BCEWithLogitsLoss()
    best_val_ap = 0.0
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()
        student.train()

        train_losses, task_losses_log, distill_losses_log = [], [], []
        kl_losses_log, mse_losses_log = [], []
        all_preds, all_labels = [], []

        if use_distillation:
            # --- Vanilla GT path: distillation + task loss ---
            # collate_attn_targets returns 5-tuple. The 4th value (attn_weights)
            # is unused on this path — it's only consumed by the GRED/hybrid branch.
            # The 5th value (opt_loss) drives sample-level distillation masking.
            for batch_idx, (pyg_batch, teacher_attns_padded, attn_masks,
                            _attn_w, opt_losses) in enumerate(active_loader):
                pyg_batch = pyg_batch.to(args.device)
                teacher_attns_padded = teacher_attns_padded.to(args.device)
                attn_masks = attn_masks.to(args.device)
                opt_losses = opt_losses.to(args.device)  # (B,)

                # Build per-sample mask: only distill from samples with good extraction
                sample_mask = (opt_losses < args.distill_loss_threshold)  # (B,) bool
                num_active = int(sample_mask.sum().item())

                optimizer.zero_grad()

                # Student forward (no proxies)
                logits, _, student_attns = student(pyg_batch, return_attention=True)

                # Filter to distilled layers
                student_attns_filtered = [student_attns[i] for i in distill_layer_indices]

                # Task loss
                task_loss = loss_fn(logits, pyg_batch.y)

                # Distillation loss against pre-extracted targets
                # Only include for samples with extraction loss below threshold
                if num_active > 0:
                    kl_loss, mse_loss = attention_distillation_loss(
                        student_attns_filtered, teacher_attns_padded,
                        attn_masks, temperature=args.temperature,
                        sample_mask=sample_mask)
                else:
                    kl_loss = torch.tensor(0.0, device=args.device)
                    mse_loss = torch.tensor(0.0, device=args.device)

                distill_loss = (args.distill_weight * kl_loss
                                + args.mse_weight * mse_loss)
                loss = task_loss + distill_loss
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())
                task_losses_log.append(task_loss.item())
                distill_losses_log.append(distill_loss.item())
                kl_losses_log.append(kl_loss.item())
                mse_losses_log.append(mse_loss.item())
                all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
                all_labels.append(pyg_batch.y.cpu().numpy())
        else:
            # --- GRED/hybrid path: task loss only ---
            for batch_data in active_loader:
                attn_weights_batch = None
                if is_gred:
                    if use_attn_weighting:
                        (pyg_batch, dist_masks_batch, node_masks_batch,
                         attn_weights_batch) = batch_data
                        attn_weights_batch = attn_weights_batch.to(args.device)
                    else:
                        pyg_batch, dist_masks_batch, node_masks_batch = batch_data
                    pyg_batch = pyg_batch.to(args.device)
                    dist_masks_batch = dist_masks_batch.to(args.device)
                    node_masks_batch = node_masks_batch.to(args.device)
                else:
                    pyg_batch = batch_data.to(args.device)
                    dist_masks_batch = None
                    node_masks_batch = None

                optimizer.zero_grad()

                if is_gred and attn_weights_batch is not None:
                    logits, _ = student(
                        pyg_batch, dist_masks_batch, node_masks_batch,
                        attn_weights=attn_weights_batch)
                else:
                    logits, _ = student(pyg_batch, dist_masks_batch, node_masks_batch)
                task_loss = loss_fn(logits, pyg_batch.y)

                task_loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()
                scheduler.step()

                train_losses.append(task_loss.item())
                task_losses_log.append(task_loss.item())
                distill_losses_log.append(0.0)
                all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
                all_labels.append(pyg_batch.y.cpu().numpy())

        train_ap = compute_macro_ap(
            np.concatenate(all_preds), np.concatenate(all_labels))
        mean_task = float(np.mean(task_losses_log))
        mean_distill = float(np.mean(distill_losses_log))
        mean_kl = float(np.mean(kl_losses_log)) if kl_losses_log else 0.0
        mean_mse = float(np.mean(mse_losses_log)) if mse_losses_log else 0.0

        val_ap = evaluate(student, val_loader, args)
        test_ap = evaluate(student, test_loader, args)

        elapsed = time.time() - epoch_start
        if use_distillation:
            print(f"Epoch {epoch:3d}/{args.max_epochs} [{elapsed:.1f}s] | "
                  f"task={mean_task:.4f} distill={mean_distill:.4f} "
                  f"(kl={mean_kl:.4f} mse={mean_mse:.6f}) "
                  f"(thresh={args.distill_loss_threshold}) | "
                  f"train_AP={train_ap:.4f} val_AP={val_ap:.4f} test_AP={test_ap:.4f}",
                  flush=True)
        else:
            print(f"Epoch {epoch:3d}/{args.max_epochs} [{elapsed:.1f}s] | "
                  f"task={mean_task:.4f} distill={mean_distill:.4f} | "
                  f"train_AP={train_ap:.4f} val_AP={val_ap:.4f} test_AP={test_ap:.4f}",
                  flush=True)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": student.state_dict(),
                "epoch": epoch, "val_ap": val_ap, "test_ap": test_ap,
            }, os.path.join(args.save_dir, "attn_distill_best.pt"))
            print(f"  -> New best val_AP={val_ap:.4f} (test={test_ap:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"\nDone. Baseline val_AP={baseline_val_ap:.4f} | "
          f"Best distilled val_AP={best_val_ap:.4f} at epoch {best_epoch}")


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate(model, loader, args):
    model.eval()
    backbone = getattr(args, 'backbone', 'vanilla_gt')
    is_gred = backbone in ("gred", "hybrid")
    all_preds, all_labels = [], []
    for batch_data in loader:
        if is_gred:
            # Loader yields 3-tuple (no attn weights) or 4-tuple (with weights).
            if len(batch_data) == 4:
                batch, dist_masks_batch, node_masks_batch, attn_weights_batch = batch_data
                attn_weights_batch = attn_weights_batch.to(args.device)
            else:
                batch, dist_masks_batch, node_masks_batch = batch_data
                attn_weights_batch = None
            batch = batch.to(args.device)
            dist_masks_batch = dist_masks_batch.to(args.device)
            node_masks_batch = node_masks_batch.to(args.device)
            if attn_weights_batch is not None:
                logits, _ = model(
                    batch, dist_masks_batch, node_masks_batch,
                    attn_weights=attn_weights_batch)
            else:
                logits, _ = model(batch, dist_masks_batch, node_masks_batch)
        else:
            batch = batch_data.to(args.device)
            logits, _ = model(batch, return_attention=False)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    return compute_macro_ap(np.concatenate(all_preds), np.concatenate(all_labels))


# ================================================================
# MAIN
# ================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["pretrain", "extract", "train"],
                   help="'pretrain' = train transformer from scratch. "
                        "'extract' = optimize proxies, save attention targets. "
                        "'train' = train student with distillation.")
    p.add_argument("--model_path", type=str, default=None,
                   help="Pretrained checkpoint. Required for extract/train. "
                        "If not provided for extract/train, looks for pretrain_best.pt in save_dir.")
    p.add_argument("--save_dir", type=str, default="checkpoints_attn_distill")

    # Backbone
    p.add_argument("--backbone", type=str, default="vanilla_gt",
                   choices=["vanilla_gt", "gred", "hybrid"],
                   help="Backbone architecture: vanilla_gt, gred, or hybrid")

    # Model architecture
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)

    # Laplacian positional encoding
    p.add_argument("--use_lap_pe", action=argparse.BooleanOptionalAction, default=False,
                   help="Add Laplacian eigenvector positional encodings to node features")
    p.add_argument("--lap_pe_dim", type=int, default=8,
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
    p.add_argument("--r_min", type=float, default=0.0)
    p.add_argument("--r_max", type=float, default=1.0)
    p.add_argument("--max_phase", type=float, default=6.28)
    p.add_argument("--gred_act", type=str, default="full-glu",
                   choices=["full-glu", "half-glu"])
    p.add_argument("--max_hops", type=int, default=40)
    p.add_argument("--dist_mask_workers", type=int, default=8)

    # Pretrain (Stage 1)
    p.add_argument("--s1_lr", type=float, default=3e-5)
    p.add_argument("--s1_weight_decay", type=float, default=3e-4)
    p.add_argument("--s1_max_epochs", type=int, default=300)
    p.add_argument("--s1_patience", type=int, default=50)
    p.add_argument("--s1_grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_min", type=float, default=1e-7)

    # Cross-attention routing (extract phase only)
    p.add_argument("--num_cross_layers", type=int, default=None)
    p.add_argument("--use_proxy_self_attn", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--use_parameter_free_proxy", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use parameter-free Form B routing (node→proxy attention "
                        "with no learnable parameters) in place of learnable "
                        "cross-attention. Affects pretrain, extract teacher, and "
                        "vanilla_gt student. No effect when no proxies are passed.")

    # Proxy optimization (extract phase only)
    p.add_argument("--num_proxies", type=int, default=64)
    p.add_argument("--proxy_lr", type=float, default=5e-2)
    p.add_argument("--proxy_opt_steps", type=int, default=300,
                   help="Maximum proxy optimization steps per batch (extract phase). "
                        "Proxy LR follows a CosineAnnealingLR schedule over this many "
                        "steps with eta_min = 0.01 * proxy_lr.")
    p.add_argument("--extract_rel_threshold", type=float, default=0.1,
                   help="Per-sample early-exit target: stop when opt_loss[i] < "
                        "rel_threshold * base_loss[i] for all i in the batch. "
                        "Replaces absolute --extract_loss_threshold.")
    p.add_argument("--proxy_init_jitter", type=float, default=0.5,
                   help="Stddev of Gaussian noise added to on-manifold proxy init. "
                        "Larger values break symmetry and let proxies follow distinct "
                        "optimization trajectories instead of collapsing.")
    p.add_argument("--proxy_mmd_lambda", type=float, default=1)

    # Extraction split selection (extract phase only)
    p.add_argument("--extract_splits", type=str, default="train",
                   choices=["train", "val", "test", "all"],
                   help="Which split(s) to run proxy optimization + attention "
                        "extraction for. Use 'all' when --use_attn_weighting "
                        "will be set during training.")

    # Distillation (train phase)
    p.add_argument("--distill_weight", type=float, default=1.0,
                   help="Weight on the KL term of the distillation loss")
    p.add_argument("--mse_weight", type=float, default=0.0,
                   help="Weight on the per-entry MSE term of the distillation "
                        "loss (uses raw softmax outputs, no temperature). Set >0 "
                        "to add uniform pressure on low-attention entries that KL "
                        "under-weights. Try 0.1*distill_weight as a starting point.")
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--distill_layers", type=str, default="all")
    p.add_argument("--distill_loss_threshold", type=float, default=0.5,
                   help="Only include attention distillation loss for samples whose "
                        "extraction opt_loss was below this threshold. Samples with "
                        "opt_loss >= threshold are excluded from distillation (but "
                        "still contribute to task BCE loss).")

    # Attention-weighted GRED aggregation (train phase, GRED/hybrid backbones)
    p.add_argument("--use_attn_weighting", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="GRED/hybrid only: weight each per-hop aggregation by "
                        "the saved teacher attention (last layer, max over "
                        "heads). Requires extracted attention for train, val, "
                        "and test splits. Stage 1 oracle setup.")

    # Training (train phase)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)

    args, unknown = p.parse_known_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    print(args.__dict__)
    print("Unknown args = ", unknown)

    # Auto-resolve model_path for extract/train if not provided
    if args.model_path is None and args.phase in ("extract", "train"):
        default_path = os.path.join(args.save_dir, "pretrain_best.pt")
        if os.path.exists(default_path):
            args.model_path = default_path
            print(f"Auto-resolved model_path: {default_path}")
        else:
            raise FileNotFoundError(
                f"No --model_path provided and no pretrain checkpoint found at {default_path}. "
                f"Run 'pretrain' phase first or provide --model_path.")

    if args.phase == "pretrain":
        run_pretrain(args)
    elif args.phase == "extract":
        run_extraction(args)
    elif args.phase == "train":
        run_training(args)


if __name__ == "__main__":
    main()


