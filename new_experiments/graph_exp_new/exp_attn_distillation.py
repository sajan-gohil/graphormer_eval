"""
Experiment: Attention Distillation from Proxy-Augmented Transformer

Idea:
  1. Load pretrained transformer (Stage 1)
  2. For each training graph, optimize proxy embeddings (Stage 2 style)
  3. Extract the attention matrices from each transformer layer when proxies are present
     (only the N×N node-to-node block, after cross-attention with proxies has enriched them)
  4. These are the "target" attention matrices — what the transformer produces when
     it has proxy-enriched embeddings
  5. Train the model (unfrozen) to produce those attention patterns WITHOUT proxies,
     by adding a KL-divergence loss on attention distributions
  6. The gradient flows back through Q, K projections into the embedding table,
     updating it to produce embeddings that naturally yield better attention patterns

The embedding table (AtomEncoder + BondEncoder) is shared across all graphs.
It can't memorize per-graph patterns — it has to learn general chemical priors
about which atom/bond types should attend to which.

Usage:
    python exp_attn_distill.py --model_path checkpoints_staged/stage1_best.pt
    python exp_attn_distill.py --model_path checkpoints_staged/stage1_best.pt --distill_weight 1.0
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool

from data import get_loaders
from models import NodeEncoder
from metrics import compute_macro_ap
from mmd import mmd_squared


# ================================================================
# MODIFIED TRANSFORMER LAYER — returns attention weights
# ================================================================

class TransformerLayerWithAttn(nn.Module):
    """Same as TransformerLayer but returns attention weights alongside output."""
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
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
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

        # Store pre-dropout attention for distillation
        attn_w_clean = attn_w.detach().clone()

        attn_w_dropped = self.attn_drop(attn_w)
        out = (attn_w_dropped @ v).transpose(1, 2).reshape(B, N, d)
        x = x + self.res_drop(self.wout(out))
        x = x + self.res_drop(self.ff(self.norm2(x)))
        return x, attn_w_clean  # (B, H, N, N)


class GraphTransformerWithAttn(nn.Module):
    """GraphTransformer variant that returns per-layer attention weights."""
    def __init__(self, num_layers=5, num_heads=8, hidden_dim=64,
                 output_dim=10, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = NodeEncoder(hidden_dim)
        self.layers = nn.ModuleList([
            TransformerLayerWithAttn(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_nodes(self, batch):
        return self.encoder(batch.x, batch.edge_index, batch.edge_attr)

    def encode_dense(self, batch):
        h = self.encode_nodes(batch)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, proxy_embeddings=None, precomputed_dense=None,
                readout_scope="nodes_only", return_attention=False):
        if precomputed_dense is not None:
            dense_x, dense_mask = precomputed_dense
        else:
            dense_x, dense_mask = self.encode_dense(batch)

        B, max_N, d = dense_x.shape

        if proxy_embeddings is not None:
            M = proxy_embeddings.shape[1]
            dense_x = torch.cat([dense_x, proxy_embeddings], dim=1)
            aug_mask = torch.cat([
                dense_mask,
                torch.ones(B, M, dtype=torch.bool, device=dense_x.device)
            ], dim=1)
        else:
            aug_mask = dense_mask

        all_attn = []
        for layer in self.layers:
            dense_x, attn_w = layer(dense_x, aug_mask)
            if return_attention:
                all_attn.append(attn_w)

        # Readout
        if readout_scope == "all_tokens" and proxy_embeddings is not None:
            valid_emb = dense_x[aug_mask]
            batch_vec = torch.arange(B, device=dense_x.device).unsqueeze(1).expand_as(aug_mask)[aug_mask]
            pooled = global_mean_pool(valid_emb, batch_vec)
        else:
            orig_x = dense_x[:, :max_N, :]
            node_emb_masked = orig_x[dense_mask]
            pooled = global_mean_pool(node_emb_masked, batch.batch)

        logits = self.head(pooled)
        orig_x = dense_x[:, :max_N, :]
        node_emb = orig_x[dense_mask]

        if return_attention:
            return logits, node_emb, all_attn
        return logits, node_emb


# ================================================================
# PROXY OPTIMIZATION (simplified Stage 2 for a single batch)
# ================================================================

def _per_sample_bce(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)


@torch.no_grad()
def optimize_proxies_for_batch(model, batch, dense_x, dense_mask, args):
    """
    Optimize proxy embeddings for a batch. Returns best proxies.
    Model must be in eval mode, all params frozen.

    Includes MMD regularization to keep proxy embeddings in the same
    distribution as the encoder's node embeddings, so the attention
    patterns produced are realistic (not from out-of-distribution inputs).
    """
    B = batch.y.size(0)
    device = batch.y.device
    mmd_lambda = getattr(args, 'proxy_mmd_lambda', 0.0)

    # We need gradients for proxy but not model
    proxy = torch.randn(B, args.num_proxies, args.hidden_dim, device=device) * 0.02
    proxy = nn.Parameter(proxy)
    opt = torch.optim.Adam([proxy], lr=args.proxy_lr)

    # Base loss without proxies
    logits_base, _, _ = model(batch, precomputed_dense=(dense_x, dense_mask),
                               return_attention=True)
    base_loss = _per_sample_bce(logits_base, batch.y)

    best_loss = base_loss.clone()
    best_proxy = proxy.detach().clone()

    # Temporarily enable grads for proxy optimization
    with torch.enable_grad():
        for step in range(args.proxy_opt_steps):
            opt.zero_grad()
            logits, _, _ = model(batch, proxy_embeddings=proxy,
                                  precomputed_dense=(dense_x, dense_mask),
                                  return_attention=True)
            task_loss = _per_sample_bce(logits, batch.y)

            # MMD regularization: keep proxies in same distribution as node embeddings
            if mmd_lambda > 0:
                mmd_losses = []
                for i in range(B):
                    nodes_i = dense_x[i][dense_mask[i]]  # (n_i, d)
                    mmd_losses.append(mmd_squared(proxy[i], nodes_i))
                mmd_batch = torch.stack(mmd_losses)
                total = (task_loss + mmd_lambda * mmd_batch).mean()
            else:
                total = task_loss.mean()

            total.backward()
            nn.utils.clip_grad_norm_([proxy], 1.0)
            opt.step()

            with torch.no_grad():
                improved = task_loss < best_loss
                if improved.any():
                    best_loss[improved] = task_loss[improved]
                    best_proxy[improved] = proxy.detach()[improved]

    return best_proxy


# ================================================================
# ATTENTION DISTILLATION LOSS
# ================================================================

def attention_distillation_loss(student_attns, teacher_attns, mask, temperature=1.0):
    """
    KL divergence between student and teacher attention distributions.

    Args:
        student_attns: list of (B, H, N, N) attention weights per layer (from model without proxies)
        teacher_attns: list of (B, H, N+M, N+M) attention weights per layer (from model with proxies)
        mask: (B, N) boolean mask for original nodes
        temperature: softening temperature (higher = softer targets)

    Returns:
        scalar loss
    """
    total_loss = 0.0
    num_layers = len(student_attns)

    for layer_idx in range(num_layers):
        s_attn = student_attns[layer_idx]  # (B, H, N, N)
        t_attn = teacher_attns[layer_idx]  # (B, H, N+M, N+M)

        B, H, N_s, _ = s_attn.shape
        N_t = t_attn.shape[2]

        # Extract the N×N node-to-node block from teacher (skip proxy rows/cols)
        # Teacher has [nodes | proxies], so node-to-node is [:N, :N]
        N = N_s  # student has only nodes
        t_attn_nn = t_attn[:, :, :N, :N]  # (B, H, N, N)

        # Re-normalize teacher's node-to-node block (since we removed proxy columns,
        # the rows no longer sum to 1)
        # But we want to preserve the RELATIVE attention pattern among nodes
        t_attn_nn = t_attn_nn / (t_attn_nn.sum(dim=-1, keepdim=True) + 1e-10)

        # Apply temperature
        if temperature != 1.0:
            # Convert back to logits, apply temperature, re-softmax
            s_log = torch.log(s_attn + 1e-10) / temperature
            t_log = torch.log(t_attn_nn + 1e-10) / temperature

            # Mask padding
            pad_mask = (~mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            s_log = s_log.masked_fill(pad_mask, float("-inf"))
            t_log = t_log.masked_fill(pad_mask, float("-inf"))

            s_soft = F.softmax(s_log, dim=-1)
            t_soft = F.softmax(t_log, dim=-1)
            s_soft = torch.nan_to_num(s_soft, nan=0.0)
            t_soft = torch.nan_to_num(t_soft, nan=0.0)
        else:
            s_soft = s_attn
            t_soft = t_attn_nn

        # KL(teacher || student) per query position, masked
        # Only compute for real nodes (not padding)
        kl = t_soft * (torch.log(t_soft + 1e-10) - torch.log(s_soft + 1e-10))
        kl = kl.sum(dim=-1)  # (B, H, N) — sum over key dim

        # Mask out padding query positions
        query_mask = mask.unsqueeze(1)  # (B, 1, N)
        kl = kl * query_mask.float()

        # Average over valid positions
        num_valid = query_mask.float().sum() * H
        layer_loss = kl.sum() / (num_valid + 1e-10)
        total_loss = total_loss + layer_loss

    return total_loss / num_layers


# ================================================================
# TRAINING LOOP
# ================================================================

def run_experiment(args):
    print("=" * 60)
    print("Attention Distillation Experiment")
    print(f"  distill_weight={args.distill_weight}")
    print(f"  temperature={args.temperature}")
    print(f"  proxy_opt_steps={args.proxy_opt_steps}")
    print(f"  num_proxies={args.num_proxies}")
    print(f"  distill_layers={args.distill_layers}")
    print(f"  proxy_mmd_lambda={args.proxy_mmd_lambda}")
    print("=" * 60)

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    # Load pretrained model as teacher (frozen, with attention extraction)
    teacher = GraphTransformerWithAttn(
        num_layers=args.num_layers, num_heads=args.num_heads,
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(args.device)

    ckpt = torch.load(args.model_path, map_location=args.device, weights_only=True)
    # Map from original model's state dict keys to our variant
    teacher.load_state_dict(ckpt["model_state"])
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # Student: same architecture, initialized from same pretrained weights
    student = GraphTransformerWithAttn(
        num_layers=args.num_layers, num_heads=args.num_heads,
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(args.device)
    student.load_state_dict(ckpt["model_state"])

    # Baseline: evaluate pretrained model before any distillation
    print("\nBaseline (pretrained, no distillation):")
    baseline_val_ap = evaluate(student, val_loader, args)
    baseline_test_ap = evaluate(student, test_loader, args)
    print(f"  val_AP={baseline_val_ap:.4f}  test_AP={baseline_test_ap:.4f}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7)

    loss_fn = nn.BCEWithLogitsLoss()
    best_val_ap = 0.0
    best_epoch = -1
    patience_counter = 0

    # Parse which layers to distill
    if args.distill_layers == "all":
        distill_layer_indices = list(range(args.num_layers))
    else:
        distill_layer_indices = [int(x) for x in args.distill_layers.split(",")]

    for epoch in range(1, args.max_epochs + 1):
        epoch_start = time.time()
        student.train()
        teacher.eval()

        train_losses, task_losses_log, distill_losses_log, mmd_losses_log = [], [], [], []
        all_preds, all_labels = [], []

        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad()

            # --- Get teacher attention targets (with optimized proxies) ---
            with torch.no_grad():
                dense_x_t, dense_mask_t = teacher.encode_dense(batch)

                # Optimize proxies against frozen teacher (with MMD regularization)
                best_proxies = optimize_proxies_for_batch(
                    teacher, batch, dense_x_t, dense_mask_t, args)

                # Log MMD of optimized proxies vs node embeddings
                if args.proxy_mmd_lambda > 0:
                    batch_mmd = []
                    for i in range(dense_x_t.size(0)):
                        nodes_i = dense_x_t[i][dense_mask_t[i]]
                        batch_mmd.append(mmd_squared(best_proxies[i], nodes_i).item())
                    mmd_losses_log.append(float(np.mean(batch_mmd)))

                # Get teacher attention with optimized proxies
                _, _, teacher_attns = teacher(
                    batch, proxy_embeddings=best_proxies,
                    precomputed_dense=(dense_x_t, dense_mask_t),
                    return_attention=True)

                # Filter to only distilled layers
                teacher_attns_filtered = [teacher_attns[i] for i in distill_layer_indices]

            # --- Student forward (no proxies) ---
            logits, _, student_attns = student(
                batch, return_attention=True)

            student_attns_filtered = [student_attns[i] for i in distill_layer_indices]

            # Task loss
            task_loss = loss_fn(logits, batch.y)

            # Distillation loss
            dense_x_s, dense_mask_s = student.encode_dense(batch)
            distill_loss = attention_distillation_loss(
                student_attns_filtered, teacher_attns_filtered,
                dense_mask_s, temperature=args.temperature)

            # Combined loss
            loss = task_loss + args.distill_weight * distill_loss
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            task_losses_log.append(task_loss.item())
            distill_losses_log.append(distill_loss.item())
            all_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())

        train_ap = compute_macro_ap(
            np.concatenate(all_preds), np.concatenate(all_labels))
        mean_task = float(np.mean(task_losses_log))
        mean_distill = float(np.mean(distill_losses_log))

        # Val / Test
        val_ap = evaluate(student, val_loader, args)
        test_ap = evaluate(student, test_loader, args)

        elapsed = time.time() - epoch_start
        log_line = (f"Epoch {epoch:3d}/{args.max_epochs} [{elapsed:.1f}s] | "
                    f"task={mean_task:.4f} distill={mean_distill:.4f} | "
                    f"train_AP={train_ap:.4f} val_AP={val_ap:.4f} test_AP={test_ap:.4f}")
        if mmd_losses_log:
            log_line += f" | proxy_mmd={float(np.mean(mmd_losses_log)):.6f}"
        print(log_line, flush=True)

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


@torch.no_grad()
def evaluate(model, loader, args):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(args.device)
        logits, _ = model(batch, return_attention=False)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())
    return compute_macro_ap(np.concatenate(all_preds), np.concatenate(all_labels))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to pretrained Stage 1 checkpoint")
    p.add_argument("--save_dir", type=str, default="checkpoints_attn_distill")

    # Model
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)

    # Proxy optimization (for generating teacher attention targets)
    p.add_argument("--num_proxies", type=int, default=4)
    p.add_argument("--proxy_lr", type=float, default=5e-2)
    p.add_argument("--proxy_opt_steps", type=int, default=75)
    p.add_argument("--proxy_mmd_lambda", type=float, default=0.01,
                   help="MMD regularization weight to keep optimized proxies "
                        "in same distribution as encoder node embeddings. "
                        "0 disables MMD constraint.")

    # Distillation
    p.add_argument("--distill_weight", type=float, default=1.0,
                   help="Weight for attention distillation loss")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Temperature for softening attention distributions")
    p.add_argument("--distill_layers", type=str, default="all",
                   help="Which layers to distill: 'all' or comma-separated indices like '2,3,4'")

    # Training
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    run_experiment(args)


if __name__ == "__main__":
    main()

