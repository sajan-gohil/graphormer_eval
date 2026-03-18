"""
5-Phase Diffusion Proxy Node Framework
=======================================
Phase 1  : Pretrain Graph Transformer backbone (with early stopping)
Phase 2  : Per-sample proxy embedding optimisation (frozen backbone, 5× samples,
           filtered to keep only significantly improved proxies)
Phase 2.5: Oracle concept validation (full data, 5% relative AP threshold)
Phase 3  : Train conditional DDPM denoiser  (T=50, consistent train/infer)
Phase 4  : Train GT backbone with diffusion proxy nodes (init from P1 weights)
Phase 4.5: Hyperedge-routed attention N→M→N (optional, --run-phase-4-5 flag)
Phase 5  : Train GNN backbone with proxy nodes added as explicit graph nodes
"""

import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import scatter

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torchmetrics.classification import MultilabelAveragePrecision


# ================================================================
# CONFIG
# ================================================================
HIDDEN_DIM  = 128
NUM_HEADS   = 4
NUM_LAYERS  = 4
NUM_PROXY   = 64
BATCH_SIZE  = 64
OUTPUT_DIM  = 10
T_DIFF      = 20        # single source-of-truth for diffusion timesteps
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR    = "./checkpoints6"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")


# ================================================================
# DATA LOADING
# ================================================================
def get_loaders(batch_size=BATCH_SIZE):
    train_ds = LRGBDataset(root="./data", name="Peptides-func", split="train")
    val_ds   = LRGBDataset(root="./data", name="Peptides-func", split="val")
    test_ds  = LRGBDataset(root="./data", name="Peptides-func", split="test")
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4),
        train_ds, val_ds, test_ds,
    )


# ================================================================
# SHARED MODULES
# ================================================================

class NodeEncoder(nn.Module):
    """Atom + bond-aggregated node embeddings."""
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim // 2)
        self.bond_encoder = BondEncoder(hidden_dim // 2)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        h = self.atom_encoder(x)
        row = edge_index[0]
        edge_emb  = self.bond_encoder(edge_attr)
        edge_aggr = scatter(edge_emb, row, dim=0, dim_size=h.size(0), reduce="add")
        h = torch.cat([h, edge_aggr], dim=-1)
        return self.proj(h)                              # [total_N, d]


# ================================================================
# PHASE 1 — GRAPH TRANSFORMER BACKBONE
# ================================================================

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads

        self.wq        = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wk        = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wv        = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wout      = nn.Linear(hidden_dim, hidden_dim)
        self.ff        = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(hidden_dim)
        self.norm2     = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        B, N, d = x.shape
        normed = self.norm1(x)
        q = self.wq(normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / self.head_dim ** 0.5

        if mask is not None:
            key_pad   = (~mask).unsqueeze(1).unsqueeze(2)    # [B,1,1,N] mask keys
            query_pad = (~mask).unsqueeze(1).unsqueeze(-1)  # [B,1,N,1] mask queries
            attn = attn.masked_fill(key_pad, float("-inf"))
            attn = attn.masked_fill(query_pad, float("-inf"))

        attn_w = F.softmax(attn, dim=-1)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        attn_w = self.attn_drop(attn_w)

        out = (attn_w @ v).transpose(1, 2).reshape(B, N, d)
        x   = x + self.wout(out)
        x   = x + self.ff(self.norm2(x))
        return attn_w, x


class GraphTransformer(nn.Module):
    def __init__(self, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                 hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder    = NodeEncoder(hidden_dim)
        self.layers     = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def encode_nodes(self, batch):
        return self.encoder(batch.x, batch.edge_index, batch.edge_attr)

    def encode_dense(self, batch):
        h = self.encode_nodes(batch)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, proxy_embeddings=None, precomputed_dense=None):
        # Allow pre-computed encoder output to avoid double encoding in
        # Phase 4 end-to-end training (encoder → denoiser → transformer).
        if precomputed_dense is not None:
            dense_x, dense_mask = precomputed_dense
        else:
            dense_x, dense_mask = self.encode_dense(batch)
        B, max_N, d = dense_x.shape

        if proxy_embeddings is not None:
            M = proxy_embeddings.shape[1]
            dense_x  = torch.cat([dense_x, proxy_embeddings], dim=1)
            aug_mask = torch.cat([
                dense_mask,
                torch.ones(B, M, dtype=torch.bool, device=dense_x.device)
            ], dim=1)
        else:
            aug_mask = dense_mask

        for layer in self.layers:
            _, dense_x = layer(dense_x, aug_mask)

        orig_x  = dense_x[:, :max_N, :]
        node_emb = orig_x[dense_mask]
        pooled   = global_mean_pool(node_emb, batch.batch)
        return self.head(pooled), node_emb


# ----------------------------------------------------------------
# Shared training helpers
# ----------------------------------------------------------------

def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)

def _unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)


def evaluate(model, loader, proxy_fn=None):
    loss_fn = nn.BCEWithLogitsLoss()
    metric  = MultilabelAveragePrecision(num_labels=OUTPUT_DIM).to(DEVICE)
    losses  = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch   = batch.to(DEVICE)
            proxies = proxy_fn(batch) if proxy_fn else None
            logits, _ = model(batch, proxies)
            loss = loss_fn(logits, batch.y)
            metric.update(torch.sigmoid(logits), batch.y.long())
            losses.append(loss.item())
    return float(metric.compute()), float(np.mean(losses))


def phase1_train(epochs=500, lr=3e-5, weight_decay=1e-5, patience=50):
    print("\n" + "=" * 60)
    print("PHASE 1: Pretraining Graph Transformer Backbone")
    print("=" * 60)

    train_loader, val_loader, test_loader, *_ = get_loaders()
    model     = GraphTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.BCEWithLogitsLoss()
    best_val  = 0.0
    no_improve = 0
    history   = dict(train_ap=[], val_ap=[], test_ap=[], train_loss=[], val_loss=[])

    for epoch in range(epochs):
        model.train()
        metric  = MultilabelAveragePrecision(num_labels=OUTPUT_DIM).to(DEVICE)
        ep_loss = []
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1:03d}", leave=False):
            batch  = batch.to(DEVICE)
            logits, _ = model(batch)
            loss   = loss_fn(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            metric.update(torch.sigmoid(logits), batch.y.long())
            ep_loss.append(loss.item())

        sched.step()
        train_ap, train_loss = float(metric.compute()), float(np.mean(ep_loss))
        val_ap, val_loss     = evaluate(model, val_loader)
        history["train_ap"].append(train_ap)
        history["val_ap"].append(val_ap)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_ap > best_val:
            best_val = val_ap
            no_improve = 0
            torch.save(model.state_dict(), f"{SAVE_DIR}/phase1_best.pt")
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            test_ap, test_loss = evaluate(model, test_loader)
            history["test_ap"].append(test_ap)
            print(f"Ep {epoch+1:3d} | Train AP {train_ap:.4f} Loss {train_loss:.4f}"
                  f" | Val AP {val_ap:.4f} Loss {val_loss:.4f}"
                  f" | Test AP {test_ap:.4f} Loss {test_loss:.4f}"
                  f" | Patience {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    model.load_state_dict(torch.load(f"{SAVE_DIR}/phase1_best.pt"))
    final_test_ap, _ = evaluate(model, test_loader)
    print(f"\nPhase 1 done. Best Val AP: {best_val:.4f}  Test AP: {final_test_ap:.4f}")
    return model, history


# ================================================================
# PHASE 2 — PER-SAMPLE PROXY OPTIMISATION  (5× samples per graph)
# ================================================================

def _count_correct_classes(logits, labels):
    """Count number of classes where prediction matches label (threshold 0.5)."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    return int((preds == labels).sum().item())


def optimize_single_graph(model, batch, num_proxy=NUM_PROXY, max_steps=1000, lr=1e-3):
    loss_fn = nn.BCEWithLogitsLoss()
    proxy   = nn.Parameter(torch.randn(1, num_proxy, model.hidden_dim, device=DEVICE) * 0.02)
    opt     = torch.optim.Adam([proxy], lr=lr)

    with torch.no_grad():
        base_logits = model(batch)[0]
        base_loss = loss_fn(base_logits, batch.y).item()
        base_correct = _count_correct_classes(base_logits, batch.y)

    best_loss, best_proxy = base_loss, proxy.data.clone()
    best_correct = base_correct
    for _ in range(max_steps):
        opt.zero_grad()
        logits, _ = model(batch, proxy_embeddings=proxy)
        loss = loss_fn(logits, batch.y)
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss, best_proxy = loss.item(), proxy.data.clone()
            with torch.no_grad():
                best_correct = _count_correct_classes(logits, batch.y)

    return best_proxy.detach(), base_loss, best_loss, base_correct, best_correct


def phase2_optimize_proxies(backbone, train_dataset,
                             subset_fraction=1.0, num_proxy=NUM_PROXY,
                             max_steps=1000, proxy_lr=1e-3, num_repeats=5,
                             save_path=None):
    """
    For each graph, run proxy optimisation `num_repeats` times (different random
    initialisations) and store all (encoder_emb, proxy_emb) pairs.
    This gives the diffusion model diverse targets for the same conditioning.

    NOTE: proxy_pairs will have len = num_repeats * len(train_dataset).
          OracleProxyDataset only uses the first len(train_dataset) entries.
    """
    print("\n" + "=" * 60)
    print(f"PHASE 2: Per-Sample Proxy Optimisation  ({num_repeats}× per graph)")
    print("=" * 60)

    if save_path is None:
        save_path = f"{SAVE_DIR}/proxy_pairs.pkl"

    _freeze(backbone)
    backbone.eval()

    single_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    n_total       = int(len(train_dataset) * subset_fraction)
    proxy_pairs   = []
    improvements  = []
    # Track which samples have found an improved proxy: 1 = found, 0 = not
    sample_improved = [0] * n_total

    for rep in range(num_repeats):
        for idx, batch in enumerate(tqdm(single_loader, total=n_total,
                                         desc=f"Proxy opt rep {rep+1}/{num_repeats}")):
            if idx >= n_total:
                break
            batch = batch.to(DEVICE)
            with torch.no_grad():
                dense_x, dense_mask = backbone.encode_dense(batch)

            best_proxy, base_loss, opt_loss, base_correct, best_correct = \
                optimize_single_graph(
                    backbone, batch, num_proxy=num_proxy,
                    max_steps=max_steps, lr=proxy_lr
                )
            improvement = (base_loss - opt_loss) / (base_loss + 1e-8)
            improvements.append(improvement)

            # Only keep sample if significant improvement:
            #   opt_loss < 0.005  OR  number of correct classes increased
            is_significant = (opt_loss < 0.005) or (best_correct > base_correct)
            if is_significant:
                sample_improved[idx] = 1
                proxy_pairs.append({
                    "encoder_emb": dense_x.squeeze(0).cpu(),
                    "mask":        dense_mask.squeeze(0).cpu(),
                    "proxy_emb":   best_proxy.squeeze(0).cpu(),
                    "base_loss":   base_loss,
                    "opt_loss":    opt_loss,
                    "sample_idx":  idx,
                })

            if (idx + 1) % 200 == 0:
                n_improved = sum(sample_improved[:idx+1])
                print(
                    f"  [rep {rep+1} | {idx+1}/{n_total}] "
                    f"Avg improvement: {np.mean(improvements[-200:]):.4f}  "
                    f"Improved: {sum(i>0 for i in improvements[-200:])}/200  "
                    f"Kept (significant): {n_improved}/{idx+1}"
                )

    _unfreeze(backbone)
    with open(save_path, "wb") as f:
        pickle.dump(proxy_pairs, f)

    n_no_improve = sum(1 for v in sample_improved if v == 0)
    pos_rate = np.mean([i > 0 for i in improvements])
    print(f"\nPhase 2 done. Saved {len(proxy_pairs)} significant pairs → {save_path}")
    print(f"Positive improvement rate: {pos_rate:.2%}  Mean: {np.mean(improvements):.4f}")
    print(f"Samples with no improved proxy: {n_no_improve}/{n_total}")
    return proxy_pairs, sample_improved


# ================================================================
# PHASE 2.5 — ORACLE EVAL
# ================================================================

class OracleProxyDataset(torch.utils.data.Dataset):
    """
    Pairs each graph with the best optimised proxy across all repeats.
    For samples with no improved proxy, uses a zero proxy (baseline-equivalent).
    """
    def __init__(self, pyg_dataset, proxy_pairs):
        n = len(pyg_dataset)
        self.dataset = pyg_dataset

        # Build best proxy per sample_idx (lowest opt_loss across all repeats)
        best_by_idx = {}
        for p in proxy_pairs:
            sid = p.get("sample_idx", None)
            if sid is None:
                continue
            if sid not in best_by_idx or p["opt_loss"] < best_by_idx[sid]["opt_loss"]:
                best_by_idx[sid] = p

        # For each dataset sample, use best proxy or zero proxy
        self.best_proxies = []
        d = proxy_pairs[0]["proxy_emb"].shape[-1] if proxy_pairs else HIDDEN_DIM
        m = proxy_pairs[0]["proxy_emb"].shape[0]  if proxy_pairs else NUM_PROXY
        zero_proxy = torch.zeros(m, d)
        for i in range(n):
            if i in best_by_idx:
                self.best_proxies.append(best_by_idx[i]["proxy_emb"])
            else:
                self.best_proxies.append(zero_proxy)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.best_proxies[idx]


def collate_oracle(batch):
    graphs, proxies = zip(*batch)
    pyg_batch   = torch_geometric.data.Batch.from_data_list(list(graphs))
    proxy_batch = torch.stack(proxies, dim=0)
    return pyg_batch, proxy_batch


def eval_oracle_proxies(backbone, pyg_dataset, proxy_pairs, batch_size=BATCH_SIZE):
    """Evaluate oracle on the full dataset, using average AP as the metric."""
    print("\n" + "=" * 60)
    print("PHASE 2.5: Oracle Proxy Concept Validation (full data)")
    print("=" * 60)

    oracle_loader = torch.utils.data.DataLoader(
        OracleProxyDataset(pyg_dataset, proxy_pairs),
        batch_size=batch_size, shuffle=False,
        collate_fn=collate_oracle, num_workers=2,
    )
    loss_fn       = nn.BCEWithLogitsLoss()
    m_oracle      = MultilabelAveragePrecision(num_labels=OUTPUT_DIM).to(DEVICE)
    m_base        = MultilabelAveragePrecision(num_labels=OUTPUT_DIM).to(DEVICE)
    lo, lb        = [], []

    backbone.eval()
    with torch.inference_mode():
        for pyg_batch, proxy_batch in tqdm(oracle_loader, desc="Oracle eval"):
            pyg_batch   = pyg_batch.to(DEVICE)
            proxy_batch = proxy_batch.to(DEVICE)
            lp, _ = backbone(pyg_batch, proxy_embeddings=proxy_batch)
            lb_logits, _ = backbone(pyg_batch)
            m_oracle.update(torch.sigmoid(lp),        pyg_batch.y.long())
            m_base.update(torch.sigmoid(lb_logits),   pyg_batch.y.long())
            lo.append(loss_fn(lp, pyg_batch.y).item())
            lb.append(loss_fn(lb_logits, pyg_batch.y).item())

    oracle_ap = float(m_oracle.compute())
    base_ap   = float(m_base.compute())
    delta     = oracle_ap - base_ap
    relative_gain = delta / (base_ap + 1e-8)
    print(f"  Baseline (no proxies) : AP={base_ap:.4f}  Loss={np.mean(lb):.4f}")
    print(f"  Oracle   (opt proxies): AP={oracle_ap:.4f}  Loss={np.mean(lo):.4f}  Δ={delta:+.4f}")
    print(f"  Relative gain: {relative_gain:.2%}")

    concept_valid = relative_gain > 0.05
    if concept_valid:
        print(f"  ✓ Concept valid — {relative_gain:.5%} above base (>5% threshold) — proceed to Phase 4/5")
    else:
        print(f"  ✗ Oracle gain {relative_gain:.5%} below 5% threshold — Phase 4/5 may not benefit")
    return oracle_ap, base_ap, concept_valid


# ================================================================
# PHASE 3 — CONDITIONAL DIFFUSION  (T=T_DIFF throughout)
# ================================================================

class DDPMScheduler:
    """
    Linear-beta DDPM with DDIM deterministic sampling.
    T_DIFF is used for BOTH training and inference — no schedule mismatch.
    """
    def __init__(self, T=T_DIFF, beta_start=1e-4, beta_end=0.02):
        self.T     = T
        betas      = torch.linspace(beta_start, beta_end, T)
        alphas     = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        # index 0 = clean (prepend 1.0)
        self.alpha_bars     = torch.cat([torch.ones(1), alpha_bars])
        self.sqrt_ab        = self.alpha_bars.sqrt()
        self.sqrt_one_minus = (1 - self.alpha_bars).sqrt()

    def to(self, device):
        self.alpha_bars     = self.alpha_bars.to(device)
        self.sqrt_ab        = self.sqrt_ab.to(device)
        self.sqrt_one_minus = self.sqrt_one_minus.to(device)
        return self

    def q_sample(self, x0, t):
        t_idx = t + 1
        sab   = self.sqrt_ab[t_idx].view(-1, 1, 1)
        s1m   = self.sqrt_one_minus[t_idx].view(-1, 1, 1)
        eps   = torch.randn_like(x0)
        return sab * x0 + s1m * eps, eps

    @torch.no_grad()
    def ddim_sample(self, denoiser, x_enc, x_mask,
                    num_proxy=NUM_PROXY, hidden_dim=HIDDEN_DIM):
        """
        DDIM reverse pass with x0-prediction parameterisation.
        Denoiser directly predicts x0; noise is derived for the DDIM step.
        """
        B      = x_enc.shape[0]
        device = x_enc.device
        x_t    = torch.randn(B, num_proxy, hidden_dim, device=device)

        ts = torch.linspace(self.T - 1, 0, self.T, dtype=torch.long)
        for i, t_val in enumerate(ts):
            t_batch = torch.full((B,), t_val.item(), device=device, dtype=torch.long)
            # Denoiser directly predicts x0
            x0_pred = denoiser(x_t, t_batch, x_enc, x_mask)
            x0_pred = x0_pred.clamp(-10, 10)
            t_cur   = t_val.item() + 1
            t_prev  = int(ts[i + 1].item()) + 1 if i + 1 < len(ts) else 0
            # Derive noise for DDIM step
            eps_derived = (x_t - self.sqrt_ab[t_cur] * x0_pred) / (self.sqrt_one_minus[t_cur] + 1e-8)
            x_t = self.sqrt_ab[t_prev] * x0_pred + self.sqrt_one_minus[t_prev] * eps_derived

        return x_t

    def single_step_denoise(self, denoiser, x_enc, x_mask,
                            num_proxy=NUM_PROXY, hidden_dim=HIDDEN_DIM):
        """
        Single-step x0 prediction from pure noise.  Fully differentiable —
        gradients flow through denoiser back to x_enc (and thus the encoder).
        Used in Phase 4 end-to-end training.
        """
        B      = x_enc.shape[0]
        device = x_enc.device
        x_T    = torch.randn(B, num_proxy, hidden_dim, device=device)
        t      = torch.full((B,), self.T - 1, device=device, dtype=torch.long)
        x0_pred = denoiser(x_T, t, x_enc, x_mask)
        return x0_pred


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim  = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half   = self.dim // 2
        freqs  = torch.exp(-np.log(10000) * torch.arange(half, device=t.device, dtype=torch.float) / half)
        angles = t[:, None].float() * freqs[None]
        return self.proj(torch.cat([angles.sin(), angles.cos()], -1))


class DenoiserBlock(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, b, x_enc, x_mask=None):
        b2, _ = self.self_attn(self.norm1(b), self.norm1(b), self.norm1(b))
        b = b + b2
        kp    = (~x_mask) if x_mask is not None else None
        b2, _ = self.cross_attn(self.norm2(b), x_enc, x_enc, key_padding_mask=kp)
        b = b + b2
        return b + self.ff(self.norm3(b))


class ProxyDenoiser(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_blocks=4):
        super().__init__()
        self.time_emb = TimestepEmbedding(hidden_dim)
        self.t_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.blocks   = nn.ModuleList([DenoiserBlock(hidden_dim, num_heads) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, b_noisy, t, x_enc, x_mask=None):
        t_emb = self.t_proj(self.time_emb(t)).unsqueeze(1)
        b     = b_noisy + t_emb
        for block in self.blocks:
            b = block(b, x_enc, x_mask)
        return self.out_proj(self.out_norm(b))


class ProxyPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        return p["encoder_emb"], p["mask"], p["proxy_emb"]


def collate_proxy_pairs(batch):
    enc_list, mask_list, proxy_list = zip(*batch)
    max_n = max(e.shape[0] for e in enc_list)
    d, B  = enc_list[0].shape[1], len(enc_list)
    enc_pad  = torch.zeros(B, max_n, d)
    mask_pad = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (e, m) in enumerate(zip(enc_list, mask_list)):
        n = e.shape[0]
        enc_pad[i, :n]  = e
        mask_pad[i, :n] = m
    return enc_pad, mask_pad, torch.stack(proxy_list)


def phase3_train_diffusion(proxy_pairs, epochs=500, lr=1e-4,
                            batch_size=64, save_path=None):
    """
    T is read from T_DIFF (global constant) — single source of truth.
    Training and inference both use the same T-step schedule.
    """
    print("\n" + "=" * 60)
    print(f"PHASE 3: Training Conditional Diffusion (T={T_DIFF})")
    print("=" * 60)

    if save_path is None:
        save_path = f"{SAVE_DIR}/phase3_diffusion_best.pt"

    sched    = DDPMScheduler(T=T_DIFF).to(DEVICE)
    denoiser = ProxyDenoiser().to(DEVICE)
    opt      = torch.optim.AdamW(denoiser.parameters(), lr=lr, weight_decay=1e-5)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = torch.utils.data.DataLoader(
        ProxyPairDataset(proxy_pairs), batch_size=batch_size, shuffle=True,
        collate_fn=collate_proxy_pairs, num_workers=2, pin_memory=True,
    )

    best_loss = float("inf")
    history   = {"loss": []}

    for epoch in range(epochs):
        denoiser.train()
        ep_losses = []
        for x_enc, x_mask, b0 in tqdm(loader, desc=f"Diffusion ep {epoch+1}", leave=False):
            x_enc, x_mask, b0 = x_enc.to(DEVICE), x_mask.to(DEVICE), b0.to(DEVICE)
            t            = torch.randint(0, T_DIFF, (b0.shape[0],), device=DEVICE)
            b_noisy, _   = sched.q_sample(b0, t)
            # x0-prediction: denoiser directly predicts the clean proxy
            x0_pred      = denoiser(b_noisy, t, x_enc, x_mask)
            loss         = F.mse_loss(x0_pred, b0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            ep_losses.append(loss.item())

        lr_sched.step()
        mean_loss = float(np.mean(ep_losses))
        history["loss"].append(mean_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(denoiser.state_dict(), save_path)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | MSE: {mean_loss:.6f}")

    denoiser.load_state_dict(torch.load(save_path))
    print(f"\nPhase 3 done. Best loss: {best_loss:.6f}")
    return denoiser, sched, history


# ================================================================
# PHASE 4 — FRESH GT BACKBONE + DIFFUSION PROXIES
# ================================================================

def _make_proxy_fn(backbone, denoiser, sched):
    """Returns a callable(batch) -> [B, M, d] that generates proxies."""
    @torch.no_grad()
    def proxy_fn(batch):
        backbone.eval()
        dense_x, dense_mask = backbone.encode_dense(batch)
        return sched.ddim_sample(denoiser, dense_x, dense_mask,
                                 num_proxy=NUM_PROXY, hidden_dim=HIDDEN_DIM)
    return proxy_fn


def phase4_train_with_proxies(denoiser, sched, train_loader, val_loader, test_loader,
                               p1_state_dict=None,
                               epochs=500, lr=3e-5, weight_decay=1e-5,
                               warmup_epochs=50):
    """
    Phase 4 — end-to-end finetuning of the full pipeline with task loss.

    Architecture (all differentiable):
        Atom+Bond Encoder
            |              |
            v              v
        Denoiser           |
            |              |
            v     +        v
              Transformer → task head → loss

    The encoder output feeds into both the denoiser (conditioning) and the
    transformer (node embeddings).  The denoiser generates proxy tokens via
    single-step x0 prediction.  Task loss backpropagates through everything:
    transformer → proxy nodes → denoiser → encoder.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: End-to-End Pipeline Finetuning (encoder + denoiser + GT)")
    print("=" * 60)

    # Unfreeze everything — full pipeline finetuning
    _unfreeze(denoiser)

    backbone = GraphTransformer().to(DEVICE)
    if p1_state_dict is not None:
        backbone.load_state_dict(p1_state_dict)
        print("  Loaded Phase 1 pretrained weights into Phase 4 backbone")

    # All parameters in a single optimizer: encoder + transformer + head + denoiser
    all_params = list(backbone.parameters()) + list(denoiser.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.BCEWithLogitsLoss()
    # Non-differentiable proxy fn for evaluation
    get_proxies = _make_proxy_fn(backbone, denoiser, sched)

    n_backbone = sum(p.numel() for p in backbone.parameters())
    n_denoiser = sum(p.numel() for p in denoiser.parameters())
    print(f"  Backbone params: {n_backbone:,}  Denoiser params: {n_denoiser:,}")
    print(f"  Total trainable: {n_backbone + n_denoiser:,}  Warmup: {warmup_epochs} epochs")

    best_val = 0.0
    history  = dict(train_ap=[], val_ap=[], test_ap=[], train_loss=[], val_loss=[])

    for epoch in range(epochs):
        backbone.train()
        denoiser.train()
        use_proxies = (epoch >= warmup_epochs)
        metric, ep_loss = MultilabelAveragePrecision(num_labels=OUTPUT_DIM).to(DEVICE), []

        for batch in tqdm(train_loader,
                          desc=f"P4 {epoch+1:03d} {'[e2e]' if use_proxies else '[warmup]'}",
                          leave=False):
            batch = batch.to(DEVICE)

            if use_proxies:
                # ── Differentiable end-to-end path ──────────────
                # 1. Encode once — shared between denoiser & transformer
                dense_x, dense_mask = backbone.encode_dense(batch)
                # 2. Single-step x0 prediction (differentiable through denoiser)
                proxies = sched.single_step_denoise(
                    denoiser, dense_x, dense_mask,
                    num_proxy=NUM_PROXY, hidden_dim=HIDDEN_DIM
                )
                # 3. Transformer forward with pre-computed encoding
                logits, _ = backbone(
                    batch, proxy_embeddings=proxies,
                    precomputed_dense=(dense_x, dense_mask)
                )
            else:
                logits, _ = backbone(batch)

            loss = loss_fn(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            metric.update(torch.sigmoid(logits), batch.y.long())
            ep_loss.append(loss.item())

        lr_sched.step()
        train_ap   = float(metric.compute())
        train_loss = float(np.mean(ep_loss))
        pfn        = get_proxies if use_proxies else None
        val_ap, val_loss = evaluate(backbone, val_loader, proxy_fn=pfn)

        history["train_ap"].append(train_ap)
        history["val_ap"].append(val_ap)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_ap > best_val:
            best_val = val_ap
            torch.save(backbone.state_dict(), f"{SAVE_DIR}/phase4_best_backbone.pt")
            torch.save(denoiser.state_dict(), f"{SAVE_DIR}/phase4_best_denoiser.pt")

        if (epoch + 1) % 10 == 0:
            test_ap, _ = evaluate(backbone, test_loader, proxy_fn=pfn)
            history["test_ap"].append(test_ap)
            print(f"Ep {epoch+1:3d} {'[e2e]' if use_proxies else '[warmup]'} "
                  f"| Train {train_ap:.4f} {train_loss:.4f} "
                  f"| Val {val_ap:.4f} {val_loss:.4f} | Test {test_ap:.4f}")

    backbone.load_state_dict(torch.load(f"{SAVE_DIR}/phase4_best_backbone.pt"))
    denoiser.load_state_dict(torch.load(f"{SAVE_DIR}/phase4_best_denoiser.pt"))
    pfn_final = _make_proxy_fn(backbone, denoiser, sched)
    final_test_ap, _ = evaluate(backbone, test_loader, proxy_fn=pfn_final)
    print(f"\nPhase 4 done. Best Val AP: {best_val:.4f}  Test AP: {final_test_ap:.4f}")
    return backbone, denoiser, history


# ================================================================
# PHASE 4.5 — HYPEREDGE-ROUTED SELF-ATTENTION  (N→M→N)
# ================================================================
#
# Instead of full N+M self-attention (Phase 4), attention is routed through
# M proxy/hyperedge nodes:
#   Step 1: Proxies (M) attend to nodes (N) → updated proxy embeddings
#   Step 2: Nodes  (N) attend to proxies (M) → updated node embeddings
# Nodes never directly attend to other nodes in this layer.
# ================================================================

class HypergraphTransformerLayer(nn.Module):
    """
    Implements the N→M→N attention routing:
      Step 1: Proxies (M) attend to nodes (N) → updated proxy (hyperedge) embeddings
      Step 2: Nodes  (N) attend to proxies (M) → updated node embeddings
    Nodes never directly attend to other nodes in this layer.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=0.1):
        super().__init__()
        # Step 1: M queries, N keys/values
        self.node_to_proxy = nn.MultiheadAttention(hidden_dim, num_heads,
                                                    dropout=dropout, batch_first=True)
        # Step 2: N queries, M keys/values
        self.proxy_to_node = nn.MultiheadAttention(hidden_dim, num_heads,
                                                    dropout=dropout, batch_first=True)
        self.ff_proxy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ff_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)  # pre-norm for node_to_proxy
        self.norm2 = nn.LayerNorm(hidden_dim)  # pre-norm for proxy_to_node
        self.norm3 = nn.LayerNorm(hidden_dim)  # ff norm for proxies
        self.norm4 = nn.LayerNorm(hidden_dim)  # ff norm for nodes

    def forward(self, nodes, proxies, node_mask=None):
        """
        nodes   : [B, N, d]  — padded original node embeddings
        proxies : [B, M, d]  — proxy / hyperedge embeddings
        node_mask: [B, N] bool — True = real node
        Returns updated (nodes, proxies).
        """
        # key_padding_mask expects True = IGNORE
        kpm = (~node_mask) if node_mask is not None else None

        # ── Step 1: N → M  (proxies query from nodes)
        p2, _ = self.node_to_proxy(
            query=self.norm1(proxies),
            key=self.norm1(nodes),
            value=self.norm1(nodes),
            key_padding_mask=kpm
        )
        proxies = proxies + p2
        proxies = proxies + self.ff_proxy(self.norm3(proxies))

        # ── Step 2: M → N  (nodes query from proxies)
        n2, _ = self.proxy_to_node(
            query=self.norm2(nodes),
            key=proxies,
            value=proxies
        )
        nodes = nodes + n2
        nodes = nodes + self.ff_node(self.norm4(nodes))

        return nodes, proxies


class HypergraphGraphTransformer(nn.Module):
    """
    Graph Transformer with hyperedge-routed attention layers.
    First `num_warmup_layers` are standard TransformerLayers (no proxies needed),
    remaining layers are HypergraphTransformerLayers (N→M→N routing).
    """
    def __init__(self, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                 hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
                 num_warmup_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder    = NodeEncoder(hidden_dim)
        self.num_warmup_layers = num_warmup_layers

        layers = []
        for i in range(num_layers):
            if i < num_warmup_layers:
                layers.append(TransformerLayer(hidden_dim, num_heads, dropout=0.1))
            else:
                layers.append(HypergraphTransformerLayer(hidden_dim, num_heads, dropout=0.1))
        self.layers = nn.ModuleList(layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def encode_nodes(self, batch):
        return self.encoder(batch.x, batch.edge_index, batch.edge_attr)

    def encode_dense(self, batch):
        h = self.encode_nodes(batch)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, proxy_embeddings=None):
        dense_x, dense_mask = self.encode_dense(batch)

        if proxy_embeddings is not None:
            proxies = proxy_embeddings  # [B, M, d]
            for layer in self.layers:
                if isinstance(layer, HypergraphTransformerLayer):
                    dense_x, proxies = layer(dense_x, proxies, dense_mask)
                else:
                    _, dense_x = layer(dense_x, dense_mask)
        else:
            for layer in self.layers:
                if isinstance(layer, HypergraphTransformerLayer):
                    # No proxies — skip hypergraph layers (warmup path)
                    continue
                else:
                    _, dense_x = layer(dense_x, dense_mask)

        node_emb = dense_x[dense_mask]
        pooled   = global_mean_pool(node_emb, batch.batch)
        return self.head(pooled), node_emb


def phase4_5_train_hypergraph(denoiser, sched, train_loader, val_loader, test_loader,
                               p1_state_dict=None,
                               epochs=500, lr=3e-5, weight_decay=1e-5,
                               warmup_epochs=50):
    """
    Phase 4.5: Hyperedge-routed self-attention (N→M→N).
    Attention through proxy hyperedge nodes instead of full self-attention.
    """
    print("\n" + "=" * 60)
    print("PHASE 4.5: Hyperedge-Routed Attention (N→M→N)")
    print("=" * 60)

    _freeze(denoiser)
    denoiser.eval()

    backbone = HypergraphGraphTransformer().to(DEVICE)
    # Initialize encoder from Phase 1 weights if available
    if p1_state_dict is not None:
        enc_state = {k.replace("encoder.", ""): v for k, v in p1_state_dict.items()
                     if k.startswith("encoder.")}
        backbone.encoder.load_state_dict(enc_state)
        print("  Loaded Phase 1 encoder weights")
    # Freeze encoder
    _freeze(backbone.encoder)
    print("  Encoder frozen — finetuning hypergraph transformer layers + head")

    trainable_params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.BCEWithLogitsLoss()
    get_proxies = _make_proxy_fn(backbone, denoiser, sched)

    print(f"Total params: {sum(p.numel() for p in backbone.parameters()):,}  "
          f"Trainable: {sum(p.numel() for p in trainable_params):,}  "
          f"Warmup: {warmup_epochs} epochs")

    best_val = 0.0
    history  = dict(train_ap=[], val_ap=[], test_ap=[], train_loss=[], val_loss=[])

    for epoch in range(epochs):
        backbone.train()
        use_proxies = (epoch >= warmup_epochs)
        metric, ep_loss = MultilabelAveragePrecision(num_labels=OUTPUT_DIM).to(DEVICE), []

        for batch in tqdm(train_loader,
                          desc=f"P4.5 {epoch+1:03d} {'[hyper]' if use_proxies else '[warmup]'}",
                          leave=False):
            batch = batch.to(DEVICE)
            if use_proxies:
                backbone.eval()
                with torch.no_grad():
                    proxies = get_proxies(batch)
                backbone.train()
                logits, _ = backbone(batch, proxy_embeddings=proxies)
            else:
                logits, _ = backbone(batch)

            loss = loss_fn(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            optimizer.step()
            metric.update(torch.sigmoid(logits), batch.y.long())
            ep_loss.append(loss.item())

        lr_sched.step()
        train_ap   = float(metric.compute())
        train_loss = float(np.mean(ep_loss))
        pfn        = get_proxies if use_proxies else None
        val_ap, val_loss = evaluate(backbone, val_loader, proxy_fn=pfn)

        history["train_ap"].append(train_ap)
        history["val_ap"].append(val_ap)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_ap > best_val:
            best_val = val_ap
            torch.save(backbone.state_dict(), f"{SAVE_DIR}/phase4_5_best_backbone.pt")

        if (epoch + 1) % 10 == 0:
            test_ap, _ = evaluate(backbone, test_loader, proxy_fn=pfn)
            history["test_ap"].append(test_ap)
            print(f"Ep {epoch+1:3d} {'[hyper]' if use_proxies else '[warmup]'} "
                  f"| Train {train_ap:.4f} {train_loss:.4f} "
                  f"| Val {val_ap:.4f} {val_loss:.4f} | Test {test_ap:.4f}")

    backbone.load_state_dict(torch.load(f"{SAVE_DIR}/phase4_5_best_backbone.pt"))
    _unfreeze(denoiser)
    pfn_final = _make_proxy_fn(backbone, denoiser, sched)
    final_test_ap, _ = evaluate(backbone, test_loader, proxy_fn=pfn_final)
    print(f"\nPhase 4.5 done. Best Val AP: {best_val:.4f}  Test AP: {final_test_ap:.4f}")
    return backbone, history


# ================================================================
# PHASE 5 — GNN BACKBONE + PROXY NODES AS EXPLICIT GRAPH NODES
# ================================================================
#
# Core difference from Phase 4:
#   Phase 4 — proxy tokens are concatenated to the dense (padded) sequence
#              for a Transformer; attention is global.
#   Phase 5 — proxy nodes are appended to the PyG graph as real nodes,
#              with explicit bidirectional edges to every original node in
#              their graph. A GNN then propagates messages along these edges.
#              No padding, no masking — GNN handles variable sizes natively.
#
# Edge augmentation is fully vectorised (no Python loop over batch items).
# ================================================================

# ----------------------------------------------------------------
# 5a. GNN building blocks
# ----------------------------------------------------------------

class GINLayer(MessagePassing):
    """
    GIN-style layer: h_v = MLP( h_v + sum_{u ∈ N(v)} h_u )
    Uses 'add' aggregation for maximum expressive power (Xu et al. 2019).
    Edge features are projected and added to source node features before
    aggregation, so bond information is not discarded.
    """
    def __init__(self, hidden_dim=HIDDEN_DIM, dropout=0.1):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # Project edge features to node space so they can be added to messages.
        # Proxy-to-proxy and proxy-to-original edges have no bond features,
        # so we use a learned zero-embedding for those.
        self.edge_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.norm         = nn.LayerNorm(hidden_dim)
        self.dropout      = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        """
        x          : [total_N, d]
        edge_index : [2, E]
        edge_attr  : [E, d] or None  (None for proxy edges)
        """
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             size=(x.size(0), x.size(0)))
        out = self.mlp(self.norm(x + out))
        return self.dropout(out)

    def message(self, x_j, edge_attr):
        """x_j : source node features, edge_attr : edge features or None."""
        if edge_attr is not None:
            return x_j + self.edge_proj(edge_attr)
        return x_j


class ProxyGNN(nn.Module):
    """
    GNN backbone that natively handles proxy nodes added to the PyG graph.

    Forward signature is compatible with GraphTransformer.forward() so the
    same evaluate() helper works for both.
    The `proxy_embeddings` argument is accepted but unused — proxy nodes
    are injected at the graph level via augment_batch_with_proxies().
    The actual proxy generation happens inside phase5_train_with_proxies().
    """
    def __init__(self, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM,
                 output_dim=OUTPUT_DIM, dropout=0.1):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.node_encoder = NodeEncoder(hidden_dim)
        # Bond encoder for original edges; proxy edges have no bond features
        self.bond_encoder = BondEncoder(hidden_dim)
        self.layers       = nn.ModuleList([
            GINLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def encode_nodes(self, batch):
        return self.node_encoder(batch.x, batch.edge_index, batch.edge_attr)

    def forward(self, batch, proxy_embeddings=None):
        """
        When proxy_embeddings is None  → standard GNN on original graph.
        When proxy_embeddings is given → augment graph, then run GNN.
        proxy_embeddings : [B, M, d]
        """
        x_orig = self.encode_nodes(batch)         # [total_N, d]
        edge_attr_enc = self.bond_encoder(batch.edge_attr)  # [E, d]

        if proxy_embeddings is not None:
            x, edge_index, edge_attr, batch_map, orig_mask = \
                augment_batch_with_proxies(
                    batch, proxy_embeddings, x_orig, edge_attr_enc
                )
        else:
            x, edge_index, edge_attr = x_orig, batch.edge_index, edge_attr_enc
            batch_map  = batch.batch
            orig_mask  = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Pool only original nodes for graph-level prediction
        node_emb = x[orig_mask]
        orig_batch_map = batch_map[orig_mask]
        pooled   = global_mean_pool(node_emb, orig_batch_map)
        return self.head(pooled), node_emb


# ----------------------------------------------------------------
# 5b. Graph augmentation — fully vectorised, no Python loop over batch
# ----------------------------------------------------------------

def augment_batch_with_proxies(batch, proxy_embeddings, x_orig, edge_attr_enc):
    """
    Appends M proxy nodes per graph to a PyG batch and adds bidirectional
    edges between every proxy node and every original node in its graph.

    Parameters
    ----------
    batch            : PyG Batch
    proxy_embeddings : [B, M, d]  — one set of M proxies per graph
    x_orig           : [total_N, d] — already-encoded node features
    edge_attr_enc    : [E, d]  — already-encoded bond features for original edges

    Returns
    -------
    aug_x         : [total_N + B*M, d]
    aug_edge_index: [2, E + 2*total_N*M]  (original + bidirectional proxy edges)
    aug_edge_attr : [E + 2*total_N*M, d]  (original bond feats + zeros for proxy edges)
    aug_batch_map : [total_N + B*M]
    orig_mask     : [total_N + B*M] bool  — True = original node
    """
    device   = x_orig.device
    B, M, d  = proxy_embeddings.shape
    total_N  = x_orig.shape[0]

    # ── Node features & batch map ────────────────────────────────
    proxy_flat      = proxy_embeddings.view(B * M, d)         # [B*M, d]
    proxy_batch_map = torch.arange(B, device=device).repeat_interleave(M)  # [B*M]
    aug_x           = torch.cat([x_orig, proxy_flat], dim=0)  # [total_N+B*M, d]
    aug_batch_map   = torch.cat([batch.batch, proxy_batch_map], dim=0)

    orig_mask = torch.zeros(total_N + B * M, dtype=torch.bool, device=device)
    orig_mask[:total_N] = True

    # ── Proxy edge construction (vectorised) ─────────────────────
    # For each original node i in graph g, it connects to all M proxies of graph g.
    # Proxy node for graph g, proxy index m  →  global index = total_N + g*M + m

    orig_idx   = torch.arange(total_N, device=device)           # [total_N]
    orig_graph = batch.batch                                     # [total_N]  graph id

    # Repeat each original node M times: [total_N * M]
    src_o = orig_idx.repeat_interleave(M)

    # For each original node, compute the global indices of its M proxies
    proxy_start = total_N + orig_graph * M                       # [total_N]
    proxy_start_rep = proxy_start.repeat_interleave(M)           # [total_N * M]
    proxy_offset    = torch.arange(M, device=device).repeat(total_N)  # [total_N * M]
    dst_p = proxy_start_rep + proxy_offset                       # [total_N * M]

    # Bidirectional: orig→proxy and proxy→orig
    new_src = torch.cat([src_o, dst_p], dim=0)                  # [2*total_N*M]
    new_dst = torch.cat([dst_p, src_o], dim=0)
    new_edges = torch.stack([new_src, new_dst], dim=0)           # [2, 2*total_N*M]

    aug_edge_index = torch.cat([batch.edge_index, new_edges], dim=1)

    # ── Edge attributes ──────────────────────────────────────────
    # Proxy edges have no bond information → use zero vectors
    n_proxy_edges  = 2 * total_N * M
    null_edge_attr = torch.zeros(n_proxy_edges, d, device=device)
    aug_edge_attr  = torch.cat([edge_attr_enc, null_edge_attr], dim=0)

    return aug_x, aug_edge_index, aug_edge_attr, aug_batch_map, orig_mask


# ----------------------------------------------------------------
# 5c. Evaluate helper for GNN (same signature as the GT version)
# ----------------------------------------------------------------

def evaluate_gnn(model, loader, proxy_fn=None):
    """Identical to evaluate() but proxy_fn receives batch and returns [B,M,d]."""
    loss_fn = nn.BCEWithLogitsLoss()
    metric  = MultilabelAveragePrecision(num_labels=OUTPUT_DIM).to(DEVICE)
    losses  = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch   = batch.to(DEVICE)
            proxies = proxy_fn(batch) if proxy_fn else None
            logits, _ = model(batch, proxies)
            loss = loss_fn(logits, batch.y)
            metric.update(torch.sigmoid(logits), batch.y.long())
            losses.append(loss.item())
    return float(metric.compute()), float(np.mean(losses))


# ----------------------------------------------------------------
# 5d. Training loop
# ----------------------------------------------------------------

def phase5_train_gnn_with_proxies(denoiser, sched,
                                   train_loader, val_loader, test_loader,
                                   p1_state_dict=None,
                                   epochs=500, lr=3e-4, weight_decay=1e-5,
                                   warmup_epochs=50):
    """
    Trains a ProxyGNN, initializing the node encoder from Phase 1 weights.

    During warmup: standard GNN on original graph (no proxies).
    After warmup : diffusion generates M proxy nodes which are inserted as
                   real graph nodes with all-to-all edges to original nodes.

    The denoiser is frozen throughout.  Gradients only flow through the GNN.

    Key design choices
    ------------------
    * Proxy generation uses the GNN's encode_nodes() as conditioning signal
      (same as Phase 4 used encode_dense()).  For Phase 5 we don't have a
      dense representation, so we pool original node embeddings per graph
      to form a sequence of length N for the denoiser cross-attention.
    * Null bond features (zeros) are used for proxy edges. The GINLayer's
      edge_proj maps these to zero, so proxy messages are purely the source
      node's current embedding — which is exactly what we want (the proxy
      acts as an information relay, not a bond type).
    * lr=3e-4 (higher than Phase 4's 3e-5) because GNNs converge faster
      than Transformers on graph tasks and benefit from larger initial steps.
    """
    print("\n" + "=" * 60)
    print("PHASE 5: GNN Backbone + Proxy Nodes as Explicit Graph Nodes")
    print("=" * 60)

    _freeze(denoiser)
    denoiser.eval()

    backbone = ProxyGNN().to(DEVICE)
    # Initialize node encoder from Phase 1 pretrained weights
    if p1_state_dict is not None:
        enc_state = {k.replace("encoder.", ""): v for k, v in p1_state_dict.items()
                     if k.startswith("encoder.")}
        backbone.node_encoder.load_state_dict(enc_state)
        print("  Loaded Phase 1 encoder weights into Phase 5 GNN node_encoder")
    # Freeze encoder — only finetune GNN layers + head
    _freeze(backbone.node_encoder)
    _freeze(backbone.bond_encoder)
    print("  Encoder (node + bond) frozen — finetuning GNN layers + head")

    trainable_params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.BCEWithLogitsLoss()

    print(f"GNN total params: {sum(p.numel() for p in backbone.parameters()):,}  "
          f"Trainable: {sum(p.numel() for p in trainable_params):,}  "
          f"Warmup: {warmup_epochs} epochs")

    @torch.no_grad()
    def get_proxies_gnn(batch):
        """
        Conditioning signal for the denoiser: use the GNN's node encoder
        output packed into a dense [B, max_N, d] tensor (same format the
        denoiser expects from Phase 4).
        """
        backbone.eval()
        h = backbone.encode_nodes(batch)                 # [total_N, d]
        dense_x, dense_mask = to_dense_batch(h, batch.batch)  # [B, max_N, d]
        return sched.ddim_sample(denoiser, dense_x, dense_mask,
                                 num_proxy=NUM_PROXY, hidden_dim=HIDDEN_DIM)

    best_val = 0.0
    history  = dict(train_ap=[], val_ap=[], test_ap=[], train_loss=[], val_loss=[])

    for epoch in range(epochs):
        backbone.train()
        use_proxies = (epoch >= warmup_epochs)
        metric, ep_loss = MultilabelAveragePrecision(num_labels=OUTPUT_DIM).to(DEVICE), []

        for batch in tqdm(train_loader,
                          desc=f"P5 {epoch+1:03d} {'[proxy]' if use_proxies else '[warmup]'}",
                          leave=False):
            batch = batch.to(DEVICE)

            if use_proxies:
                backbone.eval()
                with torch.no_grad():
                    proxies = get_proxies_gnn(batch)    # [B, M, d]
                backbone.train()
                logits, _ = backbone(batch, proxy_embeddings=proxies)
            else:
                logits, _ = backbone(batch)

            loss = loss_fn(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
            optimizer.step()
            metric.update(torch.sigmoid(logits), batch.y.long())
            ep_loss.append(loss.item())

        lr_sched.step()
        train_ap   = float(metric.compute())
        train_loss = float(np.mean(ep_loss))
        pfn        = get_proxies_gnn if use_proxies else None
        val_ap, val_loss = evaluate_gnn(backbone, val_loader, proxy_fn=pfn)

        history["train_ap"].append(train_ap)
        history["val_ap"].append(val_ap)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_ap > best_val:
            best_val = val_ap
            torch.save(backbone.state_dict(), f"{SAVE_DIR}/phase5_best_gnn.pt")

        if (epoch + 1) % 10 == 0:
            test_ap, _ = evaluate_gnn(backbone, test_loader, proxy_fn=pfn)
            history["test_ap"].append(test_ap)
            print(f"Ep {epoch+1:3d} {'[proxy]' if use_proxies else '[warmup]'} "
                  f"| Train {train_ap:.4f} {train_loss:.4f} "
                  f"| Val {val_ap:.4f} {val_loss:.4f} | Test {test_ap:.4f}")

    backbone.load_state_dict(torch.load(f"{SAVE_DIR}/phase5_best_gnn.pt"))
    _unfreeze(denoiser)
    final_test_ap, _ = evaluate_gnn(backbone, test_loader, proxy_fn=get_proxies_gnn)
    print(f"\nPhase 5 done. Best Val AP: {best_val:.4f}  Test AP: {final_test_ap:.4f}")
    return backbone, history


# ================================================================
# ABLATION HELPERS
# ================================================================

def run_ablation_num_proxy_gt(backbone, denoiser, sched, val_loader,
                               proxy_counts=(1, 4, 8, 16, 32, 64, 128, 256)):
    """Ablation for Graph Transformer (Phase 4) backbone."""
    print("\n[Ablation GT] Number of proxy nodes M:")
    results = {}
    for M in proxy_counts:
        # Temporarily override NUM_PROXY inside the closure via a wrapper
        @torch.no_grad()
        def pfn_m(batch, _M=M):
            backbone.eval()
            dense_x, dense_mask = backbone.encode_dense(batch)
            return sched.ddim_sample(denoiser, dense_x, dense_mask,
                                     num_proxy=_M, hidden_dim=HIDDEN_DIM)
        val_ap, _ = evaluate(backbone, val_loader, proxy_fn=pfn_m)
        results[M] = val_ap
        print(f"  M={M:4d}  Val AP: {val_ap:.4f}")
    return results


def run_ablation_num_proxy_gnn(backbone, denoiser, sched, val_loader,
                                proxy_counts=(1, 4, 8, 16, 32, 64, 128, 256)):
    """Ablation for GNN (Phase 5) backbone."""
    print("\n[Ablation GNN] Number of proxy nodes M:")
    results = {}
    for M in proxy_counts:
        @torch.no_grad()
        def pfn_m(batch, _M=M):
            backbone.eval()
            h = backbone.encode_nodes(batch)
            dense_x, dense_mask = to_dense_batch(h, batch.batch)
            return sched.ddim_sample(denoiser, dense_x, dense_mask,
                                     num_proxy=_M, hidden_dim=HIDDEN_DIM)
        val_ap, _ = evaluate_gnn(backbone, val_loader, proxy_fn=pfn_m)
        results[M] = val_ap
        print(f"  M={M:4d}  Val AP: {val_ap:.4f}")
    return results


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    train_loader, val_loader, test_loader, train_ds, val_ds, test_ds = get_loaders()

    # ── Phase 1 ──────────────────────────────────────────────────
    P1_CKPT = f"{SAVE_DIR}/phase1_best.pt"
    if os.path.exists(P1_CKPT):
        print(f"\nLoading Phase 1 backbone from {P1_CKPT}")
        p1_backbone = GraphTransformer().to(DEVICE)
        p1_backbone.load_state_dict(torch.load(P1_CKPT, map_location=DEVICE))
        p1_best_val = None
    else:
        p1_backbone, p1_hist = phase1_train(epochs=500, lr=3e-5, patience=50)
        p1_best_val = max(p1_hist["val_ap"])

    p1_state_dict = p1_backbone.state_dict()

    # ── Phase 2 ──────────────────────────────────────────────────
    PROXY_PATH = f"{SAVE_DIR}/proxy_pairs.pkl"
    if os.path.exists(PROXY_PATH):
        print(f"\nLoading proxy pairs from {PROXY_PATH}")
        with open(PROXY_PATH, "rb") as f:
            proxy_pairs = pickle.load(f)
        sample_improved = None
    else:
        proxy_pairs, sample_improved = phase2_optimize_proxies(
            p1_backbone, train_ds, subset_fraction=1.0,
            num_proxy=NUM_PROXY, max_steps=1000, proxy_lr=1e-3, num_repeats=2,
        )

    # ── Phase 2.5 ────────────────────────────────────────────────
    oracle_ap, base_ap, concept_valid = eval_oracle_proxies(
        p1_backbone, train_ds, proxy_pairs, batch_size=BATCH_SIZE
    )
    if not concept_valid:
        print("  WARNING: Oracle gain below 5% threshold. Continuing anyway.")

    # ── Phase 3 ──────────────────────────────────────────────────
    DIFF_CKPT = f"{SAVE_DIR}/phase3_diffusion_best.pt"
    if os.path.exists(DIFF_CKPT):
        print(f"\nLoading denoiser from {DIFF_CKPT}")
        denoiser   = ProxyDenoiser().to(DEVICE)
        denoiser.load_state_dict(torch.load(DIFF_CKPT, map_location=DEVICE))
        ddpm_sched = DDPMScheduler(T=T_DIFF).to(DEVICE)
    else:
        denoiser, ddpm_sched, p3_hist = phase3_train_diffusion(
            proxy_pairs, epochs=500, lr=1e-4, batch_size=64,
        )

    # ── Phase 4 (end-to-end finetuning with proxies) ────────────
    P4_CKPT = f"{SAVE_DIR}/phase4_best_backbone.pt"
    P4_DEN_CKPT = f"{SAVE_DIR}/phase4_best_denoiser.pt"
    if os.path.exists(P4_CKPT):
        print(f"\nLoading Phase 4 backbone from {P4_CKPT}")
        p4_backbone = GraphTransformer().to(DEVICE)
        p4_backbone.load_state_dict(torch.load(P4_CKPT, map_location=DEVICE))
        if os.path.exists(P4_DEN_CKPT):
            denoiser.load_state_dict(torch.load(P4_DEN_CKPT, map_location=DEVICE))
            print(f"  Also loaded Phase 4 finetuned denoiser from {P4_DEN_CKPT}")
    else:
        p4_backbone, denoiser, p4_hist = phase4_train_with_proxies(
            denoiser, ddpm_sched, train_loader, val_loader, test_loader,
            p1_state_dict=p1_state_dict,
            epochs=500, lr=3e-5, weight_decay=1e-5, warmup_epochs=50,
        )

    # ── Phase 4.5 (hyperedge-routed attention, optional) ─────────
    p4_5_backbone = None
    P4_5_CKPT = f"{SAVE_DIR}/phase4_5_best_backbone.pt"
    if os.path.exists(P4_5_CKPT):
        print(f"\nLoading Phase 4.5 backbone from {P4_5_CKPT}")
        p4_5_backbone = HypergraphGraphTransformer().to(DEVICE)
        p4_5_backbone.load_state_dict(torch.load(P4_5_CKPT, map_location=DEVICE))
    else:
        p4_5_backbone, p4_5_hist = phase4_5_train_hypergraph(
            denoiser, ddpm_sched, train_loader, val_loader, test_loader,
            p1_state_dict=p1_state_dict,
            epochs=500, lr=3e-5, weight_decay=1e-5, warmup_epochs=50,
        )

    # ── Phase 5 ──────────────────────────────────────────────────
    P5_CKPT = f"{SAVE_DIR}/phase5_best_gnn.pt"
    if os.path.exists(P5_CKPT):
        print(f"\nLoading Phase 5 GNN from {P5_CKPT}")
        p5_backbone = ProxyGNN().to(DEVICE)
        p5_backbone.load_state_dict(torch.load(P5_CKPT, map_location=DEVICE))
    else:
        p5_backbone, p5_hist = phase5_train_gnn_with_proxies(
            denoiser, ddpm_sched, train_loader, val_loader, test_loader,
            p1_state_dict=p1_state_dict,
            epochs=500, lr=3e-4, weight_decay=1e-5, warmup_epochs=50,
        )

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    if p1_best_val is not None:
        print(f"Phase 1  (vanilla GT)         — Best Val AP: {p1_best_val:.4f}")
    print(f"Phase 2.5 oracle delta         — ΔAP: {oracle_ap - base_ap:+.4f}")
    p4_val_ap, _ = evaluate(p4_backbone, val_loader,
                             proxy_fn=_make_proxy_fn(p4_backbone, denoiser, ddpm_sched))
    print(f"Phase 4  (GT + diff proxies)   — Val AP: {p4_val_ap:.4f}")
    if p4_5_backbone is not None:
        p4_5_val_ap, _ = evaluate(
            p4_5_backbone, val_loader,
            proxy_fn=_make_proxy_fn(p4_5_backbone, denoiser, ddpm_sched))
        print(f"Phase 4.5 (hyperedge routing)  — Val AP: {p4_5_val_ap:.4f}")
    @torch.no_grad()
    def _p5_proxy_fn(batch):
        p5_backbone.eval()
        h = p5_backbone.encode_nodes(batch)
        dense_x, dense_mask = to_dense_batch(h, batch.batch)
        return ddpm_sched.ddim_sample(denoiser, dense_x, dense_mask,
                                       num_proxy=NUM_PROXY, hidden_dim=HIDDEN_DIM)
    p5_val_ap, _ = evaluate_gnn(p5_backbone, val_loader, proxy_fn=_p5_proxy_fn)
    print(f"Phase 5  (GNN + diff proxies)  — Val AP: {p5_val_ap:.4f}")

    # Ablations
    run_ablation_num_proxy_gt(p4_backbone, denoiser, ddpm_sched, val_loader)
    run_ablation_num_proxy_gnn(p5_backbone, denoiser, ddpm_sched, val_loader)


"""
When we use the trained denoiser model to dynamically generate embeddings, 
generate for train data when doing full M+N retraining,
add edge prediction layer -> pass to GNN -> Get task loss -> optimize main GNN and also optimize the edge predictor
-> Gives a better graph that incorporates both structural and global features
-> Graph trans

"""