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
BATCH_SIZE  = 512
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
    """Count number of correctly predicted classes per sample (threshold 0.5)."""
    preds = (torch.sigmoid(logits) > 0.5).float()
    return (preds == labels).sum(dim=1)


def _per_sample_bce_loss(logits, labels):
    """Per-sample BCE loss averaged over labels."""
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)


def optimize_batch_graphs(model, batch, num_proxy=NUM_PROXY, max_steps=1000, lr=1e-3):
    """
    Jointly optimize one proxy tensor per graph in a mini-batch.

    Returns per-sample best proxy and stats, so filtering remains sample-wise.
    """
    B = batch.y.size(0)
    proxy = nn.Parameter(
        torch.randn(B, num_proxy, model.hidden_dim, device=DEVICE) * 0.02
    )
    opt = torch.optim.Adam([proxy], lr=lr)

    with torch.no_grad():
        base_logits = model(batch)[0]
        base_loss = _per_sample_bce_loss(base_logits, batch.y)
        base_correct = _count_correct_classes(base_logits, batch.y)

    best_loss = base_loss.clone()
    best_proxy = proxy.detach().clone()
    best_correct = base_correct.clone()

    for _ in range(max_steps):
        opt.zero_grad()
        logits, _ = model(batch, proxy_embeddings=proxy)
        per_sample_loss = _per_sample_bce_loss(logits, batch.y)
        loss = per_sample_loss.mean()
        loss.backward()
        opt.step()

        with torch.no_grad():
            improved = per_sample_loss < best_loss
            if improved.any():
                best_loss[improved] = per_sample_loss[improved]
                best_proxy[improved] = proxy.detach()[improved]
                cur_correct = _count_correct_classes(logits, batch.y)
                best_correct[improved] = cur_correct[improved]

    return (
        best_proxy.detach(),
        base_loss.detach(),
        best_loss.detach(),
        base_correct.detach(),
        best_correct.detach(),
    )


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
    opt_batch_size = BATCH_SIZE
    print("\n" + "=" * 60)
    print(
        f"PHASE 2: Per-Sample Proxy Optimisation  "
        f"({num_repeats}× per graph, batch={opt_batch_size})"
    )
    print("=" * 60)

    if save_path is None:
        save_path = f"{SAVE_DIR}/proxy_pairs.pkl"

    _freeze(backbone)
    backbone.eval()

    batch_loader = DataLoader(train_dataset, batch_size=opt_batch_size, shuffle=False)
    n_total       = int(len(train_dataset) * subset_fraction)
    proxy_pairs   = []
    improvements  = []
    # Track which samples have found an improved proxy: 1 = found, 0 = not
    sample_improved = [0] * n_total

    for rep in range(num_repeats):
        processed = 0
        pbar = tqdm(total=n_total//opt_batch_size, desc=f"Proxy opt rep {rep+1}/{num_repeats}")
        for batch in batch_loader:
            if processed >= n_total:
                break
            batch = batch.to(DEVICE)
            with torch.no_grad():
                dense_x, dense_mask = backbone.encode_dense(batch)

            best_proxy, base_loss, opt_loss, base_correct, best_correct = \
                optimize_batch_graphs(
                    backbone, batch, num_proxy=num_proxy,
                    max_steps=max_steps, lr=proxy_lr
                )
            improvement = (base_loss - opt_loss) / (base_loss + 1e-8)
            improvements.extend(improvement.cpu().tolist())
            batch_size_cur = batch.y.size(0)
            for j in range(len(batch.y)):
                sample_idx = processed + j
                base_loss_j = float(base_loss[j].item())
                opt_loss_j = float(opt_loss[j].item())
                base_correct_j = int(base_correct[j].item())
                best_correct_j = int(best_correct[j].item())

                # Only keep sample if significant improvement:
                #   opt_loss < 0.005  OR  number of correct classes increased
                is_significant = (opt_loss_j < 0.005) or (best_correct_j > base_correct_j)
                if is_significant:
                    sample_improved[sample_idx] = 1
                    proxy_pairs.append({
                        "encoder_emb": dense_x[j].cpu(),
                        "mask":        dense_mask[j].cpu(),
                        "proxy_emb":   best_proxy[j].cpu(),
                        "base_loss":   base_loss_j,
                        "opt_loss":    opt_loss_j,
                        "sample_idx":  sample_idx,
                    })

            processed += batch_size_cur
            pbar.update(batch_size_cur)

            if processed % 200 == 0:
                n_improved = sum(sample_improved)
                recent = improvements[-200:] if len(improvements) >= 200 else improvements
                print(
                    f"  [rep {rep+1} | {processed}/{n_total}] "
                    f"Avg improvement: {np.mean(recent):.4f}  "
                    f"Improved: {sum(i > 0 for i in recent)}/{len(recent)}  "
                    f"Kept (significant, unique): {n_improved}/{n_total}"
                )

        pbar.close()

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

