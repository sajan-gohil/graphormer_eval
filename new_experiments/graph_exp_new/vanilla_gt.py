import os
import torch_geometric
import torch
import torch.nn.functional as F
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool, global_mean_pool
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm
import numpy as np

from models import build_node_encoder

# ── Hardcoded configuration ──────────────────────────────────────────────────
HIDDEN_DIM        = 320
NUM_HEADS         = 40
FFN_RATIO         = 4
NUM_LAYERS        = 1
DROPOUT           = 0.2
GRAPH_POOL        = "sum"       # "sum" or "mean"
LR                = 0.001
WEIGHT_DECAY      = 0.0003
BATCH_SIZE        = 128
MAX_EPOCHS        = 200
PATIENCE          = 40
REDUCE_LR_PATIENCE = 10
SAVE_DIR          = "checkpoints_vanilla_func_best"
OUTPUT_DIM        = 10
DATASET_NAME      = "Peptides-func"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(SAVE_DIR, exist_ok=True)

train_dataset = LRGBDataset(root="./data", name=DATASET_NAME, split="train")
val_dataset   = LRGBDataset(root="./data", name=DATASET_NAME, split="val")
test_dataset  = LRGBDataset(root="./data", name=DATASET_NAME, split="test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)


class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, ffn_ratio=FFN_RATIO):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.wq = torch.nn.Linear(hidden_dim, hidden_dim)
        self.wk = torch.nn.Linear(hidden_dim, hidden_dim)
        self.wv = torch.nn.Linear(hidden_dim, hidden_dim)
        self.interm_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
        )
        self.attn_drop = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_dim)
        self.drop1 = torch.nn.Dropout(dropout)
        self.drop2 = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        batch_size, N, d = x.shape
        residual = x
        x = self.layer_norm(x)
        q = self.wq(x).reshape(batch_size, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).reshape(batch_size, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).reshape(batch_size, N, self.num_heads, self.head_dim).transpose(1, 2)
        qk = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Padding mask
        if mask is not None:
            key_pad = (~mask).unsqueeze(1).unsqueeze(2)      # (B, 1, 1, N)
            query_pad = (~mask).unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)
            qk = qk.masked_fill(key_pad, float("-inf"))
            qk = qk.masked_fill(query_pad, float("-inf"))

        qk = torch.nn.functional.softmax(qk, dim=-1)
        qk = torch.nan_to_num(qk, nan=0.0)
        qk = self.attn_drop(qk)
        qkv = torch.matmul(qk, v)
        qkv = qkv.transpose(1, 2).reshape(batch_size, N, d)
        qkv = self.interm_proj(qkv)
        qkv = self.drop1(qkv) + residual
        residual2 = qkv
        qkv = self.out_proj(self.layer_norm_2(qkv))
        qkv = self.drop2(qkv) + residual2
        return qkv


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                 hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
                 dropout=DROPOUT, ffn_ratio=FFN_RATIO,
                 graph_pool=GRAPH_POOL, dataset_name=DATASET_NAME):
        super().__init__()
        self.graph_pool = graph_pool

        # Use the same encoder as model_hop_masked_transformer.py
        self.encoder = build_node_encoder(
            hidden_dim=hidden_dim,
            lap_pe_dim=0,
            dataset_name=dataset_name,
        )

        self.layers = torch.nn.ModuleList([
            MultiHeadAttentionLayer(num_heads, hidden_dim, dropout, ffn_ratio)
            for _ in range(num_layers)
        ])
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch):
        lap_pe = getattr(batch, "lap_pe", None)
        x = self.encoder(batch.x, batch.edge_index, batch.edge_attr, lap_pe=lap_pe)
        dense_x, dense_mask = to_dense_batch(x, batch.batch)

        for layer in self.layers:
            dense_x = layer(dense_x, mask=dense_mask)

        node_embeddings = dense_x[dense_mask]

        B = dense_x.shape[0]
        batch_vec = (
            torch.arange(B, device=dense_x.device)
            .unsqueeze(1)
            .expand_as(dense_mask)[dense_mask]
        )
        if self.graph_pool == "mean":
            pooled = global_mean_pool(node_embeddings, batch_vec)
        else:
            pooled = global_add_pool(node_embeddings, batch_vec)

        outputs = self.head(pooled)
        return outputs, node_embeddings


device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiHeadAttention().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=REDUCE_LR_PATIENCE)
loss_function = torch.nn.BCEWithLogitsLoss()
metric = MultilabelAveragePrecision(num_labels=OUTPUT_DIM)

loss_dict = {
    "train_ap": [], "val_ap": [], "test_ap": [],
    "train_loss": [], "val_loss": [], "test_loss": [],
}

best_val_ap = 0.0
epochs_without_improvement = 0

for epoch in range(MAX_EPOCHS):
    print(f"EPOCH ===================: {epoch}")
    model.train()
    train_losses = []

    for batch in tqdm(train_loader, desc="Train set"):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs, node_embeddings = model(batch)
        loss = loss_function(outputs, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        metric.update(torch.sigmoid(outputs), batch.y.long())
        train_losses.append(loss.item())

    loss_dict["train_loss"].append(np.mean(train_losses))
    loss_dict["train_ap"].append(metric.compute().item())
    metric.reset()

    model.eval()
    val_losses = []
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Val set"):
            batch = batch.to(device)
            outputs, node_embeddings = model(batch)
            loss = loss_function(outputs, batch.y)
            metric.update(torch.sigmoid(outputs), batch.y.long())
            val_losses.append(loss.item())
        loss_dict["val_loss"].append(np.mean(val_losses))
        current_val_ap = metric.compute().item()
        loss_dict["val_ap"].append(current_val_ap)
        metric.reset()

    # ReduceLROnPlateau step
    scheduler.step(current_val_ap)

    # Early stopping & checkpoint
    if current_val_ap > best_val_ap:
        best_val_ap = current_val_ap
        epochs_without_improvement = 0
        ckpt_path = os.path.join(SAVE_DIR, "best_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_ap": best_val_ap,
        }, ckpt_path)
        print(f"  ✓ New best val_AP={best_val_ap:.4f}, saved to {ckpt_path}")
    else:
        epochs_without_improvement += 1
        print(f"  No improvement for {epochs_without_improvement}/{PATIENCE} epochs (best={best_val_ap:.4f})")

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

    test_losses = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Test set"):
            batch = batch.to(device)
            outputs, node_embeddings = model(batch)
            loss = loss_function(outputs, batch.y)
            metric.update(torch.sigmoid(outputs), batch.y.long())
            test_losses.append(loss.item())
        loss_dict["test_loss"].append(np.mean(test_losses))
        loss_dict["test_ap"].append(metric.compute().item())
        metric.reset()

    print("EPOCH RESULTS:", {k: float(v[-1]) for k, v in loss_dict.items() if v})

print(f"\nTraining complete. Best val_AP = {best_val_ap:.4f}")
