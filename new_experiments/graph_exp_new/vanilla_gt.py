import torch_geometric
import torch
import torch.nn.functional as F
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, add_self_loops, to_dense_batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm.notebook import tqdm
from torch_geometric.utils import scatter
import numpy as np


train_dataset = LRGBDataset(root="./data", name="Peptides-func", split="train")
val_dataset   = LRGBDataset(root="./data", name="Peptides-func", split="val")
test_dataset  = LRGBDataset(root="./data", name="Peptides-func", split="test")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16)
test_loader  = DataLoader(test_dataset, batch_size=16)


class EmbedNode(torch.nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        # self.embeddings = #[torch.nn.Embedding(max(xvals[i]), 128) for i in range(9)]
        self.atom_encoder = AtomEncoder(output_dim//2)
        self.bond_encoder = BondEncoder(output_dim//2) # Not utilized in Attn directly, but added to node feats
        self.node_proj = torch.nn.Linear(output_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        # Initial Embedding
        h = self.atom_encoder(x)
        row, col = edge_index
        edge_emb = self.bond_encoder(edge_attr)
        edge_aggr = scatter(
            edge_emb, row, dim=0, dim_size=h.size(0), reduce='add'
        )
        h = torch.cat([h, edge_aggr], dim=-1)
        h = self.node_proj(h)
        return h

class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, num_heads=8, hidden_dim=512):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.wq = torch.nn.Linear(hidden_dim, hidden_dim)
        self.wk = torch.nn.Linear(hidden_dim, hidden_dim)
        self.wv = torch.nn.Linear(hidden_dim, hidden_dim)
        self.interm_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        residual = x
        x = self.layer_norm(x)
        q = self.wq(x).reshape(batch_size, num_nodes, self.num_heads, self.hidden_dim//self.num_heads).transpose(-3, -2)
        k = self.wk(x).reshape(batch_size, num_nodes, self.num_heads, self.hidden_dim//self.num_heads).transpose(-3, -2)
        v = self.wv(x).reshape(batch_size, num_nodes, self.num_heads, self.hidden_dim//self.num_heads).transpose(-3, -2)
        qk = torch.matmul(q, k.transpose(-2, -1))/torch.sqrt(torch.tensor(self.hidden_dim))
        qk = torch.nn.functional.softmax(qk, dim=-1)
        qkv = torch.matmul(qk, v)
        qkv = torch.flatten(qkv.transpose(-3, -2), start_dim=2, end_dim=3)
        qkv = self.interm_proj(qkv)
        qkv = qkv + residual
        residual2 = qkv
        qkv = self.out_proj(self.layer_norm_2(qkv))
        qkv = qkv + residual2
        return qk, qkv


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_layers=4, num_heads=8, hidden_dim=512, output_dim=10):
        super().__init__()
        self.encoder = EmbedNode(hidden_dim)
        self.layers = torch.nn.ModuleList([
            MultiHeadAttentionLayer(num_heads, hidden_dim)
            for i in range(num_layers)])
        self.final = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim//2, output_dim),
        )

    def forward(self, batch):
        x, edge_index, edge_attr, batch_map = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        dense_x, dense_mask = torch_geometric.utils.to_dense_batch(x, batch.batch)
        for layer in self.layers:
            attn_scores, dense_x = layer(dense_x)
        node_embeddings = dense_x[dense_mask]
        pooled = torch_geometric.nn.global_mean_pool(node_embeddings, batch.batch)
        outputs = self.final(pooled)
        return outputs, node_embeddings


model = MultiHeadAttention()
model.to("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
loss_function = torch.nn.BCEWithLogitsLoss()
metric = MultilabelAveragePrecision(num_labels=10)


loss_dict = {
    "epoch_train_ap": [],
    "epoch_val_ap": [],
    "epoch_test_ap": [],
    "epoch_train_loss": [],
    "epoch_val_loss": [],
    "epoch_test_loss": [],
}
for epoch in range(500):
    print("EPOCH ===================:", epoch)
    model.train()
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_test_losses = []
    for batch in tqdm(train_loader, desc="Train set"):
        batch.to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer.zero_grad()
        outputs, node_embeddings = model(batch)
        loss = loss_function(outputs, batch.y)
        loss.backward()
        optimizer.step()
        metric.update(torch.nn.functional.sigmoid(outputs), batch.y.long())
        epoch_train_losses.append(loss.item())
    loss_dict["epoch_train_loss"].append(np.mean(epoch_train_losses))
    loss_dict["epoch_train_ap"].append(metric.compute())
    metric.reset()
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Val set"):
            batch.to("cuda" if torch.cuda.is_available() else "cpu")
            outputs, node_embeddings = model(batch)
            loss = loss_function(outputs, batch.y)
            metric.update(torch.nn.functional.sigmoid(outputs), batch.y.long())
            epoch_val_losses.append(loss.item())
        loss_dict["epoch_val_loss"].append(np.mean(epoch_val_losses))
        loss_dict["epoch_val_ap"].append(metric.compute())
        metric.reset()
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Test set"):
            batch.to("cuda" if torch.cuda.is_available() else "cpu")
            outputs, node_embeddings = model(batch)
            loss = loss_function(outputs, batch.y)
            metric.update(torch.nn.functional.sigmoid(outputs), batch.y.long())
            epoch_test_losses.append(loss.item())
        loss_dict["epoch_test_loss"].append(np.mean(epoch_val_losses))
        loss_dict["epoch_test_ap"].append(metric.compute())
        metric.reset()
    print("EPOCH RESULTS:", {k:float(v[-1]) for k,v in loss_dict.items()})