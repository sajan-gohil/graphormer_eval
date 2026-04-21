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
from tqdm import tqdm
from torch_geometric.utils import scatter
import numpy as np


train_dataset = LRGBDataset(root="./data", name="Peptides-func", split="train")
val_dataset   = LRGBDataset(root="./data", name="Peptides-func", split="val")
test_dataset  = LRGBDataset(root="./data", name="Peptides-func", split="test")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64)
test_loader  = DataLoader(test_dataset, batch_size=64)


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

    def forward(self, x, xk=None):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
            if xk:
                xk = xk.unsqueeze(0)
        if xk is None:xk = x
        batch_size = x.shape[0]
        residual = x
        x = self.layer_norm(x)
        q = self.wq(x).reshape(batch_size, x.shape[1], self.num_heads, self.hidden_dim//self.num_heads).transpose(-3, -2)
        k = self.wk(xk).reshape(batch_size, xk.shape[1], self.num_heads, self.hidden_dim//self.num_heads).transpose(-3, -2)
        v = self.wv(xk).reshape(batch_size, xk.shape[1], self.num_heads, self.hidden_dim//self.num_heads).transpose(-3, -2)
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
    def __init__(self, num_layers=8, num_heads=8, hidden_dim=512, output_dim=10):
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
        self.in_ln = torch.nn.LayerNorm(hidden_dim)
        self.interm_proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.out_ln = torch.nn.LayerNorm(hidden_dim)
        self.out_ffn = torch.nn.Linear(hidden_dim, hidden_dim)
        self.proxy_sims = []
        self.batch_proxy_sims = []

    def attend_proxies(self, generator, dense_x, dense_mask):
        proxies = self.in_ln(generator(dense_x))
        proxies_normalized = F.normalize(proxies, p=2, dim=-1)
        internal_proxy_sim = torch.matmul(proxies_normalized, proxies_normalized.transpose(-2, -1))
        self.proxy_sims.append(torch.mean(internal_proxy_sim).detach().item())
        self.batch_proxy_sims.append(torch.mean(internal_proxy_sim))

        # x-> proxies @ proxies -> x
        node_proxy_matmul = torch.matmul(dense_x, proxies.transpose(-2, -1))
        node_proxy_z = torch.nn.functional.softmax(node_proxy_matmul, dim=-1)  # N,M
        proxy_node_z = torch.nn.functional.softmax(node_proxy_matmul.transpose(-2, -1), dim=-1)  # M,N
        out_val = torch.matmul(node_proxy_z, torch.matmul(proxy_node_z, dense_x))
        out_val = self.interm_proj(out_val) + dense_x
        out_val = self.out_ffn(self.out_ln(out_val)) + out_val
        return proxies, out_val

    def attend_proxies(self, generator, dense_x, dense_mask):    
        proxies = generator(dense_x)
        # x-> proxies @ proxies -> x
        node_proxy_matmul = torch.matmul(dense_x, proxies.transpose(-2, -1))
        node_proxy_z = torch.nn.functional.softmax(node_proxy_matmul, dim=-1)  # N,M
        proxy_node_z = torch.nn.functional.softmax(node_proxy_matmul.transpose(-2, -1), dim=-1)  # M,N
        # attn_score = torch.matmul(node_proxy_z, proxy_node_z)  # N,N
        # out_val = torch.matmul(attn_score, dense_x)  # N,d
        out_val = torch.matmul(node_proxy_z, torch.matmul(proxy_node_z, dense_x))
        return proxies, out_val


    def forward(self, batch, generators=None):
        if not generators: generators = {}
        x, edge_index, edge_attr, batch_map = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        dense_x, dense_mask = torch_geometric.utils.to_dense_batch(x, batch.batch)
        if 0 in generators:
            proxies, dense_x = self.attend_proxies(generators[0], dense_x, dense_mask)
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx+1 in generators:
                proxies, dense_x = self.attend_proxies(generators[layer_idx+1], dense_x, dense_mask)
            attn_scores, dense_x = layer(dense_x)

        node_embeddings = dense_x[dense_mask]
        pooled = torch_geometric.nn.global_mean_pool(node_embeddings, batch.batch)
        outputs = self.final(pooled)
        return outputs, node_embeddings


class ScoreBasedGenerator(torch.nn.Module):
    def __init__(self, hidden_dim=512, num_proxies=128):
        super().__init__()
        self.squeeze = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim//2, num_proxies),
        )

    def forward(self, x):
        score_logits = self.squeeze(x)  # B, N, d -> B, N, M
        scores = torch.nn.functional.softmax(score_logits, dim=1)
        proxies = torch.bmm(scores.transpose(-1, -2), x)  # M,N @ N,d
        return proxies


model = MultiHeadAttention()
model.to("cuda" if torch.cuda.is_available() else "cpu")

generators = {idx: ScoreBasedGenerator() for idx in range(8)}
generator_params = []
for k,v in generators.items():
    v.to("cuda" if torch.cuda.is_available() else "cpu")
    generator_params.append({"params": v.parameters(), "lr": 3e-5})

optimizer = torch.optim.AdamW([{"params": model.parameters(), "lr": 3e-5}]+generator_params)
loss_function = torch.nn.BCEWithLogitsLoss()
metric = MultilabelAveragePrecision(num_labels=10)


loss_dict = {
    "train_ap": [],
    "val_ap": [],
    "test_ap": [],
    "train_proxy_sims": [],
    "val_proxy_sims": [],
    "test_proxy_sims": [],
    "train_loss": [],
    "val_loss": [],
    "test_loss": [],

}
for epoch in range(500):
    print("EPOCH ===================:", epoch)
    model.train()
    train_losses = []
    val_losses = []
    test_losses = []
    for batch in tqdm(train_loader, desc="Train set"):
        batch.to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer.zero_grad()
        outputs, node_embeddings = model(batch, generators)
        loss = loss_function(outputs, batch.y)
        loss += torch.mean(torch.stack(model.batch_proxy_sims))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        for gen in generators.values():
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        optimizer.step()
        metric.update(torch.nn.functional.sigmoid(outputs), batch.y.long())
        train_losses.append(loss.item())
        model.batch_proxy_sims = []

    loss_dict["train_loss"].append(np.mean(train_losses))
    loss_dict["train_ap"].append(metric.compute())
    loss_dict["train_proxy_sims"].append(np.mean(model.proxy_sims))
    model.proxy_sims = []
    metric.reset()
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Val set"):
            batch.to("cuda" if torch.cuda.is_available() else "cpu")
            outputs, node_embeddings = model(batch, generators)
            loss = loss_function(outputs, batch.y)
            metric.update(torch.nn.functional.sigmoid(outputs), batch.y.long())
            val_losses.append(loss.item())
        loss_dict["val_loss"].append(np.mean(val_losses))
        loss_dict["val_ap"].append(metric.compute())
        loss_dict["val_proxy_sims"].append(np.mean(model.proxy_sims))
        model.proxy_sims = []
        metric.reset()
        model.batch_proxy_sims = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Test set"):
            batch.to("cuda" if torch.cuda.is_available() else "cpu")
            outputs, node_embeddings = model(batch, generators)
            loss = loss_function(outputs, batch.y)
            metric.update(torch.nn.functional.sigmoid(outputs), batch.y.long())
            test_losses.append(loss.item())
        loss_dict["test_loss"].append(np.mean(test_losses))
        loss_dict["test_ap"].append(metric.compute())
        loss_dict["test_proxy_sims"].append(np.mean(model.proxy_sims))
        model.proxy_sims = []
        model.batch_proxy_sims = []
        metric.reset()
    print("EPOCH RESULTS:", {k:float(v[-1]) for k,v in loss_dict.items()})
