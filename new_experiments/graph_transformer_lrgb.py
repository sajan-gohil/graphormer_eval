import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import accuracy_score, f1_score
import argparse
import random
import numpy as np

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def attention_improvement_loss(node_embeddings, denoised_embeddings, edge_index, batch, tau=0.2):
    """
    Compute attention improvement loss to encourage better structural associations.
    
    Args:
        node_embeddings: Initial node embeddings [N, D]
        denoised_embeddings: Embeddings after QKV transformation [N, D]
        edge_index: Edge indices [2, E]
        batch: Batch assignment for each node [N]
        tau: Temperature for sigmoid
    """
    # Normalize embeddings
    if not isinstance(denoised_embeddings, torch.Tensor):
        denoised_embeddings = torch.stack(denoised_embeddings, dim=0).mean(dim=0)

    node_emb_normed = F.normalize(node_embeddings, p=2, dim=-1)
    denoised_emb_normed = F.normalize(denoised_embeddings, p=2, dim=-1)
    
    src, dst = edge_index
    
    # Compute similarity scores for connected nodes
    initial_scores = (node_emb_normed[src] * node_emb_normed[dst]).sum(-1)
    final_scores = (denoised_emb_normed[src] * denoised_emb_normed[dst]).sum(-1)
    
    # Compute per-graph threshold as mean initial score
    edge_batch = batch[src]  # batch assignment for each edge
    num_graphs = batch.max().item() + 1
    
    with torch.no_grad():
        threshold = torch.zeros_like(initial_scores)
        for b in range(num_graphs):
            mask = (edge_batch == b)
            if mask.sum() > 0:
                threshold[mask] = initial_scores[mask].mean()
    
    # Soft recall using sigmoid
    recall_final = torch.sigmoid((final_scores - threshold) / tau)
    
    # Compute per-graph loss
    per_graph_loss = torch.zeros(num_graphs, device=node_embeddings.device)
    per_graph_loss.index_add_(0, edge_batch, -torch.log(recall_final + 1e-8))
    
    # Normalize by edge count per graph
    counts = torch.bincount(edge_batch, minlength=num_graphs).float()
    counts[counts == 0] = 1  # avoid division by zero
    
    return (per_graph_loss / counts).mean()

# Graph Multi-Head Attention
class GraphMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, return_qkv=False):
        N = x.size(0)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        src, dst = edge_index
        Q = self.W_q(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.W_k(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.W_v(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Store QKV embeddings (mean over heads) for attention improvement loss
        qkv_embeddings = (Q + K + V).mean(dim=0)  # [N, head_dim] -> average representation
        qkv_embeddings = qkv_embeddings.view(N, -1)  # Flatten if needed
        
        # Edge-wise attention
        scores = torch.einsum("hnd,hmd->hnm", Q, K) / np.sqrt(self.head_dim)

        # make diagonal zero
        diag_mask = torch.eye(N, device=scores.device).bool()
        scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))
        scores = torch.softmax(scores, dim=-1)
        scores = self.dropout(scores)

        out = torch.einsum("hnm,hmd->hnd", scores, V)
        out = out.transpose(0, 1).contiguous().view(N, self.embed_dim)
        # print("Attention output shape:", out.shape)
        
        if return_qkv:
            return self.out_proj(out), qkv_embeddings
        return self.out_proj(out)


# Transformer Layer
class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = GraphMultiHeadAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, return_qkv=False):
        # print("Layer input:", x.shape)
        if return_qkv:
            attn_out, qkv_emb = self.attn(self.norm1(x), edge_index, return_qkv=True)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x, qkv_emb
        else:
            x = x + self.dropout(self.attn(self.norm1(x), edge_index))
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x


# Graph Transformer
class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers, heads, dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, heads, dropout)
            for _ in range(layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch, return_attn_loss=False):
        x = self.input_proj(x)
        initial_embeddings = x.clone()  # Store initial embeddings for attention loss
        
        all_qkv_embeddings = []
        for layer in self.layers:
            if return_attn_loss:
                x, qkv_emb = layer(x, edge_index, return_qkv=True)
                all_qkv_embeddings.append(qkv_emb)
            else:
                x = layer(x, edge_index)

        # graph-level pooling
        pooled = global_mean_pool(x, batch)
        out = self.output_proj(pooled)
        
        if return_attn_loss:
            # Average QKV embeddings across layers as denoised embeddings
            denoised_embeddings = all_qkv_embeddings  # torch.stack(all_qkv_embeddings, dim=0).mean(dim=0)
            return out, initial_embeddings, denoised_embeddings
        return out


def train_epoch(model, loader, optimizer, device, attn_loss_weight=0.1):
    model.train()
    total_loss = 0
    total_task_loss = 0
    total_attn_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass with attention loss components
        out, initial_emb, denoised_emb = model(
            data.x.float(), data.edge_index, data.batch, return_attn_loss=True
        )
        
        # Task loss (binary cross-entropy)
        task_loss = F.binary_cross_entropy_with_logits(out, data.y)
        
        # Attention improvement loss
        attn_loss = torch.tensor(0)
        if attn_loss_weight > 0:
            attn_loss = attention_improvement_loss(
                initial_emb, denoised_emb, data.edge_index, data.batch
            )
        
        # Combined loss
        loss = task_loss + attn_loss_weight * attn_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_task_loss += task_loss.item() * data.num_graphs
        total_attn_loss += attn_loss.item() * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_task_loss / n, total_attn_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []

    for data in loader:
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.batch)
        preds.append(out.cpu())
        ys.append(data.y.cpu())

    y = torch.cat(ys, dim=0)
    pred = torch.cat(preds, dim=0)
    pred = (pred > 0).int()

    return f1_score(y.numpy(), pred.numpy(), average="micro")


# Driver
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = LRGBDataset(root="./data", name="Peptides-func", split="train")
    val_dataset   = LRGBDataset(root="./data", name="Peptides-func", split="val")
    test_dataset  = LRGBDataset(root="./data", name="Peptides-func", split="test")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(train_dataset[0], flush=True)

    model = GraphTransformer(
        in_dim=train_dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=train_dataset.num_classes,
        layers=args.num_layers,
        heads=args.num_heads,
        dropout=args.dropout
    ).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss, task_loss, attn_loss = train_epoch(
            model, train_loader, optimizer, device, attn_loss_weight=args.attn_loss_weight
        )
        val_f1 = evaluate(model, val_loader, device)

        if val_f1 > best_val:
            best_val = val_f1
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} | Task {task_loss:.4f} | "
                f"Attn {attn_loss:.4f} | Val Micro-F1 {val_f1:.4f}"
            , flush=True)

    model.load_state_dict(best_state)
    test_f1 = evaluate(model, test_loader, device)
    print(f"\nTest Micro-F1: {test_f1:.4f}", flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--attn_loss_weight", type=float, default=0.1,
                        help="Weight for attention improvement loss")
    args = parser.parse_args()
    print(args.__dict__, flush=True)
    main(args)
