import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops
# from torch_scatter import scatter
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

    def forward(self, x, edge_index):
        N = x.size(0)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=N)
        src, dst = edge_index

        Q = self.W_q(x).view(N, self.num_heads, self.head_dim)
        K = self.W_k(x).view(N, self.num_heads, self.head_dim)
        V = self.W_v(x).view(N, self.num_heads, self.head_dim)
        # Edge-wise attention
        scores = (Q * K).sum(dim=-1) / (self.head_dim ** 0.5)
        # attn = scatter(scores, dst, dim=0, reduce="softmax")
        attn = self.dropout(scores)

        out = V * attn.unsqueeze(-1)
        # out = scatter(out, dst, dim=0, reduce="sum")
        out = out.reshape(N, self.embed_dim)
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

    def forward(self, x, edge_index):
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

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        return self.output_proj(x)


# Training & Evaluation
def train_epoch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)[mask]
    preds = logits.argmax(dim=-1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    return {
        "accuracy": accuracy_score(labels, preds),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


# Driver
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root="./data", name="Cora")
    data = dataset[0].to(device)

    model = GraphTransformer(
        in_dim=data.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=dataset.num_classes,
        layers=args.num_layers,
        heads=args.num_heads,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, data, optimizer)
        val = evaluate(model, data, data.val_mask)

        if val["micro_f1"] > best_val:
            best_val = val["micro_f1"]
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                f"Val Acc {val['accuracy']:.4f} | "
                f"Val Micro-F1 {val['micro_f1']:.4f}"
            )

    model.load_state_dict(best_state)

    print("\nFinal Metrics")
    for name, mask in [
        ("Train", data.train_mask),
        ("Val", data.val_mask),
        ("Test", data.test_mask),
    ]:
        m = evaluate(model, data, mask)
        print(
            f"{name:5s} | Acc {m['accuracy']:.4f} | "
            f"Micro-F1 {m['micro_f1']:.4f} | "
            f"Macro-F1 {m['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    main(args)
