import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import accuracy_score, f1_score
import argparse
import random
import numpy as np

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GATv2Block(nn.Module):
    def __init__(self, in_dim, out_dim, heads, dropout):
        super().__init__()

        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            dropout=dropout,
            concat=True,        # heads are concatenated
            add_self_loops=True
        )

        self.norm = nn.LayerNorm(out_dim * heads)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim * heads, out_dim * heads),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_index):
        x = self.norm(self.gat(x, edge_index))
        return self.ffn(x)


# GATv2 Model
class GATv2Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 layers, heads, dropout, use_unet):
        super().__init__()

        self.use_unet = use_unet
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        if not use_unet:
            self.layers = nn.ModuleList([
                GATv2Block(hidden_dim, hidden_dim // heads, heads, dropout)
                for _ in range(layers)
            ])
        else:
            dims = [hidden_dim // (2 ** i) for i in range(layers)]
            dims = [max(d, heads) for d in dims]

            self.enc = nn.ModuleList([
                GATv2Block(
                    dims[i],
                    dims[i] // heads,
                    heads,
                    dropout
                ) for i in range(layers)
            ])

            self.dec = nn.ModuleList([
                GATv2Block(
                    dims[i],
                    dims[i] // heads,
                    heads,
                    dropout
                ) for i in reversed(range(layers))
            ])

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.input_proj(x)

        if not self.use_unet:
            for layer in self.layers:
                x = x + layer(x, edge_index)
            return self.output_proj(x)

        skips = []
        for layer in self.enc:
            x = layer(x, edge_index)
            skips.append(x)

        for layer, skip in zip(self.dec, reversed(skips)):
            x = layer(x + skip, edge_index)

        return self.output_proj(x)


# Training & Eval
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
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root="./data", name="Cora")
    data = dataset[0].to(device)

    model = GATv2Model(
        in_dim=data.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=args.hidden_dim if args.use_unet else dataset.num_classes,
        layers=args.num_layers,
        heads=args.num_heads,
        dropout=args.dropout,
        use_unet=args.use_unet
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
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--use_unet", action="store_true")

    args = parser.parse_args()
    main(args)

