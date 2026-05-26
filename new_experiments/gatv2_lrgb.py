import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import accuracy_score, f1_score
import argparse
import random
import numpy as np


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# GATv2 Model (Variable Depth + UNet)
class GATv2Model(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers,
        num_heads,
        dropout,
        use_unet=False,
    ):
        super().__init__()

        assert num_layers >= 1
        self.dropout = dropout
        self.use_unet = use_unet and num_layers > 1

        # ---- Single-layer case
        if num_layers == 1:
            self.single = GATv2Conv(
                in_channels=in_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
        else:
            self.enc = nn.ModuleList()

            self.enc.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=True,
                )
            )

            for _ in range(num_layers - 1):
                self.enc.append(
                    GATv2Conv(
                        in_channels=hidden_dim * num_heads,
                        out_channels=hidden_dim,
                        heads=num_heads,
                        concat=True,
                        dropout=dropout,
                        add_self_loops=True,
                    )
                )

            if self.use_unet:
                self.dec = nn.ModuleList([
                    GATv2Conv(
                        in_channels=hidden_dim * num_heads,
                        out_channels=hidden_dim,
                        heads=num_heads,
                        concat=True,
                        dropout=dropout,
                        add_self_loops=True,
                    )
                    for _ in range(num_layers - 1)
                ])

        self.graph_proj = nn.Linear(hidden_dim * num_heads, out_dim)

    def forward(self, x, edge_index, batch):
        # ---- Single layer
        if hasattr(self, "single"):
            x = self.single(x, edge_index)
            x = F.elu(x)
        else:
            skips = []
            for layer in self.enc:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.elu(layer(x, edge_index))
                skips.append(x)

            if self.use_unet:
                for layer, skip in zip(self.dec, reversed(skips[:-1])):
                    x = F.dropout(x, p=self.dropout, training=self.training)
                    x = F.elu(layer(x, edge_index))
                    x = x + skip

        # ---- Graph pooling
        x = global_mean_pool(x, batch)
        return self.graph_proj(x)


# Training & Evaluation
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x.float(), data.edge_index, data.batch)
        loss = F.binary_cross_entropy_with_logits(out, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


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

    print(train_dataset[0])

    model = GATv2Model(
        in_dim=train_dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=train_dataset.num_classes,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_unet=args.use_unet,
    ).to(device)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4,
    )

    best_val = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_f1 = evaluate(model, val_loader, device)

        if val_f1 > best_val:
            best_val = val_f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f} | "
                f"Val Micro-F1 {val_f1:.4f}"
            )

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_f1 = evaluate(model, test_loader, device)
    print(f"\nTest Micro-F1: {test_f1:.4f}")



# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--use_unet", action="store_true")

    args = parser.parse_args()
    main(args)

