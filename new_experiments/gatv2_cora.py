import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATv2Conv
from torch_geometric.transforms import NormalizeFeatures
from sklearn.metrics import accuracy_score, f1_score
import argparse
import random
import numpy as np


# Reproducibility
# -------------------------
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
                out_channels=out_dim,
                heads=1,
                concat=False,
                dropout=dropout,
                add_self_loops=True,
            )
            return

        # ---- Encoder
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

        # ---- Decoder (same feature dim)
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

        # ---- Final output layer
        self.out_layer = GATv2Conv(
            in_channels=hidden_dim * num_heads,
            out_channels=out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

    def forward(self, x, edge_index):
        # ---- Single layer shortcut
        if hasattr(self, "single"):
            return self.single(x, edge_index)

        skips = []

        # ---- Encoder
        for layer in self.enc:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
            x = F.elu(x)
            skips.append(x)

        # ---- Decoder
        if self.use_unet:
            for layer, skip in zip(self.dec, reversed(skips[:-1])):
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = layer(x, edge_index)
                x = F.elu(x)
                x = x + skip

        # ---- Final layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out_layer(x, edge_index)


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


# -------------------------
# Driver
# -------------------------
def main(args):
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(
        root="./data",
        name="Cora",
        transform=NormalizeFeatures(),
    )
    data = dataset[0].to(device)

    model = GATv2Model(
        in_dim=data.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=dataset.num_classes,
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
        loss = train_epoch(model, data, optimizer)
        val = evaluate(model, data, data.val_mask)

        if val["micro_f1"] > best_val:
            best_val = val["micro_f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f} | "
                f"Val Acc {val['accuracy']:.4f} | "
                f"Val Micro-F1 {val['micro_f1']:.4f}"
            )

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    print("\nFinal Metrics (Best Validation Model)")
    for name, mask in [
        ("Train", data.train_mask),
        ("Val", data.val_mask),
        ("Test", data.test_mask),
    ]:
        m = evaluate(model, data, mask)
        print(
            f"{name:5s} | "
            f"Acc {m['accuracy']:.4f} | "
            f"Micro-F1 {m['micro_f1']:.4f} | "
            f"Macro-F1 {m['macro_f1']:.4f}"
        )


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--use_unet", action="store_true")

    args = parser.parse_args()
    main(args)

