import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import LRGBDataset
from sklearn.metrics import average_precision_score
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class DeterministicHyperedges(nn.Module):
    def __init__(self, num_hyperedges, dim, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.hyperedges = nn.Parameter(torch.randn(num_hyperedges, dim))

    def forward(self, node_embeddings, batch):
        """
        node_embeddings: [N, d]
        batch: [N]
        """
        # Graph-level summaries
        graph_emb = global_mean_pool(node_embeddings, batch)  # [B, d]
        pooled = graph_emb.mean(dim=0, keepdim=True)          # [1, d]

        # Deterministic diffusion update
        with torch.no_grad():
            self.hyperedges.data = (
                (1 - self.alpha) * self.hyperedges.data
                + self.alpha * pooled
            )

        return self.hyperedges


class HyperedgeAttentionLayer(nn.Module):
    def __init__(self, dim, num_hyperedges, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hyperedges):
        """
        x: [N, d]
        hyperedges: [M, d]
        """
        Q = self.q(x)                      # [N, d]
        V = self.v(x)                      # [N, d]

        scores = Q @ hyperedges.t() / (x.size(-1) ** 0.5)  # [N, M]
        B = torch.softmax(scores, dim=-1)                  # assignments

        Z = B.t() @ V                                      # [M, d]
        x_out = B @ Z                                      # [N, d]

        return self.dropout(self.out(x_out))

class DeterministicHypergraphTransformer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_tasks,
        num_layers,
        num_hyperedges,
        dropout,
        alpha
    ):
        super().__init__()

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)
        self.hyperedge_gen = DeterministicHyperedges(
            num_hyperedges, hidden_dim, alpha
        )

        self.layers = nn.ModuleList([
            HyperedgeAttentionLayer(hidden_dim, num_hyperedges, dropout)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, num_tasks)

    def forward(self, data):
        x, edge_index, edge_attr, batch = \
            data.x, data.edge_index, data.edge_attr, data.batch
    
        # Atom embedding
        h = self.atom_encoder(x)  # [N, d]
    
        # Bond embedding
        edge_emb = self.bond_encoder(edge_attr)  # [E, d]
    
        # Inject bond information into node representations
        row, col = edge_index
        h = h + scatter(edge_emb, row, dim=0,
                        dim_size=h.size(0),
                        reduce='add')
    
        # Generate hyperedges
        hyperedges = self.hyperedge_gen(h, batch)
    
        # Hyperedge attention layers
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + layer(h, hyperedges))

        # Graph pooling
        g = global_mean_pool(h, batch)

        return self.head(g)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def eval_ap(model, loader, device):
    model.eval()
    ys, preds = [], []

    for data in loader:
        data = data.to(device)
        logits = model(data)
        ys.append(data.y.cpu())
        preds.append(torch.sigmoid(logits).cpu())

    y_true = torch.cat(ys, dim=0).numpy()
    y_pred = torch.cat(preds, dim=0).numpy()

    return average_precision_score(y_true, y_pred, average="macro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_hyperedges", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = LRGBDataset(root="./data", name="Peptides-func", split="train")
    val_dataset   = LRGBDataset(root="./data", name="Peptides-func", split="val")
    test_dataset  = LRGBDataset(root="./data", name="Peptides-func", split="test")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=128)
    test_loader  = DataLoader(test_dataset, batch_size=128)

    model = DeterministicHypergraphTransformer(
        in_dim=train_dataset.num_features,
        hidden_dim=args.hidden_dim,
        num_tasks=10,
        num_layers=args.num_layers,
        num_hyperedges=args.num_hyperedges,
        dropout=args.dropout,
        alpha=args.alpha
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_ap = eval_ap(model, val_loader, device)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss:.4f} | Val AP {val_ap:.4f}")

    test_ap = eval_ap(model, test_loader, device)
    print(f"Final Test AP: {test_ap:.4f}")


if __name__ == "__main__":
    main()

