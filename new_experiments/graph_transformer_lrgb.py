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
        Q = self.W_q(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.W_k(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.W_v(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
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
        # print("Layer input:", x.shape)
        x = x + self.dropout(self.attn(self.norm1(x), edge_index))
        # print("After attention:", x.shape)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        # print("After FFN:", x.shape)
        return x


# Node Embedding VAE (without reconstruction loss)
class NodeEmbeddingVAE(nn.Module):
    """
    Variational Autoencoder for node embeddings.
    Encodes to latent space and decodes to generate refined embeddings.
    No reconstruction loss - trained end-to-end with task loss only.
    """
    def __init__(self, embed_dim, latent_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        
        # Encoder: maps embeddings to latent distribution
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(embed_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim // 2, latent_dim)
        
        # Decoder: generates refined embeddings from latent samples
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent sample to refined embedding."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass: encode, sample, decode."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)


# Node Embedding Generator using VAE
class NodeEmbeddingGenerator(nn.Module):
    """
    Implements EP(Ni | N-Ni) for all nodes in a graph.
    For each node i:
    - Mask out the i'th node embedding
    - Aggregate information from all other nodes
    - Use VAE to predict/generate refined embedding for node i
    """
    def __init__(self, embed_dim, latent_dim, num_context_heads=4, context_dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.vae = NodeEmbeddingVAE(embed_dim, latent_dim)
        
        # Attention-based aggregation for context (all nodes except i)
        # num_context_heads and context_dropout can be tuned for performance
        self.context_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_context_heads,
            dropout=context_dropout,
            batch_first=True
        )
        
    def forward(self, x, batch):
        """
        Generate refined embeddings for all nodes.
        
        This implementation batches all node contexts together and processes them
        in a single forward pass through the VAE to ensure proper gradient propagation.
        
        Args:
            x: Node embeddings [num_nodes, embed_dim]
            batch: Batch assignment for each node [num_nodes]
            
        Returns:
            Refined node embeddings [num_nodes, embed_dim]
        """
        all_aggregated_contexts = []
        node_indices = []  # Track which nodes to process
        
        # Process each graph in the batch separately to collect contexts
        unique_batches = torch.unique(batch)
        for batch_idx in unique_batches:
            # Get nodes for this graph
            mask = batch == batch_idx
            graph_nodes = x[mask]  # [num_nodes_in_graph, embed_dim]
            num_nodes = graph_nodes.size(0)
            
            # For each node i in the graph
            for i in range(num_nodes):
                # Special case: single node graph
                if num_nodes == 1:
                    # Use the node's own embedding as context
                    aggregated = graph_nodes[0]
                else:
                    # Create mask for all nodes except i
                    context_mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)
                    context_mask[i] = False
                    
                    # Get context (all nodes except i)
                    context = graph_nodes[context_mask].unsqueeze(0)  # [1, num_nodes-1, embed_dim]
                    
                    # Aggregate context using attention
                    # Query: mean of context, Key/Value: context nodes
                    query = context.mean(dim=1, keepdim=True)  # [1, 1, embed_dim]
                    aggregated, _ = self.context_attn(query, context, context)  # [1, 1, embed_dim]
                    aggregated = aggregated.squeeze(0).squeeze(0)  # [embed_dim]
                
                all_aggregated_contexts.append(aggregated)
                node_indices.append((batch_idx, i))
        
        # Batch all aggregated contexts and process through VAE in one forward pass
        # This ensures gradients propagate correctly during backpropagation
        if len(all_aggregated_contexts) > 0:
            batched_contexts = torch.stack(all_aggregated_contexts, dim=0)  # [total_nodes, embed_dim]
            batched_refined = self.vae(batched_contexts)  # [total_nodes, embed_dim]
        else:
            batched_refined = torch.empty(0, self.embed_dim, device=x.device)
        
        # Reconstruct the refined embeddings in the original order
        refined_embeddings = batched_refined
        
        return refined_embeddings


# Graph Transformer
class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers, heads, dropout, use_vae_refiner=True):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, heads, dropout)
            for _ in range(layers)
        ])
        
        # VAE-based embedding generator (optional)
        self.use_vae_refiner = use_vae_refiner
        if use_vae_refiner:
            latent_dim = hidden_dim // 2  # Latent dimension is half of embedding dimension
            self.embedding_generator = NodeEmbeddingGenerator(hidden_dim, latent_dim)

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        
        # Apply VAE-based embedding refinement before pooling
        if self.use_vae_refiner:
            x = self.embedding_generator(x, batch)

        # graph-level pooling
        x = global_mean_pool(x, batch)
        return self.output_proj(x)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

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

    model = GraphTransformer(
        in_dim=train_dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=train_dataset.num_classes,
        layers=args.num_layers,
        heads=args.num_heads,
        dropout=args.dropout,
        use_vae_refiner=not args.no_vae_refiner
    ).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_f1 = evaluate(model, val_loader, device)

        if val_f1 > best_val:
            best_val = val_f1
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                f"Val Micro-F1 {val_f1:.4f}"
            )

    model.load_state_dict(best_state)
    test_f1 = evaluate(model, test_loader, device)
    print(f"\nTest Micro-F1: {test_f1:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no_vae_refiner", action='store_true', default=False,
                        help="Disable VAE-based node embedding refinement")
    args = parser.parse_args()
    main(args)
