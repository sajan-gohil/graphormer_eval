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
import sys
import os
os.environ["WANDB_MODE"] = "disabled"
import wandb
wandb.init(mode="disabled")
from tqdm import tqdm
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_diffusion import GraphLatentDiffusion

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class DiffusionConfig:
    """Simple config class for diffusion parameters."""
    def __init__(self, **kwargs):
        # Diffusion settings
        self.diffusion_type = kwargs.get("diffusion_type", "x0")  # "x0", "noise_pred", "ddim"
        self.num_denoising_steps = kwargs.get("num_denoising_steps", 100)
        self.num_denoiser_layers = kwargs.get("num_denoiser_layers", 3)
        self.denoiser_type = kwargs.get("denoiser_type", "mha")  # "mha", "gat", "linear"
        self.reconstruction_scale = kwargs.get("reconstruction_scale", 0.5)
        self.structure_scale = kwargs.get("structure_scale", 0.5)
        self.gnn_only = kwargs.get("gnn_only", False)
        self.detached_denoiser = kwargs.get("detached_denoiser", False)
        self.mask_random_input_prob = kwargs.get("mask_random_input_prob", 0.0)
        self.aug_loss_scale = kwargs.get("aug_loss_scale", 0.0)
        self.log_memory = kwargs.get("log_memory", False)
        self.experiment_dir = kwargs.get("experiment_dir", "./experiments")
        
        # Runtime state
        self.current_split = "train"
        self.current_step = 0

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


# Graph Transformer
class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers, heads, dropout, use_diffusion=False, diffusion_config=None):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.use_diffusion = use_diffusion

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, heads, dropout)
            for _ in range(layers)
        ])

        # Initialize diffusion module if enabled
        if use_diffusion and diffusion_config is not None:
            self.diffusion = GraphLatentDiffusion(
                input_dim=hidden_dim,
                latent_dim=hidden_dim,
                num_denoising_steps=diffusion_config.num_denoising_steps,
                config=diffusion_config
            )
            self.diffusion_config = diffusion_config
        else:
            self.diffusion = None
            self.diffusion_config = None

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index)

        # Apply diffusion refinement if enabled
        diffusion_loss = None
        if self.use_diffusion and self.diffusion is not None:
            # Convert flat node features to batched format for diffusion
            # x: [total_nodes, hidden_dim], batch: [total_nodes]
            batch_size = batch.max().item() + 1
            max_nodes = max((batch == i).sum().item() for i in range(batch_size))
            
            # Pad and reshape to [B, N, D] for diffusion
            x_batched = torch.zeros(batch_size, max_nodes, x.size(-1), device=x.device)
            edge_index_list = []
            
            node_offset = 0
            for i in range(batch_size):
                mask = (batch == i)
                num_nodes = mask.sum().item()
                x_batched[i, :num_nodes] = x[mask]
                
                # Extract edge indices for this graph (adjust to local indices)
                graph_nodes = torch.where(mask)[0]
                node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(graph_nodes)}
                
                # Filter edges belonging to this graph
                edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
                graph_edges = edge_index[:, edge_mask]
                
                # Remap to local indices
                local_edges = torch.tensor(
                    [[node_map[e.item()] for e in graph_edges[0]],
                     [node_map[e.item()] for e in graph_edges[1]]],
                    device=x.device, dtype=torch.long
                )
                edge_index_list.append(local_edges)
            
            # Apply diffusion
            denoised_x, diffusion_loss = self.diffusion(x_batched, edge_index_list)
            
            # Unpack back to flat format
            x_refined = torch.zeros_like(x)
            for i in range(batch_size):
                mask = (batch == i)
                num_nodes = mask.sum().item()
                x_refined[mask] = denoised_x[i, :num_nodes]
            
            x = x_refined

        # Graph-level pooling
        x = global_mean_pool(x, batch)
        logits = self.output_proj(x)
        
        if diffusion_loss is not None:
            return logits, diffusion_loss
        return logits


def train_epoch(model, loader, optimizer, device, use_diffusion=False, diffusion_weight=0.1):
    model.train()
    if hasattr(model, 'diffusion_config') and model.diffusion_config is not None:
        model.diffusion_config.current_split = "train"
    total_loss = 0
    total_diffusion_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass with optional diffusion
        output = model(data.x.float(), data.edge_index, data.batch)
        
        if use_diffusion and isinstance(output, tuple):
            out, diff_loss = output
            loss = F.binary_cross_entropy_with_logits(out, data.y)
            # Combine classification loss with diffusion loss
            combined_loss = loss + diffusion_weight * diff_loss
            total_diffusion_loss += diff_loss.item() * data.num_graphs
        else:
            out = output if not isinstance(output, tuple) else output[0]
            loss = F.binary_cross_entropy_with_logits(out, data.y)
            combined_loss = loss
        
        combined_loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
        # Update step counter for diffusion config
        if hasattr(model, 'diffusion_config') and model.diffusion_config is not None:
            model.diffusion_config.current_step += 1

    avg_loss = total_loss / len(loader.dataset)
    avg_diff_loss = total_diffusion_loss / len(loader.dataset) if use_diffusion else 0
    return avg_loss, avg_diff_loss


@torch.no_grad()
def evaluate(model, loader, device, use_diffusion=False):
    model.eval()
    if hasattr(model, 'diffusion_config') and model.diffusion_config is not None:
        model.diffusion_config.current_split = "eval"
    ys, preds = [], []

    for data in loader:
        data = data.to(device)
        output = model(data.x.float(), data.edge_index, data.batch)
        
        # Handle tuple output from diffusion model
        out = output[0] if isinstance(output, tuple) else output
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

    # Setup diffusion config if enabled
    diffusion_config = None
    if args.use_diffusion:
        diffusion_config = DiffusionConfig(
            diffusion_type=args.diffusion_type,
            num_denoising_steps=args.num_denoising_steps,
            num_denoiser_layers=args.num_denoiser_layers,
            denoiser_type=args.denoiser_type,
            reconstruction_scale=args.reconstruction_scale,
            structure_scale=args.structure_scale,
            gnn_only=args.gnn_only,
            mask_random_input_prob=args.mask_random_input_prob,
        )
        print(f"Diffusion enabled: type={args.diffusion_type}, steps={args.num_denoising_steps}, "
              f"denoiser={args.denoiser_type}, rec_scale={args.reconstruction_scale}, "
              f"struct_scale={args.structure_scale}")

    model = GraphTransformer(
        in_dim=train_dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=train_dataset.num_classes,
        layers=args.num_layers,
        heads=args.num_heads,
        dropout=args.dropout,
        use_diffusion=args.use_diffusion,
        diffusion_config=diffusion_config
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val = 0
    best_state = None

    for epoch in tqdm(list(range(1, args.epochs + 1))):
        loss, diff_loss = train_epoch(model, train_loader, optimizer, device, 
                                       use_diffusion=args.use_diffusion,
                                       diffusion_weight=args.diffusion_weight)
        val_f1 = evaluate(model, val_loader, device, use_diffusion=args.use_diffusion)

        if val_f1 > best_val:
            best_val = val_f1
            best_state = model.state_dict()

        # if epoch % 10 == 0 or epoch == 1:
        if args.use_diffusion:
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} | Diff Loss {diff_loss:.4f} | "
                f"Val Micro-F1 {val_f1:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} | "
                f"Val Micro-F1 {val_f1:.4f}"
            )

    model.load_state_dict(best_state)
    test_f1 = evaluate(model, test_loader, device, use_diffusion=args.use_diffusion)
    print(f"\nTest Micro-F1: {test_f1:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    
    # Diffusion arguments
    parser.add_argument("--use_diffusion", action="store_true", help="Enable diffusion refinement")
    parser.add_argument("--diffusion_type", type=str, default="x0", 
                        choices=["x0", "noise_pred", "delta"], help="Type of diffusion")
    parser.add_argument("--num_denoising_steps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--num_denoiser_layers", type=int, default=3, help="Number of denoiser layers")
    parser.add_argument("--denoiser_type", type=str, default="mha", 
                        choices=["mha", "gat", "linear"], help="Denoiser architecture type")
    parser.add_argument("--reconstruction_scale", type=float, default=0.5, 
                        help="Weight for reconstruction loss")
    parser.add_argument("--structure_scale", type=float, default=0.5, 
                        help="Weight for structure preservation loss")
    parser.add_argument("--diffusion_weight", type=float, default=0.1, 
                        help="Weight for diffusion loss in total loss")
    parser.add_argument("--gnn_only", action="store_true", 
                        help="Use GNN-only mode (no noise addition)")
    parser.add_argument("--mask_random_input_prob", type=float, default=0.0, 
                        help="Probability of masking input for J-invariant training")
    
    args = parser.parse_args()
    main(args)
