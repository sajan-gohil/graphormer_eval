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
import math

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# ===================== Diffusion Components =====================

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Sinusoidal timestep embeddings for diffusion.
    
    Args:
        timesteps: Tensor of shape [B] with timestep indices
        embedding_dim: Dimension of the embedding
    
    Returns:
        Embeddings of shape [B, embedding_dim]
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class DiffusionScheduler:
    """
    Simple diffusion scheduler for attention matrices.
    Uses cosine schedule for noise levels.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # Cosine schedule
        steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alpha_bar = torch.cos(((steps / num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
        
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x, t, noise=None):
        """Add noise to x at timestep t."""
        device = x.device
        if noise is None:
            noise = torch.randn_like(x)
        
        # Convert tensor index to int
        t_idx = t.item() if isinstance(t, torch.Tensor) else t
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t_idx].to(device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t_idx].to(device)
        
        # Reshape for broadcasting
        while sqrt_alpha.dim() < x.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise
    
    def get_alpha_cumprod(self, t):
        return self.alphas_cumprod[t]


class AttentionDenoiser(nn.Module):
    """
    Simplified denoiser that refines attention scores using node embeddings.
    """
    def __init__(self, embed_dim, num_heads, hidden_dim=128, num_denoise_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Time embedding MLP - output to hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Node projection
        self.node_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Per-pair refinement network
        self.denoise_mlp = nn.Sequential(
            nn.Linear(1 + 2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, noisy_attn, node_embeddings, t):
        """
        Args:
            noisy_attn: Noisy attention [H, N, N]
            node_embeddings: Node embeddings [N, D]
            t: Timestep tensor [1]
        
        Returns:
            Denoised attention [H, N, N]
        """
        H, N, _ = noisy_attn.shape
        
        # Time embedding
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_emb = get_timestep_embedding(t, self.embed_dim)  # [1, D]
        t_emb = self.time_embed(t_emb).squeeze(0)  # [D]
        
        # Project node embeddings
        node_proj = self.node_proj(node_embeddings)  # [N, hidden_dim]
        
        # Expand for pairwise operations
        node_i = node_proj.unsqueeze(1).expand(-1, N, -1)  # [N, N, hidden_dim]
        node_j = node_proj.unsqueeze(0).expand(N, -1, -1)  # [N, N, hidden_dim]
        
        # Broadcast time embedding
        t_emb_expanded = t_emb.unsqueeze(0).unsqueeze(0).expand(N, N, -1)  # [N, N, D]
        
        # Reshape for batch processing all heads at once
        noisy_attn_flat = noisy_attn.reshape(H * N * N, 1)  # [H*N*N, 1]
        
        # Replicate node embeddings for H heads
        node_i_rep = node_i.unsqueeze(0).expand(H, -1, -1, -1).reshape(H * N * N, -1)  # [H*N*N, hidden_dim]
        node_j_rep = node_j.unsqueeze(0).expand(H, -1, -1, -1).reshape(H * N * N, -1)  # [H*N*N, hidden_dim]
        t_emb_rep = t_emb_expanded.unsqueeze(0).expand(H, -1, -1, -1).reshape(H * N * N, -1)  # [H*N*N, D]
        
        # Combine features for all pairs and heads in single batch
        combined = torch.cat([
            noisy_attn_flat,  # [H*N*N, 1]
            node_i_rep,  # [H*N*N, hidden_dim]
            node_j_rep + t_emb_rep  # [H*N*N, hidden_dim]
        ], dim=-1)  # [H*N*N, 1 + 2*hidden_dim]
        
        # Refine attention scores in one pass
        delta = self.denoise_mlp(combined).squeeze(-1)  # [H*N*N]
        delta = delta.reshape(H, N, N)
        
        # Residual connection
        return noisy_attn + delta


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
    
    # Loss: encourage final scores to be higher than threshold
    loss = F.relu(threshold - final_scores).mean()
    
    return loss


def structure_reconstruction_loss(node_embeddings, edge_index, batch):
    """
    Compute structure reconstruction loss by predicting adjacency matrix
    from node embeddings using self dot product + sigmoid.
    
    Vectorized implementation for efficiency.
    
    Args:
        node_embeddings: Node embeddings after first transformer layer [N, D]
        edge_index: Original edge indices [2, E]
        batch: Batch assignment for each node [N]
    
    Returns:
        BCE loss between predicted and original adjacency
    """
    device = node_embeddings.device
    num_nodes = node_embeddings.size(0)
    
    # Normalize embeddings for numerical stability
    node_emb_normed = F.normalize(node_embeddings, p=2, dim=-1)
    
    # Sample negative edges (non-connected node pairs within same graph)
    src, dst = edge_index
    num_pos = edge_index.size(1)
    
    # Positive edge predictions (connected pairs)
    pos_scores = (node_emb_normed[src] * node_emb_normed[dst]).sum(dim=-1)
    pos_pred = torch.sigmoid(pos_scores)
    
    # Sample negative edges: random pairs within the same graph
    # Use ~2x negative samples for balance
    num_neg = min(num_pos * 2, num_nodes * 10)
    
    # Random node indices
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    
    # Keep only pairs in the same graph and not self-loops
    same_graph = batch[neg_src] == batch[neg_dst]
    not_self = neg_src != neg_dst
    valid_neg = same_graph & not_self
    
    neg_src = neg_src[valid_neg]
    neg_dst = neg_dst[valid_neg]
    
    # Filter out actual edges (optional but more accurate)
    # Create edge set for fast lookup
    edge_set = src * num_nodes + dst
    neg_edge_ids = neg_src * num_nodes + neg_dst
    is_not_edge = ~torch.isin(neg_edge_ids, edge_set)
    
    neg_src = neg_src[is_not_edge]
    neg_dst = neg_dst[is_not_edge]
    
    if neg_src.size(0) == 0:
        return torch.tensor(0.0, device=device)
    
    # Negative edge predictions (non-connected pairs)
    neg_scores = (node_emb_normed[neg_src] * node_emb_normed[neg_dst]).sum(dim=-1)
    neg_pred = torch.sigmoid(neg_scores)
    
    # BCE loss: positive edges should be 1, negative should be 0
    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    
    return (pos_loss + neg_loss) / 2


# Graph Multi-Head Attention with Diffusion
class GraphMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, use_diffusion=False, num_diff_steps=10):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_diffusion = use_diffusion
        self.num_diff_steps = num_diff_steps

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Diffusion components
        if use_diffusion:
            self.denoiser = AttentionDenoiser(embed_dim, num_heads)
            self.scheduler = DiffusionScheduler(num_timesteps=1000)

    def compute_base_attention(self, x):
        """Compute the base attention scores from Q and K."""
        N = x.size(0)
        Q = self.W_q(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        K = self.W_k(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        V = self.W_v(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Compute raw attention scores
        scores = torch.einsum("hnd,hmd->hnm", Q, K) / np.sqrt(self.head_dim)
        
        return Q, K, V, scores

    def forward(self, x, edge_index, return_qkv=False):
        N = x.size(0)
        Q, K, V, scores = self.compute_base_attention(x)
        
        # Store QKV embeddings for attention improvement loss
        qkv_embeddings = (Q + K + V).mean(dim=0).view(N, -1)
        
        # Make diagonal zero (no self-attention)
        diag_mask = torch.eye(N, device=scores.device).bool()
        scores = scores.masked_fill(diag_mask.unsqueeze(0), float('-inf'))
        
        if self.use_diffusion and self.training:
            # During training: use diffusion to refine attention
            # Sample random timestep
            t = torch.randint(0, self.num_diff_steps, (1,), device=x.device)
            
            # Get base attention (before softmax) - this is what we want to denoise
            base_attn = scores.clone()
            
            # Add noise to attention scores
            noise = torch.randn_like(base_attn)
            noisy_attn = self.scheduler.add_noise(base_attn, t, noise)
            
            # Denoise the attention
            denoised_attn = self.denoiser(noisy_attn, x, t)
            
            # Mask diagonal again after denoising
            denoised_attn = denoised_attn.masked_fill(diag_mask.unsqueeze(0), float('-inf'))
            
            # Use denoised attention
            attn_weights = torch.softmax(denoised_attn, dim=-1)
        elif self.use_diffusion and not self.training:
            # During inference: run full denoising process
            attn = scores.clone()
            
            # Iterative denoising from T to 0
            for t_val in reversed(range(self.num_diff_steps)):
                t = torch.tensor([t_val], device=x.device)
                attn = self.denoiser(attn, x, t)
            
            # Mask diagonal
            attn = attn.masked_fill(diag_mask.unsqueeze(0), float('-inf'))
            attn_weights = torch.softmax(attn, dim=-1)
        else:
            # No diffusion
            attn_weights = torch.softmax(scores, dim=-1)
        
        attn_weights = self.dropout(attn_weights)

        out = torch.einsum("hnm,hmd->hnd", attn_weights, V)
        out = out.transpose(0, 1).contiguous().view(N, self.embed_dim)
        
        if return_qkv:
            return self.out_proj(out), qkv_embeddings
        return self.out_proj(out)


# Transformer Layer
class GraphTransformerLayer(nn.Module):
    def __init__(self, dim, heads, dropout, use_diffusion=False, num_diff_steps=10):
        super().__init__()
        self.attn = GraphMultiHeadAttention(dim, heads, dropout, use_diffusion, num_diff_steps)
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
    def __init__(self, in_dim, hidden_dim, out_dim, layers, heads, dropout, 
                 use_diffusion=False, num_diff_steps=10):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, heads, dropout, use_diffusion, num_diff_steps)
            for _ in range(layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch, return_attn_loss=False):
        x = self.input_proj(x)
        initial_embeddings = x.clone()  # Store initial embeddings for attention loss
        
        all_qkv_embeddings = []
        first_layer_embeddings = None
        
        for i, layer in enumerate(self.layers):
            if return_attn_loss:
                x, qkv_emb = layer(x, edge_index, return_qkv=True)
                all_qkv_embeddings.append(qkv_emb)
                # Store embeddings after first transformer layer for structure loss
                if i == 0:
                    first_layer_embeddings = x.clone()
            else:
                x = layer(x, edge_index)

        # graph-level pooling
        pooled = global_mean_pool(x, batch)
        out = self.output_proj(pooled)
        
        if return_attn_loss:
            # Average QKV embeddings across layers as denoised embeddings
            denoised_embeddings = all_qkv_embeddings  # torch.stack(all_qkv_embeddings, dim=0).mean(dim=0)
            return out, initial_embeddings, denoised_embeddings, first_layer_embeddings
        return out


def train_epoch(model, loader, optimizer, device, attn_loss_weight=0.0, struct_loss_weight=0.0):
    model.train()
    total_loss = 0
    total_task_loss = 0
    total_attn_loss = 0
    total_struct_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass with attention loss components
        out, initial_emb, denoised_emb, first_layer_emb = model(
            data.x.float(), data.edge_index, data.batch, return_attn_loss=True
        )
        
        # Task loss (binary cross-entropy)
        task_loss = F.binary_cross_entropy_with_logits(out, data.y)
        
        # Attention improvement loss (computed for each layer separately and summed)
        attn_loss = torch.tensor(0.0, device=device)
        if attn_loss_weight > 0:
            input_emb = initial_emb
            for layer_qkv_emb in denoised_emb:
                attn_loss = attn_loss + attention_improvement_loss(
                    input_emb, layer_qkv_emb, data.edge_index, data.batch
                )
                input_emb = layer_qkv_emb  # Update input for next layer
        
        # Structure reconstruction loss
        struct_loss = torch.tensor(0.0, device=device)
        if struct_loss_weight > 0 and first_layer_emb is not None:
            struct_loss = structure_reconstruction_loss(
                first_layer_emb, data.edge_index, data.batch
            )
        
        # Combined loss
        loss = task_loss + attn_loss_weight * attn_loss + struct_loss_weight * struct_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_task_loss += task_loss.item() * data.num_graphs
        total_attn_loss += attn_loss.item() * data.num_graphs
        total_struct_loss += struct_loss.item() * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_task_loss / n, total_attn_loss / n, total_struct_loss / n


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

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)

    print(train_dataset[0], flush=True)

    model = GraphTransformer(
        in_dim=train_dataset.num_node_features,
        hidden_dim=args.hidden_dim,
        out_dim=train_dataset.num_classes,
        layers=args.num_layers,
        heads=args.num_heads,
        dropout=args.dropout,
        use_diffusion=args.use_diffusion,
        num_diff_steps=args.num_diff_steps
    ).to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_val = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss, task_loss, attn_loss, struct_loss = train_epoch(
            model, train_loader, optimizer, device, 
            attn_loss_weight=args.attn_loss_weight,
            struct_loss_weight=args.struct_loss_weight
        )
        val_f1 = evaluate(model, val_loader, device)

        if val_f1 > best_val:
            best_val = val_f1
            best_state = model.state_dict()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | Loss {loss:.4f} | Task {task_loss:.4f} | "
                f"Attn {attn_loss:.4f} | Struct {struct_loss:.4f} | Val Micro-F1 {val_f1:.4f}"
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
    parser.add_argument("--attn_loss_weight", type=float, default=0,
                        help="Weight for attention improvement loss")
    parser.add_argument("--struct_loss_weight", type=float, default=0,
                        help="Weight for structure reconstruction loss")
    parser.add_argument("--use_diffusion", action="store_true",
                        help="Use diffusion pipeline for attention prediction")
    parser.add_argument("--num_diff_steps", type=int, default=10,
                        help="Number of diffusion steps for attention denoising")
    args = parser.parse_args()
    print(args.__dict__, flush=True)
    main(args)
