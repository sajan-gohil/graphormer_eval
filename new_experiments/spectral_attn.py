import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, add_self_loops
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torchmetrics.classification import MultilabelAveragePrecision
from tqdm import tqdm

# ==========================================
# 1. Core Component: Chebyshev Filter
# ==========================================
class ChebyshevFilter(nn.Module):
    """
    Applies a learnable spectral filter using Chebyshev polynomials.
    Output = Sum_k (theta_k * T_k(L) * X)
    """
    def __init__(self, in_channels, K=3):
        super().__init__()
        self.K = K
        # Learnable coefficients for the filter (one per order k)
        # We initialize them to mimic a low-pass filter (decaying with k)
        self.coeffs = nn.Parameter(torch.randn(K))
        nn.init.normal_(self.coeffs, mean=0.0, std=0.1)

    def forward(self, x, edge_index, batch=None):
        """
        x: [N, dim]
        edge_index: [2, E]
        """
        # 1. Compute Normalized Laplacian: L = I - D^-0.5 A D^-0.5
        # We assume edge_index describes the A matrix.
        
        # Ensure self loops are present for stability
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Compute L (PyG util) - returns L values and indices
        edge_index_L, edge_weight_L = get_laplacian(edge_index, normalization='sym')
        
        # Rescale L for Chebyshev: L_tilde = L - I  (shifts eigenvalues from [0, 2] to [-1, 1])
        # Note: get_laplacian returns L. We need to shift it manually or use the recurrence relation carefully.
        # Standard ChebNet recurrence: T_0(x)=x, T_1(x)=Lx, T_k(x) = 2L T_{k-1} - T_{k-2}
        # But L must be scaled to [-1, 1]. The eigenvalues of L_sym are in [0, 2].
        # So L_hat = L_sym - I maps [0, 2] -> [-1, 1].
        
        # Perform Sparse Matrix-Vector Multiply (L * x)
        def sparse_mm(idx, wt, mat):
            return torch.sparse.mm(
                torch.sparse_coo_tensor(idx, wt, (mat.size(0), mat.size(0))), 
                mat
            )
            
        # We need to explicitly construct the "L - I" operator or handle it in the loop.
        # It's cleaner to calculate Tx_0, Tx_1...
        
        Tx_0 = x
        
        # For Tx_1, we need (L - I)x = Lx - x
        Lx = sparse_mm(edge_index_L, edge_weight_L, x)
        Tx_1 = Lx - x 
        
        out = self.coeffs[0] * Tx_0 + self.coeffs[1] * Tx_1
        
        Tx_prev = Tx_1
        Tx_prev2 = Tx_0
        
        # Recurrence: T_k(x) = 2 * (L-I) * T_{k-1} - T_{k-2}
        # Note: 2*(L-I)x = 2Lx - 2x
        for k in range(2, self.K):
            L_Tx_prev = sparse_mm(edge_index_L, edge_weight_L, Tx_prev)
            term1 = 2 * (L_Tx_prev - Tx_prev)
            Tx_k = term1 - Tx_prev2
            
            out = out + self.coeffs[k] * Tx_k
            
            Tx_prev2 = Tx_prev
            Tx_prev = Tx_k
            
        return out

# ==========================================
# 2. Spectrally-Decoupled Attention Layer
# ==========================================
class SDA_Layer(nn.Module):
    def __init__(self, embed_dim, num_heads, K=3, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear Projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        # Spectral Filters for Q and K
        # Each head gets its own learnable band-pass filter
        self.filters_q = nn.ModuleList([ChebyshevFilter(self.head_dim, K) for _ in range(num_heads)])
        self.filters_k = nn.ModuleList([ChebyshevFilter(self.head_dim, K) for _ in range(num_heads)])
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim)
        )

    def forward(self, x, edge_index, batch):
        # x: [N, d]
        resid = x
        x = self.norm1(x)
        
        # 1. Linear Projection
        q = self.W_q(x).view(-1, self.num_heads, self.head_dim)
        k = self.W_k(x).view(-1, self.num_heads, self.head_dim)
        v = self.W_v(x).view(-1, self.num_heads, self.head_dim)
        
        # 2. Spectral Filtering (The Novelty)
        # We apply the filter per head.
        # This is a bit slow in python loop, but valid for prototype.
        q_spec_list = []
        k_spec_list = []
        
        for h in range(self.num_heads):
            # Extract head specific features: [N, head_dim]
            q_h = q[:, h, :]
            k_h = k[:, h, :]
            
            # Apply Band-Pass Filter
            q_filt = self.filters_q[h](q_h, edge_index)
            k_filt = self.filters_k[h](k_h, edge_index)
            
            q_spec_list.append(q_filt)
            k_spec_list.append(k_filt)
            
        # Re-stack: [N, H, D]
        q_spec = torch.stack(q_spec_list, dim=1)
        k_spec = torch.stack(k_spec_list, dim=1)
        
        # 3. Attention Mechanism (Standard from here, but with Filtered Inputs)
        # We use PyTorch's scaled_dot_product_attention for efficiency if available (Torch 2.0+)
        # We need [Batch, Heads, Seq, Dim] format. But graphs are varying size packed in one tensor.
        # We must use sparse attention or mask padding. 
        # For simplicity in this script: Dense attention with mask per graph in batch.
        
        # Unbatching to [B, Max_Nodes, H, D] is expensive.
        # Let's use a simplified global attention approach:
        # Since this is a "Global" layer, we ideally want Full Attention.
        # With packed batch, we can't do simple matmul.
        
        # IMPLEMENTATION TRICK:
        # To avoid padding issues, we iterate over unique batch indices (slow but correct)
        # Or use torch_geometric.utils.to_dense_batch
        
        from torch_geometric.utils import to_dense_batch
        
        # [B, max_N, H, D]
        q_dense, mask = to_dense_batch(q_spec.reshape(x.size(0), -1), batch)
        k_dense, _ = to_dense_batch(k_spec.reshape(x.size(0), -1), batch)
        v_dense, _ = to_dense_batch(v.reshape(x.size(0), -1), batch)
        
        # Reshape for torch attention: [B, H, N, D_head]
        B, max_N, _ = q_dense.size()
        q_dense = q_dense.view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)
        k_dense = k_dense.view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)
        v_dense = v_dense.view(B, max_N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create attn_mask from mask [B, N] -> [B, 1, 1, N]
        # mask is True for real nodes, False for padding
        attn_mask = mask.view(B, 1, 1, max_N).expand(B, self.num_heads, max_N, max_N)
        
        # PyTorch Scaled Dot Product
        # Note: torch SDP expects mask where True = Keep (if bool) or 0/-inf (if float).
        # We generally set padding positions to -inf.
        
        # Compute scores manually to be safe with versions
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q_dense, k_dense.transpose(-2, -1)) * scale # [B, H, N, N]
        
        # Apply Mask
        # mask is [B, N]. We want to mask keys that are padding.
        # mask_broadcast: [B, 1, 1, N]
        mask_broadcast = mask.unsqueeze(1).unsqueeze(2) 
        scores = scores.masked_fill(~mask_broadcast, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out_dense = torch.matmul(attn, v_dense) # [B, H, N, D]
        
        # Re-pack to [Total_N, H, D]
        out_dense = out_dense.transpose(1, 2).reshape(B, max_N, self.embed_dim)
        
        # Select valid nodes
        out = out_dense[mask] # [Total_N, Embed]
        
        out = self.W_o(out)
        
        # Residual
        x = resid + self.dropout(out)
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x

# ==========================================
# 3. Full Model
# ==========================================
class SDA_GraphTransformer(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, num_heads, num_classes):
        super().__init__()
        
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim) # Not utilized in Attn directly, but added to node feats
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(SDA_Layer(hidden_dim, num_heads, K=3))
            
        self.pool = global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Initial Embedding
        h = self.atom_encoder(x)
        
        # Naive integration of edge attributes: Add to node features of source/target
        # A more complex model would use them in the Attention bias
        row, col = edge_index
        edge_emb = self.bond_encoder(edge_attr)
        # Scatter add edge features to nodes (simple message passing step 0)
        from torch_geometric.utils import scatter
        h = h + scatter(edge_emb, row, dim=0, dim_size=h.size(0), reduce='add')

        for layer in self.layers:
            h = layer(h, edge_index, batch)
            
        h_graph = self.pool(h, batch)
        out = self.classifier(h_graph)
        return out

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100) # Reduced for testing
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=32) # Smaller batch for dense Attn memory
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    print("Loading Peptides-func...")
    train_dataset = LRGBDataset(root="./data", name="Peptides-func", split="train")
    val_dataset   = LRGBDataset(root="./data", name="Peptides-func", split="val")
    test_dataset  = LRGBDataset(root="./data", name="Peptides-func", split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)

    # Model Setup
    model = SDA_GraphTransformer(
        in_channels=9, # AtomEncoder handles input dim internally
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_classes=train_dataset.num_classes
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    metric = MultilabelAveragePrecision(num_labels=train_dataset.num_classes, average="macro").to(device)

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        # TRAIN
        model.train()
        loss_epoch = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y.float())
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        
        avg_train_loss = loss_epoch / len(train_loader)

        # VAL
        model.eval()
        val_metric_score = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                metric.update(out, data.y.long())
            
            val_ap = metric.compute()
            metric.reset()
            
        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f} | Val AP {val_ap:.4f}")

    # TEST
    print("Testing...")
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            metric.update(out, data.y.long())
        test_ap = metric.compute()
        
    print(f"Final Test AP: {test_ap:.4f}")

if __name__ == "__main__":
    main()

