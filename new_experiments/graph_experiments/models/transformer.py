"""
Phase 1 Model: GPS (General, Powerful, Scalable) Graph Transformer.

GPS interleaves local MPNN layers with global self-attention.
This design makes it natural for proxy insertion in later phases:
proxies participate in global attention but not the local MPNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch, degree
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.utils import scatter


class PEEncoder(nn.Module):
    """
    Encode Laplacian PE and RWSE into hidden_dim, then add to node embeddings.
    """
    def __init__(self, lap_dim, rwse_dim, hidden_dim, pe_hidden_dim=64):
        super().__init__()
        # Laplacian PE encoder: linear -> ReLU -> linear
        self.lap_encoder = nn.Sequential(
            nn.Linear(lap_dim, pe_hidden_dim),
            nn.ReLU(),
            nn.Linear(pe_hidden_dim, hidden_dim),
        )
        # RWSE encoder: linear -> ReLU -> linear
        self.rwse_encoder = nn.Sequential(
            nn.Linear(rwse_dim, pe_hidden_dim),
            nn.ReLU(),
            nn.Linear(pe_hidden_dim, hidden_dim),
        )

    def forward(self, lap_pe, rwse):
        """
        Args:
            lap_pe: (N, lap_dim) Laplacian eigenvectors
            rwse: (N, rwse_dim) Random walk structural encoding

        Returns:
            (N, hidden_dim) positional encoding to add to node embeddings
        """
        return self.lap_encoder(lap_pe) + self.rwse_encoder(rwse)


class NodeEmbedding(nn.Module):
    """
    Embed atom features + aggregated bond features into hidden_dim.
    Follows the pattern from vanilla_gt.py.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim // 2)
        self.bond_encoder = BondEncoder(hidden_dim // 2)
        self.node_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        h = self.atom_encoder(x)
        row, col = edge_index
        edge_emb = self.bond_encoder(edge_attr)
        edge_aggr = scatter(
            edge_emb, row, dim=0, dim_size=h.size(0), reduce='add'
        )
        h = torch.cat([h, edge_aggr], dim=-1)
        h = self.node_proj(h)
        return h


class GlobalSelfAttention(nn.Module):
    """
    Multi-head self-attention for dense batched node embeddings.
    Operates on (B, N_max, d) tensors with a mask for padding.
    """
    def __init__(self, hidden_dim, num_heads, attn_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, d) dense node embeddings
            mask: (B, N) boolean mask — True for real nodes, False for padding

        Returns:
            out: (B, N, d) attention output
            attn_weights: (B, H, N, N) attention weights (for diagnostics)
        """
        B, N, d = x.shape
        H = self.num_heads

        q = self.q_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)  # (B, H, N, hd)
        k = self.k_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, H, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Apply mask: padding nodes should not attend or be attended to
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N) for key masking
            key_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(~key_mask, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        # NaN guard: if all keys are masked, softmax gives NaN -> set to 0
        attn_weights = attn_weights.nan_to_num(0.0)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, N, hd)
        out = out.transpose(1, 2).reshape(B, N, d)  # (B, N, d)
        out = self.out_proj(out)

        return out, attn_weights


class GINELayer(nn.Module):
    """
    GIN-E (Graph Isomorphism Network with Edge features) for the local MPNN component.
    Uses edge attributes in message passing.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.conv = GINEConv(mlp, edge_dim=hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: (N_total, d) node features in sparse (concatenated) format
            edge_index: (2, E) edge indices
            edge_attr: raw edge attributes
            batch: (N_total,) batch assignment

        Returns:
            (N_total, d) updated node features
        """
        edge_emb = self.bond_encoder(edge_attr)
        return self.conv(x, edge_index, edge_emb)


class GPSLayer(nn.Module):
    """
    A single GPS layer: local MPNN + global self-attention + FFN,
    with residual connections and LayerNorm.

    Architecture:
        h = h + LocalMPNN(LN(h))
        h = h + GlobalAttn(LN(h))
        h = h + FFN(LN(h))
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1, attn_dropout=0.1,
                 local_gnn_type="GIN"):
        super().__init__()

        # Local MPNN
        self.local_norm = nn.LayerNorm(hidden_dim)
        if local_gnn_type == "GIN":
            self.local_model = GINELayer(hidden_dim)
        else:
            raise ValueError(f"Unsupported local_gnn_type: {local_gnn_type}")

        # Global self-attention
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.global_attn = GlobalSelfAttention(hidden_dim, num_heads, attn_dropout)

        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_sparse, edge_index, edge_attr, batch, dense_x=None, dense_mask=None):
        """
        Args:
            x_sparse: (N_total, d) node features in sparse format (for local MPNN)
            edge_index: (2, E) edges
            edge_attr: edge attributes
            batch: (N_total,) batch assignment
            dense_x: (B, N_max, d) dense batched embeddings (for global attention)
            dense_mask: (B, N_max) boolean mask

        Returns:
            x_sparse: updated sparse features
            dense_x: updated dense features
            attn_weights: attention weight matrix
        """
        # 1. Local MPNN (sparse domain)
        local_out = self.local_model(self.local_norm(x_sparse), edge_index, edge_attr, batch)
        x_sparse = x_sparse + self.dropout(local_out)

        # Convert to dense for global attention
        dense_x, dense_mask = to_dense_batch(x_sparse, batch)

        # 2. Global self-attention (dense domain)
        attn_input = self.global_norm(dense_x)
        attn_out, attn_weights = self.global_attn(attn_input, dense_mask)
        dense_x = dense_x + self.dropout(attn_out)

        # 3. FFN (dense domain)
        ffn_out = self.ffn(self.ffn_norm(dense_x))
        dense_x = dense_x + ffn_out

        # Convert back to sparse
        x_sparse = dense_x[dense_mask]

        return x_sparse, dense_x, dense_mask, attn_weights


class GPSModel(nn.Module):
    """
    Full GPS model for graph classification.

    Architecture:
        NodeEmbedding + PE → stack of GPSLayers → global mean pool → MLP classifier

    The model exposes `get_initial_embeddings(batch)` to extract the
    post-PE, pre-GPSLayer node embeddings — used as conditioning input
    for the flow matching model in later stages.
    """
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim

        # Node embedding (atom + bond features)
        self.node_emb = NodeEmbedding(d)

        # Positional encoding
        self.pe_encoder = PEEncoder(
            lap_dim=config.lap_dim,
            rwse_dim=config.rwse_dim,
            hidden_dim=d,
            pe_hidden_dim=config.pe_hidden_dim,
        )

        # GPS layers
        self.layers = nn.ModuleList([
            GPSLayer(
                hidden_dim=d,
                num_heads=config.num_heads,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                local_gnn_type=config.local_gnn_type,
            )
            for _ in range(config.num_layers)
        ])

        # Classification head
        self.post_norm = nn.LayerNorm(d)
        self.classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d // 2, config.num_classes),
        )

        self.hidden_dim = d

    def get_initial_embeddings(self, batch):
        """
        Extract node embeddings after embedding + PE injection, before GPS layers.
        This is the conditioning input for later flow matching stages.

        Args:
            batch: PyG Batch object with x, edge_index, edge_attr, batch, lap_pe, rwse.

        Returns:
            h: (N_total, d) initial node embeddings in sparse format.
            batch_idx: (N_total,) batch assignment vector.
        """
        h = self.node_emb(batch.x, batch.edge_index, batch.edge_attr)
        pe = self.pe_encoder(batch.lap_pe, batch.rwse)
        h = h + pe
        return h, batch.batch

    def forward(self, batch):
        """
        Full forward pass for graph classification.

        Args:
            batch: PyG Batch with x, edge_index, edge_attr, batch, lap_pe, rwse, y.

        Returns:
            logits: (B, num_classes) raw logits.
        """
        # Initial embeddings
        h, batch_idx = self.get_initial_embeddings(batch)

        # GPS layers
        dense_x, dense_mask = None, None
        for layer in self.layers:
            h, dense_x, dense_mask, _ = layer(
                h, batch.edge_index, batch.edge_attr, batch_idx,
                dense_x, dense_mask
            )

        # Readout: global mean pool over real nodes
        h = self.post_norm(h)
        graph_emb = global_mean_pool(h, batch_idx)

        # Classify
        logits = self.classifier(graph_emb)
        return logits

    def forward_with_attention(self, batch):
        """
        Forward pass that also returns attention weights from each layer.
        Used for diagnostics.

        Returns:
            logits: (B, num_classes)
            all_attn_weights: list of (B, H, N_max, N_max) per layer
        """
        h, batch_idx = self.get_initial_embeddings(batch)

        all_attn_weights = []
        dense_x, dense_mask = None, None
        for layer in self.layers:
            h, dense_x, dense_mask, attn_w = layer(
                h, batch.edge_index, batch.edge_attr, batch_idx,
                dense_x, dense_mask
            )
            all_attn_weights.append(attn_w)

        h = self.post_norm(h)
        graph_emb = global_mean_pool(h, batch_idx)
        logits = self.classifier(graph_emb)
        return logits, all_attn_weights
