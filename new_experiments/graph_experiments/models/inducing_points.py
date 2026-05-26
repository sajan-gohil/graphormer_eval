"""
Phase 3.5: Set Transformer Inducing Points.

Replaces global self-attention in GPS layers with Set Transformer-style
ISAB (Induced Set Attention Block) using M learnable inducing points:

    H = MultiheadAttn(I, X, X)  — inducing points attend to nodes
    O = MultiheadAttn(X, H, H)  — nodes attend to updated inducing points

Complexity: O(NM) instead of O(N²).

The inducing points I ∈ ℝ^{M×d} are learnable parameters (fixed across
graphs, shared across layers or per-layer depending on config).

Trained from scratch, try M ∈ {16, 64}.

Reference: Lee et al., "Set Transformer: A Framework for Attention-based
Permutation-Invariant Input", ICML 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from models.transformer import NodeEmbedding, PEEncoder, GINELayer


class InducedMultiheadAttention(nn.Module):
    """
    Standard multi-head cross-attention: Q attends to K/V.

    Used as a building block for ISAB:
        Step 1: Q=I, K=V=X  →  inducing points attend to nodes
        Step 2: Q=X, K=V=H  →  nodes attend to updated inducing points
    """

    def __init__(self, hidden_dim, num_heads, attn_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key_value, kv_mask=None):
        """
        Args:
            query: (B, Nq, d) query embeddings
            key_value: (B, Nkv, d) key/value embeddings
            kv_mask: (B, Nkv) boolean mask — True for valid, False for padding

        Returns:
            out: (B, Nq, d) attention output
            attn_weights: (B, H, Nq, Nkv) attention weights
        """
        B, Nq, d = query.shape
        Nkv = key_value.shape[1]
        H = self.num_heads

        q = self.q_proj(query).reshape(B, Nq, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, Nkv, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, Nkv, H, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, Nq, Nkv)

        if kv_mask is not None:
            kv_mask_exp = kv_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Nkv)
            attn = attn.masked_fill(~kv_mask_exp, float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, Nq, hd)
        out = out.transpose(1, 2).reshape(B, Nq, d)
        out = self.out_proj(out)

        return out, attn_weights


class ISABLayer(nn.Module):
    """
    Induced Set Attention Block (ISAB) — replaces O(N²) self-attention
    with O(NM) two-step cross-attention through M inducing points.

    Architecture:
        H = LN(I) + CrossAttn(LN(I), LN(X), LN(X))   — inducing points aggregate from nodes
        O = LN(X) + CrossAttn(LN(X), LN(H), LN(H))   — nodes read from inducing points

    This is a single GPS-style layer: local MPNN + ISAB + FFN.
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

        # ISAB step 1: inducing points attend to nodes
        self.norm_i1 = nn.LayerNorm(hidden_dim)
        self.norm_x1 = nn.LayerNorm(hidden_dim)
        self.cross_attn_i2x = InducedMultiheadAttention(hidden_dim, num_heads, attn_dropout)

        # ISAB step 2: nodes attend to inducing points
        self.norm_x2 = nn.LayerNorm(hidden_dim)
        self.norm_h2 = nn.LayerNorm(hidden_dim)
        self.cross_attn_x2h = InducedMultiheadAttention(hidden_dim, num_heads, attn_dropout)

        # FFN
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_sparse, edge_index, edge_attr, batch, inducing_points):
        """
        Args:
            x_sparse: (N_total, d) node features in sparse format.
            edge_index: (2, E) edge indices.
            edge_attr: edge attributes.
            batch: (N_total,) batch assignment.
            inducing_points: (B, M, d) inducing point embeddings.

        Returns:
            x_sparse_out: (N_total, d) updated node features.
            inducing_out: (B, M, d) updated inducing point embeddings.
            attn_i2x: (B, H, M, N_max) attention weights (inducing→nodes).
            attn_x2h: (B, H, N_max, M) attention weights (nodes→inducing).
        """
        # 1. Local MPNN on nodes only
        local_out = self.local_model(self.local_norm(x_sparse), edge_index, edge_attr, batch)
        x_sparse = x_sparse + self.dropout(local_out)

        # Convert to dense for cross-attention
        dense_x, node_mask = to_dense_batch(x_sparse, batch)  # (B, N_max, d), (B, N_max)

        # 2. ISAB step 1: inducing points attend to nodes
        #    H = I + CrossAttn(I, X)
        i_normed = self.norm_i1(inducing_points)
        x_normed = self.norm_x1(dense_x)
        h_attn, attn_i2x = self.cross_attn_i2x(i_normed, x_normed, kv_mask=node_mask)
        H = inducing_points + self.dropout(h_attn)  # (B, M, d)

        # 3. ISAB step 2: nodes attend to updated inducing points
        #    O = X + CrossAttn(X, H)
        #    No mask needed for inducing points (all valid)
        x_normed2 = self.norm_x2(dense_x)
        h_normed2 = self.norm_h2(H)
        o_attn, attn_x2h = self.cross_attn_x2h(x_normed2, h_normed2)
        dense_x = dense_x + self.dropout(o_attn)

        # 4. FFN on nodes
        ffn_out = self.ffn(self.ffn_norm(dense_x))
        dense_x = dense_x + ffn_out

        # Convert back to sparse
        x_sparse_out = dense_x[node_mask]

        # Update inducing points with their own FFN-like residual
        # (inducing points also get refined through the layers)
        inducing_out = H

        return x_sparse_out, inducing_out, attn_i2x, attn_x2h


class GPSModelISAB(nn.Module):
    """
    GPS model with Set Transformer-style ISAB replacing global self-attention.

    Instead of O(N²) self-attention per layer, uses M inducing points for
    O(NM) two-step cross-attention. The inducing points I ∈ ℝ^{M×d} are
    learnable parameters shared across all graphs.

    Two modes:
        - shared_inducing=True:  same I used at every layer (updated through layers)
        - shared_inducing=False: each layer gets its own independent I

    Args:
        config: ModelConfig with hidden_dim, num_layers, num_heads, etc.
        M: Number of inducing points.
        shared_inducing: Whether inducing points are shared across layers.
    """

    def __init__(self, config, M=16, shared_inducing=True):
        super().__init__()
        d = config.hidden_dim
        self.hidden_dim = d
        self.M = M
        self.shared_inducing = shared_inducing

        # Node embedding (same as base GPS)
        self.node_emb = NodeEmbedding(d)
        self.pe_encoder = PEEncoder(
            lap_dim=config.lap_dim,
            rwse_dim=config.rwse_dim,
            hidden_dim=d,
            pe_hidden_dim=config.pe_hidden_dim,
        )

        # ISAB layers (replace standard GPSLayers)
        self.layers = nn.ModuleList([
            ISABLayer(
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

        # Inducing points
        if shared_inducing:
            # Single set of M inducing points, propagated through all layers
            self.inducing = nn.Parameter(torch.randn(1, M, d) * 0.02)
        else:
            # Independent inducing points per layer
            self.inducing = nn.ParameterList([
                nn.Parameter(torch.randn(1, M, d) * 0.02)
                for _ in range(config.num_layers)
            ])

    def forward(self, batch):
        """
        Forward pass with ISAB layers replacing global self-attention.
        """
        # Initial embeddings
        h = self.node_emb(batch.x, batch.edge_index, batch.edge_attr)
        pe = self.pe_encoder(batch.lap_pe, batch.rwse)
        h = h + pe
        batch_idx = batch.batch
        num_graphs = batch_idx.max().item() + 1

        if self.shared_inducing:
            # Expand shared inducing points to batch size
            I = self.inducing.expand(num_graphs, -1, -1)  # (B, M, d)

            for layer in self.layers:
                h, I, _, _ = layer(
                    h, batch.edge_index, batch.edge_attr, batch_idx, I
                )
        else:
            for i, layer in enumerate(self.layers):
                I = self.inducing[i].expand(num_graphs, -1, -1)
                h, _, _, _ = layer(
                    h, batch.edge_index, batch.edge_attr, batch_idx, I
                )

        # Readout: pool original nodes only
        h = self.post_norm(h)
        graph_emb = global_mean_pool(h, batch_idx)
        logits = self.classifier(graph_emb)
        return logits
