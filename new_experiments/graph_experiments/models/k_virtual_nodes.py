"""
Phase 3.2: K Fixed Virtual Nodes Baseline.

M fixed learnable embeddings (nn.Parameter), shared across all graphs,
optimized during training as part of the model parameters. Each graph
receives the same M proxy embeddings regardless of its content.

This is the *critical baseline* — it tests whether graph-conditional
generation adds value over fixed tokens.

Trained from scratch, try M ∈ {4, 8, 16}.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from models.transformer import NodeEmbedding, PEEncoder, GPSLayer
from models.virtual_node import _forward_gps_layer_with_proxies


class GPSModelKVN(nn.Module):
    """
    GPS model with M fixed learnable virtual node embeddings.

    The M embeddings are nn.Parameters shared across all graphs — every graph
    gets the same M proxies. They participate in global self-attention at every
    GPS layer but not in the local MPNN.

    Args:
        config: ModelConfig with hidden_dim, num_layers, num_heads, etc.
        M: Number of fixed virtual nodes.
    """

    def __init__(self, config, M=8):
        super().__init__()
        d = config.hidden_dim
        self.hidden_dim = d
        self.M = M

        # Node embedding (same as base GPS)
        self.node_emb = NodeEmbedding(d)
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

        # M fixed virtual node embeddings — learnable, shared across all graphs
        self.vn_embeddings = nn.Parameter(torch.randn(1, M, d) * 0.02)

    def forward(self, batch):
        """
        Forward pass with M fixed virtual nodes at every layer.
        """
        # Initial embeddings
        h = self.node_emb(batch.x, batch.edge_index, batch.edge_attr)
        pe = self.pe_encoder(batch.lap_pe, batch.rwse)
        h = h + pe
        batch_idx = batch.batch
        num_graphs = batch_idx.max().item() + 1

        # Expand fixed VN embeddings to batch: (1, M, d) -> (B, M, d)
        vn = self.vn_embeddings.expand(num_graphs, -1, -1)

        # Process through GPS layers with VN injection
        for layer in self.layers:
            h, vn, _ = _forward_gps_layer_with_proxies(
                layer, h, batch.edge_index, batch.edge_attr,
                batch_idx, vn, num_graphs
            )

        # Readout: pool original nodes only
        h = self.post_norm(h)
        graph_emb = global_mean_pool(h, batch_idx)
        logits = self.classifier(graph_emb)
        return logits
