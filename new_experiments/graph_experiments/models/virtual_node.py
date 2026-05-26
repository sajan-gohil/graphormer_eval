"""
Phase 3.1: Virtual Node Baselines.

Two variants of a single virtual node that participates in every GPS layer's
global attention:

  - VN-Fixed:      A fixed learnable parameter shared across all graphs,
                   initialized from N(0, σ). Updated through attention at
                   each layer.
  - VN-Aggregated: Re-initialized at each layer as the mean of node embeddings,
                   then updated via attention. (Standard virtual node approach.)

Both are trained end-to-end from scratch with the same hyperparameters as Phase 1.
The model has the same forward(batch) -> logits interface as GPSModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from models.transformer import NodeEmbedding, PEEncoder, GPSLayer


class GPSModelVN(nn.Module):
    """
    GPS model with a single virtual node.

    Args:
        config: ModelConfig with hidden_dim, num_layers, num_heads, etc.
        vn_mode: "fixed" or "aggregated".
    """

    def __init__(self, config, vn_mode="fixed"):
        super().__init__()
        assert vn_mode in ("fixed", "aggregated"), \
            f"vn_mode must be 'fixed' or 'aggregated', got '{vn_mode}'"

        d = config.hidden_dim
        self.vn_mode = vn_mode
        self.hidden_dim = d

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

        # Virtual node embedding (learnable, shared across all graphs)
        if vn_mode == "fixed":
            self.vn_embedding = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        else:
            # For aggregated mode, we still have a learnable projection
            # applied after computing the mean
            self.vn_proj = nn.Linear(d, d)

    def _get_vn_embeddings(self, h_sparse, batch_idx, num_graphs, layer_idx):
        """
        Get virtual node embeddings for this layer.

        For 'fixed': expand the single learnable embedding to (B, 1, d).
        For 'aggregated': compute mean of node embeddings per graph, project.

        Returns:
            vn: (B, 1, d) virtual node embeddings.
        """
        if self.vn_mode == "fixed":
            return self.vn_embedding.expand(num_graphs, -1, -1)
        else:
            # Compute per-graph mean of node embeddings
            graph_means = global_mean_pool(h_sparse, batch_idx)  # (B, d)
            vn = self.vn_proj(graph_means).unsqueeze(1)  # (B, 1, d)
            return vn

    def forward(self, batch):
        """
        Forward pass with virtual node at every layer.

        Proxies (VN) participate in global attention but NOT local MPNN.
        Readout pools only original nodes.
        """
        # Initial embeddings
        h = self.node_emb(batch.x, batch.edge_index, batch.edge_attr)
        pe = self.pe_encoder(batch.lap_pe, batch.rwse)
        h = h + pe
        batch_idx = batch.batch
        num_graphs = batch_idx.max().item() + 1

        # Process through GPS layers with VN injection
        for layer_idx, layer in enumerate(self.layers):
            # Get VN embedding for this layer
            vn = self._get_vn_embeddings(h, batch_idx, num_graphs, layer_idx)

            # Run layer with proxy (VN) injection
            h, vn, _ = _forward_gps_layer_with_proxies(
                layer, h, batch.edge_index, batch.edge_attr,
                batch_idx, vn, num_graphs
            )

        # Readout: pool original nodes only (exclude VN)
        h = self.post_norm(h)
        graph_emb = global_mean_pool(h, batch_idx)
        logits = self.classifier(graph_emb)
        return logits


def _forward_gps_layer_with_proxies(layer, x_sparse_nodes, edge_index, edge_attr,
                                    node_batch, proxy_embs, num_graphs):
    """
    Shared helper: run a GPSLayer with proxy embeddings in attention only.

    Identical logic to proxy_optimizer.forward_gps_layer_with_proxies but
    importable without pulling in the optimizer module.
    """
    B = num_graphs
    M = proxy_embs.shape[1]

    # 1. Local MPNN on original nodes only
    local_out = layer.local_model(layer.local_norm(x_sparse_nodes), edge_index, edge_attr, node_batch)
    x_sparse_nodes = x_sparse_nodes + layer.dropout(local_out)

    # 2. Build combined dense [nodes | proxies]
    dense_nodes, node_mask = to_dense_batch(x_sparse_nodes, node_batch)
    N_max = dense_nodes.shape[1]
    dense_combined = torch.cat([dense_nodes, proxy_embs], dim=1)

    proxy_mask = torch.ones(B, M, dtype=torch.bool, device=node_mask.device)
    combined_mask = torch.cat([node_mask, proxy_mask], dim=1)

    # 3. Global self-attention
    attn_input = layer.global_norm(dense_combined)
    attn_out, attn_weights = layer.global_attn(attn_input, combined_mask)
    dense_combined = dense_combined + layer.dropout(attn_out)

    # 4. FFN
    ffn_out = layer.ffn(layer.ffn_norm(dense_combined))
    dense_combined = dense_combined + ffn_out

    # 5. Separate
    dense_nodes_out = dense_combined[:, :N_max, :]
    proxy_embs_out = dense_combined[:, N_max:, :]
    x_sparse_nodes_out = dense_nodes_out[node_mask]

    return x_sparse_nodes_out, proxy_embs_out, attn_weights
