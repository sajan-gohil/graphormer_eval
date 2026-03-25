"""
Phase 4.1 & 4.2: Combined ablation forward passes.

Provides forward_with_proxies_ablation() which supports:
  - insertion_point: {0, 1, 2, "all"} — which layer(s) to insert proxies at.
  - routing_mode: {"full", "routed", "hybrid"} — attention routing strategy.

insertion_point semantics:
  0:     Proxies inserted before layer 0, propagated through all layers. (Default)
  1:     Proxies skip layer 0; inserted at layer 1, propagated through remaining.
  2:     Proxies skip layers 0-1; inserted at layer 2, propagated through remaining.
  "all": Fresh proxies added at each layer independently. Each layer's set of
         M proxies is a separate learnable parameter, optimized jointly.
         Returns require proxy_embs to be shape (num_layers, B, M, d).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from models.ablation_attention import get_layer_forward_fn


def forward_with_proxies_ablation(model, batch, proxy_embs, insertion_point=0,
                                  routing_mode='full', return_attention=False):
    """
    Run a frozen GPSModel with proxy embeddings, supporting ablation options.

    Args:
        model: Frozen GPSModel.
        batch: PyG Batch object.
        proxy_embs: Proxy embeddings. Shape depends on insertion_point:
            - insertion_point in {0, 1, 2}: (B, M, d)
            - insertion_point == "all": (num_layers, B, M, d)
        insertion_point: Layer at which to start inserting proxies, or "all".
        routing_mode: "full", "routed", or "hybrid".
        return_attention: If True, return attention weights from each layer.

    Returns:
        logits: (B, num_classes)
        all_attn_weights: list (if return_attention)
    """
    layer_forward_fn = get_layer_forward_fn(routing_mode)

    # Get initial node embeddings
    h, batch_idx = model.get_initial_embeddings(batch)
    num_graphs = batch_idx.max().item() + 1
    num_layers = len(model.layers)

    all_attn_weights = []

    if insertion_point == "all":
        # Fresh proxies at every layer — proxy_embs is (num_layers, B, M, d)
        assert proxy_embs.dim() == 4 and proxy_embs.shape[0] == num_layers, \
            f"For insertion_point='all', proxy_embs must be (num_layers, B, M, d), " \
            f"got {proxy_embs.shape}"

        for layer_idx, layer in enumerate(model.layers):
            current_proxies = proxy_embs[layer_idx]  # (B, M, d) fresh for this layer
            h, _, attn_w = layer_forward_fn(
                layer, h, batch.edge_index, batch.edge_attr,
                batch_idx, current_proxies, num_graphs
            )
            if return_attention:
                all_attn_weights.append(attn_w)

    else:
        # insertion_point is an int: insert at that layer and propagate
        insert_at = int(insertion_point)
        assert 0 <= insert_at < num_layers, \
            f"insertion_point={insert_at} but model has {num_layers} layers"

        current_proxies = proxy_embs  # (B, M, d)

        for layer_idx, layer in enumerate(model.layers):
            if layer_idx < insert_at:
                # Run layer without proxies (standard forward)
                h, _, attn_w = _forward_layer_no_proxy(
                    layer, h, batch.edge_index, batch.edge_attr, batch_idx
                )
                if return_attention:
                    all_attn_weights.append(attn_w)
            else:
                # Run layer with proxies using the specified routing mode
                h, current_proxies, attn_w = layer_forward_fn(
                    layer, h, batch.edge_index, batch.edge_attr,
                    batch_idx, current_proxies, num_graphs
                )
                if return_attention:
                    all_attn_weights.append(attn_w)

    # Readout: pool original nodes only
    h = model.post_norm(h)
    graph_emb = global_mean_pool(h, batch_idx)
    logits = model.classifier(graph_emb)

    if return_attention:
        return logits, all_attn_weights
    return logits


def _forward_layer_no_proxy(layer, x_sparse_nodes, edge_index, edge_attr, node_batch):
    """
    Standard GPSLayer forward without any proxies.
    Used for layers before the insertion point.

    Returns:
        x_sparse_out, dummy_proxy (None), attn_weights
    """
    # Local MPNN
    local_out = layer.local_model(layer.local_norm(x_sparse_nodes), edge_index, edge_attr, node_batch)
    x_sparse_nodes = x_sparse_nodes + layer.dropout(local_out)

    # Dense for attention
    dense_x, dense_mask = to_dense_batch(x_sparse_nodes, node_batch)

    # Global self-attention
    attn_input = layer.global_norm(dense_x)
    attn_out, attn_weights = layer.global_attn(attn_input, dense_mask)
    dense_x = dense_x + layer.dropout(attn_out)

    # FFN
    ffn_out = layer.ffn(layer.ffn_norm(dense_x))
    dense_x = dense_x + ffn_out

    # Back to sparse
    x_sparse_out = dense_x[dense_mask]

    return x_sparse_out, None, attn_weights
