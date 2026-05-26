"""
Phase 4.2: Attention routing ablation.

Three routing modes for how proxies interact with nodes in global attention:

  Full:   Standard self-attention on [nodes | proxies] (N+M)×(N+M).
          Bidirectional: nodes see proxies, proxies see nodes, nodes see nodes.
          This is the default mode already implemented.

  Routed: Two-step cross-attention only. No direct node-node attention.
          Step 1: proxy queries attend to node keys/values → updated proxies.
          Step 2: node queries attend to proxy keys/values → updated nodes.
          Reuses the frozen layer's projection weights.

  Hybrid: Full node-node self-attention + bidirectional cross-attention with proxies.
          Two separate attention operations per layer, combined via residual.
          Op 1: Self-attention on nodes only.
          Op 2: Cross-attention between nodes and proxies (bidirectional).

All three functions have the same signature as forward_gps_layer_with_proxies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


def _cross_attention(attn_module, queries, keys_values, q_mask=None, kv_mask=None):
    """
    Cross-attention using an existing GlobalSelfAttention module's weights.

    Args:
        attn_module: GlobalSelfAttention instance (provides q/k/v/out projections).
        queries: (B, Nq, d)
        keys_values: (B, Nkv, d)
        q_mask: (B, Nq) boolean — True for real query positions.
        kv_mask: (B, Nkv) boolean — True for real key/value positions.

    Returns:
        out: (B, Nq, d) attended output for queries.
        attn_weights: (B, H, Nq, Nkv)
    """
    B, Nq, d = queries.shape
    Nkv = keys_values.shape[1]
    H = attn_module.num_heads
    hd = attn_module.head_dim

    q = attn_module.q_proj(queries).reshape(B, Nq, H, hd).transpose(1, 2)   # (B, H, Nq, hd)
    k = attn_module.k_proj(keys_values).reshape(B, Nkv, H, hd).transpose(1, 2)
    v = attn_module.v_proj(keys_values).reshape(B, Nkv, H, hd).transpose(1, 2)

    attn = torch.matmul(q, k.transpose(-2, -1)) * attn_module.scale  # (B, H, Nq, Nkv)

    if kv_mask is not None:
        key_mask = kv_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Nkv)
        attn = attn.masked_fill(~key_mask, float('-inf'))

    attn_weights = F.softmax(attn, dim=-1).nan_to_num(0.0)
    attn_weights = attn_module.attn_dropout(attn_weights)

    out = torch.matmul(attn_weights, v)  # (B, H, Nq, hd)
    out = out.transpose(1, 2).reshape(B, Nq, d)
    out = attn_module.out_proj(out)

    return out, attn_weights


# =========================================================================
# Mode: FULL (default — same as forward_gps_layer_with_proxies)
# =========================================================================

def forward_layer_full(layer, x_sparse_nodes, edge_index, edge_attr,
                       node_batch, proxy_embs, num_graphs):
    """Full mode: standard self-attention on [nodes | proxies]."""
    B = num_graphs
    M = proxy_embs.shape[1]

    # Local MPNN on nodes only
    local_out = layer.local_model(layer.local_norm(x_sparse_nodes), edge_index, edge_attr, node_batch)
    x_sparse_nodes = x_sparse_nodes + layer.dropout(local_out)

    # Dense combined
    dense_nodes, node_mask = to_dense_batch(x_sparse_nodes, node_batch)
    N_max = dense_nodes.shape[1]
    dense_combined = torch.cat([dense_nodes, proxy_embs], dim=1)
    proxy_mask = torch.ones(B, M, dtype=torch.bool, device=node_mask.device)
    combined_mask = torch.cat([node_mask, proxy_mask], dim=1)

    # Self-attention on combined
    attn_input = layer.global_norm(dense_combined)
    attn_out, attn_weights = layer.global_attn(attn_input, combined_mask)
    dense_combined = dense_combined + layer.dropout(attn_out)

    # FFN
    ffn_out = layer.ffn(layer.ffn_norm(dense_combined))
    dense_combined = dense_combined + ffn_out

    # Separate
    dense_nodes_out = dense_combined[:, :N_max, :]
    proxy_embs_out = dense_combined[:, N_max:, :]
    x_sparse_nodes_out = dense_nodes_out[node_mask]

    return x_sparse_nodes_out, proxy_embs_out, attn_weights


# =========================================================================
# Mode: ROUTED — node→proxy then proxy→node cross-attention only
# =========================================================================

def forward_layer_routed(layer, x_sparse_nodes, edge_index, edge_attr,
                         node_batch, proxy_embs, num_graphs):
    """
    Routed mode: two-step cross-attention only, no direct node-node attention.

    Step 1: proxies (queries) attend to nodes (keys/values) → updated proxies.
    Step 2: nodes (queries) attend to updated proxies (keys/values) → updated nodes.
    """
    B = num_graphs
    M = proxy_embs.shape[1]

    # Local MPNN on nodes only
    local_out = layer.local_model(layer.local_norm(x_sparse_nodes), edge_index, edge_attr, node_batch)
    x_sparse_nodes = x_sparse_nodes + layer.dropout(local_out)

    # Convert to dense
    dense_nodes, node_mask = to_dense_batch(x_sparse_nodes, node_batch)
    N_max = dense_nodes.shape[1]
    proxy_mask = torch.ones(B, M, dtype=torch.bool, device=node_mask.device)

    # Apply LayerNorm before attention
    normed_nodes = layer.global_norm(dense_nodes)
    normed_proxies = layer.global_norm(proxy_embs)

    # Step 1: proxies attend to nodes → updated proxies
    proxy_update, attn_p2n = _cross_attention(
        layer.global_attn, normed_proxies, normed_nodes,
        q_mask=proxy_mask, kv_mask=node_mask
    )
    proxy_embs = proxy_embs + layer.dropout(proxy_update)

    # Step 2: nodes attend to (updated) proxies → updated nodes
    normed_proxies_updated = layer.global_norm(proxy_embs)
    node_update, attn_n2p = _cross_attention(
        layer.global_attn, normed_nodes, normed_proxies_updated,
        q_mask=node_mask, kv_mask=proxy_mask
    )
    dense_nodes = dense_nodes + layer.dropout(node_update)

    # FFN on both (separately)
    dense_nodes = dense_nodes + layer.ffn(layer.ffn_norm(dense_nodes))
    proxy_embs = proxy_embs + layer.ffn(layer.ffn_norm(proxy_embs))

    # Build a combined attention weight matrix for diagnostics
    # Shape: (B, H, N_max+M, N_max+M) — fill in cross-attention blocks
    H = layer.global_attn.num_heads
    combined_len = N_max + M
    attn_combined = torch.zeros(B, H, combined_len, combined_len, device=dense_nodes.device)
    # node→proxy block (rows=nodes, cols=proxies)
    attn_combined[:, :, :N_max, N_max:] = attn_n2p
    # proxy→node block (rows=proxies, cols=nodes)
    attn_combined[:, :, N_max:, :N_max] = attn_p2n

    # Back to sparse
    x_sparse_nodes_out = dense_nodes[node_mask]

    return x_sparse_nodes_out, proxy_embs, attn_combined


# =========================================================================
# Mode: HYBRID — full node-node self-attention + cross-attention with proxies
# =========================================================================

def forward_layer_hybrid(layer, x_sparse_nodes, edge_index, edge_attr,
                         node_batch, proxy_embs, num_graphs):
    """
    Hybrid mode: full node-node self-attention + bidirectional cross-attention.

    Op 1: Standard self-attention on nodes only (preserves original behavior).
    Op 2: Bidirectional cross-attention between nodes and proxies.
    Results combined via residual connections.
    """
    B = num_graphs
    M = proxy_embs.shape[1]

    # Local MPNN on nodes only
    local_out = layer.local_model(layer.local_norm(x_sparse_nodes), edge_index, edge_attr, node_batch)
    x_sparse_nodes = x_sparse_nodes + layer.dropout(local_out)

    # Convert to dense
    dense_nodes, node_mask = to_dense_batch(x_sparse_nodes, node_batch)
    N_max = dense_nodes.shape[1]
    proxy_mask = torch.ones(B, M, dtype=torch.bool, device=node_mask.device)

    # Op 1: Self-attention on nodes only
    normed_nodes = layer.global_norm(dense_nodes)
    self_attn_out, self_attn_weights = layer.global_attn(normed_nodes, node_mask)
    dense_nodes = dense_nodes + layer.dropout(self_attn_out)

    # Op 2a: Proxies attend to nodes
    normed_nodes_2 = layer.global_norm(dense_nodes)
    normed_proxies = layer.global_norm(proxy_embs)
    proxy_update, attn_p2n = _cross_attention(
        layer.global_attn, normed_proxies, normed_nodes_2,
        q_mask=proxy_mask, kv_mask=node_mask
    )
    proxy_embs = proxy_embs + layer.dropout(proxy_update)

    # Op 2b: Nodes attend to updated proxies
    normed_proxies_2 = layer.global_norm(proxy_embs)
    node_cross_update, attn_n2p = _cross_attention(
        layer.global_attn, layer.global_norm(dense_nodes), normed_proxies_2,
        q_mask=node_mask, kv_mask=proxy_mask
    )
    dense_nodes = dense_nodes + layer.dropout(node_cross_update)

    # FFN on both
    dense_nodes = dense_nodes + layer.ffn(layer.ffn_norm(dense_nodes))
    proxy_embs = proxy_embs + layer.ffn(layer.ffn_norm(proxy_embs))

    # Build combined attention matrix for diagnostics
    H = layer.global_attn.num_heads
    combined_len = N_max + M
    attn_combined = torch.zeros(B, H, combined_len, combined_len, device=dense_nodes.device)
    attn_combined[:, :, :N_max, :N_max] = self_attn_weights
    attn_combined[:, :, :N_max, N_max:] = attn_n2p
    attn_combined[:, :, N_max:, :N_max] = attn_p2n

    x_sparse_nodes_out = dense_nodes[node_mask]

    return x_sparse_nodes_out, proxy_embs, attn_combined


# =========================================================================
# Dispatch
# =========================================================================

ROUTING_MODES = {
    'full': forward_layer_full,
    'routed': forward_layer_routed,
    'hybrid': forward_layer_hybrid,
}


def get_layer_forward_fn(routing_mode):
    """Get the appropriate layer forward function for the given routing mode."""
    if routing_mode not in ROUTING_MODES:
        raise ValueError(f"Unknown routing mode '{routing_mode}'. "
                         f"Choose from: {list(ROUTING_MODES.keys())}")
    return ROUTING_MODES[routing_mode]
