import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from mmd import mmd_squared


# ================================================================
# BASE GENERATOR
# ================================================================

class BaseGenerator(ABC, nn.Module):
    """
    Common interface for all proxy generators.

    Input:  node_embeddings (B, N, d), mask (B, N), optional targets (B, M, d), **kwargs
    Output: proxy_embeddings (B, M, d), aux_loss (scalar or None)
    """
    def __init__(self, num_proxies, hidden_dim):
        super().__init__()
        self.num_proxies = num_proxies
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(self, node_embeddings, mask, targets=None, **kwargs):
        """
        Args:
            node_embeddings: (B, N, d) dense batched node embeddings.
            mask: (B, N) boolean mask (True = real node, False = padding).
            targets: optional (B, M, d) target proxy embeddings for training loss.
            **kwargs: additional inputs (e.g. edge_index, batch_vec for GNN).
        Returns:
            proxy_embeddings: (B, M, d)
            aux_loss: scalar tensor or None
        """
        ...

    def generate(self, node_embeddings, mask, **kwargs):
        """Inference-only: return just proxy embeddings, no loss."""
        proxies, _ = self.forward(node_embeddings, mask, targets=None, **kwargs)
        return proxies


# ================================================================
# SCORE-BASED GENERATOR
# ================================================================

class ScoreBasedGenerator(BaseGenerator):
    """
    Deterministic generator: node scoring + softmax aggregation + self-attention refinement.

    Step 1: Value projection V = Linear(X)
    Step 2: Score MLP: S = MLP(X) -> (B, N, M)
    Step 3: Masked softmax over node dim -> attention weights A (B, N, M)
    Step 4: Aggregate: B0 = A^T @ V -> (B, M, d)
    Steps 5-6: L refinement layers (self-attention + FFN among M proxies)
    """
    def __init__(self, num_proxies, input_dim, hidden_dim=128, num_layers=2,
                 num_heads=4, dropout=0.2):
        super().__init__(num_proxies, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_proxies),
        )

        # Refinement layers: self-attention among M proxies
        self.refinement_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.refinement_layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(input_dim),
                "attn": nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True),
                "norm2": nn.LayerNorm(input_dim),
                "ffn": nn.Sequential(
                    nn.Linear(input_dim, input_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim * 4, input_dim),
                ),
            }))

    def forward(self, node_embeddings, mask, targets=None, **kwargs):
        # Step 1: Value projection
        V = self.value_proj(node_embeddings)  # (B, N, d)

        # Step 2: Score MLP
        S = self.score_mlp(node_embeddings)  # (B, N, M)

        # Step 3: Masked softmax over node dim
        S = S.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        A = F.softmax(S, dim=1)  # (B, N, M)
        A = torch.nan_to_num(A, nan=0.0)

        # Step 4: Weighted aggregation
        B0 = torch.bmm(A.transpose(1, 2), V)  # (B, M, d)

        # Steps 5-6: Refinement layers
        x = B0
        for layer in self.refinement_layers:
            # Self-attention
            normed = layer["norm1"](x)
            attn_out, _ = layer["attn"](normed, normed, normed)
            x = x + attn_out
            # FFN
            x = x + layer["ffn"](layer["norm2"](x))

        proxy_embeddings = x  # (B, M, d) — no final activation/norm

        # Compute aux_loss if targets provided
        aux_loss = None
        if targets is not None:
            # Per-graph MMD, averaged over batch
            batch_size = proxy_embeddings.shape[0]
            mmd_losses = []
            for i in range(batch_size):
                mmd_losses.append(mmd_squared(proxy_embeddings[i], targets[i]))
            aux_loss = torch.stack(mmd_losses).mean()

        return proxy_embeddings, aux_loss


# ================================================================
# FLOW MATCHING GENERATOR (Pipeline A only)
# ================================================================

class _DenoiserLayer(nn.Module):
    """Single denoiser layer: time conditioning + cross-attention + FFN."""
    def __init__(self, denoiser_dim, num_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            denoiser_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(denoiser_dim)
        self.norm2 = nn.LayerNorm(denoiser_dim)
        self.ffn = nn.Sequential(
            nn.Linear(denoiser_dim, denoiser_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(denoiser_dim * 4, denoiser_dim),
        )

    def forward(self, x_t, t_emb, node_emb, node_key_padding_mask):
        """
        Args:
            x_t: (B, M, denoiser_dim) — noisy proxy tokens
            t_emb: (B, 1, denoiser_dim) — time embedding
            node_emb: (B, N, denoiser_dim) — conditioning node embeddings
            node_key_padding_mask: (B, N) — True = padding (inverted for MHA)
        """
        # Additive time conditioning
        x_t = x_t + t_emb

        # Cross-attention: Q=x_t, K=node_emb, V=node_emb
        normed = self.norm1(x_t)
        attn_out, _ = self.cross_attn(
            normed, node_emb, node_emb,
            key_padding_mask=node_key_padding_mask,
        )
        x_t = x_t + attn_out

        # FFN
        x_t = x_t + self.ffn(self.norm2(x_t))
        return x_t


class FlowMatchingGenerator(BaseGenerator):
    """
    Conditional flow matching generator (Pipeline A only — needs target embeddings).

    Learns a vector field v_theta that transports N(0,I) to target proxy distribution.
    Training: CFM loss = MSE(v_theta(x_t, t | X), u_t) where u_t = targets - x_0.
    Inference: Euler integration from z_0 ~ N(0,I) over euler_steps.
    """
    def __init__(self, num_proxies, node_dim, denoiser_dim=128,
                 denoiser_layers=4, denoiser_heads=8, dropout=0.2,
                 euler_steps=1):
        super().__init__(num_proxies, node_dim)
        self.denoiser_dim = denoiser_dim
        self.euler_steps = euler_steps

        # Projections if node_dim != denoiser_dim
        self.input_proj = nn.Linear(node_dim, denoiser_dim) if node_dim != denoiser_dim else nn.Identity()
        self.output_proj = nn.Linear(denoiser_dim, node_dim) if node_dim != denoiser_dim else nn.Identity()
        self.node_proj = nn.Linear(node_dim, denoiser_dim) if node_dim != denoiser_dim else nn.Identity()

        # Sinusoidal time embedding -> MLP
        sin_dim = denoiser_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(sin_dim, denoiser_dim),
            nn.GELU(),
            nn.Linear(denoiser_dim, denoiser_dim),
        )

        # Denoiser layers
        self.layers = nn.ModuleList([
            _DenoiserLayer(denoiser_dim, denoiser_heads, dropout)
            for _ in range(denoiser_layers)
        ])

    def _sinusoidal_embedding(self, t, dim):
        """Sinusoidal positional encoding for scalar time t. t: (B,)"""
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)

    def _denoise(self, x_t, t, node_embeddings, mask):
        """
        Run denoiser to predict vector field.

        Args:
            x_t: (B, M, node_dim) — noisy proxies
            t: (B,) — time values
            node_embeddings: (B, N, node_dim)
            mask: (B, N) — True = real node
        Returns:
            v: (B, M, node_dim) — predicted vector field
        """
        # Project to denoiser dim
        x_t_proj = self.input_proj(x_t)  # (B, M, denoiser_dim)
        node_proj = self.node_proj(node_embeddings)  # (B, N, denoiser_dim)

        # Time embedding
        t_emb = self._sinusoidal_embedding(t, self.denoiser_dim)  # (B, denoiser_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)  # (B, 1, denoiser_dim)

        # MHA expects key_padding_mask where True = ignore
        key_padding_mask = ~mask  # (B, N)

        # Run denoiser layers
        h = x_t_proj
        for layer in self.layers:
            h = layer(h, t_emb, node_proj, key_padding_mask)

        # Project back to node_dim
        return self.output_proj(h)  # (B, M, node_dim)

    def forward(self, node_embeddings, mask, targets=None, **kwargs):
        B, N, d = node_embeddings.shape
        M = self.num_proxies
        device = node_embeddings.device

        if targets is not None:
            # Training: CFM objective
            t = torch.rand(B, device=device)  # (B,)
            x_0 = torch.randn(B, M, d, device=device)  # noise
            t_expand = t.view(B, 1, 1)  # (B, 1, 1)
            x_t = (1 - t_expand) * x_0 + t_expand * targets  # interpolation
            u = targets - x_0  # conditional vector field

            v = self._denoise(x_t, t, node_embeddings, mask)
            aux_loss = F.mse_loss(v, u)

            # Also generate proxies via Euler for return value
            with torch.no_grad():
                proxy_embeddings = self._euler_sample(node_embeddings, mask)
        else:
            # Inference: Euler integration
            proxy_embeddings = self._euler_sample(node_embeddings, mask)
            aux_loss = None

        return proxy_embeddings, aux_loss

    def _euler_sample(self, node_embeddings, mask):
        """Generate proxies via Euler integration from t=0 to t=1."""
        B, N, d = node_embeddings.shape
        M = self.num_proxies
        dt = 1.0 / self.euler_steps
        z = torch.randn(B, M, d, device=node_embeddings.device)

        for step in range(self.euler_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=node_embeddings.device)
            v = self._denoise(z, t, node_embeddings, mask)
            z = z + dt * v

        return z  # (B, M, d)

    def forward_differentiable(self, node_embeddings, mask, euler_steps=1):
        """
        Single-step differentiable generation for Stage 4 finetuning.
        Gradients flow through v_theta.
        """
        B, N, d = node_embeddings.shape
        M = self.num_proxies
        z_0 = torch.randn(B, M, d, device=node_embeddings.device)
        dt = 1.0 / euler_steps

        z = z_0
        for step in range(euler_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=node_embeddings.device)
            v = self._denoise(z, t, node_embeddings, mask)
            z = z + dt * v

        return z  # (B, M, d) — fully differentiable


# ================================================================
# GNN POOLING GENERATOR
# ================================================================

class _GNNLayer(nn.Module):
    """Single GNN layer with residual + LayerNorm. Supports GCN/GIN/GINE/GAT."""
    def __init__(self, hidden_dim, gnn_type="GINE", dropout=0.2):
        super().__init__()
        self.gnn_type = gnn_type
        self.dropout = dropout

        if gnn_type == "GCN":
            from torch_geometric.nn import GCNConv
            self.conv = GCNConv(hidden_dim, hidden_dim)
        elif gnn_type == "GIN":
            from torch_geometric.nn import GINConv
            gin_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv = GINConv(gin_nn)
        elif gnn_type == "GINE":
            from torch_geometric.nn import GINEConv
            gine_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv = GINEConv(gine_nn, edge_dim=hidden_dim)
        elif gnn_type == "GAT":
            from torch_geometric.nn import GATConv
            assert hidden_dim % 4 == 0
            self.conv = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True)
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        if self.gnn_type == "GINE":
            h = self.conv(x, edge_index, edge_attr=edge_attr)
        elif self.gnn_type == "GAT":
            h = self.conv(x, edge_index)
        else:
            h = self.conv(x, edge_index)
        h = self.drop(F.relu(h))
        return self.norm(x + h)  # residual + LayerNorm


class GNNPoolingGenerator(BaseGenerator):
    """
    GNN-based proxy generator: multi-hop GNN + multi-scale pooling + shared MLP decode.

    Step 1: K GNN layers collecting H^0...H^K (K+1 hop levels)
    Step 2: Multi-scale pooling (mean/max/std) per hop -> graph descriptor g
    Step 3: Shared MLP with proxy index embeddings to decode M proxies
    """
    def __init__(self, num_proxies, input_dim, gnn_layers=4, gnn_type="GINE",
                 pool_types=("mean", "max", "std"), decode_hidden=256,
                 decode_layers=3, idx_emb_dim=32, dropout=0.2,
                 decode_mode="shared"):
        super().__init__(num_proxies, input_dim)
        self.gnn_layers_count = gnn_layers
        self.pool_types = list(pool_types)
        self.decode_mode = decode_mode
        self.input_dim = input_dim

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            _GNNLayer(input_dim, gnn_type, dropout)
            for _ in range(gnn_layers)
        ])

        # Descriptor dimension: P * (K+1) * d
        P = len(self.pool_types)
        K_plus_1 = gnn_layers + 1
        descriptor_dim = P * K_plus_1 * input_dim

        if decode_mode == "shared":
            # Proxy index embeddings
            self.idx_embeddings = nn.Embedding(num_proxies, idx_emb_dim)

            # Shared decoder MLP
            layers = []
            in_dim = descriptor_dim + idx_emb_dim
            for i in range(decode_layers):
                out_dim = input_dim if i == decode_layers - 1 else decode_hidden
                layers.append(nn.Linear(in_dim, out_dim))
                if i < decode_layers - 1:
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
                in_dim = out_dim
            self.decoder = nn.Sequential(*layers)

        elif decode_mode == "grouped":
            # Grouped decoding: split M into num_groups groups
            self.num_groups = min(8, num_proxies)
            assert num_proxies % self.num_groups == 0
            proxies_per_group = num_proxies // self.num_groups
            self.group_decoders = nn.ModuleList()
            for _ in range(self.num_groups):
                layers = []
                in_dim = descriptor_dim
                for i in range(decode_layers):
                    out_dim = proxies_per_group * input_dim if i == decode_layers - 1 else decode_hidden
                    layers.append(nn.Linear(in_dim, out_dim))
                    if i < decode_layers - 1:
                        layers.append(nn.GELU())
                        layers.append(nn.Dropout(dropout))
                    in_dim = out_dim
                self.group_decoders.append(nn.Sequential(*layers))

    def _pool(self, h, batch_vec, num_graphs):
        """Apply pooling functions to flat node features. Returns (B, P*d)."""
        from torch_geometric.nn import global_mean_pool, global_max_pool
        results = []
        for pool_type in self.pool_types:
            if pool_type == "mean":
                results.append(global_mean_pool(h, batch_vec, size=num_graphs))
            elif pool_type == "max":
                results.append(global_max_pool(h, batch_vec, size=num_graphs))
            elif pool_type == "std":
                # Compute per-graph std
                mean = global_mean_pool(h, batch_vec, size=num_graphs)  # (B, d)
                mean_expanded = mean[batch_vec]  # (total_N, d)
                sq_diff = (h - mean_expanded) ** 2
                var = global_mean_pool(sq_diff, batch_vec, size=num_graphs)
                results.append(torch.sqrt(var + 1e-8))
        return torch.cat(results, dim=-1)  # (B, P*d)

    def forward(self, node_embeddings, mask, targets=None, **kwargs):
        """
        Args:
            node_embeddings: (total_N, d) FLAT pyg node embeddings (not dense batch).
            mask: ignored for GNN (uses batch_vec instead).
            targets: optional (B, M, d) target proxies.
            **kwargs: must contain 'edge_index', 'batch_vec'; optionally 'edge_attr'.
        """
        edge_index = kwargs["edge_index"]
        batch_vec = kwargs["batch_vec"]
        edge_attr = kwargs.get("edge_attr", None)
        num_graphs = int(batch_vec.max().item()) + 1

        # Step 1: Multi-hop GNN, collect H^0...H^K
        hop_representations = [node_embeddings]  # H^0
        h = node_embeddings
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_index, edge_attr)
            hop_representations.append(h)

        # Step 2: Multi-scale pooling per hop level
        pool_results = []
        for h_k in hop_representations:
            pool_results.append(self._pool(h_k, batch_vec, num_graphs))
        # Each pool result is (B, P*d), concatenate across hops
        g = torch.cat(pool_results, dim=-1)  # (B, P*(K+1)*d)

        # Step 3: Decode M proxies
        B = num_graphs
        M = self.num_proxies
        d = self.input_dim

        if self.decode_mode == "shared":
            # g_repeated: (B, M, D), E: (M, idx_emb_dim) -> (B, M, idx_emb_dim)
            g_repeated = g.unsqueeze(1).expand(B, M, -1)  # (B, M, D)
            idx = torch.arange(M, device=g.device)
            E = self.idx_embeddings(idx).unsqueeze(0).expand(B, -1, -1)  # (B, M, idx_emb_dim)
            decoder_input = torch.cat([g_repeated, E], dim=-1)  # (B, M, D+idx_emb_dim)
            proxy_embeddings = self.decoder(decoder_input)  # (B, M, d)

        elif self.decode_mode == "grouped":
            proxies_per_group = M // self.num_groups
            group_outputs = []
            for group_decoder in self.group_decoders:
                out = group_decoder(g)  # (B, proxies_per_group * d)
                out = out.view(B, proxies_per_group, d)
                group_outputs.append(out)
            proxy_embeddings = torch.cat(group_outputs, dim=1)  # (B, M, d)

        # Compute aux_loss if targets provided
        aux_loss = None
        if targets is not None:
            mmd_losses = []
            for i in range(B):
                mmd_losses.append(mmd_squared(proxy_embeddings[i], targets[i]))
            aux_loss = torch.stack(mmd_losses).mean()

        return proxy_embeddings, aux_loss
