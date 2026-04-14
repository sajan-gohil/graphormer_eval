import copy
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


# ================================================================
# PMA GENERATOR (Per-Graph Query Cross-Attention)
# ================================================================

class PMAGenerator(BaseGenerator):
    """
    Pooling by Multihead Attention with per-graph query generation.

    Step 1: Generate M diverse per-graph query seeds (detached) via FPS or soft k-means.
    Step 2: Cross-attention from queries to node embeddings (learnable).
    Step 3: FFN per cross-attention layer.

    Gradients flow through cross-attention parameters, not through the seed selection.
    """
    def __init__(self, num_proxies, input_dim, num_heads=4, num_layers=2,
                 dropout=0.2, query_mode="farthest_point"):
        super().__init__(num_proxies, input_dim)
        self.query_mode = query_mode

        self.cross_attn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_attn_layers.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(input_dim),
                "norm_kv": nn.LayerNorm(input_dim),
                "cross_attn": nn.MultiheadAttention(
                    input_dim, num_heads, dropout=dropout, batch_first=True
                ),
                "norm_ff": nn.LayerNorm(input_dim),
                "ffn": nn.Sequential(
                    nn.Linear(input_dim, input_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim * 4, input_dim),
                ),
            }))

    def _get_query_seeds(self, node_embeddings, mask):
        """Per-graph diverse query seeds (detached). Returns (B, M, d)."""
        B, N, d = node_embeddings.shape
        M = self.num_proxies

        with torch.no_grad():
            seeds = []
            for b in range(B):
                valid = node_embeddings[b][mask[b]]  # (n_valid, d)
                n_valid = valid.shape[0]
                if n_valid == 0:
                    seeds.append(torch.zeros(M, d, device=valid.device))
                    continue
                if n_valid <= M:
                    idx = torch.arange(n_valid, device=valid.device)
                    idx = idx.repeat(M // n_valid + 1)[:M]
                    seeds.append(valid[idx])
                    continue
                if self.query_mode == "farthest_point":
                    idx = [torch.randint(n_valid, (1,)).item()]
                    for _ in range(M - 1):
                        dists = torch.cdist(valid[idx], valid).min(dim=0).values
                        idx.append(dists.argmax().item())
                    idx = torch.tensor(idx, device=valid.device)
                    seeds.append(valid[idx])
                elif self.query_mode == "soft_kmeans":
                    seeds.append(self._soft_kmeans(valid, M))
                else:
                    raise ValueError(f"Unknown query_mode: {self.query_mode}")
            return torch.stack(seeds).detach()  # (B, M, d)

    @staticmethod
    def _soft_kmeans(X, K, n_iters=3, temp=1.0):
        n = X.shape[0]
        if n <= K:
            idx = torch.arange(n, device=X.device).repeat(K // n + 1)[:K]
            return X[idx]
        idx = [torch.randint(n, (1,)).item()]
        for _ in range(K - 1):
            dists = torch.cdist(X[idx], X).min(dim=0).values
            idx.append(dists.argmax().item())
        centers = X[torch.tensor(idx, device=X.device)]
        for _ in range(n_iters):
            dists = torch.cdist(X, centers)
            weights = F.softmax(-dists / temp, dim=1)
            centers = (weights.T @ X) / (weights.sum(0, keepdim=True).T + 1e-8)
        return centers

    def forward(self, node_embeddings, mask, targets=None, **kwargs):
        queries = self._get_query_seeds(node_embeddings, mask)  # (B, M, d)
        key_padding_mask = ~mask  # True = padding for MHA
        x = queries
        for layer in self.cross_attn_layers:
            q_normed = layer["norm_q"](x)
            kv_normed = layer["norm_kv"](node_embeddings)
            attn_out, _ = layer["cross_attn"](
                q_normed, kv_normed, kv_normed,
                key_padding_mask=key_padding_mask
            )
            x = x + attn_out
            x = x + layer["ffn"](layer["norm_ff"](x))
        return x, None  # No aux_loss — task loss is the only signal


# ================================================================
# GRAPH COARSENING GENERATOR
# ================================================================

class GraphCoarseningGenerator(BaseGenerator):
    """
    Generates proxies via learnt graph coarsening (DiffPool/MinCutPool style).

    Uses a GNN to compute soft assignment of N nodes to M clusters,
    then aggregates node features per cluster to produce proxy embeddings.
    Optionally applies orthogonality regularization to encourage non-degenerate clusters.

    Input: flat node embeddings (total_N, d) + edge_index + batch_vec
    Output: (B, M, d) proxy embeddings
    """
    def __init__(self, num_proxies, input_dim, gnn_layers=2,
                 gnn_type="GIN", dropout=0.2, reg_type="mincut",
                 reg_weight=0.1, num_refine_layers=1, num_heads=4):
        super().__init__(num_proxies, input_dim)
        self.reg_type = reg_type
        self.reg_weight = reg_weight

        # GNN layers for assignment logits
        self.assign_gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            if gnn_type == "GIN":
                from torch_geometric.nn import GINConv
                gin_nn = nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.ReLU(),
                    nn.Linear(input_dim, input_dim),
                )
                self.assign_gnn_layers.append(GINConv(gin_nn))
            elif gnn_type == "GCN":
                from torch_geometric.nn import GCNConv
                self.assign_gnn_layers.append(GCNConv(input_dim, input_dim))
            else:
                raise ValueError(f"GraphCoarseningGenerator: unsupported gnn_type={gnn_type}")

        self.assign_norms = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(gnn_layers)
        ])
        self.assign_proj = nn.Linear(input_dim, num_proxies)

        # Optional self-attention refinement among proxies
        self.refinement_layers = nn.ModuleList()
        for _ in range(num_refine_layers):
            self.refinement_layers.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(input_dim),
                "attn": nn.MultiheadAttention(
                    input_dim, num_heads, dropout=dropout, batch_first=True
                ),
                "norm2": nn.LayerNorm(input_dim),
                "ffn": nn.Sequential(
                    nn.Linear(input_dim, input_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(input_dim * 4, input_dim),
                ),
            }))

    def forward(self, node_embeddings, mask, targets=None, **kwargs):
        """
        Args:
            node_embeddings: (total_N, d) FLAT node embeddings (not dense-batched).
            mask: ignored (GNN uses batch_vec instead).
            targets: optional (B, M, d) — unused; aux_loss is orthogonality only.
            **kwargs: must contain 'edge_index' and 'batch_vec'.
        Returns:
            proxy_embeddings: (B, M, d)
            aux_loss: orthogonality regularization loss or None
        """
        edge_index = kwargs["edge_index"]
        batch_vec = kwargs["batch_vec"]
        num_graphs = int(batch_vec.max().item()) + 1

        # GNN for assignment
        h = node_embeddings
        for gnn_layer, norm in zip(self.assign_gnn_layers, self.assign_norms):
            h = norm(h + F.relu(gnn_layer(h, edge_index)))

        assign_logits = self.assign_proj(h)       # (total_N, M)
        S = F.softmax(assign_logits, dim=-1)       # (total_N, M)

        # Aggregate per cluster per graph
        proxies_list = []
        for g in range(num_graphs):
            g_mask = (batch_vec == g)
            S_g = S[g_mask]                                        # (n_g, M)
            X_g = node_embeddings[g_mask]                          # (n_g, d)
            S_g_norm = S_g / (S_g.sum(dim=0, keepdim=True) + 1e-8)
            proxies_list.append(S_g_norm.T @ X_g)                  # (M, d)

        proxy_embeddings = torch.stack(proxies_list)  # (B, M, d)

        # Refinement layers
        x = proxy_embeddings
        for layer in self.refinement_layers:
            normed = layer["norm1"](x)
            attn_out, _ = layer["attn"](normed, normed, normed)
            x = x + attn_out
            x = x + layer["ffn"](layer["norm2"](x))
        proxy_embeddings = x

        # Orthogonality regularization
        aux_loss = None
        if self.reg_type == "mincut" and self.reg_weight > 0:
            ortho_losses = []
            for g in range(num_graphs):
                g_mask = (batch_vec == g)
                S_g = S[g_mask]
                StS = (S_g.T @ S_g) / S_g.shape[0]
                I_M = torch.eye(self.num_proxies, device=S_g.device) / self.num_proxies
                ortho_losses.append(torch.norm(StS - I_M))
            aux_loss = self.reg_weight * torch.stack(ortho_losses).mean()

        return proxy_embeddings, aux_loss


# ================================================================
# GNN POOLING GENERATOR
# ================================================================

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
        self.gnn_type = gnn_type

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

    def _prepare_gine_edge_attr(self, edge_attr, edge_index, ref_tensor):
        """Convert/project edge attrs to float (E, input_dim) for GINEConv."""
        num_edges = edge_index.size(1)

        if edge_attr is None:
            return torch.zeros(
                num_edges,
                self.input_dim,
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        edge_attr = edge_attr.to(device=ref_tensor.device)

        if edge_attr.shape[-1] == self.input_dim and edge_attr.is_floating_point():
            return edge_attr.to(dtype=ref_tensor.dtype)

        edge_attr = edge_attr.to(dtype=ref_tensor.dtype)
        feat_dim = edge_attr.shape[-1]
        if feat_dim < self.input_dim:
            # Zero-pad feature width to match GINE edge_dim.
            edge_attr = F.pad(edge_attr, (0, self.input_dim - feat_dim))
        elif feat_dim > self.input_dim:
            # Truncate extra channels if edge feature width exceeds hidden dim.
            edge_attr = edge_attr[:, :self.input_dim]
        return edge_attr

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

        if self.gnn_type == "GINE":
            edge_attr = self._prepare_gine_edge_attr(edge_attr, edge_index, node_embeddings)

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


# ================================================================
# CROSS-ATTENTION ROUTER (N→M→N Hyperedge Routing)
# ================================================================

class _CrossAttentionRouterLayer(nn.Module):
    """Single N→M→N routing layer: proxy gather, proxy self-refine, node readback."""
    def __init__(self, hidden_dim, num_heads, dropout, use_proxy_self_attn):
        super().__init__()
        self.use_proxy_self_attn = use_proxy_self_attn

        # Step 1: Proxy Gather — N→M cross-attention (Q=proxy, K/V=node)
        self.gather_norm_q = nn.LayerNorm(hidden_dim)
        self.gather_norm_kv = nn.LayerNorm(hidden_dim)
        self.gather_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gather_norm_ff = nn.LayerNorm(hidden_dim)
        self.gather_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Step 2: Proxy Self-Refine — M×M self-attention (optional)
        if use_proxy_self_attn:
            self.self_norm = nn.LayerNorm(hidden_dim)
            self.self_attn = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.self_norm_ff = nn.LayerNorm(hidden_dim)
            self.self_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )

        # Step 3: Node Readback — M→N cross-attention (Q=node, K/V=proxy)
        self.readback_norm_q = nn.LayerNorm(hidden_dim)
        self.readback_norm_kv = nn.LayerNorm(hidden_dim)
        self.readback_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.readback_norm_ff = nn.LayerNorm(hidden_dim)
        self.readback_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, node_emb, proxy_emb, node_key_pad_mask):
        """
        Args:
            node_emb: (B, N, d)
            proxy_emb: (B, M, d)
            node_key_pad_mask: (B, N) — True = padding (for MHA key_padding_mask)
        Returns:
            node_out: (B, N, d)
            proxy_out: (B, M, d)
        """
        # Step 1: Proxy Gather (N→M)
        q = self.gather_norm_q(proxy_emb)
        kv = self.gather_norm_kv(node_emb)
        attn_out, _ = self.gather_cross_attn(
            q, kv, kv, key_padding_mask=node_key_pad_mask
        )
        proxy_emb = proxy_emb + attn_out
        proxy_emb = proxy_emb + self.gather_ffn(self.gather_norm_ff(proxy_emb))

        # Step 2: Proxy Self-Refine (M×M)
        if self.use_proxy_self_attn:
            normed = self.self_norm(proxy_emb)
            attn_out, _ = self.self_attn(normed, normed, normed)
            proxy_emb = proxy_emb + attn_out
            proxy_emb = proxy_emb + self.self_ffn(self.self_norm_ff(proxy_emb))

        # Step 3: Node Readback (M→N)
        q = self.readback_norm_q(node_emb)
        kv = self.readback_norm_kv(proxy_emb)
        attn_out, _ = self.readback_cross_attn(q, kv, kv)
        node_emb = node_emb + attn_out
        node_emb = node_emb + self.readback_ffn(self.readback_norm_ff(node_emb))

        return node_emb, proxy_emb


class CrossAttentionRouter(nn.Module):
    """
    N→M→N cross-attention routing using M proxies as hyperedges.

    Takes M proxy embeddings from any generator and N node embeddings,
    performs bidirectional cross-attention routing, and returns N refined
    node embeddings. The M proxies never enter the main transformer.

    Args:
        hidden_dim: embedding dimension d.
        num_heads: attention heads for cross- and self-attention.
        num_cross_layers: number of N→M→N routing iterations.
        dropout: dropout rate.
        use_proxy_self_attn: if True, include M×M self-attention in each layer.
    """
    def __init__(self, hidden_dim, num_heads=8, num_cross_layers=2,
                 dropout=0.2, use_proxy_self_attn=True):
        super().__init__()
        self.layers = nn.ModuleList([
            _CrossAttentionRouterLayer(
                hidden_dim, num_heads, dropout, use_proxy_self_attn
            )
            for _ in range(num_cross_layers)
        ])

    def forward(self, node_embeddings, proxy_embeddings, node_mask):
        """
        Args:
            node_embeddings: (B, N, d)
            proxy_embeddings: (B, M, d)
            node_mask: (B, N) boolean — True = real node, False = padding
        Returns:
            refined_nodes: (B, N, d) — proxy-informed node embeddings
        """
        # MHA expects key_padding_mask where True = ignore
        key_pad_mask = ~node_mask  # (B, N)

        node_emb = node_embeddings
        proxy_emb = proxy_embeddings
        for layer in self.layers:
            node_emb, proxy_emb = layer(node_emb, proxy_emb, key_pad_mask)

        return node_emb


# ================================================================
# MULTI-POINT PROXY WRAPPER
# ================================================================

class MultiPointProxyWrapper(nn.Module):
    """
    Manages proxy generation and cross-attention routing at multiple
    insertion points within a transformer stack.

    At each insertion point the wrapper generates fresh proxies from the
    current node representations (Option A: shared generator) or from a
    dedicated per-point generator (Option B: separate generators), then
    either routes via N->M->N cross-attention or concatenates M proxy
    tokens to the sequence.

    Args:
        generator: BaseGenerator instance (used directly if shared, deep-copied
                   per insertion point if ``separate_generators`` is True).
        router: CrossAttentionRouter instance for N->M->N mode, or ``None``
                for N+M concatenation mode.  Shared by default; deep-copied
                per insertion point when ``separate_routers`` is True.
        insertion_layers: sorted list of ints — before which transformer
                          layers to inject a proxy block (0-indexed).
        separate_generators: if True, create independent generator copies
                             per insertion point (Option B).
        separate_routers: if True, create independent router copies per
                          insertion point.
        aux_loss_decay: geometric decay factor applied to the aux_loss of
                        successive insertion points (1.0 = no decay).
    """

    def __init__(self, generator, router, insertion_layers,
                 separate_generators=False, separate_routers=False,
                 aux_loss_decay=1.0):
        super().__init__()
        self.insertion_layers = sorted(insertion_layers)
        self.aux_loss_decay = aux_loss_decay
        K = len(insertion_layers)

        # Generators
        if separate_generators:
            self.generators = nn.ModuleList([
                copy.deepcopy(generator) for _ in range(K)
            ])
        else:
            self.generators = nn.ModuleList([generator])

        # Routers (None when using N+M concat mode)
        if router is not None:
            if separate_routers:
                self.routers = nn.ModuleList([
                    copy.deepcopy(router) for _ in range(K)
                ])
            else:
                self.routers = nn.ModuleList([router])
        else:
            self.routers = None

    def get_generator(self, point_idx):
        if len(self.generators) == 1:
            return self.generators[0]
        return self.generators[point_idx]

    def get_router(self, point_idx):
        if self.routers is None:
            return None
        if len(self.routers) == 1:
            return self.routers[0]
        return self.routers[point_idx]

    def run_proxy_block(self, node_emb, mask, point_idx, **gen_kwargs):
        """Run one proxy-generation + routing pass.

        Args:
            node_emb: (B, N_current, d) current node / token embeddings.
            mask: (B, N_current) boolean mask (True = real token).
            point_idx: index into ``self.insertion_layers``.
            **gen_kwargs: extra keyword arguments forwarded to the generator.
                For GNN-based generators (GraphCoarseningGenerator,
                GNNPoolingGenerator) this must include 'edge_index' and
                'batch_vec', and optionally 'edge_attr'.  Those generators
                also expect *flat* node embeddings, so the caller is
                responsible for flattening dense (B, N, d) back to
                (total_N, d) before passing to this method (or the
                generator handles the mask internally).

        Returns:
            out_tokens: (B, N', d) — either refined nodes (cross-attn,
                        N' = N_current) or augmented tokens (concat,
                        N' = N_current + M).
            out_mask: (B, N') — updated mask.
            aux_loss: scalar auxiliary loss from the generator.
        """
        gen = self.get_generator(point_idx)
        if gen_kwargs:
            # GNN-based generators expect flat (total_N, d) node embeddings.
            # Flatten from dense (B, N, d) using the boolean mask.
            flat_node_emb = node_emb[mask]  # (total_N, d)
            proxy_emb, aux_loss = gen(flat_node_emb, mask=None, **gen_kwargs)
        else:
            proxy_emb, aux_loss = gen(node_emb, mask)

        router = self.get_router(point_idx)
        if router is not None:
            # Cross-attention mode: N->M->N, token count unchanged
            refined = router(node_emb, proxy_emb, mask)
            return refined, mask, aux_loss
        else:
            # Concat mode: append M proxy tokens
            B, M, _d = proxy_emb.shape
            aug_tokens = torch.cat([node_emb, proxy_emb], dim=1)
            aug_mask = torch.cat([
                mask,
                torch.ones(B, M, dtype=torch.bool, device=mask.device),
            ], dim=1)
            return aug_tokens, aug_mask, aux_loss
