import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, scatter
from torch_geometric.nn import global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class NodeEncoder(nn.Module):
    """Atom + bond-aggregated node embeddings with optional Laplacian PE."""
    def __init__(self, hidden_dim=64, lap_pe_dim=0):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim // 2)
        self.bond_encoder = BondEncoder(hidden_dim // 2)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        # Optional Laplacian positional encoding
        self.lap_pe_dim = lap_pe_dim
        if lap_pe_dim > 0:
            self.lap_pe_encoder = nn.Linear(lap_pe_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr, lap_pe=None):
        h = self.atom_encoder(x)
        row = edge_index[0]
        edge_emb = self.bond_encoder(edge_attr)
        edge_aggr = scatter(edge_emb, row, dim=0, dim_size=h.size(0), reduce="add")
        h = torch.cat([h, edge_aggr], dim=-1)
        h = self.proj(h)  # (total_N, d)

        # Add Laplacian positional encoding if available
        if self.lap_pe_dim > 0 and lap_pe is not None:
            h = h + self.lap_pe_encoder(lap_pe)

        return h


class TransformerLayer(nn.Module):
    """Pre-norm transformer layer with configurable dropout and attention masking."""
    def __init__(self, hidden_dim=64, num_heads=8, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.wout = nn.Linear(hidden_dim, hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.res_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask=None):
        B, N, d = x.shape
        normed = self.norm1(x)

        q = self.wq(normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(normed).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            # Mask both key and query positions for padding
            key_pad = (~mask).unsqueeze(1).unsqueeze(2)    # (B, 1, 1, N)
            query_pad = (~mask).unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            attn = attn.masked_fill(key_pad, float("-inf"))
            attn = attn.masked_fill(query_pad, float("-inf"))

        attn_w = F.softmax(attn, dim=-1)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        attn_w = self.attn_drop(attn_w)

        out = (attn_w @ v).transpose(1, 2).reshape(B, N, d)
        x = x + self.res_drop(self.wout(out))
        x = x + self.res_drop(self.ff(self.norm2(x)))
        return x


class GraphTransformer(nn.Module):
    def __init__(self, num_layers=5, num_heads=8, hidden_dim=64,
                 output_dim=10, dropout=0.3, lap_pe_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = NodeEncoder(hidden_dim, lap_pe_dim=lap_pe_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_nodes(self, batch):
        """Flat node embeddings: (total_N, d)."""
        lap_pe = getattr(batch, 'lap_pe', None)
        return self.encoder(batch.x, batch.edge_index, batch.edge_attr,
                            lap_pe=lap_pe)

    def encode_dense(self, batch):
        """Dense batched node embeddings: (B, max_N, d), (B, max_N) mask."""
        h = self.encode_nodes(batch)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, proxy_embeddings=None, precomputed_dense=None,
                readout_scope="nodes_only"):
        """
        Args:
            batch: PyG Batch object.
            proxy_embeddings: optional (B, M, d) proxy tokens to concatenate.
            precomputed_dense: optional (dense_x, dense_mask) tuple to skip encoding.
            readout_scope: "nodes_only" pools over N original nodes,
                           "all_tokens" pools over N+M (requires proxy_embeddings).
        Returns:
            logits: (B, output_dim)
            node_embeddings: (total_N, d) flat node embeddings from encoder
        """
        if precomputed_dense is not None:
            dense_x, dense_mask = precomputed_dense
        else:
            dense_x, dense_mask = self.encode_dense(batch)

        B, max_N, d = dense_x.shape

        if proxy_embeddings is not None:
            M = proxy_embeddings.shape[1]
            dense_x = torch.cat([dense_x, proxy_embeddings], dim=1)
            aug_mask = torch.cat([
                dense_mask,
                torch.ones(B, M, dtype=torch.bool, device=dense_x.device)
            ], dim=1)
        else:
            aug_mask = dense_mask

        for layer in self.layers:
            dense_x = layer(dense_x, aug_mask)

        # Readout: pool and classify
        if readout_scope == "all_tokens" and proxy_embeddings is not None:
            valid_emb = dense_x[aug_mask]
            batch_vec = torch.arange(B, device=dense_x.device).unsqueeze(1).expand_as(aug_mask)[aug_mask]
            pooled = global_mean_pool(valid_emb, batch_vec)
        else:
            orig_x = dense_x[:, :max_N, :]
            node_emb_masked = orig_x[dense_mask]
            # Build pooling indices from the active dense node mask so this
            # works for both full batches and subsampled precomputed_dense.
            try:
                pooled = global_mean_pool(node_emb_masked, batch.batch)
            except:
                batch_vec = (
                    torch.arange(B, device=dense_x.device)
                    .unsqueeze(1)
                    .expand_as(dense_mask)[dense_mask]
                )
                pooled = global_mean_pool(node_emb_masked, batch_vec)

        logits = self.head(pooled)

        # Always return original node embeddings
        orig_x = dense_x[:, :max_N, :]
        node_emb = orig_x[dense_mask]
        return logits, node_emb


# ================================================================
# GRED: Recurrent Distance Filtering (PyTorch port of JAX/Flax)
# ================================================================

class DiagonalLRU(nn.Module):
    """
    Diagonal Linear Recurrent Unit with complex eigenvalues.
    Processes a sequence (far hops → near hops) via diagonal recurrence with
    log-polar parameterized eigenvalues for stable training.
    Uses parallel associative scan for efficiency.

    Args:
        input_dim: dimension of input features (hidden_dim of the model).
        state_dim: dimension of the complex recurrent state.
        r_min, r_max: range for eigenvalue magnitude initialization.
        max_phase: maximum phase for eigenvalue initialization.
        dropout: dropout rate on output.
        act: GLU activation variant ("full-glu" or "half-glu").
    """
    def __init__(self, input_dim, state_dim, r_min=0.0, r_max=1.0,
                 max_phase=6.28, dropout=0.2, act="full-glu"):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.act = act

        # Eigenvalue parameterization (log-polar): lambda = exp(-exp(nu_log) + i*exp(theta_log))
        # Initialize nu_log so that |lambda| is uniform in [r_min, r_max]
        u = torch.rand(state_dim)
        nu_log = torch.log(-0.5 * torch.log(u * (r_max ** 2 - r_min ** 2) + r_min ** 2 + 1e-8))
        self.nu_log = nn.Parameter(nu_log)

        # Initialize theta_log so that phase is uniform in [0, max_phase]
        u2 = torch.rand(state_dim)
        theta_log = torch.log(u2 * max_phase + 1e-8)
        self.theta_log = nn.Parameter(theta_log)

        # Input projection B (complex: B_re + i*B_im)
        self.B_re = nn.Parameter(torch.empty(input_dim, state_dim))
        self.B_im = nn.Parameter(torch.empty(input_dim, state_dim))
        nn.init.trunc_normal_(self.B_re, std=(0.5 / input_dim) ** 0.5)
        nn.init.trunc_normal_(self.B_im, std=(0.5 / input_dim) ** 0.5)

        # Output projection C (complex: C_re + i*C_im)
        self.C_re = nn.Parameter(torch.empty(state_dim, input_dim))
        self.C_im = nn.Parameter(torch.empty(state_dim, input_dim))
        nn.init.normal_(self.C_re, std=(1.0 / state_dim) ** 0.5)
        nn.init.normal_(self.C_im, std=(1.0 / state_dim) ** 0.5)

        # GLU output projections
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        if act == "full-glu":
            self.gate_proj = nn.Linear(input_dim, input_dim)
            self.out_proj = nn.Linear(input_dim, input_dim)
        elif act == "half-glu":
            self.gate_proj = nn.Linear(input_dim, input_dim)
        else:
            raise ValueError(f"Unknown act: {act}")

    def _get_lambda(self):
        """Compute complex eigenvalues from log-polar parameterization."""
        return torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))

    def _get_gamma(self, diag_lambda):
        """Normalization factor: gamma = sqrt(1 - |lambda|^2)."""
        return torch.sqrt(1 - torch.abs(diag_lambda) ** 2 + 1e-8)

    def forward(self, xs):
        """
        Args:
            xs: (B, K+1, N, input_dim) — per-hop aggregated features, ordered far→near.
                Actually reshaped to (B*N, K+1, input_dim) for the LRU.
        Returns:
            out: (B*N, input_dim) — the final hidden state mapped to output dim.
        """
        # xs: (L, input_dim) where L is sequence length (K+1 hops)
        # But we receive (B*N, K+1, d) — process each node's hop sequence

        BN, K_plus_1, d = xs.shape

        normed = self.norm(xs)  # (BN, K+1, d)

        # Complex eigenvalues and input matrix
        diag_lambda = self._get_lambda()  # (state_dim,)
        gamma = self._get_gamma(diag_lambda)  # (state_dim,)
        B_complex = (self.B_re + 1j * self.B_im) * gamma.unsqueeze(0)  # (d, state_dim)

        # Project input: Bu = normed @ B_complex -> (BN, K+1, state_dim) complex
        Bu = torch.einsum("bkd,ds->bks", normed.to(torch.cfloat), B_complex)

        # Parallel scan: associative scan with (lambda, Bu) pairs
        # For efficiency, use the unrolled power-sum form:
        # s_K = sum_{k=0}^{K} lambda^k * Bu_{K-k}
        # Since xs is already far→near, Bu[0] is farthest, Bu[K] is self.
        # The scan reverses: s accumulates from far to near.
        # Use iterative scan (vectorized over BN dimension):
        out = self._parallel_scan(diag_lambda, Bu)  # (BN, state_dim) complex

        # Output projection: C
        C_complex = self.C_re + 1j * self.C_im  # (state_dim, d)
        x = torch.einsum("bs,sd->bd", out, C_complex).real  # (BN, d)

        x = F.gelu(x)
        x = self.dropout(x)

        # GLU activation
        if self.act == "full-glu":
            x = self.out_proj(x) * torch.sigmoid(self.gate_proj(x))
        elif self.act == "half-glu":
            x = x * torch.sigmoid(self.gate_proj(x))

        x = self.dropout(x)

        # Residual: add the first element of the input sequence (nearest hop = self)
        # In GRED, residual is added from inputs[0] which is the target node itself
        return x + xs[:, -1, :]  # xs[:, -1] = hop 0 (self), since ordered far→near

    def _parallel_scan(self, diag_lambda, Bu):
        """
        Associative parallel scan for diagonal LRU.
        Implements reverse scan: s_k = lambda * s_{k-1} + Bu_k
        (scanning from k=0 to K, which is far→near).

        Uses the O(K log K) parallel scan algorithm, but for simplicity
        and correctness, implements the sequential O(K) version which is
        fast enough for K<=40.

        Args:
            diag_lambda: (state_dim,) complex eigenvalues.
            Bu: (BN, K+1, state_dim) complex projected inputs.
        Returns:
            final_state: (BN, state_dim) complex.
        """
        BN, K_plus_1, state_dim = Bu.shape
        # Sequential scan (reverse order in the paper, but our input is already far→near)
        s = torch.zeros(BN, state_dim, device=Bu.device, dtype=Bu.dtype)
        for k in range(K_plus_1):
            s = diag_lambda.unsqueeze(0) * s + Bu[:, k, :]
        return s  # (BN, state_dim)


class GREDLayer(nn.Module):
    """
    Single GRED layer: DeepSets per-hop aggregation → LRU over hops → output.

    Given node features H ∈ (B, N, d) and distance masks (B, K, N, N):
    1. For each target node v and hop k: aggregate neighbors at distance k via MLP.
    2. Feed the hop sequence (far→near) into a diagonal LRU.
    3. GLU output MLP produces updated node features.

    Args:
        hidden_dim: feature dimension d.
        state_dim: LRU complex state dimension.
        expand: FFN expansion factor for the DeepSets MLP.
        r_min, r_max: eigenvalue magnitude range.
        max_phase: max eigenvalue phase.
        dropout: dropout rate.
        act: GLU variant.
    """
    def __init__(self, hidden_dim, state_dim, expand=1, r_min=0.0, r_max=1.0,
                 max_phase=6.28, dropout=0.2, act="full-glu"):
        super().__init__()
        # DeepSets aggregation MLP (applied to aggregated per-hop features)
        self.deepsets_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, expand * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.deepsets_residual = True  # residual on aggregated features

        # LRU over the hop sequence
        self.lru = DiagonalLRU(
            input_dim=hidden_dim,
            state_dim=state_dim,
            r_min=r_min,
            r_max=r_max,
            max_phase=max_phase,
            dropout=dropout,
            act=act,
        )

    def forward(self, h, dist_masks, node_masks=None):
        """
        Args:
            h: (B, N, d) node features.
            dist_masks: (B, K, N, N) float distance masks (1 where d(v,u)==k).
            node_masks: (B, N) boolean — True for real nodes. Optional.
        Returns:
            h_out: (B, N, d) updated node features.
        """
        B, N, d = h.shape
        K = dist_masks.shape[1]

        # Step 1: Per-hop aggregation using distance masks
        # dist_masks[:, k] is (B, N, N): for target v, which nodes u are at distance k
        # Aggregate: x_{v,k} = sum_{u in N_k(v)} h_u = dist_masks[:, k] @ h
        # (B, K, N, N) @ (B, 1, N, d) -> need to batch-matmul per hop

        # Reshape for batched matmul: (B*K, N, N) @ (B*K, N, d)
        # Expand h to (B, K, N, d) then reshape
        dm = dist_masks  # (B, K, N, N)
        h_expanded = h.unsqueeze(1).expand(B, K, N, d)  # (B, K, N, d)
        dm_flat = dm.reshape(B * K, N, N)
        h_flat = h_expanded.reshape(B * K, N, d)
        agg = torch.bmm(dm_flat, h_flat)  # (B*K, N, d)
        agg = agg.reshape(B, K, N, d)  # (B, K, N, d) — per-hop aggregated features

        # Apply DeepSets MLP to aggregated features
        agg_reshaped = agg.reshape(B * K * N, d)
        mlp_out = self.deepsets_mlp(agg_reshaped)
        if self.deepsets_residual:
            mlp_out = mlp_out + agg_reshaped
        mlp_out = mlp_out.reshape(B, K, N, d)  # (B, K, N, d)

        # Step 2: LRU over hop sequence per node
        # Reorder to (B, N, K, d) then reshape to (B*N, K, d)
        # Hops are already ordered 0→K-1 (near→far), reverse to far→near for LRU
        hop_seq = mlp_out.permute(0, 2, 1, 3)  # (B, N, K, d)
        hop_seq = hop_seq.flip(dims=[2])  # reverse: far→near
        hop_seq_flat = hop_seq.reshape(B * N, K, d)

        # Run LRU
        h_out_flat = self.lru(hop_seq_flat)  # (B*N, d)
        h_out = h_out_flat.reshape(B, N, d)

        # Mask padding nodes
        if node_masks is not None:
            h_out = h_out * node_masks.unsqueeze(-1).float()

        return h_out


class GREDEncoder(nn.Module):
    """
    Full GRED encoder: node embedding + stack of GREDLayers + readout.

    Replaces the Transformer as the backbone for encoding node features.
    Produces per-node embeddings that are topology-aware via distance filtering.

    Args:
        hidden_dim: model hidden dimension.
        state_dim: LRU state dimension (complex).
        num_layers: number of GRED layers.
        expand: FFN expansion factor.
        r_min, r_max: eigenvalue magnitude init range.
        max_phase: eigenvalue max phase.
        dropout: dropout rate.
        act: GLU variant ("full-glu" or "half-glu").
        output_dim: number of classes (for standalone classification head).
    """
    def __init__(self, hidden_dim=88, state_dim=88, num_layers=8, expand=1,
                 r_min=0.0, r_max=1.0, max_phase=6.28, dropout=0.2,
                 act="full-glu", output_dim=10, lap_pe_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = NodeEncoder(hidden_dim, lap_pe_dim=lap_pe_dim)

        self.layers = nn.ModuleList([
            GREDLayer(hidden_dim, state_dim, expand, r_min, r_max, max_phase, dropout, act)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_nodes(self, batch):
        """Flat node embeddings: (total_N, d)."""
        lap_pe = getattr(batch, 'lap_pe', None)
        return self.encoder(batch.x, batch.edge_index, batch.edge_attr,
                            lap_pe=lap_pe)

    def encode_dense(self, batch):
        """Dense batched node embeddings: (B, max_N, d), (B, max_N) mask."""
        h = self.encode_nodes(batch)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, dist_masks, node_masks, proxy_embeddings=None,
                precomputed_dense=None, readout_scope="nodes_only"):
        """
        Args:
            batch: PyG Batch object.
            dist_masks: (B, K, max_N, max_N) float distance masks.
            node_masks: (B, max_N) boolean node masks.
            proxy_embeddings: optional (B, M, d) — if provided, passes through
                              transformer layers after GRED encoding.
            precomputed_dense: optional (dense_x, dense_mask) to skip re-encoding.
            readout_scope: "nodes_only" or "all_tokens".
        Returns:
            logits: (B, output_dim)
            node_embeddings: (total_N, d) flat
        """
        if precomputed_dense is not None:
            dense_x, dense_mask = precomputed_dense
        else:
            dense_x, dense_mask = self.encode_dense(batch)

        B, max_N, d = dense_x.shape

        # Run GRED layers
        h = dense_x
        for layer in self.layers:
            h = layer(h, dist_masks, node_masks)

        # If proxy embeddings provided, we need transformer layers to integrate them.
        # For standalone GRED (no proxies), skip this.
        if proxy_embeddings is not None:
            # This path is for the hybrid model — handled by GREDHybridTransformer
            raise NotImplementedError(
                "GREDEncoder alone does not support proxy integration. "
                "Use GREDHybridTransformer instead."
            )

        # Readout: mean pool over valid nodes
        valid_emb = h[node_masks]
        batch_vec = torch.arange(B, device=h.device).unsqueeze(1).expand_as(node_masks)[node_masks]
        pooled = global_mean_pool(valid_emb, batch_vec)

        logits = self.head(pooled)

        # Return flat node embeddings
        node_emb = h[node_masks]
        return logits, node_emb


class GREDHybridTransformer(nn.Module):
    """
    Hybrid model: GRED layers for structure-aware node encoding, then
    Transformer layers for proxy integration.

    Pipeline: NodeEncoder → GRED layers → (optional: generate proxies) →
              concat [H_gred; proxies] → Transformer layers → readout → classify.

    This combines GRED's topology-aware distance filtering with the proxy
    mechanism's ability to create learned information hubs.

    Args:
        hidden_dim: shared hidden dimension across GRED and Transformer.
        state_dim: LRU complex state dimension.
        num_gred_layers: number of GRED layers for node encoding.
        num_transformer_layers: number of Transformer layers for proxy integration.
        num_heads: attention heads for Transformer layers.
        expand: FFN expansion factor for GRED layers.
        r_min, r_max: eigenvalue magnitude range.
        max_phase: max eigenvalue phase.
        dropout: dropout rate.
        act: GLU variant.
        output_dim: number of output classes.
    """
    def __init__(self, hidden_dim=88, state_dim=88, num_gred_layers=6,
                 num_transformer_layers=2, num_heads=8, expand=1,
                 r_min=0.0, r_max=1.0, max_phase=6.28, dropout=0.2,
                 act="full-glu", output_dim=10, lap_pe_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = NodeEncoder(hidden_dim, lap_pe_dim=lap_pe_dim)

        # GRED layers for structure-aware encoding
        self.gred_layers = nn.ModuleList([
            GREDLayer(hidden_dim, state_dim, expand, r_min, r_max, max_phase, dropout, act)
            for _ in range(num_gred_layers)
        ])

        # Transformer layers for proxy integration (only used when proxies present)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_transformer_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_nodes(self, batch):
        """Flat node embeddings: (total_N, d)."""
        lap_pe = getattr(batch, 'lap_pe', None)
        return self.encoder(batch.x, batch.edge_index, batch.edge_attr,
                            lap_pe=lap_pe)

    def encode_dense(self, batch):
        """Dense batched node embeddings: (B, max_N, d), (B, max_N) mask."""
        h = self.encode_nodes(batch)
        return to_dense_batch(h, batch.batch)

    def encode_gred(self, dense_x, dist_masks, node_masks):
        """
        Run GRED layers on dense node embeddings.
        Returns: (B, max_N, d) GRED-encoded node features.
        """
        h = dense_x
        for layer in self.gred_layers:
            h = layer(h, dist_masks, node_masks)
        return h

    def forward(self, batch, dist_masks, node_masks, proxy_embeddings=None,
                precomputed_dense=None, precomputed_gred=None,
                readout_scope="nodes_only"):
        """
        Args:
            batch: PyG Batch object.
            dist_masks: (B, K, max_N, max_N) float distance masks.
            node_masks: (B, max_N) boolean node masks.
            proxy_embeddings: optional (B, M, d) proxy tokens to concatenate.
            precomputed_dense: optional (dense_x, dense_mask) to skip node encoding.
            precomputed_gred: optional (gred_h,) to skip GRED encoding.
            readout_scope: "nodes_only" or "all_tokens".
        Returns:
            logits: (B, output_dim)
            node_embeddings: (total_N, d) flat
        """
        if precomputed_dense is not None:
            dense_x, dense_mask = precomputed_dense
        else:
            dense_x, dense_mask = self.encode_dense(batch)

        B, max_N, d = dense_x.shape

        # GRED encoding
        if precomputed_gred is not None:
            h = precomputed_gred
        else:
            h = self.encode_gred(dense_x, dist_masks, node_masks)

        # Transformer layers with optional proxy integration
        if proxy_embeddings is not None:
            M = proxy_embeddings.shape[1]
            # Concatenate proxies to GRED-encoded node features
            h_aug = torch.cat([h, proxy_embeddings], dim=1)  # (B, N+M, d)
            aug_mask = torch.cat([
                dense_mask,
                torch.ones(B, M, dtype=torch.bool, device=h.device)
            ], dim=1)
        else:
            h_aug = h
            aug_mask = dense_mask

        # Run transformer layers (for proxy-node interaction)
        for layer in self.transformer_layers:
            h_aug = layer(h_aug, aug_mask)

        # Readout
        if readout_scope == "all_tokens" and proxy_embeddings is not None:
            valid_emb = h_aug[aug_mask]
            batch_vec = torch.arange(B, device=h_aug.device).unsqueeze(1).expand_as(aug_mask)[aug_mask]
            pooled = global_mean_pool(valid_emb, batch_vec)
        else:
            orig_h = h_aug[:, :max_N, :]
            node_emb_masked = orig_h[dense_mask]
            # Keep pooling indices aligned with dense_mask when nodes are dropped
            # and precomputed_dense is passed from Phase 3.
            try:
                pooled = global_mean_pool(node_emb_masked, batch.batch)
            except:
                batch_vec = (
                    torch.arange(B, device=h_aug.device)
                    .unsqueeze(1)
                    .expand_as(dense_mask)[dense_mask]
                )
                pooled = global_mean_pool(node_emb_masked, batch_vec)

        logits = self.head(pooled)

        # Flat node embeddings (from the GRED-encoded nodes, post-transformer)
        orig_h = h_aug[:, :max_N, :]
        node_emb = orig_h[dense_mask]
        return logits, node_emb
