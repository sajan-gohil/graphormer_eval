# models.py
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_add_pool

from data import get_dataset_info


# OGB peptides categorical feature dimensions.
FULL_ATOM_FEATURE_DIMS = [119, 5, 12, 12, 10, 6, 6, 2, 2]

# OGB bond (edge) categorical feature dimensions: bond type, bond stereo,
# is-conjugated. Used by BondEncoder to embed edge_attr for molecular datasets.
FULL_BOND_FEATURE_DIMS = [5, 6, 2]


class BondEncoder(nn.Module):
    """Categorical bond/edge-feature encoder (OGB-style).

    Mirrors the atom encoder: sums per-feature embeddings of the integer
    edge_attr columns into a single ``hidden_dim`` edge embedding. Robust to
    edge_attr that has fewer/more columns than ``FULL_BOND_FEATURE_DIMS`` and
    clamps out-of-range indices, so it degrades gracefully across datasets.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_feature_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=hidden_dim)
            for dim in FULL_BOND_FEATURE_DIMS
        ])
        for emb in self.bond_feature_embeddings:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        # edge_attr: (E, n_bond_features) integer -> (E, hidden_dim).
        n_cols = min(edge_attr.size(-1), len(self.bond_feature_embeddings))
        h = 0
        for i in range(n_cols):
            emb = self.bond_feature_embeddings[i]
            feat_i = edge_attr[:, i].long().clamp(min=0, max=emb.num_embeddings - 1)
            h = h + emb(feat_i)
        return h


class LinearBondEncoder(nn.Module):
    """Continuous edge-feature encoder: Linear projection of edge_attr.

    Fallback for non-molecular datasets whose ``edge_attr`` is continuous (or
    integer but not OGB-categorical). Pads/truncates to ``in_dim`` so a dataset
    switch does not require re-instantiation.
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        x = edge_attr.float()
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.size(-1) < self.in_dim:
            x = F.pad(x, (0, self.in_dim - x.size(-1)))
        elif x.size(-1) > self.in_dim:
            x = x[..., :self.in_dim]
        return self.proj(x)


def build_bond_encoder(hidden_dim, dataset_name="Peptides-func", edge_feat_dim=None):
    """Factory: categorical BondEncoder for molecular (atom_categorical)
    datasets, LinearBondEncoder otherwise."""
    info = get_dataset_info(dataset_name)
    if info["node_encoder"] == "atom_categorical":
        return BondEncoder(hidden_dim)
    in_dim = edge_feat_dim if edge_feat_dim not in (None, "auto") else hidden_dim
    return LinearBondEncoder(in_dim=in_dim, hidden_dim=hidden_dim)


def aggregate_edges_to_nodes(
    node_h: torch.Tensor,
    edge_index: torch.Tensor,
    edge_emb: torch.Tensor,
) -> torch.Tensor:
    """Sum incident edge embeddings into their destination nodes, normalised
    by node degree, and add to ``node_h``.

    node_h:     (N, d) node features.
    edge_index: (2, E) — edges aggregated onto ``edge_index[1]`` (destination).
    edge_emb:   (E, d) per-edge embeddings.
    """
    N, d = node_h.shape
    dst = edge_index[1]
    agg = node_h.new_zeros(N, d)
    agg.index_add_(0, dst, edge_emb)
    deg = node_h.new_zeros(N)
    deg.index_add_(0, dst, node_h.new_ones(dst.shape[0]))
    agg = agg / deg.clamp(min=1.0).unsqueeze(-1)
    return node_h + agg


class NodeEncoder(nn.Module):
    """Peptides-style atom feature encoder with optional Laplacian PE.

    For OGB peptides categorical node features, this matches the official
    implementation by summing per-feature embeddings. For other datasets,
    a simple dimension-matching fallback keeps the pipelines usable.

    For non-Peptides LRGB datasets (e.g. PascalVOC-SP) prefer building via
    :func:`build_node_encoder` — it returns ``LinearNodeEncoder`` with an
    actual learnable input projection rather than this class's pad/truncate
    fallback.
    """
    def __init__(self, hidden_dim=64, lap_pe_dim=0, use_edge_features=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_atom_features = len(FULL_ATOM_FEATURE_DIMS)

        self.atom_feature_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=dim, embedding_dim=hidden_dim)
            for dim in FULL_ATOM_FEATURE_DIMS
        ])
        for emb in self.atom_feature_embeddings:
            nn.init.normal_(emb.weight, std=0.01)

        self.atom_post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Non-peptides fallback for continuous or differently-shaped node features.
        # Keep this parameter-free to avoid lazy-module initialization issues during
        # parameter counting/optimizer construction before the first forward pass.

        # Optional Laplacian positional encoding
        self.lap_pe_dim = lap_pe_dim
        if lap_pe_dim > 0:
            self.lap_pe_encoder = nn.Linear(lap_pe_dim, hidden_dim)

        # Optional edge-feature incorporation: embed bonds and aggregate the
        # incident bond embeddings into each node.
        self.use_edge_features = use_edge_features
        if use_edge_features:
            self.bond_encoder = BondEncoder(hidden_dim)

    def _encode_categorical_atom_features(self, x):
        h = 0
        for i, emb in enumerate(self.atom_feature_embeddings):
            feat_i = x[:, i].long().clamp(min=0, max=emb.num_embeddings - 1)
            h = h + emb(feat_i)
        return self.atom_post(h)

    def forward(self, x, edge_index, edge_attr, lap_pe=None):
        is_integral = x.dtype in (
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.uint8, torch.bool,
        )
        if x.dim() == 2 and x.size(1) == self.num_atom_features and is_integral:
            h = self._encode_categorical_atom_features(x)
        else:
            x_float = x.float()
            if x_float.size(-1) == self.hidden_dim:
                h = x_float
            elif x_float.size(-1) > self.hidden_dim:
                h = x_float[:, :self.hidden_dim]
            else:
                h = F.pad(x_float, (0, self.hidden_dim - x_float.size(-1)))

        # Incorporate edge features: embed bonds, aggregate onto nodes.
        if self.use_edge_features and edge_attr is not None and edge_index is not None:
            edge_emb = self.bond_encoder(edge_attr)
            h = aggregate_edges_to_nodes(h, edge_index, edge_emb)

        # Add Laplacian positional encoding if available
        if self.lap_pe_dim > 0 and lap_pe is not None:
            h = h + self.lap_pe_encoder(lap_pe)

        return h


class LinearNodeEncoder(nn.Module):
    """Continuous-feature node encoder for LRGB datasets like PascalVOC-SP.

    Projects per-node feature vectors of size ``in_dim`` into ``hidden_dim``
    via a learnable Linear + GELU, then optionally adds a Laplacian PE term.
    Mirrors the interface of ``NodeEncoder`` so it can be swapped in via
    :func:`build_node_encoder` without touching the downstream encoders.
    """
    def __init__(self, in_dim, hidden_dim, lap_pe_dim=0, use_edge_features=False,
                 edge_feat_dim=None, dataset_name="PascalVOC-SP"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )
        self.lap_pe_dim = lap_pe_dim
        if lap_pe_dim > 0:
            self.lap_pe_encoder = nn.Linear(lap_pe_dim, hidden_dim)

        # Optional edge-feature incorporation (continuous-feature datasets).
        self.use_edge_features = use_edge_features
        if use_edge_features:
            self.bond_encoder = build_bond_encoder(
                hidden_dim, dataset_name=dataset_name, edge_feat_dim=edge_feat_dim,
            )

    def forward(self, x, edge_index, edge_attr, lap_pe=None):
        x_float = x.float()
        # Robust to feature-dim mismatch: pad/truncate so the experiment script
        # can switch datasets without re-instantiating the model when in_dim
        # was inferred from the registry but the raw feature width differs.
        if x_float.size(-1) < self.in_dim:
            x_float = F.pad(x_float, (0, self.in_dim - x_float.size(-1)))
        elif x_float.size(-1) > self.in_dim:
            x_float = x_float[..., :self.in_dim]
        h = self.proj(x_float)

        if self.use_edge_features and edge_attr is not None and edge_index is not None:
            edge_emb = self.bond_encoder(edge_attr)
            h = aggregate_edges_to_nodes(h, edge_index, edge_emb)

        if self.lap_pe_dim > 0 and lap_pe is not None:
            h = h + self.lap_pe_encoder(lap_pe)
        return h


def build_node_encoder(hidden_dim, lap_pe_dim=0, dataset_name="Peptides-func",
                       node_feat_dim=None, use_edge_features=False,
                       edge_feat_dim=None):
    """Factory: pick the right node encoder for a registered dataset.

    Reads ``data.GRAPH_DATASETS[dataset_name]`` to decide between the
    Peptides-style categorical encoder and a learnable Linear projection
    for continuous features (e.g. PascalVOC-SP). Other call sites should
    use this rather than instantiating ``NodeEncoder`` directly so that
    flipping ``--dataset`` re-routes to the correct encoder automatically.
    """
    info = get_dataset_info(dataset_name)
    kind = info["node_encoder"]
    if kind == "atom_categorical":
        return NodeEncoder(hidden_dim, lap_pe_dim=lap_pe_dim,
                           use_edge_features=use_edge_features)
    if kind == "linear":
        # Precedence: explicit arg > registry info > hidden_dim fallback.
        if node_feat_dim not in (None, "auto"):
            in_dim = node_feat_dim
        else:
            in_dim = info.get("node_feat_dim")
        if in_dim in (None, "auto"):
            warnings.warn(
                f"node_feat_dim not set for {dataset_name}; falling back to hidden_dim.",
                RuntimeWarning,
            )
            in_dim = hidden_dim
        print("IN DIM NODE ENCODER = ==================== ", in_dim)
        return LinearNodeEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            lap_pe_dim=lap_pe_dim,
            use_edge_features=use_edge_features,
            edge_feat_dim=edge_feat_dim,
            dataset_name=info["name"],
        )
    raise ValueError(f"Unknown node_encoder kind: {kind}")


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

    def forward(self, x, mask=None, return_attention=False):
        """Pre-norm self-attention + FFN.

        Args:
            x:    (B, N, d) input.
            mask: (B, N) bool or None. True = real token.
            return_attention: if True, also return the post-softmax,
                pre-dropout attention weights (B, H, N, N) detached from
                the autograd graph. Used by attention distillation.
        Returns:
            x                       if return_attention is False.
            (x, attn_w_clean)       if return_attention is True.
        """
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
        # Snapshot the clean post-softmax weights BEFORE dropout, so distillation
        # targets reflect the deterministic attention pattern (the dropout mask
        # is a stochastic train-time artefact we don't want to distill).
        attn_w_clean = attn_w.detach().clone() if return_attention else None
        attn_w_dropped = self.attn_drop(attn_w)

        out = (attn_w_dropped @ v).transpose(1, 2).reshape(B, N, d)
        x = x + self.res_drop(self.wout(out))
        x = x + self.res_drop(self.ff(self.norm2(x)))
        if return_attention:
            return x, attn_w_clean
        return x


class GraphTransformer(nn.Module):
    def __init__(self, num_layers=5, num_heads=8, hidden_dim=64,
                 output_dim=10, dropout=0.3, lap_pe_dim=0,
                 cross_attn_router=None, multi_point_proxy=None,
                 dataset_name="Peptides-func", task_level="graph"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dataset_name = dataset_name
        self.task_level = task_level
        self.encoder = build_node_encoder(hidden_dim, lap_pe_dim=lap_pe_dim,
                                          dataset_name=dataset_name)
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
        # Optional N→M→N cross-attention router (None = use N+M concat default)
        self.cross_attn_router = cross_attn_router
        # Optional multi-point proxy wrapper (None = single-point or no proxies)
        self.multi_point_proxy = multi_point_proxy
        self._last_mp_aux_loss = 0.0

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
                readout_scope="all_tokens", disable_proxy_injection=False):
        """
        Args:
            batch: PyG Batch object.
            proxy_embeddings: optional (B, M, d) proxy tokens to concatenate.
            precomputed_dense: optional (dense_x, dense_mask) tuple to skip encoding.
            readout_scope: "nodes_only" pools over N original nodes,
                           "all_tokens" pools over N+M (requires proxy_embeddings).
            disable_proxy_injection: if True, force the no-proxy path regardless
                of ``proxy_embeddings`` or an attached ``multi_point_proxy``.
                Useful for running paired with/without proxy forwards on the
                same model instance.
        Returns:
            logits: (B, output_dim)
            node_embeddings: (total_N, d) flat node embeddings from encoder
        """
        if precomputed_dense is not None:
            dense_x, dense_mask = precomputed_dense
        else:
            dense_x, dense_mask = self.encode_dense(batch)

        B, max_N, d = dense_x.shape

        if disable_proxy_injection:
            proxy_embeddings = None
        use_multi_point = (self.multi_point_proxy is not None
                           and not disable_proxy_injection)

        if use_multi_point:
            # ── Multi-point mode: interleave proxy blocks with TF layers ──
            insertion_set = set(self.multi_point_proxy.insertion_layers)
            point_idx = 0
            total_aux = 0.0
            aug_x = dense_x
            aug_mask = dense_mask

            # Build kwargs for GNN-based generators (graph_coarsening / gnn_pooling).
            # For attention-based generators the dict is empty and ignored.
            _gen = self.multi_point_proxy.get_generator(0)
            from generators import GraphCoarseningGenerator, GNNPoolingGenerator
            _needs_graph_kwargs = isinstance(_gen, (GraphCoarseningGenerator, GNNPoolingGenerator))
            gen_kwargs = {
                "edge_index": batch.edge_index,
                "batch_vec": batch.batch,
                "edge_attr": getattr(batch, "edge_attr", None),
            } if _needs_graph_kwargs else {}

            for i, layer in enumerate(self.layers):
                if i in insertion_set:
                    aug_x, aug_mask, aux = self.multi_point_proxy.run_proxy_block(
                        aug_x, aug_mask, point_idx, **gen_kwargs
                    )
                    decay = self.multi_point_proxy.aux_loss_decay ** point_idx
                    total_aux = total_aux + aux * decay
                    point_idx += 1

                aug_x = layer(aug_x, aug_mask)

            self._last_mp_aux_loss = total_aux
            dense_x = aug_x
            aug_mask_final = aug_mask

        elif proxy_embeddings is not None and self.cross_attn_router is not None:
            # ── Single-point cross-attention (existing behaviour) ──
            dense_x = self.cross_attn_router(dense_x, proxy_embeddings, dense_mask)
            aug_mask_final = dense_mask
            for layer in self.layers:
                dense_x = layer(dense_x, aug_mask_final)

        elif proxy_embeddings is not None:
            # ── Single-point N+M concat (existing behaviour) ──
            M = proxy_embeddings.shape[1]
            dense_x = torch.cat([dense_x, proxy_embeddings], dim=1)
            aug_mask_final = torch.cat([
                dense_mask,
                torch.ones(B, M, dtype=torch.bool, device=dense_x.device)
            ], dim=1)
            for layer in self.layers:
                dense_x = layer(dense_x, aug_mask_final)

        else:
            # ── No proxies at all ──
            aug_mask_final = dense_mask
            for layer in self.layers:
                dense_x = layer(dense_x, aug_mask_final)

        # Always extract original-N node embeddings (used by both readouts).
        orig_x = dense_x[:, :max_N, :]
        node_emb = orig_x[dense_mask]

        if self.task_level == "node":
            # Per-node prediction (e.g. PascalVOC-SP): apply head directly to
            # flat real-node embeddings — no graph pooling. Logits shape is
            # (total_real_nodes, num_classes), aligned with the flattened
            # batch.y produced by PyG node-level batching.
            logits = self.head(node_emb)
            return logits, node_emb

        # Graph-level readout: pool and classify
        if readout_scope == "all_tokens" and (
            proxy_embeddings is not None or self.multi_point_proxy is not None
        ) and self.cross_attn_router is None:
            valid_emb = dense_x[aug_mask_final]
            batch_vec = torch.arange(B, device=dense_x.device).unsqueeze(1).expand_as(aug_mask_final)[aug_mask_final]
            pooled = global_add_pool(valid_emb, batch_vec)
        else:
            node_emb_masked = orig_x[dense_mask]
            # Build pooling indices from the active dense node mask so this
            # works for both full batches and subsampled precomputed_dense.
            try:
                pooled = global_add_pool(node_emb_masked, batch.batch)
            except:
                batch_vec = (
                    torch.arange(B, device=dense_x.device)
                    .unsqueeze(1)
                    .expand_as(dense_mask)[dense_mask]
                )
                pooled = global_add_pool(node_emb_masked, batch_vec)

        logits = self.head(pooled)
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

        with torch.no_grad():
            diag_lambda_init = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log))
            gamma_log_init = torch.log(torch.sqrt(1 - torch.abs(diag_lambda_init) ** 2 + 1e-8))
        self.gamma_log = nn.Parameter(gamma_log_init)

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

    def _get_gamma(self):
        """Trainable input scaling from log parameterization."""
        return torch.exp(self.gamma_log)

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
        gamma = self._get_gamma()  # (state_dim,)
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

    def forward(self, h, dist_masks, node_masks=None, attn_weights=None):
        """
        Args:
            h: (B, N, d) node features.
            dist_masks: (B, K, N, N) float distance masks (1 where d(v,u)==k).
            node_masks: (B, N) boolean — True for real nodes. Optional.
            attn_weights: optional (B, N, N) — multiplicative weight on each
                (target=v, source=u) pair, broadcast over the K hop axis. The
                aggregation becomes
                    agg[v, k] = sum_{u : d(v,u)=k} attn_weights[v, u] * h_u
                No normalization is applied; weights need not sum to 1.
                Pass None for vanilla GRED behavior.
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
        if attn_weights is not None:
            # Broadcast (B, N, N) over the hop axis: every (v, u) pair across
            # all hops gets multiplied by the same attn_weights[v, u].
            dm = dm * attn_weights.unsqueeze(1)  # (B, K, N, N)
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
                 act="full-glu", output_dim=10, lap_pe_dim=0,
                 dataset_name="Peptides-func", task_level="graph"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dataset_name = dataset_name
        self.task_level = task_level
        self.encoder = build_node_encoder(hidden_dim, lap_pe_dim=lap_pe_dim,
                                          dataset_name=dataset_name)

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
                precomputed_dense=None, readout_scope="all_tokens",
                attn_weights=None):
        """
        Args:
            batch: PyG Batch object.
            dist_masks: (B, K, max_N, max_N) float distance masks.
            node_masks: (B, max_N) boolean node masks.
            proxy_embeddings: optional (B, M, d) — if provided, passes through
                              transformer layers after GRED encoding.
            precomputed_dense: optional (dense_x, dense_mask) to skip re-encoding.
            readout_scope: "nodes_only" or "all_tokens".
            attn_weights: optional (B, max_N, max_N) — passed unchanged to every
                GRED layer; weights the per-hop aggregation. None = vanilla GRED.
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
            h = layer(h, dist_masks, node_masks, attn_weights=attn_weights)

        # If proxy embeddings provided, we need transformer layers to integrate them.
        # For standalone GRED (no proxies), skip this.
        if proxy_embeddings is not None:
            # This path is for the hybrid model — handled by GREDHybridTransformer
            raise NotImplementedError(
                "GREDEncoder alone does not support proxy integration. "
                "Use GREDHybridTransformer instead."
            )

        # Flat node embeddings used by both readouts.
        node_emb = h[node_masks]

        if self.task_level == "node":
            # Per-node prediction: skip pooling entirely.
            logits = self.head(node_emb)
            return logits, node_emb

        # Graph-level readout: sum pool over valid nodes.
        valid_emb = h[node_masks]
        batch_vec = torch.arange(B, device=h.device).unsqueeze(1).expand_as(node_masks)[node_masks]
        pooled = global_add_pool(valid_emb, batch_vec)

        logits = self.head(pooled)
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
                 act="full-glu", output_dim=10, lap_pe_dim=0,
                 cross_attn_router=None, multi_point_proxy=None,
                 dataset_name="Peptides-func", task_level="graph"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dataset_name = dataset_name
        self.task_level = task_level
        self.encoder = build_node_encoder(hidden_dim, lap_pe_dim=lap_pe_dim,
                                          dataset_name=dataset_name)

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
        # Optional N→M→N cross-attention router (None = use N+M concat default)
        self.cross_attn_router = cross_attn_router
        # Optional multi-point proxy wrapper (None = single-point or no proxies)
        self.multi_point_proxy = multi_point_proxy
        self._last_mp_aux_loss = 0.0

    def encode_nodes(self, batch):
        """Flat node embeddings: (total_N, d)."""
        lap_pe = getattr(batch, 'lap_pe', None)
        return self.encoder(batch.x, batch.edge_index, batch.edge_attr,
                            lap_pe=lap_pe)

    def encode_dense(self, batch):
        """Dense batched node embeddings: (B, max_N, d), (B, max_N) mask."""
        h = self.encode_nodes(batch)
        return to_dense_batch(h, batch.batch)

    def encode_gred(self, dense_x, dist_masks, node_masks, attn_weights=None):
        """
        Run GRED layers on dense node embeddings.

        attn_weights: optional (B, max_N, max_N) — passed to every GRED layer.
        Returns: (B, max_N, d) GRED-encoded node features.
        """
        h = dense_x
        for layer in self.gred_layers:
            h = layer(h, dist_masks, node_masks, attn_weights=attn_weights)
        return h

    def forward(self, batch, dist_masks, node_masks, proxy_embeddings=None,
                precomputed_dense=None, precomputed_gred=None,
                readout_scope="all_tokens", disable_proxy_injection=False,
                attn_weights=None, return_attention=False):
        """
        Args:
            batch: PyG Batch object.
            dist_masks: (B, K, max_N, max_N) float distance masks.
            node_masks: (B, max_N) boolean node masks.
            proxy_embeddings: optional (B, M, d) proxy tokens to concatenate.
            precomputed_dense: optional (dense_x, dense_mask) to skip node encoding.
            precomputed_gred: optional (gred_h,) to skip GRED encoding.
            readout_scope: "nodes_only" or "all_tokens".
            disable_proxy_injection: if True, force the no-proxy path regardless
                of ``proxy_embeddings`` or an attached ``multi_point_proxy``.
            attn_weights: optional (B, max_N, max_N) — multiplicative weight on
                each (target, source) pair, applied uniformly across GRED hops
                in every GRED layer. None = vanilla GRED behavior. Ignored when
                ``precomputed_gred`` is supplied (the GRED stack has already run).
            return_attention: if True, also return a list of per-transformer-layer
                post-softmax attention weights (B, H, N_aug, N_aug), captured
                BEFORE attention dropout and detached from the autograd graph.
                Used for attention distillation against a hybrid teacher.
        Returns:
            logits: (B, output_dim)
            node_embeddings: (total_N, d) flat
            (optional) attentions: list of length len(transformer_layers), each
                tensor (B, H, N_aug, N_aug). Only returned if return_attention.
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
            h = self.encode_gred(dense_x, dist_masks, node_masks,
                                  attn_weights=attn_weights)

        if disable_proxy_injection:
            proxy_embeddings = None
        use_multi_point = (self.multi_point_proxy is not None
                           and not disable_proxy_injection)

        # Optional per-layer attention collector for distillation.
        attentions = [] if return_attention else None

        # Transformer layers with optional proxy integration
        if use_multi_point:
            # ── Multi-point mode: interleave proxy blocks with TF layers ──
            insertion_set = set(self.multi_point_proxy.insertion_layers)
            point_idx = 0
            total_aux = 0.0
            h_aug = h
            aug_mask = dense_mask

            # Build kwargs for GNN-based generators (graph_coarsening / gnn_pooling).
            # For attention-based generators the dict is empty and ignored.
            _gen = self.multi_point_proxy.get_generator(0)
            from generators import GraphCoarseningGenerator, GNNPoolingGenerator
            _needs_graph_kwargs = isinstance(_gen, (GraphCoarseningGenerator, GNNPoolingGenerator))
            gen_kwargs = {
                "edge_index": batch.edge_index,
                "batch_vec": batch.batch,
                "edge_attr": getattr(batch, "edge_attr", None),
            } if _needs_graph_kwargs else {}

            for i, layer in enumerate(self.transformer_layers):
                if i in insertion_set:
                    h_aug, aug_mask, aux = self.multi_point_proxy.run_proxy_block(
                        h_aug, aug_mask, point_idx, **gen_kwargs
                    )
                    decay = self.multi_point_proxy.aux_loss_decay ** point_idx
                    total_aux = total_aux + aux * decay
                    point_idx += 1

                if return_attention:
                    h_aug, attn_w = layer(h_aug, aug_mask, return_attention=True)
                    attentions.append(attn_w)
                else:
                    h_aug = layer(h_aug, aug_mask)

            self._last_mp_aux_loss = total_aux

        elif proxy_embeddings is not None and self.cross_attn_router is not None:
            # ── Single-point cross-attention (existing behaviour) ──
            h = self.cross_attn_router(h, proxy_embeddings, dense_mask)
            h_aug = h
            aug_mask = dense_mask
            for layer in self.transformer_layers:
                if return_attention:
                    h_aug, attn_w = layer(h_aug, aug_mask, return_attention=True)
                    attentions.append(attn_w)
                else:
                    h_aug = layer(h_aug, aug_mask)

        elif proxy_embeddings is not None:
            # ── Single-point N+M concat (existing behaviour) ──
            M = proxy_embeddings.shape[1]
            h_aug = torch.cat([h, proxy_embeddings], dim=1)  # (B, N+M, d)
            aug_mask = torch.cat([
                dense_mask,
                torch.ones(B, M, dtype=torch.bool, device=h.device)
            ], dim=1)
            for layer in self.transformer_layers:
                if return_attention:
                    h_aug, attn_w = layer(h_aug, aug_mask, return_attention=True)
                    attentions.append(attn_w)
                else:
                    h_aug = layer(h_aug, aug_mask)

        else:
            # ── No proxies ──
            h_aug = h
            aug_mask = dense_mask
            for layer in self.transformer_layers:
                if return_attention:
                    h_aug, attn_w = layer(h_aug, aug_mask, return_attention=True)
                    attentions.append(attn_w)
                else:
                    h_aug = layer(h_aug, aug_mask)

        # Flat node embeddings (from the GRED-encoded nodes, post-transformer).
        orig_h = h_aug[:, :max_N, :]
        node_emb = orig_h[dense_mask]

        if self.task_level == "node":
            # Per-node prediction: skip pooling and apply head to flat embeddings.
            logits = self.head(node_emb)
            if return_attention:
                return logits, node_emb, attentions
            return logits, node_emb

        # Graph-level readout
        if readout_scope == "all_tokens" and (
            proxy_embeddings is not None or self.multi_point_proxy is not None
        ) and self.cross_attn_router is None:
            valid_emb = h_aug[aug_mask]
            batch_vec = torch.arange(B, device=h_aug.device).unsqueeze(1).expand_as(aug_mask)[aug_mask]
            pooled = global_add_pool(valid_emb, batch_vec)
        else:
            node_emb_masked = orig_h[dense_mask]
            # Keep pooling indices aligned with dense_mask when nodes are dropped
            # and precomputed_dense is passed from Phase 3.
            try:
                pooled = global_add_pool(node_emb_masked, batch.batch)
            except:
                batch_vec = (
                    torch.arange(B, device=h_aug.device)
                    .unsqueeze(1)
                    .expand_as(dense_mask)[dense_mask]
                )
                pooled = global_add_pool(node_emb_masked, batch_vec)

        logits = self.head(pooled)
        if return_attention:
            return logits, node_emb, attentions
        return logits, node_emb


