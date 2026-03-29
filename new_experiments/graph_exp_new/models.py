import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, scatter
from torch_geometric.nn import global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class NodeEncoder(nn.Module):
    """Atom + bond-aggregated node embeddings."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden_dim // 2)
        self.bond_encoder = BondEncoder(hidden_dim // 2)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        h = self.atom_encoder(x)
        row = edge_index[0]
        edge_emb = self.bond_encoder(edge_attr)
        edge_aggr = scatter(edge_emb, row, dim=0, dim_size=h.size(0), reduce="add")
        h = torch.cat([h, edge_aggr], dim=-1)
        return self.proj(h)  # (total_N, d)


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
                 output_dim=10, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = NodeEncoder(hidden_dim)
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
        return self.encoder(batch.x, batch.edge_index, batch.edge_attr)

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
            pooled = global_mean_pool(node_emb_masked, batch.batch)

        logits = self.head(pooled)

        # Always return original node embeddings
        orig_x = dense_x[:, :max_N, :]
        node_emb = orig_x[dense_mask]
        return logits, node_emb
