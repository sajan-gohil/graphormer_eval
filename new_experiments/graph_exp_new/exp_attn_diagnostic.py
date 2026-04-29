"""
Diagnostic: What do optimized proxies actually change in the attention patterns?

Instead of looking for specific patterns, we compute a broad set of structural
statistics about the DIFFERENCE between vanilla and proxy-augmented attention.
Then we aggregate across all training graphs to find consistent patterns.

Computed per graph, per layer, per head:
  1. Attention shift by hop distance: do proxies increase/decrease attention
     to nodes at specific hop distances? (histogram of Δattention vs hop distance)
  2. Attention entropy change: does proxy enrichment make attention sharper or more diffuse?
  3. Top-K attention shift: which node PAIRS see the largest attention increase?
     What structural properties do those pairs share? (hop distance, degree product, etc.)
  4. Degree-dependent attention shift: do high-degree or low-degree nodes benefit more?
  5. Proxy attention analysis: which nodes do the proxies attend to most?
     What structural properties do those "proxy-attended" nodes have?
  6. Attention flow through proxies: for the N→M→N path, which node pairs
     gain the most indirect connectivity?

Output: a pickle file with raw statistics + printed summary of findings.

Usage:
    python exp_attn_diagnostic.py --model_path checkpoints_staged/stage1_best.pt
    python exp_attn_diagnostic.py --model_path checkpoints_staged/stage1_best.pt --max_graphs 200
"""

import argparse
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
import scipy.sparse.csgraph as csgraph

from data import get_loaders
from models import NodeEncoder
from metrics import compute_macro_ap


# ================================================================
# MODIFIED TRANSFORMER — returns attention weights
# ================================================================

class TransformerLayerWithAttn(nn.Module):
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
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim),
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
            key_pad = (~mask).unsqueeze(1).unsqueeze(2)
            query_pad = (~mask).unsqueeze(1).unsqueeze(-1)
            attn = attn.masked_fill(key_pad, float("-inf"))
            attn = attn.masked_fill(query_pad, float("-inf"))
        attn_w = F.softmax(attn, dim=-1)
        attn_w = torch.nan_to_num(attn_w, nan=0.0)
        attn_w_clean = attn_w.clone()
        attn_w_dropped = self.attn_drop(attn_w)
        out = (attn_w_dropped @ v).transpose(1, 2).reshape(B, N, d)
        x = x + self.res_drop(self.wout(out))
        x = x + self.res_drop(self.ff(self.norm2(x)))
        return x, attn_w_clean


class GraphTransformerWithAttn(nn.Module):
    def __init__(self, num_layers=5, num_heads=8, hidden_dim=64,
                 output_dim=10, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = NodeEncoder(hidden_dim)
        self.layers = nn.ModuleList([
            TransformerLayerWithAttn(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
        )

    def encode_dense(self, batch):
        h = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        return to_dense_batch(h, batch.batch)

    def forward(self, batch, proxy_embeddings=None, precomputed_dense=None):
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

        all_attn = []
        for layer in self.layers:
            dense_x, attn_w = layer(dense_x, aug_mask)
            all_attn.append(attn_w)

        orig_x = dense_x[:, :max_N, :]
        node_emb_masked = orig_x[dense_mask]
        pooled = global_mean_pool(node_emb_masked, batch.batch)
        logits = self.head(pooled)
        return logits, all_attn, dense_mask


# ================================================================
# PROXY OPTIMIZATION
# ================================================================

def _per_sample_bce(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)


@torch.no_grad()
def optimize_proxies_for_batch(model, batch, dense_x, dense_mask, args):
    B = batch.y.size(0)
    device = batch.y.device
    proxy = torch.randn(B, args.num_proxies, args.hidden_dim, device=device) * 0.02
    proxy = nn.Parameter(proxy)
    opt = torch.optim.Adam([proxy], lr=args.proxy_lr)

    logits_base, _, _ = model(batch, precomputed_dense=(dense_x, dense_mask))
    base_loss = _per_sample_bce(logits_base, batch.y)
    best_loss = base_loss.clone()
    best_proxy = proxy.detach().clone()

    with torch.enable_grad():
        for step in range(args.proxy_opt_steps):
            opt.zero_grad()
            logits, _, _ = model(batch, proxy_embeddings=proxy,
                                  precomputed_dense=(dense_x, dense_mask))
            task_loss = _per_sample_bce(logits, batch.y)
            task_loss.mean().backward()
            nn.utils.clip_grad_norm_([proxy], 1.0)
            opt.step()
            with torch.no_grad():
                improved = task_loss < best_loss
                if improved.any():
                    best_loss[improved] = task_loss[improved]
                    best_proxy[improved] = proxy.detach()[improved]

    return best_proxy, base_loss, best_loss


# ================================================================
# GRAPH STRUCTURAL FEATURES
# ================================================================

def compute_hop_distances(edge_index, num_nodes):
    """Compute all-pairs shortest path distances using BFS."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    ei = edge_index.cpu().numpy()
    adj[ei[0], ei[1]] = 1.0
    adj[ei[1], ei[0]] = 1.0  # undirected
    dist_matrix = csgraph.shortest_path(adj, method='BF', unweighted=True)
    dist_matrix[np.isinf(dist_matrix)] = -1  # unreachable
    return dist_matrix.astype(np.int32)


def compute_node_degrees(edge_index, num_nodes):
    """Compute degree of each node."""
    degrees = np.zeros(num_nodes, dtype=np.int32)
    ei = edge_index.cpu().numpy()
    for i in range(ei.shape[1]):
        degrees[ei[0, i]] += 1
    return degrees


# ================================================================
# DIAGNOSTIC ANALYSIS PER GRAPH
# ================================================================

def analyze_single_graph(vanilla_attns, proxy_attns, mask, hop_dist, degrees,
                         num_proxies, num_layers, num_heads):
    """
    Analyze attention differences for a single graph.

    Args:
        vanilla_attns: list of (H, N, N) per layer — attention WITHOUT proxies
        proxy_attns: list of (H, N+M, N+M) per layer — attention WITH proxies
        mask: (max_N,) boolean
        hop_dist: (n, n) shortest path distances for real nodes
        degrees: (n,) node degrees
        num_proxies: M

    Returns:
        dict of diagnostic statistics
    """
    n = int(mask.sum())
    stats = {}

    # --- 1. Attention shift by hop distance ---
    # For each layer and head, compute mean Δattention at each hop distance
    max_hop = min(int(hop_dist.max()), 20)
    hop_delta = np.zeros((num_layers, max_hop + 1))
    hop_counts = np.zeros(max_hop + 1)

    for h in range(max_hop + 1):
        pairs = np.argwhere(hop_dist == h)
        hop_counts[h] = len(pairs)

    for layer_idx in range(num_layers):
        v_attn = vanilla_attns[layer_idx][:, :n, :n]  # (H, n, n)
        p_attn = proxy_attns[layer_idx][:, :n, :n]     # (H, n, n) node-to-node block

        # Re-normalize proxy attention's node block
        p_attn_sum = p_attn.sum(dim=-1, keepdim=True)
        p_attn_norm = p_attn / (p_attn_sum + 1e-10)

        # Head-averaged delta
        delta = (p_attn_norm - v_attn).mean(dim=0).cpu().numpy()  # (n, n)

        for h in range(max_hop + 1):
            pairs = np.argwhere(hop_dist == h)
            if len(pairs) > 0:
                hop_delta[layer_idx, h] = delta[pairs[:, 0], pairs[:, 1]].mean()

    stats["hop_delta"] = hop_delta  # (num_layers, max_hop+1)
    stats["hop_counts"] = hop_counts
    stats["max_hop"] = max_hop

    # --- 2. Attention entropy change ---
    entropy_vanilla = np.zeros(num_layers)
    entropy_proxy = np.zeros(num_layers)

    for layer_idx in range(num_layers):
        v_attn = vanilla_attns[layer_idx][:, :n, :n]  # (H, n, n)
        p_attn = proxy_attns[layer_idx][:, :n, :n]

        p_attn_sum = p_attn.sum(dim=-1, keepdim=True)
        p_attn_norm = p_attn / (p_attn_sum + 1e-10)

        # Entropy per query, averaged over heads and queries
        v_ent = -(v_attn * torch.log(v_attn + 1e-10)).sum(dim=-1).mean().item()
        p_ent = -(p_attn_norm * torch.log(p_attn_norm + 1e-10)).sum(dim=-1).mean().item()
        entropy_vanilla[layer_idx] = v_ent
        entropy_proxy[layer_idx] = p_ent

    stats["entropy_vanilla"] = entropy_vanilla
    stats["entropy_proxy"] = entropy_proxy
    stats["entropy_delta"] = entropy_proxy - entropy_vanilla

    # --- 3. Top attention shifts: which node pairs gain the most? ---
    # Use last layer, head-averaged
    v_attn_last = vanilla_attns[-1][:, :n, :n].mean(dim=0).cpu().numpy()
    p_attn_last = proxy_attns[-1][:, :n, :n]
    p_attn_sum = p_attn_last.sum(dim=-1, keepdim=True)
    p_attn_last = (p_attn_last / (p_attn_sum + 1e-10)).mean(dim=0).cpu().numpy()

    delta_last = p_attn_last - v_attn_last
    # Top 20 increased pairs
    flat_idx = np.argsort(delta_last.ravel())[-20:]
    top_pairs = np.array(np.unravel_index(flat_idx, delta_last.shape)).T
    top_pair_stats = []
    for i, j in top_pairs:
        top_pair_stats.append({
            "hop_dist": int(hop_dist[i, j]),
            "degree_i": int(degrees[i]),
            "degree_j": int(degrees[j]),
            "delta_attn": float(delta_last[i, j]),
            "vanilla_attn": float(v_attn_last[i, j]),
            "proxy_attn": float(p_attn_last[i, j]),
        })
    stats["top_increased_pairs"] = top_pair_stats

    # --- 4. Degree-dependent attention shift ---
    degree_bins = [0, 2, 3, 4, 10, 100]  # bin edges
    degree_delta = {}
    for layer_idx in range(num_layers):
        v_attn = vanilla_attns[layer_idx][:, :n, :n].mean(dim=0).cpu().numpy()
        p_attn = proxy_attns[layer_idx][:, :n, :n]
        p_attn_sum = p_attn.sum(dim=-1, keepdim=True)
        p_attn = (p_attn / (p_attn_sum + 1e-10)).mean(dim=0).cpu().numpy()
        delta = p_attn - v_attn

        # Average incoming attention delta by degree of receiving node
        for b_idx in range(len(degree_bins) - 1):
            lo, hi = degree_bins[b_idx], degree_bins[b_idx + 1]
            nodes_in_bin = np.where((degrees >= lo) & (degrees < hi))[0]
            if len(nodes_in_bin) > 0:
                key = f"L{layer_idx}_deg_{lo}-{hi}"
                incoming_delta = delta[:, nodes_in_bin].mean()
                degree_delta[key] = float(incoming_delta)

    stats["degree_delta"] = degree_delta

    # --- 5. Proxy attention analysis: which nodes do proxies attend to? ---
    # In the proxy-augmented attention, rows N:N+M are proxy queries attending to nodes
    proxy_attn_to_nodes = []
    for layer_idx in range(num_layers):
        p_full = proxy_attns[layer_idx]  # (H, N+M, N+M)
        # Proxy rows attending to node columns
        proxy_to_node = p_full[:, n:n+num_proxies, :n]  # (H, M, n)
        # Average over heads and proxies
        avg_proxy_attn = proxy_to_node.mean(dim=(0, 1)).cpu().numpy()  # (n,)
        proxy_attn_to_nodes.append(avg_proxy_attn)

    # Correlate proxy attention with node degree
    last_proxy_attn = proxy_attn_to_nodes[-1]
    degree_corr = float(np.corrcoef(last_proxy_attn, degrees[:n])[0, 1]) if n > 2 else 0.0
    stats["proxy_degree_corr"] = degree_corr
    stats["proxy_attn_to_nodes_per_layer"] = proxy_attn_to_nodes

    # --- 6. Effective N×N connectivity through proxies ---
    # The indirect path: node i → proxy m → node j
    # Effective connectivity: sum_m (attn[i→m] * attn[m→j])
    # This is the N×N matrix of "how much info flows between i and j via proxies"
    p_full_last = proxy_attns[-1]  # (H, N+M, N+M)
    node_to_proxy = p_full_last[:, :n, n:n+num_proxies]  # (H, n, M)
    proxy_to_node = p_full_last[:, n:n+num_proxies, :n]  # (H, M, n)
    # Head-averaged indirect connectivity
    indirect = torch.bmm(node_to_proxy, proxy_to_node).mean(dim=0).cpu().numpy()  # (n, n)

    # Distribution of indirect connectivity by hop distance
    indirect_by_hop = np.zeros(max_hop + 1)
    for h in range(max_hop + 1):
        pairs = np.argwhere(hop_dist == h)
        if len(pairs) > 0:
            indirect_by_hop[h] = indirect[pairs[:, 0], pairs[:, 1]].mean()

    stats["indirect_connectivity_by_hop"] = indirect_by_hop

    stats["num_nodes"] = n

    return stats


# ================================================================
# MAIN ANALYSIS LOOP
# ================================================================

def run_diagnostic(args):
    print("=" * 60)
    print("Attention Diagnostic Analysis")
    print(f"  model_path={args.model_path}")
    print(f"  max_graphs={args.max_graphs}")
    print(f"  proxy_opt_steps={args.proxy_opt_steps}")
    print(f"  num_proxies={args.num_proxies}")
    print("=" * 60)

    train_loader, _, _, _, _, _ = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = GraphTransformerWithAttn(
        num_layers=args.num_layers, num_heads=args.num_heads,
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        dropout=args.dropout,
    ).to(args.device)

    ckpt = torch.load(args.model_path, map_location=args.device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    all_stats = []
    graphs_processed = 0
    total_base_loss, total_opt_loss = 0.0, 0.0

    for batch in train_loader:
        if graphs_processed >= args.max_graphs:
            break

        batch = batch.to(args.device)
        B = batch.y.size(0)

        with torch.no_grad():
            dense_x, dense_mask = model.encode_dense(batch)

            # Vanilla attention (no proxies)
            _, vanilla_attns, _ = model(batch, precomputed_dense=(dense_x, dense_mask))

        # Optimize proxies
        best_proxy, base_loss, opt_loss = optimize_proxies_for_batch(
            model, batch, dense_x, dense_mask, args)
        total_base_loss += base_loss.sum().item()
        total_opt_loss += opt_loss.sum().item()

        with torch.no_grad():
            # Proxy-augmented attention
            _, proxy_attns, _ = model(batch, proxy_embeddings=best_proxy,
                                       precomputed_dense=(dense_x, dense_mask))

        # Analyze each graph in the batch
        graphs_list = batch.to_data_list()
        offset = 0
        for i in range(B):
            if graphs_processed >= args.max_graphs:
                break

            g = graphs_list[i]
            n = g.x.size(0)
            mask_i = dense_mask[i]

            # Compute graph structural features
            hop_dist = compute_hop_distances(g.edge_index, n)
            degrees = compute_node_degrees(g.edge_index, n)

            # Extract per-graph attention
            v_attns_i = [a[i] for a in vanilla_attns]     # list of (H, N, N)
            p_attns_i = [a[i] for a in proxy_attns]       # list of (H, N+M, N+M)

            graph_stats = analyze_single_graph(
                v_attns_i, p_attns_i, mask_i, hop_dist, degrees,
                args.num_proxies, args.num_layers, args.num_heads,
            )
            graph_stats["graph_idx"] = graphs_processed
            graph_stats["base_loss"] = float(base_loss[i])
            graph_stats["opt_loss"] = float(opt_loss[i])
            all_stats.append(graph_stats)

            graphs_processed += 1
            if graphs_processed % 50 == 0:
                print(f"  Processed {graphs_processed}/{args.max_graphs} graphs", flush=True)

    # ================================================================
    # AGGREGATE STATISTICS
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"AGGREGATE RESULTS ({len(all_stats)} graphs)")
    print(f"{'=' * 60}")
    print(f"\nAvg base loss: {total_base_loss / graphs_processed:.4f}")
    print(f"Avg opt loss:  {total_opt_loss / graphs_processed:.4f}")

    # 1. Attention shift by hop distance
    print(f"\n--- Attention Δ by Hop Distance (averaged across graphs) ---")
    max_hop_global = max(s["max_hop"] for s in all_stats)
    max_hop_show = min(max_hop_global, 15)
    hop_deltas_by_layer = defaultdict(lambda: np.zeros(max_hop_show + 1))
    hop_denoms = np.zeros(max_hop_show + 1)

    for s in all_stats:
        for layer_idx in range(args.num_layers):
            for h in range(min(s["max_hop"] + 1, max_hop_show + 1)):
                if s["hop_counts"][h] > 0:
                    hop_deltas_by_layer[layer_idx][h] += s["hop_delta"][layer_idx, h]
        for h in range(min(s["max_hop"] + 1, max_hop_show + 1)):
            if s["hop_counts"][h] > 0:
                hop_denoms[h] += 1

    print(f"{'Hop':>4}", end="")
    for l in range(args.num_layers):
        print(f"  {'L' + str(l):>8}", end="")
    print()

    for h in range(max_hop_show + 1):
        if hop_denoms[h] > 0:
            print(f"{h:4d}", end="")
            for l in range(args.num_layers):
                val = hop_deltas_by_layer[l][h] / hop_denoms[h]
                print(f"  {val:+8.5f}", end="")
            print()

    # 2. Entropy change
    print(f"\n--- Attention Entropy Change (proxy - vanilla) per Layer ---")
    for l in range(args.num_layers):
        deltas = [s["entropy_delta"][l] for s in all_stats]
        print(f"  Layer {l}: mean={np.mean(deltas):+.4f}  std={np.std(deltas):.4f}"
              f"  (vanilla={np.mean([s['entropy_vanilla'][l] for s in all_stats]):.4f}"
              f"  proxy={np.mean([s['entropy_proxy'][l] for s in all_stats]):.4f})")

    # 3. Top increased pairs: structural properties
    print(f"\n--- Top Attention-Increased Pairs: Structural Properties ---")
    all_top_hops = []
    all_top_degree_products = []
    for s in all_stats:
        for pair in s["top_increased_pairs"]:
            all_top_hops.append(pair["hop_dist"])
            all_top_degree_products.append(pair["degree_i"] * pair["degree_j"])

    if all_top_hops:
        hops_arr = np.array(all_top_hops)
        print(f"  Hop distance distribution of top-increased pairs:")
        for h in range(min(int(hops_arr.max()) + 1, 15)):
            count = (hops_arr == h).sum()
            if count > 0:
                frac = count / len(hops_arr)
                bar = "#" * int(frac * 50)
                print(f"    hop {h:2d}: {count:5d} ({frac:.1%}) {bar}")
        print(f"  Mean hop distance: {np.mean(all_top_hops):.2f}")
        print(f"  Mean degree product: {np.mean(all_top_degree_products):.2f}")

    # 4. Degree-dependent shift
    print(f"\n--- Degree-Dependent Attention Shift ---")
    degree_keys = set()
    for s in all_stats:
        degree_keys.update(s["degree_delta"].keys())
    for key in sorted(degree_keys):
        vals = [s["degree_delta"].get(key, 0) for s in all_stats if key in s["degree_delta"]]
        if vals:
            print(f"  {key}: mean={np.mean(vals):+.6f}")

    # 5. Proxy-degree correlation
    print(f"\n--- Proxy Attention vs Node Degree ---")
    corrs = [s["proxy_degree_corr"] for s in all_stats]
    print(f"  Correlation (proxy_attn, node_degree): "
          f"mean={np.mean(corrs):.4f}  std={np.std(corrs):.4f}")
    print(f"  Interpretation: {'proxies preferentially attend to high-degree nodes' if np.mean(corrs) > 0.1 else 'proxies attend to low-degree nodes' if np.mean(corrs) < -0.1 else 'no clear degree preference'}")

    # 6. Indirect connectivity through proxies by hop
    print(f"\n--- Indirect Connectivity Through Proxies by Hop Distance ---")
    indirect_by_hop_agg = np.zeros(max_hop_show + 1)
    indirect_counts = np.zeros(max_hop_show + 1)
    for s in all_stats:
        for h in range(min(len(s["indirect_connectivity_by_hop"]), max_hop_show + 1)):
            indirect_by_hop_agg[h] += s["indirect_connectivity_by_hop"][h]
            indirect_counts[h] += 1

    for h in range(max_hop_show + 1):
        if indirect_counts[h] > 0:
            val = indirect_by_hop_agg[h] / indirect_counts[h]
            bar = "#" * int(val * 5000)
            print(f"  hop {h:2d}: {val:.6f} {bar}")

    # Save raw stats
    save_path = os.path.join(args.save_dir, "attn_diagnostic_stats.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(all_stats, f)
    print(f"\nRaw statistics saved to {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="checkpoints_attn_distill")

    # Model (must match pretrained checkpoint)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=5)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--output_dim", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.1)

    # Proxy optimization
    p.add_argument("--num_proxies", type=int, default=4)
    p.add_argument("--proxy_lr", type=float, default=5e-2)
    p.add_argument("--proxy_opt_steps", type=int, default=75)

    # Analysis
    p.add_argument("--max_graphs", type=int, default=500,
                   help="Max training graphs to analyze")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)

    args = p.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    run_diagnostic(args)


if __name__ == "__main__":
    main()
