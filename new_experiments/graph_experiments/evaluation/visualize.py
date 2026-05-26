"""
Phase 4: Visualization — t-SNE / UMAP of proxy + node embeddings.

Generates visualizations to determine whether optimized proxy embeddings
show meaningful variation across graphs (supporting flow matching) or are
similar across graphs (favoring fixed tokens).

Produces:
  1. t-SNE scatter plot: proxy + node embeddings colored by graph identity.
  2. Proxy diversity analysis: inter-graph vs intra-graph proxy variance.
  3. MMD Pareto frontier: task AP vs distribution alignment for λ sweep.

Usage:
    python evaluation/visualize.py --checkpoint ./checkpoints/best_model.pt
    python evaluation/visualize.py --phase4_log ./logs/phase4_*.json  # for Pareto plot
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.phase1_config import Phase1Config
from data.peptides_func import get_peptides_func_loaders
from models.transformer import GPSModel
from models.proxy_optimizer import ProxyOptimizer
from training.validate_premise import load_frozen_model, set_seed
from torch_geometric.loader import DataLoader


def collect_proxy_and_node_embeddings(model, train_data, indices, device,
                                      M=8, proxy_lr=1e-2, proxy_iterations=500,
                                      mmd_lambda=0.05, num_restarts=3):
    """
    For each graph, optimize proxy embeddings and collect:
      - Node embeddings (initial, from the frozen encoder).
      - Optimized proxy embeddings.
      - Graph identity label.

    Args:
        model: Frozen GPSModel.
        train_data: Dataset list.
        indices: Graph indices to process.
        device: torch device.

    Returns:
        all_node_embs: (N_total, d) numpy array.
        all_proxy_embs: (len(indices)*M, d) numpy array.
        node_graph_ids: (N_total,) graph identity for each node.
        proxy_graph_ids: (len(indices)*M,) graph identity for each proxy.
    """
    optimizer = ProxyOptimizer(
        model, M=M, lr=proxy_lr, num_iterations=proxy_iterations,
        mmd_lambda=mmd_lambda, gradient_clip=1.0, num_restarts=num_restarts,
    )

    all_node_embs = []
    all_proxy_embs = []
    node_graph_ids = []
    proxy_graph_ids = []

    for i, idx in enumerate(indices):
        data = train_data[idx]
        loader = DataLoader([data], batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)

        # Get node embeddings
        with torch.no_grad():
            h, batch_idx = model.get_initial_embeddings(batch)

        # Optimize proxies
        result = optimizer.optimize(batch, device, log_trajectory=False)
        proxy_B = result['best_B'].squeeze(0)  # (M, d)

        all_node_embs.append(h.cpu().numpy())
        all_proxy_embs.append(proxy_B.numpy())
        node_graph_ids.extend([i] * h.shape[0])
        proxy_graph_ids.extend([i] * M)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(indices)} graphs")

    all_node_embs = np.concatenate(all_node_embs, axis=0)
    all_proxy_embs = np.concatenate(all_proxy_embs, axis=0)
    node_graph_ids = np.array(node_graph_ids)
    proxy_graph_ids = np.array(proxy_graph_ids)

    return all_node_embs, all_proxy_embs, node_graph_ids, proxy_graph_ids


def compute_proxy_diversity(all_proxy_embs, proxy_graph_ids, M):
    """
    Analyze whether proxies show meaningful per-graph variation.

    Computes:
      - Intra-graph variance: avg variance of proxies within a graph.
      - Inter-graph variance: variance of per-graph mean proxies across graphs.
      - Ratio: inter/intra — high ratio means proxies are graph-specific.

    Returns:
        dict with diversity metrics.
    """
    unique_graphs = np.unique(proxy_graph_ids)
    num_graphs = len(unique_graphs)

    # Per-graph proxy sets
    per_graph_proxies = []
    per_graph_means = []
    for g in unique_graphs:
        mask = proxy_graph_ids == g
        proxies_g = all_proxy_embs[mask]  # (M, d)
        per_graph_proxies.append(proxies_g)
        per_graph_means.append(proxies_g.mean(axis=0))

    per_graph_means = np.stack(per_graph_means)  # (num_graphs, d)

    # Intra-graph variance (average across graphs)
    intra_vars = []
    for proxies_g in per_graph_proxies:
        intra_vars.append(np.var(proxies_g, axis=0).mean())
    intra_var = np.mean(intra_vars)

    # Inter-graph variance
    inter_var = np.var(per_graph_means, axis=0).mean()

    # Pairwise cosine similarity between per-graph mean proxies
    norms = np.linalg.norm(per_graph_means, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = per_graph_means / norms
    cosine_matrix = normed @ normed.T

    # Off-diagonal mean
    mask = ~np.eye(num_graphs, dtype=bool)
    mean_cross_cosine = cosine_matrix[mask].mean()

    return {
        'intra_graph_variance': float(intra_var),
        'inter_graph_variance': float(inter_var),
        'variance_ratio': float(inter_var / (intra_var + 1e-8)),
        'mean_cross_graph_cosine': float(mean_cross_cosine),
        'num_graphs': num_graphs,
        'interpretation': (
            "HIGH ratio + LOW cosine → proxies are graph-specific → flow matching justified. "
            "LOW ratio + HIGH cosine → proxies similar across graphs → fixed tokens may suffice."
        ),
    }


def generate_tsne_data(all_node_embs, all_proxy_embs, node_graph_ids, proxy_graph_ids):
    """
    Run t-SNE on combined node + proxy embeddings.

    Returns:
        coords_2d: (N_total + N_proxies, 2) t-SNE coordinates.
        is_proxy: boolean array indicating proxy vs node.
        graph_ids: graph identity for each point.
    """
    from sklearn.manifold import TSNE

    combined = np.concatenate([all_node_embs, all_proxy_embs], axis=0)
    is_proxy = np.concatenate([
        np.zeros(len(all_node_embs), dtype=bool),
        np.ones(len(all_proxy_embs), dtype=bool),
    ])
    graph_ids = np.concatenate([node_graph_ids, proxy_graph_ids])

    print(f"  Running t-SNE on {combined.shape[0]} points ({combined.shape[1]}d)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    coords_2d = tsne.fit_transform(combined)

    return coords_2d, is_proxy, graph_ids


def save_tsne_plot(coords_2d, is_proxy, graph_ids, output_path):
    """
    Save t-SNE visualization as a matplotlib figure.
    Nodes as small dots, proxies as large stars, colored by graph identity.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("  matplotlib not available; skipping plot generation.")
        print("  t-SNE data saved to JSON — plot externally.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    unique_graphs = np.unique(graph_ids)
    cmap = plt.cm.get_cmap('tab20', len(unique_graphs))
    color_map = {g: cmap(i) for i, g in enumerate(unique_graphs)}

    # Plot nodes (small, semi-transparent)
    node_mask = ~is_proxy
    for g in unique_graphs:
        g_mask = (graph_ids == g) & node_mask
        if g_mask.any():
            ax.scatter(
                coords_2d[g_mask, 0], coords_2d[g_mask, 1],
                c=[color_map[g]], s=8, alpha=0.3, label=f'Graph {g} nodes'
            )

    # Plot proxies (large stars)
    proxy_mask = is_proxy
    for g in unique_graphs:
        g_mask = (graph_ids == g) & proxy_mask
        if g_mask.any():
            ax.scatter(
                coords_2d[g_mask, 0], coords_2d[g_mask, 1],
                c=[color_map[g]], s=200, marker='*', edgecolors='black',
                linewidths=0.5, zorder=5
            )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=6, label='Node embeddings'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markersize=15, markeredgecolor='black', label='Proxy embeddings'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    ax.set_title('t-SNE: Optimized Proxy Embeddings vs Node Embeddings\n'
                 '(colored by graph identity)', fontsize=13)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  t-SNE plot saved to: {output_path}")


def save_pareto_plot(mmd_results, output_path):
    """
    Plot the MMD Pareto frontier: task AP vs distribution alignment.

    Args:
        mmd_results: Dict from Phase 4 ablation (mmd_sensitivity key).
        output_path: Where to save the figure.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping Pareto plot.")
        return

    lambdas = []
    aps = []
    mmds = []
    norms = []

    for label, summary in mmd_results.items():
        lam = float(label.split('_')[1])
        lambdas.append(lam)
        aps.append(summary['mean_ap_improvement'])
        mmds.append(summary['mean_final_mmd'])
        norms.append(summary.get('mean_proxy_norms', 0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: AP improvement vs λ
    axes[0].plot(lambdas, aps, 'bo-', markersize=8)
    axes[0].set_xlabel('λ_MMD')
    axes[0].set_ylabel('Mean AP Improvement')
    axes[0].set_title('AP Improvement vs MMD Weight')
    axes[0].set_xscale('symlog', linthresh=0.01)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Final MMD vs λ
    axes[1].plot(lambdas, mmds, 'rs-', markersize=8)
    axes[1].set_xlabel('λ_MMD')
    axes[1].set_ylabel('Final MMD²')
    axes[1].set_title('Distribution Alignment vs MMD Weight')
    axes[1].set_xscale('symlog', linthresh=0.01)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Pareto frontier (AP vs MMD)
    axes[2].scatter(mmds, aps, c='green', s=100, zorder=5)
    for i, lam in enumerate(lambdas):
        axes[2].annotate(f'λ={lam}', (mmds[i], aps[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=9)
    axes[2].set_xlabel('Final MMD² (distribution alignment)')
    axes[2].set_ylabel('Mean AP Improvement')
    axes[2].set_title('Pareto Frontier: Performance vs Alignment')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Pareto plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Visualization")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--num_graphs", type=int, default=25,
                        help="Number of graphs for t-SNE visualization (20-30)")
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--phase4_log", type=str, default=None,
                        help="Path to Phase 4 ablation log (for Pareto plot)")
    parser.add_argument("--output_dir", type=str, default="./logs/visualizations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = Phase1Config()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_seed(args.seed)

    # Load data and model
    train_loader, val_loader, test_loader, dataset_info = get_peptides_func_loaders(config.data)
    train_data = train_loader.dataset
    model = load_frozen_model(args.checkpoint, config, device)

    # Select graphs for visualization
    indices = list(range(min(args.num_graphs, len(train_data))))

    # 1. Collect embeddings
    print("\n[1/3] Collecting proxy and node embeddings...")
    node_embs, proxy_embs, node_gids, proxy_gids = collect_proxy_and_node_embeddings(
        model, train_data, indices, device, M=args.M
    )

    # 2. Diversity analysis
    print("\n[2/3] Computing proxy diversity metrics...")
    diversity = compute_proxy_diversity(proxy_embs, proxy_gids, args.M)
    print(f"  Intra-graph variance: {diversity['intra_graph_variance']:.6f}")
    print(f"  Inter-graph variance: {diversity['inter_graph_variance']:.6f}")
    print(f"  Variance ratio (inter/intra): {diversity['variance_ratio']:.4f}")
    print(f"  Mean cross-graph cosine sim: {diversity['mean_cross_graph_cosine']:.4f}")

    # 3. t-SNE
    print("\n[3/3] Generating t-SNE visualization...")
    coords, is_proxy, gids = generate_tsne_data(node_embs, proxy_embs, node_gids, proxy_gids)

    tsne_plot_path = os.path.join(args.output_dir, "tsne_proxy_node_embeddings.png")
    save_tsne_plot(coords, is_proxy, gids, tsne_plot_path)

    # Save raw data
    tsne_data_path = os.path.join(args.output_dir, "tsne_data.json")
    with open(tsne_data_path, 'w') as f:
        json.dump({
            'coords': coords.tolist(),
            'is_proxy': is_proxy.tolist(),
            'graph_ids': gids.tolist(),
            'diversity': diversity,
        }, f, indent=2)
    print(f"  t-SNE data saved to: {tsne_data_path}")

    # 4. Pareto plot (if Phase 4 log provided)
    if args.phase4_log and os.path.exists(args.phase4_log):
        print("\n[Bonus] Generating MMD Pareto frontier plot...")
        with open(args.phase4_log) as f:
            phase4_data = json.load(f)
        if 'mmd_sensitivity' in phase4_data:
            pareto_path = os.path.join(args.output_dir, "mmd_pareto_frontier.png")
            save_pareto_plot(phase4_data['mmd_sensitivity'], pareto_path)

    # Summary
    print(f"\n{'='*60}")
    print("VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Graphs analyzed: {len(indices)}")
    print(f"  Proxies per graph: {args.M}")
    print(f"  Variance ratio: {diversity['variance_ratio']:.4f}")
    if diversity['variance_ratio'] > 1.0 and diversity['mean_cross_graph_cosine'] < 0.8:
        print("  → Proxies show graph-specific variation → flow matching justified")
    elif diversity['mean_cross_graph_cosine'] > 0.9:
        print("  → Proxies very similar across graphs → fixed tokens may suffice")
    else:
        print("  → Moderate variation → investigate further")


if __name__ == "__main__":
    main()
