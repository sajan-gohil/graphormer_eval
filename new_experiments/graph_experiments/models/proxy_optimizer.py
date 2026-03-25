"""
Phase 2: Proxy Optimizer.

Optimizes M learnable proxy embeddings per graph (or batch) to improve
a frozen GPS model's predictions. Proxies participate in global self-attention
but NOT in the local MPNN.

Key design decision:
    The frozen GPSLayer does local MPNN + attention + FFN internally. For proxy
    insertion, we run a *custom forward pass* that:
      1. Runs local MPNN on original nodes only (proxies have no edges).
      2. Builds a combined dense batch [nodes | proxies] for global attention.
      3. Runs attention and FFN on the combined representation.
      4. Separates node and proxy features for the next layer.
    This ensures proxies only influence the model through the attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_dense_batch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mmd import mmd_squared


def forward_gps_layer_with_proxies(layer, x_sparse_nodes, edge_index, edge_attr,
                                   node_batch, proxy_embs, num_graphs):
    """
    Run a single GPSLayer with proxy embeddings injected into the attention step.

    Proxies participate in global self-attention + FFN, but NOT in the local MPNN.

    Args:
        layer: A frozen GPSLayer instance.
        x_sparse_nodes: (N_total, d) node features in sparse format (no proxies).
        edge_index: (2, E) edge indices for original graph.
        edge_attr: Edge attributes for original graph.
        node_batch: (N_total,) batch assignment for original nodes.
        proxy_embs: (B, M, d) proxy embeddings for each graph in the batch.
        num_graphs: B, number of graphs in the batch.

    Returns:
        x_sparse_nodes_out: (N_total, d) updated node features (sparse).
        proxy_embs_out: (B, M, d) updated proxy embeddings.
        attn_weights: (B, H, N_max+M, N_max+M) attention weights.
    """
    B = num_graphs
    M = proxy_embs.shape[1]
    d = proxy_embs.shape[2]

    # --- Step 1: Local MPNN on original nodes only ---
    local_out = layer.local_model(layer.local_norm(x_sparse_nodes), edge_index, edge_attr, node_batch)
    x_sparse_nodes = x_sparse_nodes + layer.dropout(local_out)

    # --- Step 2: Build combined dense batch [nodes | proxies] ---
    # Convert nodes to dense: (B, N_max, d) with mask (B, N_max)
    dense_nodes, node_mask = to_dense_batch(x_sparse_nodes, node_batch)
    N_max = dense_nodes.shape[1]

    # Concatenate proxies: (B, N_max + M, d)
    dense_combined = torch.cat([dense_nodes, proxy_embs], dim=1)

    # Build combined mask: nodes have their mask, proxies are always valid
    proxy_mask = torch.ones(B, M, dtype=torch.bool, device=node_mask.device)
    combined_mask = torch.cat([node_mask, proxy_mask], dim=1)  # (B, N_max + M)

    # --- Step 3: Global self-attention on combined ---
    attn_input = layer.global_norm(dense_combined)
    attn_out, attn_weights = layer.global_attn(attn_input, combined_mask)
    dense_combined = dense_combined + layer.dropout(attn_out)

    # --- Step 4: FFN on combined ---
    ffn_out = layer.ffn(layer.ffn_norm(dense_combined))
    dense_combined = dense_combined + ffn_out

    # --- Step 5: Separate nodes and proxies ---
    dense_nodes_out = dense_combined[:, :N_max, :]   # (B, N_max, d)
    proxy_embs_out = dense_combined[:, N_max:, :]     # (B, M, d)

    # Convert nodes back to sparse using the original mask
    x_sparse_nodes_out = dense_nodes_out[node_mask]   # (N_total, d)

    return x_sparse_nodes_out, proxy_embs_out, attn_weights


def forward_with_proxies(model, batch, proxy_embs, return_attention=False):
    """
    Run the full frozen GPSModel with proxy embeddings injected at every layer.

    Proxies participate in global attention but not local MPNN.
    Readout (mean pool) is over original nodes only — proxies excluded.

    Args:
        model: Frozen GPSModel instance.
        batch: PyG Batch object.
        proxy_embs: (B, M, d) proxy embeddings.
        return_attention: If True, also return attention weights from each layer.

    Returns:
        logits: (B, num_classes) classification logits.
        all_attn_weights: (optional) list of attention weight tensors per layer.
    """
    # Get initial node embeddings from frozen encoder
    h, batch_idx = model.get_initial_embeddings(batch)

    num_graphs = batch_idx.max().item() + 1
    current_proxies = proxy_embs  # (B, M, d)

    all_attn_weights = []
    for layer in model.layers:
        h, current_proxies, attn_w = forward_gps_layer_with_proxies(
            layer, h, batch.edge_index, batch.edge_attr,
            batch_idx, current_proxies, num_graphs
        )
        if return_attention:
            all_attn_weights.append(attn_w)

    # Readout: mean pool original nodes only (exclude proxies)
    h = model.post_norm(h)
    from torch_geometric.nn import global_mean_pool
    graph_emb = global_mean_pool(h, batch_idx)

    # Classify
    logits = model.classifier(graph_emb)

    if return_attention:
        return logits, all_attn_weights
    return logits


class ProxyOptimizer:
    """
    Optimizes M learnable proxy embeddings to improve a frozen GPS model's
    predictions on a given batch of graphs.

    The proxy embeddings B ∈ R^{M × d} are initialized from N(0, σ) where σ is
    the empirical std of the node embeddings. They are optimized via Adam to
    minimize: L_task + λ * MMD²(B, X).

    Usage:
        optimizer = ProxyOptimizer(frozen_model, config)
        result = optimizer.optimize(batch, device)
    """

    def __init__(self, model, M=8, lr=1e-2, num_iterations=500,
                 mmd_lambda=0.05, gradient_clip=1.0, num_restarts=5):
        """
        Args:
            model: Frozen GPSModel (must have requires_grad=False on all params).
            M: Number of proxy nodes per graph.
            lr: Learning rate for proxy optimization.
            num_iterations: Number of optimization steps per restart.
            mmd_lambda: Weight for MMD regularization.
            gradient_clip: Max gradient norm for clipping.
            num_restarts: Number of random restarts (keep best).
        """
        self.model = model
        self.M = M
        self.lr = lr
        self.num_iterations = num_iterations
        self.mmd_lambda = mmd_lambda
        self.gradient_clip = gradient_clip
        self.num_restarts = num_restarts
        self.d = model.hidden_dim

    def _init_proxies(self, batch, device):
        """
        Initialize proxy embeddings from N(0, σ) where σ is the empirical
        std of the initial node embeddings.

        Returns:
            B: nn.Parameter of shape (num_graphs, M, d)
        """
        with torch.no_grad():
            h, batch_idx = self.model.get_initial_embeddings(batch)
            sigma = h.std().item()
            num_graphs = batch_idx.max().item() + 1

        B = torch.randn(num_graphs, self.M, self.d, device=device) * sigma
        B = nn.Parameter(B)
        return B, h, batch_idx

    def _compute_loss(self, batch, B, h_init, batch_idx, criterion):
        """
        Compute task loss + MMD regularization.

        Returns:
            total_loss, task_loss, mmd_loss, logits
        """
        logits = forward_with_proxies(self.model, batch, B)
        task_loss = criterion(logits, batch.y.float())

        # MMD: compare proxy embeddings to node embeddings (per-graph average)
        if self.mmd_lambda > 0:
            # Flatten proxies: (B*M, d) vs all node embeddings (N_total, d)
            B_flat = B.reshape(-1, self.d)
            mmd_loss = mmd_squared(B_flat, h_init.detach())
            total_loss = task_loss + self.mmd_lambda * mmd_loss
        else:
            mmd_loss = torch.tensor(0.0, device=B.device)
            total_loss = task_loss

        return total_loss, task_loss, mmd_loss, logits

    @staticmethod
    def _count_class_corrections(baseline_probs, optimized_probs, labels, threshold=0.5):
        """
        Count how many classes were corrected by proxy optimization.

        A class is "corrected" if:
          - For a positive class (label=1): baseline sigmoid < 0.5, optimized >= 0.5
          - For a negative class (label=0): baseline sigmoid >= 0.5, optimized < 0.5

        Args:
            baseline_probs: (N, C) baseline sigmoid probabilities.
            optimized_probs: (N, C) optimized sigmoid probabilities.
            labels: (N, C) binary ground truth.
            threshold: decision threshold (default 0.5).

        Returns:
            dict with correction counts and details.
        """
        baseline_pred = (baseline_probs >= threshold).astype(int)
        optimized_pred = (optimized_probs >= threshold).astype(int)

        baseline_correct = (baseline_pred == labels)
        optimized_correct = (optimized_pred == labels)

        # Classes that went from wrong → right
        newly_correct = (~baseline_correct & optimized_correct).sum()
        # Classes that went from right → wrong
        newly_wrong = (baseline_correct & ~optimized_correct).sum()

        total_classes = labels.size
        baseline_num_correct = baseline_correct.sum()
        optimized_num_correct = optimized_correct.sum()

        return {
            'newly_correct': int(newly_correct),
            'newly_wrong': int(newly_wrong),
            'net_corrections': int(newly_correct) - int(newly_wrong),
            'baseline_correct': int(baseline_num_correct),
            'optimized_correct': int(optimized_num_correct),
            'total_classes': int(total_classes),
        }

    def optimize(self, batch, device, log_trajectory=False, log_every=50):
        """
        Run proxy optimization with multiple random restarts.

        Args:
            batch: PyG Batch object (single batch of graphs).
            device: torch device.
            log_trajectory: If True, log loss at every `log_every` iterations.
            log_every: Logging frequency for trajectory.

        Returns:
            dict with keys:
                'best_B': (num_graphs, M, d) optimized proxy embeddings
                'baseline_loss': float, loss without proxies
                'optimized_loss': float, best loss with proxies
                'baseline_probs': np.ndarray (N, C) baseline sigmoid probs
                'optimized_probs': np.ndarray (N, C) optimized sigmoid probs
                'labels': np.ndarray (N, C) ground truth binary labels
                'loss_improvement': float, baseline_loss - optimized_loss
                'class_corrections': dict with newly_correct, newly_wrong, etc.
                'trajectory': list of dicts (if log_trajectory=True)
                'final_mmd': float, MMD² at convergence
                'init_mmd': float, MMD² at initialization
                'proxy_norms': (num_graphs, M) L2 norms of final proxies
                'proxy_cosine_sims': (M, M) pairwise cosine similarities of avg proxies
                'last_attn_weights': attention weights from last layer (if log_trajectory)
        """
        batch = batch.to(device)
        criterion = nn.BCEWithLogitsLoss()

        # --- Compute baseline (no proxies) ---
        self.model.eval()
        with torch.no_grad():
            baseline_logits = self.model(batch)
            baseline_loss = criterion(baseline_logits, batch.y.float()).item()
            baseline_probs = torch.sigmoid(baseline_logits).cpu().numpy()
            baseline_labels = batch.y.cpu().numpy()

        # --- Multiple random restarts ---
        best_result = None
        best_total_loss = float('inf')

        for restart in range(self.num_restarts):
            B, h_init, batch_idx = self._init_proxies(batch, device)
            opt = torch.optim.Adam([B], lr=self.lr)

            trajectory = []

            # Record initial MMD
            with torch.no_grad():
                init_mmd = mmd_squared(
                    B.reshape(-1, self.d), h_init.detach()
                ).item()

            for step in range(self.num_iterations):
                opt.zero_grad()
                total_loss, task_loss, mmd_loss, logits = self._compute_loss(
                    batch, B, h_init, batch_idx, criterion
                )
                total_loss.backward()

                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_([B], self.gradient_clip)

                opt.step()

                # Trajectory logging — use loss and class corrections, not degenerate AP
                if log_trajectory and (step % log_every == 0 or step == self.num_iterations - 1):
                    with torch.no_grad():
                        step_probs = torch.sigmoid(logits).cpu().numpy()
                        step_corrections = self._count_class_corrections(
                            baseline_probs, step_probs, baseline_labels
                        )
                        trajectory.append({
                            'step': step,
                            'total_loss': total_loss.item(),
                            'task_loss': task_loss.item(),
                            'mmd_loss': mmd_loss.item(),
                            'net_corrections': step_corrections['net_corrections'],
                            'newly_correct': step_corrections['newly_correct'],
                        })

            # Evaluate this restart
            with torch.no_grad():
                final_total_loss, final_task_loss, final_mmd, final_logits = \
                    self._compute_loss(batch, B, h_init, batch_idx, criterion)
                final_probs = torch.sigmoid(final_logits).cpu().numpy()
                final_mmd_val = final_mmd.item()

            if final_total_loss.item() < best_total_loss:
                best_total_loss = final_total_loss.item()

                # Class correction analysis
                corrections = self._count_class_corrections(
                    baseline_probs, final_probs, baseline_labels
                )

                # Compute diagnostics
                proxy_norms = B.detach().norm(dim=-1)  # (B_graphs, M)
                avg_proxy = B.detach().mean(dim=0)  # (M, d)
                avg_proxy_normed = F.normalize(avg_proxy, dim=-1)
                cosine_sims = torch.mm(avg_proxy_normed, avg_proxy_normed.t())  # (M, M)

                # Get attention weights from last layer
                last_attn = None
                if log_trajectory:
                    _, all_attn = forward_with_proxies(
                        self.model, batch, B.detach(), return_attention=True
                    )
                    last_attn = all_attn[-1].cpu()  # (B, H, N+M, N+M)

                best_result = {
                    'best_B': B.detach().cpu(),
                    'baseline_loss': baseline_loss,
                    'optimized_loss': final_task_loss.item(),
                    'loss_improvement': baseline_loss - final_task_loss.item(),
                    'baseline_probs': baseline_probs,
                    'optimized_probs': final_probs,
                    'labels': baseline_labels,
                    'class_corrections': corrections,
                    'trajectory': trajectory if log_trajectory else None,
                    'init_mmd': init_mmd,
                    'final_mmd': final_mmd_val,
                    'proxy_norms': proxy_norms.cpu(),
                    'proxy_cosine_sims': cosine_sims.cpu(),
                    'last_attn_weights': last_attn,
                    'restart_idx': restart,
                }

        return best_result
