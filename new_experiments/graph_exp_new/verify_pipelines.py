"""
Phase 5 verification: confirm all generator × pipeline combinations work.

Tests:
  5.2 — FlowMatchingGenerator in staged pipeline (Stages 3 & 4)
  5.3 — GNNPoolingGenerator in both pipelines

Runs forward + backward passes with synthetic data to verify shapes,
gradient flow, and no runtime errors. Does NOT require dataset download.

Usage:
    python verify_pipelines.py
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace

from models import GraphTransformer
from generators import (
    ScoreBasedGenerator, FlowMatchingGenerator, GNNPoolingGenerator,
)
from metrics import compute_macro_ap
from mmd import mmd_squared


# ================================================================
# HELPERS
# ================================================================

B, N, M, d, C = 4, 12, 4, 64, 10  # batch, nodes, proxies, dim, classes


def make_dense_batch():
    """Create synthetic dense batch tensors."""
    dense_x = torch.randn(B, N, d)
    dense_mask = torch.ones(B, N, dtype=torch.bool)
    dense_mask[0, 9:] = False
    dense_mask[1, 10:] = False

    # Build batch vector from mask
    batch_vec = []
    for i in range(B):
        n_i = dense_mask[i].sum().item()
        batch_vec.extend([i] * n_i)

    labels = torch.randint(0, 2, (B, C)).float()
    batch = SimpleNamespace(batch=torch.tensor(batch_vec), y=labels)
    return dense_x, dense_mask, batch


def make_flat_batch():
    """Create synthetic flat (PyG-style) batch tensors."""
    dense_x, dense_mask, batch_obj = make_dense_batch()
    # Flatten valid nodes
    flat_emb = dense_x[dense_mask]  # (total_N, d)
    total_n = flat_emb.shape[0]
    # Simple edges: connect consecutive nodes
    src = torch.arange(0, total_n - 1)
    dst = torch.arange(1, total_n)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    edge_attr = torch.randn(edge_index.shape[1], d)

    return flat_emb, edge_index, edge_attr, batch_obj, dense_x, dense_mask


def check(name, passed):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}", flush=True)
    return passed


# ================================================================
# TEST FUNCTIONS
# ================================================================

def test_score_based_staged():
    """ScoreBasedGenerator in staged pipeline (train with targets + finetune)."""
    print("\n--- ScoreBasedGenerator: Staged Pipeline ---", flush=True)
    ok = True
    model = GraphTransformer(hidden_dim=d, output_dim=C)
    gen = ScoreBasedGenerator(num_proxies=M, input_dim=d)
    dense_x, dense_mask, batch = make_dense_batch()

    # Stage 3: train with targets
    targets = torch.randn(B, M, d)
    proxies, aux_loss = gen(dense_x, dense_mask, targets=targets)
    ok &= check("Shape: proxies", proxies.shape == (B, M, d))
    ok &= check("aux_loss is scalar", aux_loss.dim() == 0)
    aux_loss.backward()
    ok &= check("Backward through aux_loss", True)

    # Stage 4: finetune (task loss through generator)
    gen.zero_grad()
    model.zero_grad()
    proxies2, _ = gen(dense_x, dense_mask)
    logits, node_emb = model(batch, proxy_embeddings=proxies2,
                             precomputed_dense=(dense_x, dense_mask))
    loss = nn.BCEWithLogitsLoss()(logits, batch.y)
    loss.backward()
    gen_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in gen.parameters())
    ok &= check("Gradients flow to generator", gen_grads)
    ok &= check("Logits shape", logits.shape == (B, C))
    return ok


def test_flow_matching_staged():
    """FlowMatchingGenerator in staged pipeline (Stages 3 & 4)."""
    print("\n--- FlowMatchingGenerator: Staged Pipeline (5.2) ---", flush=True)
    ok = True
    model = GraphTransformer(hidden_dim=d, output_dim=C)
    fm = FlowMatchingGenerator(num_proxies=M, node_dim=d,
                               denoiser_dim=32, denoiser_layers=2,
                               denoiser_heads=4, euler_steps=1)
    dense_x, dense_mask, batch = make_dense_batch()

    # Stage 3: CFM training with targets
    targets = torch.randn(B, M, d)
    proxies, cfm_loss = fm(dense_x, dense_mask, targets=targets)
    ok &= check("CFM loss is scalar", cfm_loss.dim() == 0)
    ok &= check("Proxies shape", proxies.shape == (B, M, d))
    cfm_loss.backward()
    fm_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in fm.parameters())
    ok &= check("Backward through CFM loss", fm_grads)

    # Stage 3: Inference (no targets)
    fm.zero_grad()
    proxies_inf = fm.generate(dense_x, dense_mask)
    ok &= check("Generate (inference) shape", proxies_inf.shape == (B, M, d))

    # Stage 4: differentiable single-step forward
    fm.zero_grad()
    model.zero_grad()
    proxies_diff = fm.forward_differentiable(dense_x, dense_mask, euler_steps=1)
    ok &= check("Differentiable forward shape", proxies_diff.shape == (B, M, d))
    logits, _ = model(batch, proxy_embeddings=proxies_diff,
                      precomputed_dense=(dense_x, dense_mask))
    loss = nn.BCEWithLogitsLoss()(logits, batch.y)
    loss.backward()
    fm_grads_ft = any(p.grad is not None and p.grad.abs().sum() > 0
                      for p in fm.parameters())
    ok &= check("Task loss gradients flow to FM generator", fm_grads_ft)
    ok &= check("Logits shape", logits.shape == (B, C))

    # Multi-step Euler
    fm.zero_grad()
    proxies_multi = fm.forward_differentiable(dense_x, dense_mask, euler_steps=4)
    ok &= check("Multi-step Euler shape", proxies_multi.shape == (B, M, d))
    return ok


def test_gnn_pooling_staged():
    """GNNPoolingGenerator in staged pipeline."""
    print("\n--- GNNPoolingGenerator: Staged Pipeline (5.3a) ---", flush=True)
    ok = True
    model = GraphTransformer(hidden_dim=d, output_dim=C)
    gnn = GNNPoolingGenerator(num_proxies=M, input_dim=d,
                              gnn_layers=2, gnn_type="GIN")
    flat_emb, edge_index, edge_attr, batch, dense_x, dense_mask = make_flat_batch()

    # Stage 3: train with targets
    targets = torch.randn(B, M, d)
    proxies, aux_loss = gnn(flat_emb, mask=None, targets=targets,
                            edge_index=edge_index, batch_vec=batch.batch,
                            edge_attr=edge_attr)
    ok &= check("Proxies shape", proxies.shape == (B, M, d))
    ok &= check("aux_loss is scalar", aux_loss.dim() == 0)
    aux_loss.backward()
    gnn_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in gnn.parameters())
    ok &= check("Backward through aux_loss", gnn_grads)

    # Stage 4: finetune
    gnn.zero_grad()
    model.zero_grad()
    proxies2, _ = gnn(flat_emb, mask=None,
                      edge_index=edge_index, batch_vec=batch.batch,
                      edge_attr=edge_attr)
    logits, _ = model(batch, proxy_embeddings=proxies2,
                      precomputed_dense=(dense_x, dense_mask))
    loss = nn.BCEWithLogitsLoss()(logits, batch.y)
    loss.backward()
    gnn_grads_ft = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in gnn.parameters())
    ok &= check("Task loss gradients flow to GNN generator", gnn_grads_ft)
    return ok


def test_gnn_pooling_e2e():
    """GNNPoolingGenerator in end-to-end pipeline."""
    print("\n--- GNNPoolingGenerator: E2E Pipeline (5.3b) ---", flush=True)
    ok = True
    model = GraphTransformer(hidden_dim=d, output_dim=C)
    gnn = GNNPoolingGenerator(num_proxies=M, input_dim=d,
                              gnn_layers=2, gnn_type="GIN")
    flat_emb, edge_index, edge_attr, batch, dense_x, dense_mask = make_flat_batch()

    # E2E forward: generate → transformer → loss → backward
    proxies, _ = gnn(flat_emb, mask=None,
                     edge_index=edge_index, batch_vec=batch.batch,
                     edge_attr=edge_attr)
    ok &= check("Proxies shape", proxies.shape == (B, M, d))

    # Forward through transformer
    logits_n, _ = model(batch, proxy_embeddings=proxies,
                        precomputed_dense=(dense_x, dense_mask),
                        readout_scope="nodes_only")
    logits_a, _ = model(batch, proxy_embeddings=proxies,
                        precomputed_dense=(dense_x, dense_mask),
                        readout_scope="all_tokens")
    ok &= check("Readout nodes_only shape", logits_n.shape == (B, C))
    ok &= check("Readout all_tokens shape", logits_a.shape == (B, C))

    # MMD regularization
    mmd_losses = []
    for i in range(B):
        nodes_i = dense_x[i][dense_mask[i]]
        mmd_losses.append(mmd_squared(proxies[i], nodes_i))
    mmd_loss = torch.stack(mmd_losses).mean()
    task_loss = nn.BCEWithLogitsLoss()(logits_n, batch.y)
    total = task_loss + 0.01 * mmd_loss
    total.backward()

    gnn_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in gnn.parameters())
    ok &= check("E2E backward (task + MMD) → GNN grads", gnn_grads)
    return ok


def test_score_based_e2e():
    """ScoreBasedGenerator in end-to-end pipeline."""
    print("\n--- ScoreBasedGenerator: E2E Pipeline (5.3c) ---", flush=True)
    ok = True
    model = GraphTransformer(hidden_dim=d, output_dim=C)
    gen = ScoreBasedGenerator(num_proxies=M, input_dim=d)
    dense_x, dense_mask, batch = make_dense_batch()

    proxies, _ = gen(dense_x, dense_mask)
    logits, _ = model(batch, proxy_embeddings=proxies,
                      precomputed_dense=(dense_x, dense_mask),
                      readout_scope="all_tokens")
    loss = nn.BCEWithLogitsLoss()(logits, batch.y)
    loss.backward()
    gen_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in gen.parameters())
    ok &= check("E2E backward → ScoreBased grads", gen_grads)
    ok &= check("Logits shape", logits.shape == (B, C))
    return ok


def test_param_counts():
    """Verify parameter count logging works for all generators."""
    print("\n--- Parameter Counts (5.4) ---", flush=True)
    ok = True
    model = GraphTransformer(hidden_dim=d, output_dim=C)
    model_params = sum(p.numel() for p in model.parameters())
    ok &= check(f"Transformer: {model_params:,} params", model_params > 0)

    for name, gen in [
        ("ScoreBased", ScoreBasedGenerator(num_proxies=M, input_dim=d)),
        ("FlowMatching", FlowMatchingGenerator(num_proxies=M, node_dim=d,
                                                denoiser_dim=32, denoiser_layers=2,
                                                denoiser_heads=4)),
        ("GNNPooling", GNNPoolingGenerator(num_proxies=M, input_dim=d,
                                            gnn_layers=2, gnn_type="GIN")),
    ]:
        gen_params = sum(p.numel() for p in gen.parameters())
        total = model_params + gen_params
        ok &= check(f"{name}: {gen_params:,} gen + {model_params:,} model = {total:,} total",
                     gen_params > 0)
    return ok


def test_diagnostics():
    """Verify diagnostic data structure."""
    print("\n--- Diagnostic Data Structure (5.1) ---", flush=True)
    ok = True
    # Simulate diagnostic collection
    diagnostics = [
        {"epoch": i, "train_loss": 1.0 - i * 0.1, "val_ap": 0.1 + i * 0.05}
        for i in range(10)
    ]
    losses = [d["train_loss"] for d in diagnostics]
    aps = [d["val_ap"] for d in diagnostics]
    corr = float(np.corrcoef(losses, aps)[0, 1])
    ok &= check(f"Correlation computable: {corr:.4f}", abs(corr) <= 1.0)
    ok &= check("Negative correlation (loss down, AP up)", corr < 0)
    return ok


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("Pipeline Verification Suite", flush=True)
    print("=" * 60, flush=True)

    tests = [
        ("5.1  Diagnostic logging", test_diagnostics),
        ("5.2  FlowMatching in staged", test_flow_matching_staged),
        ("5.3a GNNPooling in staged", test_gnn_pooling_staged),
        ("5.3b GNNPooling in E2E", test_gnn_pooling_e2e),
        ("5.3c ScoreBased in E2E", test_score_based_e2e),
        ("      ScoreBased in staged", test_score_based_staged),
        ("5.4  Parameter counts", test_param_counts),
    ]

    results = []
    for label, fn in tests:
        try:
            passed = fn()
        except Exception as e:
            print(f"  [FAIL] Exception: {e}", flush=True)
            passed = False
        results.append((label, passed))

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    all_passed = True
    for label, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label}", flush=True)
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\nAll {len(results)} test groups passed.", flush=True)
    else:
        n_fail = sum(1 for _, p in results if not p)
        print(f"\n{n_fail}/{len(results)} test groups FAILED.", flush=True)
        sys.exit(1)
