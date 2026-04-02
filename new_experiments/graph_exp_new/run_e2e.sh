#!/usr/bin/env bash
set -eux pipefail

# =============================================================
# End-to-End experiments (Pipeline B)
# Varies: generator type, readout scope
# =============================================================

BASE_DIR="checkpoints_e2e"

# --- Score-based generator ---
echo "=== E2E: score_based, nodes_only ==="
python main_e2e.py \
    --generator score_based \
    --readout_scope nodes_only \
    --save_dir "${BASE_DIR}/score_based_nodes_only" \
    --batch_size 64

echo "=== E2E: score_based, all_tokens ==="
python main_e2e.py \
    --generator score_based \
    --readout_scope all_tokens \
    --save_dir "${BASE_DIR}/score_based_all_tokens" \
    --batch_size 64

# --- GNN pooling generator ---
echo "=== E2E: gnn_pooling, nodes_only ==="
python main_e2e.py \
    --generator gnn_pooling \
    --readout_scope nodes_only \
    --save_dir "${BASE_DIR}/gnn_pooling_nodes_only" \
    --batch_size 64

echo "=== E2E: gnn_pooling, all_tokens ==="
python main_e2e.py \
    --generator gnn_pooling \
    --readout_scope all_tokens \
    --save_dir "${BASE_DIR}/gnn_pooling_all_tokens" \
    --batch_size 64

echo "All E2E experiments complete."
