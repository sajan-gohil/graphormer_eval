#!/usr/bin/env bash
set -eux pipefail

# =============================================================
# Three-Staged experiments
# Stage 1 is shared — train once, reuse for all generators.
# Stage 2+3 run per generator type.
# =============================================================

BASE_DIR="checkpoints_three_staged"
SHARED_DIR="${BASE_DIR}/shared"

# --- Stage 1: Pretrain transformer (shared) ---
echo "=== Three-Staged: Stage 1 (shared pretrain) ==="
python3 main_three_staged.py \
    --stage 1 \
    --save_dir "${SHARED_DIR}"

MODEL_PATH="${SHARED_DIR}/stage1_best.pt"

# =============================================================
# Stage 2 + 3: Per-generator experiments
# =============================================================

# --- Score-based generator (highest priority) ---
echo "=== Three-Staged: score_based (stage 2) ==="
python3 main_three_staged.py \
    --stage 2 \
    --generator score_based \
    --model_path "${MODEL_PATH}" \
    --save_dir "${BASE_DIR}/score_based"

echo "=== Three-Staged: score_based (stage 3) ==="
python3 main_three_staged.py \
    --stage 3 \
    --generator score_based \
    --model_path "${MODEL_PATH}" \
    --generator_path "${BASE_DIR}/score_based/stage2_generator.pt" \
    --save_dir "${BASE_DIR}/score_based"

# --- PMA generator (farthest_point queries) ---
echo "=== Three-Staged: pma/farthest_point (stage 2) ==="
python3 main_three_staged.py \
    --stage 2 \
    --generator pma \
    --pma_query_mode farthest_point \
    --model_path "${MODEL_PATH}" \
    --save_dir "${BASE_DIR}/pma_fp"

echo "=== Three-Staged: pma/farthest_point (stage 3) ==="
python3 main_three_staged.py \
    --stage 3 \
    --generator pma \
    --pma_query_mode farthest_point \
    --model_path "${MODEL_PATH}" \
    --generator_path "${BASE_DIR}/pma_fp/stage2_generator.pt" \
    --save_dir "${BASE_DIR}/pma_fp"

# --- PMA generator (soft_kmeans queries) ---
echo "=== Three-Staged: pma/soft_kmeans (stage 2) ==="
python3 main_three_staged.py \
    --stage 2 \
    --generator pma \
    --pma_query_mode soft_kmeans \
    --model_path "${MODEL_PATH}" \
    --save_dir "${BASE_DIR}/pma_sk"

echo "=== Three-Staged: pma/soft_kmeans (stage 3) ==="
python3 main_three_staged.py \
    --stage 3 \
    --generator pma \
    --pma_query_mode soft_kmeans \
    --model_path "${MODEL_PATH}" \
    --generator_path "${BASE_DIR}/pma_sk/stage2_generator.pt" \
    --save_dir "${BASE_DIR}/pma_sk"

# --- Graph coarsening generator ---
echo "=== Three-Staged: graph_coarsening (stage 2) ==="
python3 main_three_staged.py \
    --stage 2 \
    --generator graph_coarsening \
    --model_path "${MODEL_PATH}" \
    --save_dir "${BASE_DIR}/graph_coarsening"

echo "=== Three-Staged: graph_coarsening (stage 3) ==="
python3 main_three_staged.py \
    --stage 3 \
    --generator graph_coarsening \
    --model_path "${MODEL_PATH}" \
    --generator_path "${BASE_DIR}/graph_coarsening/stage2_generator.pt" \
    --save_dir "${BASE_DIR}/graph_coarsening"

# --- GNN pooling (ablation baseline) ---
echo "=== Three-Staged: gnn_pooling (stage 2) ==="
python3 main_three_staged.py \
    --stage 2 \
    --generator gnn_pooling \
    --model_path "${MODEL_PATH}" \
    --save_dir "${BASE_DIR}/gnn_pooling"

echo "=== Three-Staged: gnn_pooling (stage 3) ==="
python3 main_three_staged.py \
    --stage 3 \
    --generator gnn_pooling \
    --model_path "${MODEL_PATH}" \
    --generator_path "${BASE_DIR}/gnn_pooling/stage2_generator.pt" \
    --save_dir "${BASE_DIR}/gnn_pooling"

echo "All three-staged experiments complete."
