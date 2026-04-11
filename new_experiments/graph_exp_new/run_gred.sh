#!/usr/bin/env bash
set -eux pipefail

# =============================================================
# GRED backbone experiments
# 1. GRED-only baseline (no proxies) — standalone distance filtering
# 2. GRED+proxy hybrid (GRED encoder + transformer layers for proxy integration)
# =============================================================

BASE_DIR="checkpoints_gred"

# =============================================================
# A. End-to-End Pipeline (main_e2e.py)
# =============================================================

# --- GRED-only baseline (no proxies) ---
echo "=== E2E: GRED-only baseline ==="
python3 main_e2e.py \
    --backbone gred \
    --generator score_based \
    --num_proxies 0 \
    --hidden_dim 88 \
    --state_dim 88 \
    --num_gred_layers 8 \
    --gred_expand 1 \
    --r_min 0.95 \
    --r_max 1.0 \
    --max_phase 6.28 \
    --gred_act full-glu \
    --max_hops 40 \
    --dropout 0.2 \
    --lr 1e-3 \
    --weight_decay 0.2 \
    --batch_size 32 \
    --max_epochs 200 \
    --patience 50 \
    --mmd_lambda 0 \
    --proxy_warmup_epochs 0 \
    --save_dir "${BASE_DIR}/gred_only_e2e"

# --- Hybrid: GRED + score_based generator (E2E) ---
echo "=== E2E: hybrid + score_based ==="
python3 main_e2e.py \
    --backbone hybrid \
    --generator score_based \
    --num_proxies 32 \
    --hidden_dim 88 \
    --state_dim 88 \
    --num_gred_layers 8 \
    --num_transformer_layers 2 \
    --gred_expand 1 \
    --r_min 0.95 \
    --r_max 1.0 \
    --max_phase 6.28 \
    --gred_act full-glu \
    --max_hops 40 \
    --num_heads 8 \
    --dropout 0.2 \
    --lr 1e-3 \
    --weight_decay 3e-4 \
    --batch_size 32 \
    --max_epochs 200 \
    --patience 50 \
    --mmd_lambda 0.01 \
    --proxy_warmup_epochs 20 \
    --readout_scope nodes_only \
    --save_dir "${BASE_DIR}/hybrid_score_e2e"

# =============================================================
# B. Three-Staged Pipeline (main_three_staged.py)
# =============================================================

STAGED_DIR="${BASE_DIR}/three_staged"
SHARED_DIR="${STAGED_DIR}/shared"

# --- Stage 1: Pretrain GRED backbone (no proxies) ---
echo "=== Three-Staged: Stage 1 (GRED pretrain) ==="
python3 main_three_staged.py \
    --stage 1 \
    --backbone gred \
    --hidden_dim 88 \
    --state_dim 88 \
    --num_gred_layers 8 \
    --gred_expand 1 \
    --r_min 0.95 \
    --r_max 1.0 \
    --max_phase 6.28 \
    --gred_act full-glu \
    --max_hops 40 \
    --dropout 0.2 \
    --batch_size 32 \
    --s1_lr 1e-3 \
    --s1_weight_decay 0.2 \
    --s1_max_epochs 200 \
    --s1_patience 50 \
    --save_dir "${SHARED_DIR}"

# --- Stage 1: Pretrain hybrid backbone ---
echo "=== Three-Staged: Stage 1 (hybrid pretrain) ==="
python3 main_three_staged.py \
    --stage 1 \
    --backbone hybrid \
    --hidden_dim 88 \
    --state_dim 88 \
    --num_gred_layers 8 \
    --num_transformer_layers 2 \
    --num_heads 8 \
    --gred_expand 1 \
    --r_min 0.95 \
    --r_max 1.0 \
    --max_phase 6.28 \
    --gred_act full-glu \
    --max_hops 40 \
    --dropout 0.2 \
    --batch_size 32 \
    --s1_lr 1e-3 \
    --s1_weight_decay 3e-4 \
    --s1_max_epochs 200 \
    --s1_patience 50 \
    --save_dir "${STAGED_DIR}/hybrid_shared"

HYBRID_MODEL="${STAGED_DIR}/hybrid_shared/stage1_best.pt"

# --- Stage 2+3: Hybrid + score_based generator ---
echo "=== Three-Staged: hybrid + score_based (stage 2) ==="
python3 main_three_staged.py \
    --stage 2 \
    --backbone hybrid \
    --generator score_based \
    --num_proxies 32 \
    --hidden_dim 88 \
    --state_dim 88 \
    --num_gred_layers 8 \
    --num_transformer_layers 2 \
    --num_heads 8 \
    --max_hops 40 \
    --dropout 0.2 \
    --batch_size 32 \
    --model_path "${HYBRID_MODEL}" \
    --save_dir "${STAGED_DIR}/hybrid_score"

echo "=== Three-Staged: hybrid + score_based (stage 3) ==="
python3 main_three_staged.py \
    --stage 3 \
    --backbone hybrid \
    --generator score_based \
    --num_proxies 32 \
    --hidden_dim 88 \
    --state_dim 88 \
    --num_gred_layers 8 \
    --num_transformer_layers 2 \
    --num_heads 8 \
    --max_hops 40 \
    --dropout 0.2 \
    --batch_size 32 \
    --model_path "${HYBRID_MODEL}" \
    --generator_path "${STAGED_DIR}/hybrid_score/stage2_generator.pt" \
    --save_dir "${STAGED_DIR}/hybrid_score"

# --- Stage 2+3: Hybrid + PMA generator ---
echo "=== Three-Staged: hybrid + pma (stage 2) ==="
python3 main_three_staged.py \
    --stage 2 \
    --backbone hybrid \
    --generator pma \
    --pma_query_mode farthest_point \
    --num_proxies 32 \
    --hidden_dim 88 \
    --state_dim 88 \
    --num_gred_layers 8 \
    --num_transformer_layers 2 \
    --num_heads 8 \
    --max_hops 40 \
    --dropout 0.2 \
    --batch_size 32 \
    --model_path "${HYBRID_MODEL}" \
    --save_dir "${STAGED_DIR}/hybrid_pma"

echo "=== Three-Staged: hybrid + pma (stage 3) ==="
python3 main_three_staged.py \
    --stage 3 \
    --backbone hybrid \
    --generator pma \
    --pma_query_mode farthest_point \
    --num_proxies 32 \
    --hidden_dim 88 \
    --state_dim 88 \
    --num_gred_layers 8 \
    --num_transformer_layers 2 \
    --num_heads 8 \
    --max_hops 40 \
    --dropout 0.2 \
    --batch_size 32 \
    --model_path "${HYBRID_MODEL}" \
    --generator_path "${STAGED_DIR}/hybrid_pma/stage2_generator.pt" \
    --save_dir "${STAGED_DIR}/hybrid_pma"

echo "All GRED experiments complete."
