#!/usr/bin/env bash
set -eux pipefail


python3 main_staged.py --backbone vanilla_gt \
    --generator flow_matching --hidden_dim 512 \
    --num_layers 1 --no-use_lap_pe \
    --s1_lr 3e-5 --s1_patience 20 --s1_max_epochs 300 \
    --num_proxies 8 --s2_num_steps 200 \
    --s2_cross_moment_lambda 0.0 \
    --s2_prior_moment_lambda 2 \
    --s2_novelty_alpha 0.2 --s2_num_restarts 5 \
    --s2_prior_target_var 0.15 \
    --s2_proxy_lr 5e-3 \
    --s2_loss_threshold 0.01 --gen_dropout 0.1 \
    --s3_lr 3e-5 --s3_max_epochs 500 --s3_patience 200 \
    --denoiser_dim 512 --s4_lr_transformer 1e-5 \
    --s4_lr_generator 1e-4 --s4_weight_decay 5e-4 \
    --s4_grad_clip 0.5 --s4_max_epochs 1000 \
    --s4_patience 100 --novelty_alpha 0.1 \
    --diversity_weight 0.1 \
    --save_dir checkpoints_staged_novelty_vanilla_3/ \
    --batch_size 16
# # =============================================================
# # Staged experiments (Pipeline A)
# # Varies: generator type
# # Stage 1+2 are shared — train once, reuse for all generators.
# # =============================================================

# BASE_DIR="checkpoints_staged_2"
# SHARED_DIR="${BASE_DIR}/shared"

# # --- Stage 1: Pretrain transformer (shared) ---
# echo "=== Staged: Stage 1 (shared pretrain) ==="
# python main_staged.py \
#     --stage 1 \
#     --save_dir "${SHARED_DIR}"

# MODEL_PATH="${SHARED_DIR}/stage1_best.pt"

# # --- Stage 2: Proxy optimization (shared) ---
# echo "=== Staged: Stage 2 (shared proxy optimization) ==="
# python main_staged.py \
#     --stage 2 \
#     --model_path "${MODEL_PATH}" \
#     --save_dir "${SHARED_DIR}" \
#     --s2_num_restarts 20

# PROXY_PATH="${SHARED_DIR}/proxy_pairs.pkl"

# # =============================================================
# # Stage 3 + 4: Per-generator experiments
# # =============================================================

# # --- Score-based generator ---
# echo "=== Staged: score_based (stages 3+4) ==="
# python main_staged.py \
#     --stage 3 \
#     --generator score_based \
#     --model_path "${MODEL_PATH}" \
#     --proxy_pairs_path "${PROXY_PATH}" \
#     --save_dir "${BASE_DIR}/score_based"

# python main_staged.py \
#     --stage 4 \
#     --generator score_based \
#     --model_path "${MODEL_PATH}" \
#     --generator_path "${BASE_DIR}/score_based/stage3_generator.pt" \
#     --save_dir "${BASE_DIR}/score_based"

# # --- Flow matching generator ---
# echo "=== Staged: flow_matching (stages 3+4) ==="
# python main_staged.py \
#     --stage 3 \
#     --generator flow_matching \
#     --model_path "${MODEL_PATH}" \
#     --proxy_pairs_path "${PROXY_PATH}" \
#     --save_dir "${BASE_DIR}/flow_matching"

# python main_staged.py \
#     --stage 4 \
#     --generator flow_matching \
#     --model_path "${MODEL_PATH}" \
#     --generator_path "${BASE_DIR}/flow_matching/stage3_generator.pt" \
#     --save_dir "${BASE_DIR}/flow_matching"

# # --- GNN pooling generator ---
# echo "=== Staged: gnn_pooling (stages 3+4) ==="
# python main_staged.py \
#     --stage 3 \
#     --generator gnn_pooling \
#     --model_path "${MODEL_PATH}" \
#     --proxy_pairs_path "${PROXY_PATH}" \
#     --save_dir "${BASE_DIR}/gnn_pooling"

# python main_staged.py \
#     --stage 4 \
#     --generator gnn_pooling \
#     --model_path "${MODEL_PATH}" \
#     --generator_path "${BASE_DIR}/gnn_pooling/stage3_generator.pt" \
#     --save_dir "${BASE_DIR}/gnn_pooling"

# # --- PMA generator (farthest_point queries) ---
# echo "=== Staged: pma/farthest_point (stages 3+4) ==="
# python main_staged.py \
#     --stage 3 \
#     --generator pma \
#     --pma_query_mode farthest_point \
#     --model_path "${MODEL_PATH}" \
#     --proxy_pairs_path "${PROXY_PATH}" \
#     --save_dir "${BASE_DIR}/pma_fp"

# python main_staged.py \
#     --stage 4 \
#     --generator pma \
#     --pma_query_mode farthest_point \
#     --model_path "${MODEL_PATH}" \
#     --generator_path "${BASE_DIR}/pma_fp/stage3_generator.pt" \
#     --save_dir "${BASE_DIR}/pma_fp"

# # --- PMA generator (soft_kmeans queries) ---
# echo "=== Staged: pma/soft_kmeans (stages 3+4) ==="
# python main_staged.py \
#     --stage 3 \
#     --generator pma \
#     --pma_query_mode soft_kmeans \
#     --model_path "${MODEL_PATH}" \
#     --proxy_pairs_path "${PROXY_PATH}" \
#     --save_dir "${BASE_DIR}/pma_sk"

# python main_staged.py \
#     --stage 4 \
#     --generator pma \
#     --pma_query_mode soft_kmeans \
#     --model_path "${MODEL_PATH}" \
#     --generator_path "${BASE_DIR}/pma_sk/stage3_generator.pt" \
#     --save_dir "${BASE_DIR}/pma_sk"

# # --- Graph coarsening generator ---
# echo "=== Staged: graph_coarsening (stages 3+4) ==="
# python main_staged.py \
#     --stage 3 \
#     --generator graph_coarsening \
#     --model_path "${MODEL_PATH}" \
#     --proxy_pairs_path "${PROXY_PATH}" \
#     --save_dir "${BASE_DIR}/graph_coarsening"

# python main_staged.py \
#     --stage 4 \
#     --generator graph_coarsening \
#     --model_path "${MODEL_PATH}" \
#     --generator_path "${BASE_DIR}/graph_coarsening/stage3_generator.pt" \
#     --save_dir "${BASE_DIR}/graph_coarsening"

# echo "All staged experiments complete."
