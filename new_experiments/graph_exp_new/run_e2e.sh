#!/usr/bin/env bash
set -eux pipefail

python3 main_e2e.py --num_proxies 128 --hidden_dim 512 --num_workers 2 --batch_size 64 --num_layers 4 --gen_hidden_dim 512 --gen_num_layers 1 --lr 5e-5 --weight_decay 3e-4 --dropout 0.1 --max_epochs 500 --proxy_warmup_epochs 2 --novelty_alpha_node 0.1 --diversity_weight 0.1 --save_dir checkpoints_e2e_nov >> checkpoint_e2e_nov.log

# # =============================================================
# # End-to-End experiments (Pipeline B)
# # Varies: generator type, readout scope
# # =============================================================

# BASE_DIR="checkpoints_e2e"

# # --- Score-based generator ---
# echo "=== E2E: score_based, nodes_only ==="
# python main_e2e.py \
#     --generator score_based \
#     --readout_scope nodes_only \
#     --save_dir "${BASE_DIR}/score_based_nodes_only" \
#     --batch_size 32 \
#     --max_epochs 200 \
#     --patience 50 \
#     --use_lap_pe

# echo "=== E2E: score_based, all_tokens ==="
# python main_e2e.py \
#     --generator score_based \
#     --readout_scope all_tokens \
#     --save_dir "${BASE_DIR}/score_based_all_tokens" \
#     --batch_size 32 \
#     --max_epochs 200 \
#     --patience 50 \
#     --use_lap_pe

# # --- GNN pooling generator ---
# echo "=== E2E: gnn_pooling, nodes_only ==="
# python main_e2e.py \
#     --generator gnn_pooling \
#     --readout_scope nodes_only \
#     --save_dir "${BASE_DIR}/gnn_pooling_nodes_only" \
#     --batch_size 32 \
#     --max_epochs 200 \
#     --patience 50 \
#     --use_lap_pe

# echo "=== E2E: gnn_pooling, all_tokens ==="
# python main_e2e.py \
#     --generator gnn_pooling \
#     --readout_scope all_tokens \
#     --save_dir "${BASE_DIR}/gnn_pooling_all_tokens" \
#     --batch_size 32 \
#     --max_epochs 200 \
#     --patience 50 \
#     --use_lap_pe

# echo "All E2E experiments complete."
