# run_exp4_1_oracle_attn.sh
#!/usr/bin/env bash
# ============================================================================
# Experiment 1: Hybrid backbone + ORACLE attention weighting.
#
# Uses the saved teacher attention (extracted in phase 1) as a per-hop
# weighting matrix inside GRED aggregation. This is the stage-1 oracle:
# the "predicted" attention is just the teacher's softmax read off disk
# and reduced via reduce_attn_to_weights (last distilled layer, max-over-heads).
#
# Phases (sequential):
#   0. pretrain — train the hybrid teacher from scratch (no proxies).
#   1. extract  — optimize proxies + save attention targets for ALL splits
#                  (use_attn_weighting consumes weights at train/val/test time).
#   2. train    — train hybrid student with --use_attn_weighting.
# ============================================================================
set -eux pipefail

save_dir="checkpoints_attn_distill_4/exp1_oracle_attn"
mkdir -p "${save_dir}"

# --- Phase 0: pretrain hybrid teacher -----------------------------------
python exp_attn_distillation_4.py pretrain \
    --backbone hybrid \
    --hidden_dim 96 --num_layers 1 --num_heads 8 --dropout 0.2 \
    --num_gred_layers 4 --num_transformer_layers 2 --max_hops 40 \
    --save_dir "${save_dir}" \
    --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 \
    --no-use_proxy_self_attn --proxy_lr 8e-2 \
    --proxy_mmd_lambda 1.5 --proxy_opt_steps 1000 \
    --distill_weight 1 --temperature 2 \
    --s1_max_epochs 1000 --s1_weight_decay 5e-4 \
    --batch_size 128 --num_proxies 4

# --- Phase 1: extract attention targets for ALL splits ------------------
# --use_attn_weighting requires attention to exist for train, val, AND test.
python exp_attn_distillation_4.py extract \
    --backbone hybrid \
    --hidden_dim 96 --num_layers 1 --num_heads 8 --dropout 0.2 \
    --num_gred_layers 4 --num_transformer_layers 2 --max_hops 40 \
    --model_path "${save_dir}/pretrain_best.pt" \
    --save_dir "${save_dir}" \
    --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 \
    --no-use_proxy_self_attn --proxy_lr 1e-2 \
    --proxy_mmd_lambda 200 --proxy_opt_steps 5000 \
    --distill_weight 1 --temperature 1 --mse_weight 0.2 \
    --batch_size 256 --num_proxies 128 \
    --extract_splits all

# --- Phase 2: train hybrid student with oracle attn weighting -----------
python exp_attn_distillation_4.py train \
    --backbone hybrid \
    --hidden_dim 96 --num_layers 1 --num_heads 8 --dropout 0.2 \
    --num_gred_layers 4 --num_transformer_layers 2 --max_hops 40 \
    --model_path "${save_dir}/pretrain_best.pt" \
    --save_dir "${save_dir}" \
    --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 \
    --no-use_proxy_self_attn \
    --use_attn_weighting \
    --distill_weight 1 --mse_weight 0.2 --temperature 1 \
    --lr 5e-5 --weight_decay 1e-4 \
    --max_epochs 300 --patience 50 --batch_size 64

