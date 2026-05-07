# run_exp4_2_predicted_attn_no_diff.sh
#!/usr/bin/env bash
# ============================================================================
# Experiment 2: Hybrid backbone + STUDENT-SELF attention weighting (no diffusion).
#
# The student's transformer attention is already trained to match the
# teacher's proxy-optimized attention via the existing attention KL/MSE
# distillation channel (use_attn_distill). At each training step we route
# that learned attention back into GRED's per-hop aggregation:
#
#   pass A (no-grad): forward with no GRED weighting, capture student's
#                      transformer attention.
#   pass B (with grad): re-run with attn_weights = reduce(pass-A attn) so
#                       GRED hop sums are weighted by the student's own
#                       (teacher-distilled) attention.
#
# Reduction matches the oracle path: last transformer layer, max-over-heads,
# sliced to the real-node region. Pass-A attention is detached, so gradient
# only flows through how attention is consumed in pass B (standard
# self-conditioning).
#
# Phases (sequential):
#   0. pretrain — train the hybrid teacher from scratch.
#   1. extract  — optimize proxies + save attention targets for ALL splits
#                  (the saved attention is the supervision signal for the
#                  student's transformer attention via KL/MSE).
#   2. train    — train hybrid student with attention KL/MSE +
#                  --use_self_attn_weighting. NOTE: --use_attn_weighting and
#                  --use_self_attn_weighting are mutually exclusive.
# ============================================================================
set -eux pipefail

save_dir="checkpoints_attn_distill_4/exp2_self_attn_no_diff"
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
# Self-attn-weighting itself does not consume saved attention at val/test
# time, but the train phase needs attention targets for the KL/MSE channel
# that supervises the student's transformer attention. Using 'all' is fine
# and lets you A/B against exp1 (oracle) without re-extracting.
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

# --- Phase 2: train hybrid student with self-attn weighting -------------
python exp_attn_distillation_4.py train \
    --backbone hybrid \
    --hidden_dim 96 --num_layers 1 --num_heads 8 --dropout 0.2 \
    --num_gred_layers 4 --num_transformer_layers 2 --max_hops 40 \
    --model_path "${save_dir}/pretrain_best.pt" \
    --save_dir "${save_dir}" \
    --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 \
    --no-use_proxy_self_attn \
    --no-use_attn_weighting \
    --use_self_attn_weighting \
    --distill_weight 1 --mse_weight 0.2 --temperature 1 \
    --lr 5e-5 --weight_decay 1e-4 \
    --max_epochs 300 --patience 50 --batch_size 64

