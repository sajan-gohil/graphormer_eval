# run_exp4_3_predicted_attn_diff.sh
#!/usr/bin/env bash
# ============================================================================
# Experiment 3: Hybrid backbone + LEARNED attention predictor WITH diffusion.
#
# Same idea as experiment 2 (a head that emits an (n, n) weight matrix
# consumed by GRED), but the predictor is trained as a conditional
# flow-matching denoiser: it learns the vector field from N(0, I) to the
# saved teacher attention, conditioned on the student's upstream features.
# At inference, run a few Euler steps to sample an attention matrix, then
# feed that into GRED.
#
# Supervision:
#   * CFM regression on the saved teacher attention (as the target sample).
#   * Optional KL term against the teacher attention as in exp2.
#
# ⚠ NOT YET IMPLEMENTED IN exp_attn_distillation_4.py.
# Placeholder flags below. To make this runnable:
#
#   --use_predicted_attn                      (shared with exp2)
#   --use_attn_diffusion                      (BooleanOptionalAction, default False)
#       Train the predictor as a CFM denoiser instead of a point estimator.
#       Requires --use_predicted_attn to be on.
#   --attn_denoiser_layers / --attn_denoiser_dim / --attn_denoiser_dropout
#   --attn_denoiser_euler_steps INT
#       Number of Euler integration steps at inference (1–8 typical).
#   --attn_uncond_train_prob FLOAT            (CFG drop-out probability)
#   --attn_guidance_scale FLOAT               (CFG scale at inference)
#
# Architecturally analogous to the existing latent-embedding flow-matching
# code (NodeEmbeddingFlowDenoiser); the new module operates on (n, n)
# attention rather than (n, d) embeddings.
#
# Phases (sequential):
#   0. pretrain — same hybrid teacher pretrain as exp1/exp2.
#   1. extract  — saves teacher attention (CFM target).
#   2. train    — student + CFM attention predictor; sampled attention
#                  feeds GRED.
# ============================================================================
set -eux pipefail

save_dir="checkpoints_attn_distill_4/exp3_predicted_attn_diff"
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

# --- Phase 2: train student + CFM attention predictor -------------------
# NOTE: --use_predicted_attn, --use_attn_diffusion, and --attn_denoiser_*
# are placeholder flags.
python exp_attn_distillation_4.py train \
    --backbone hybrid \
    --hidden_dim 96 --num_layers 1 --num_heads 8 --dropout 0.2 \
    --num_gred_layers 4 --num_transformer_layers 2 --max_hops 40 \
    --model_path "${save_dir}/pretrain_best.pt" \
    --save_dir "${save_dir}" \
    --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 \
    --no-use_proxy_self_attn \
    --use_predicted_attn \
    --use_attn_diffusion \
    --no-use_attn_weighting \
    --attn_denoiser_layers 4 \
    --attn_denoiser_dim 96 \
    --attn_denoiser_dropout 0.1 \
    --attn_denoiser_euler_steps 4 \
    --attn_uncond_train_prob 0.1 \
    --attn_guidance_scale 1.0 \
    --attn_predictor_supervision_weight 1.0 \
    --distill_weight 1 --mse_weight 0.2 --temperature 1 \
    --lr 5e-5 --weight_decay 1e-4 \
    --max_epochs 300 --patience 50 --batch_size 64

