# run_exp4_4_node_emb_diff.sh
#!/usr/bin/env bash
# ============================================================================
# Experiment 4: Hybrid backbone, NO attention weighting, WITH node-embedding
# diffusion (CFM).
#
# Trains the existing latent-embedding flow-matching denoiser
# (NodeEmbeddingFlowDenoiser) as an auxiliary head: it learns the vector field
# from N(0, I) to the teacher's post-proxy / pre-attention node embeddings
# (post_emb), conditioned on the student's matching upstream features
# (post-GRED for hybrid). The denoiser output is regressed against the saved
# post_emb target. No attention weighting on GRED aggregation.
#
# All flags here exist in exp_attn_distillation_4.py — runs as-is.
#
# Phases (sequential):
#   0. pretrain — train the hybrid teacher from scratch.
#   1. extract  — optimize proxies, save attention + post_emb (latent target).
#                  Only the train split is strictly required (CFM is a
#                  train-time auxiliary loss); using 'all' is also fine.
#   2. train    — train hybrid student + CFM denoiser on post_emb. Attention
#                  KL/MSE distillation is also enabled (uses the saved
#                  attention targets from extract; orthogonal to CFM).
# ============================================================================
set -eux pipefail

save_dir="checkpoints_attn_distill_4/exp4_node_emb_diff"
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

# --- Phase 1: extract attention + post_emb (train split is sufficient) --
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
    --extract_splits train

# --- Phase 2: train hybrid student + CFM node-embedding denoiser --------
python exp_attn_distillation_4.py train \
    --backbone hybrid \
    --hidden_dim 96 --num_layers 1 --num_heads 8 --dropout 0.2 \
    --num_gred_layers 4 --num_transformer_layers 2 --max_hops 40 \
    --model_path "${save_dir}/pretrain_best.pt" \
    --save_dir "${save_dir}" \
    --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 \
    --no-use_proxy_self_attn \
    --no-use_attn_weighting \
    --use_latent_embedding_distill --latent_distill_weight 0.5 \
    --denoiser_layers 4 \
    --denoiser_euler_steps 4 \
    --latent_uncond_train_prob 0.1 \
    --latent_guidance_scale 1.0 \
    --distill_weight 1 --mse_weight 0.2 --temperature 1 \
    --lr 5e-5 --weight_decay 1e-4 \
    --max_epochs 300 --patience 50 --batch_size 64

