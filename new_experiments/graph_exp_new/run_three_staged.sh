#!/usr/bin/env bash
set -eux pipefail


python3 main_three_staged.py --backbone vanilla_gt \
    --stage "1,2,3" \
    --generator gred_layers --hidden_dim 256 --num_transformer_layers 1 \
    --num_layers 1 --no-use_lap_pe \
    --dropout 0.3 \
    --s1_lr 3e-5 --s1_patience 20 --s1_max_epochs 300 \
    --s1_weight_decay 1e-3 \
    --num_proxies 64 \
    --s2_lr 1e-4 --s2_max_epochs 500 --s2_patience 50 --s2_grad_clip 2.0 \
    --s2_weight_decay 1e-3 \
    --s3_lr_transformer 3e-6 --s3_lr_gen 1e-5 --s3_weight_decay 1e-4 \
    --s3_grad_clip 2 --s3_max_epochs 1000 --s3_patience 100 \
    --s3_phase_a_epochs 10 --s3_proxy_dropout 0.2 \
    --diversity_weight 0.01 --novelty_alpha 0.1 \
    --gen_hidden_dim 88 --gen_num_layers 2 --gen_dropout 0.3 \
    --denoiser_layers 2 \
    --save_dir checkpoints_ts_gred_vanilla_3 \
    --batch_size 64 \
    --use_cross_attn_routing --num_cross_layers 1 --no-cross_attn_proxy_self_attn