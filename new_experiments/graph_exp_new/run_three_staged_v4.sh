#!/usr/bin/env bash
set -eux pipefail

# v7: Based on V6 analysis (S2 recovered, S3 stagnant)
#   - Restore V3's S3 LRs: s3_lr_transformer 1e-6→3e-6, s3_lr_gen 3e-6→1e-5
#   - Restore V3's s3_weight_decay: 3e-4→1e-4 (was too strong)
#   - Restore V3's s2_grad_clip: 1.0→2.0 (was truncating updates)
#   - Keep V6 improvements: s2_max_epochs=300, s2_patience=40,
#     s3_patience=50, s3_phase_a=15, proxy_dropout=0.3,
#     novelty_alpha=0.1, diversity_weight=0.01

python3 main_three_staged.py --backbone vanilla_gt \
    --stage "1,2,3" \
    --generator gred_layers --hidden_dim 256 --num_transformer_layers 1 \
    --num_layers 1 --no-use_lap_pe \
    --dropout 0.3 \
    --s1_lr 3e-5 --s1_patience 30 --s1_max_epochs 500 \
    --s1_weight_decay 1e-3 \
    --num_proxies 64 \
    --s2_lr 1e-4 --s2_max_epochs 1000 --s2_patience 40 --s2_grad_clip 3.0 \
    --s2_weight_decay 1e-3 \
    --s3_lr_transformer 3e-6 --s3_lr_gen 1e-5 --s3_weight_decay 1e-4 \
    --s3_grad_clip 2 --s3_max_epochs 500 --s3_patience 50 \
    --s3_phase_a_epochs 15 --s3_proxy_dropout 0.3 \
    --diversity_weight 0.01 --novelty_alpha 0.1 \
    --gen_hidden_dim 88 --gen_num_layers 4 --gen_dropout 0.3 \
    --save_dir checkpoints_ts_gred_vanilla_7 \
    --batch_size 64 \
#    --use_cross_attn_routing --num_cross_layers 1 --no-cross_attn_proxy_self_attn
