#!/usr/bin/env bash
set -eux pipefail


python3 main_three_staged.py --backbone vanilla_gt \
    --stage "1,2,3" \
    --generator graph_coarsening --hidden_dim 512 \
    --num_layers 1 --no-use_lap_pe \
    --s1_lr 3e-5 --s1_patience 20 --s1_max_epochs 300 \
    --num_proxies 32  \
    --s2_lr 3e-5 --s2_max_epochs 1500 --s2_patience 200 \
    --s3_lr_transformer 1e-5 \
    --s3_lr_gen 1e-4 --s3_weight_decay 5e-4 \
    --s3_grad_clip 0.5 --s3_max_epochs 1000 \
    --s3_patience 100 --novelty_alpha 0.1 \
    --diversity_weight 0.1 \
    --gen_hidden_dim 512 --gen_num_layers 1 --gen_dropout 0.1 \
    --save_dir checkpoints_three_staged_cros_gph_coarsening_vanilla_real/ \
    --batch_size 32 \
    --use_cross_attn_routing --num_cross_layers 1 --no-cross_attn_proxy_self_attn
