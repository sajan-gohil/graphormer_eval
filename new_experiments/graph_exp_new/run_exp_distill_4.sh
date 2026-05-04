#!/usr/bin/env bash
set -eux pipefail

save_dir="checkpoints_attn_distill_gred"

python exp_attn_distillation_2.py pretrain --backbone hybrid --num_gred_layers 2 \
    --hidden_dim 512 --num_layers 1 --num_heads 16 --save_dir ${save_dir} \
    --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 \
    --no-use_proxy_self_attn --proxy_mmd_lambda 1.5 --proxy_opt_steps 500 \
    --distill_weight 1 --temperature 2 --s1_max_epochs 1000 \
    --batch_size 32 --num_proxies 64


python exp_attn_distillation_2.py extract --backbone hybrid --num_gred_layers 2 \
    --hidden_dim 512 --num_layers 1 --num_heads 16\
    --model_path ${save_dir}/pretrain_best.pt --save_dir ${save_dir} \
    --no-use_lap_pe --warmup_ratio 0 --num_cross_layers 1 \
    --no-use_proxy_self_attn --proxy_mmd_lambda 1.5 --proxy_opt_steps 750 \
    --distill_weight 1 --temperature 2 \
    --batch_size 32 --num_proxies 64 \
    --proxy_lr 8e-2


python exp_attn_distillation_2.py train --bacbone hybrid --num_gred_layers 2 \
    --hidden_dim 512 --num_layers 1 --num_heads 16 \
    --model_path ${save_dir}/pretrain_best.pt --save_dir ${save_dir} \
    --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 \
    --no-use_proxy_self_attn --proxy_mmd_lambda 1.5 --proxy_opt_steps 500 \
    --distill_weight 1 --temperature 2  --max_epochs 500 \
    --patience 50 --lr 1e-5 --batch_size 32 --num_proxies 64
