#!/usr/bin/env bash
set -eux pipefail

save_dir="checkpoints_attn_distill_5"

#python exp_attn_distillation_2.py pretrain --hidden_dim 64 \
#    --save_dir ${save_dir} \
#    --num_layers 1 --num_heads 16 --proxy_lr 8e-2 \
#    --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 \
#    --no-use_proxy_self_attn --proxy_mmd_lambda 1.5 --proxy_opt_steps 1000 \
#    --distill_weight 1 --temperature 2 --s1_max_epochs 1000 --s1_weight_decay 5e-4 \
#    --batch_size 128 --num_proxies 4 --dropout 0.2


python exp_attn_distillation_2.py extract  --hidden_dim 64 \
    --model_path ${save_dir}/pretrain_best.pt --save_dir ${save_dir} \
    --num_layers 1 --num_heads 16 --proxy_lr 1e-3 \
    --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 \
    --no-use_proxy_self_attn --proxy_mmd_lambda 20 --proxy_opt_steps 5000 \
    --distill_weight 1 --temperature 2 \
    --batch_size 256 --num_proxies 32


python exp_attn_distillation_2.py train  --hidden_dim 64 \
    --model_path ${save_dir}/pretrain_best.pt --save_dir ${save_dir} \
    --num_layers 1 --num_heads 16 --proxy_lr 8e-2 \
    --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 \
    --no-use_proxy_self_attn --proxy_mmd_lambda 20 --proxy_opt_steps 5000 \
    --distill_weight 1 --temperature 2 --max_epochs 500 \
    --patience 50 --lr 1e-5 --batch_size 64 --num_proxies 32

