#!/usr/bin/env bash
set -eux pipefail

save_dir="checkpoints_attn_distill_4"

python exp_attn_distillation_2.py pretrain --hidden_dim 512 --num_layers 1 --num_heads 8 --save_dir ${save_dir} --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 --no-use_proxy_self_attn --proxy_mmd_lambda 1.5 --proxy_opt_steps 500 --distill_weight 1 --temperature 1.5 --s1_max_epochs 1000 --batch_size 128 --num_proxies 128


python exp_attn_distillation_2.py extract --model_path ${save_dir}/pretrain_best.pt --save_dir ${save_dir} --num_layers 1 --num_heads 8 --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 --no-use_proxy_self_attn --proxy_mmd_lambda 1.5 --proxy_opt_steps 500 --distill_weight 1 --temperature 2 --hidden_dim 512 --batch_size 512 --num_proxies 128


python exp_attn_distillation_2.py train --model_path ${save_dir}/pretrain_best.pt --save_dir ${save_dir} --num_layers 1 --num_heads 8 --no-use_lap_pe --warmup_ratio 0.05 --num_cross_layers 1 --no-use_proxy_self_attn --proxy_mmd_lambda 1.5 --proxy_opt_steps 500 --distill_weight 1 --temperature 2 --hidden_dim 512 --max_epochs 500 --patience 50 --lr 1e-5 --batch_size 128 --num_proxies 128
