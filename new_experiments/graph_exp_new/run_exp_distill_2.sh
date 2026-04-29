# Phase 1: Extract (run once, ~30-60 min depending on proxy_opt_steps)
python exp_attn_distill_2.py extract \
    --model_path checkpoints_staged/stage1_best.pt \
    --proxy_opt_steps 100 \
    --num_proxies 4 \
    --proxy_mmd_lambda 1  --hidden_dim 512 --num_layers 1 --num_proxies 8 --proxy_mmd_lambda 1 --batch_size 128

# Phase 2: Train (run many times, fast — each epoch is just normal training speed)
python exp_attn_distill_2.py train \
    --model_path checkpoints_staged/stage1_best.pt \
    --distill_weight 1.0 \
    --temperature 2.0  --hidden_dim 512 --num_layers 1 --num_proxies 8 --proxy_mmd_lambda 1 --batch_size 128
