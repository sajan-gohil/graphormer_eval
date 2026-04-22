#!/usr/bin/env bash
set -eux pipefail
# Sample run for the self-novelty proxy pipeline.
#
# Trains a graph_coarsening generator on the hybrid (GRED + transformer) backbone,
# using task_loss + 1.0 * output_penalty + 1.0 * node_penalty with temperature 1.0.
# Runs all three stages end-to-end; checkpoints land in ./checkpoints_self_novelty/.
#
# Override individual args on the command line, e.g.:
#   ./run_self_novelty.sh --backbone vanilla_gt --generator score_based
#
# To run a single stage (requires prior checkpoints):
#   python main_self_novelty.py --stage 2 \
#       --model_path checkpoints_self_novelty/stage1_best.pt
#   python main_self_novelty.py --stage 3 \
#       --model_path checkpoints_self_novelty/stage1_best.pt \
#       --generator_path checkpoints_self_novelty/stage2_generator.pt


python main_self_novelty.py \
    --stage all \
    --backbone vanilla_gt \
    --generator flow_matching \
    --num_proxies 32 \
    --hidden_dim 128 \
    --num_layers 1 \
    --num_heads 8 \
    --batch_size 128 \
    --num_workers 4 \
    --s1_lr 5e-4 --s1_max_epochs 200 --s1_patience 20 \
    --s1_weight_decay 1e-3 \
    --s2_lr 1e-4 --s2_max_epochs 500 --s2_patience 20 \
    --s3_lr_gen 3e-4 --s3_lr_transformer 1e-4 --s3_max_epochs 500 --s3_patience 30 \
    --novelty_temperature 0.5 \
    --novelty_alpha 0.1 \
    --novelty_alpha_node 0.0 \
    --diversity_weight 0.01 \
    --euler_steps 5 \
    --readout_scope all_tokens \
    --save_dir checkpoints_self_novelty_vanilla_fm_v2 \
    "$@"

