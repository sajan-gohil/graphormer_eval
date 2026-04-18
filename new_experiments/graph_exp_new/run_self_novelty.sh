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
    --generator graph_coarsening \
    --num_proxies 32 \
    --hidden_dim 256 \
    --num_heads 8 \
    --num_gred_layers 8 \
    --num_transformer_layers 2 \
    --state_dim 88 \
    --use_lap_pe --lap_pe_dim 8 \
    --batch_size 64 \
    --num_workers 4 \
    --s1_lr 1e-4 --s1_max_epochs 300 --s1_patience 50 \
    --s2_lr 1e-4 --s2_max_epochs 500 --s2_patience 50 \
    --s3_lr_gen 5e-4 --s3_lr_transformer 1e-4 --s3_max_epochs 500 --s3_patience 50 \
    --novelty_temperature 1.0 \
    --novelty_alpha 1.0 \
    --novelty_alpha_node 1.0 \
    --save_dir checkpoints_self_novelty_vanilla 
    # "$@"

