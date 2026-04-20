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
<<<<<<< Updated upstream
    --backbone hybrid \
    --generator graph_coarsening \
    --num_proxies 32 \
    --hidden_dim 128 \
=======
    --backbone vanilla_gt \
    --generator score_based \
    --num_proxies 64 \
    --hidden_dim 256 \
>>>>>>> Stashed changes
    --num_heads 8 \
    --num_gred_layers 6 \
    --num_transformer_layers 2 \
    --state_dim 88 \
<<<<<<< Updated upstream
    --batch_size 16 \
    --num_workers 4 \
    --s1_lr 5e-4 --s1_max_epochs 500 --s1_patience 20 \
    --s2_lr 5e-4 --s2_max_epochs 500 --s2_patience 50 \
    --s3_lr_gen 5e-4 --s3_lr_transformer 5e-4 --s3_max_epochs 500 --s3_patience 50 \
    --novelty_temperature 1.0 \
    --novelty_alpha 1.0 \
    --novelty_alpha_node 1.0 \
    --diversity_weight 1 \
    --readout_scope all_tokens \
    --save_dir checkpoints_self_novelty_hybrid_graph_coarsening
=======
    --batch_size 32 \
    --num_workers 4 \
    --s1_lr 5e-4 --s1_max_epochs 200 --s1_patience 20 \
    --s2_lr 5e-4 --s2_max_epochs 500 --s2_patience 50 \
    --s3_lr_gen 5e-4 --s3_lr_transformer 5e-4 --s3_max_epochs 500 --s3_patience 50 \
    --novelty_temperature 2.0 \
    --novelty_alpha 2.0 \
    --novelty_alpha_node 0.2 \
    --diversity_weight 0.5 \
    --readout_scope all_tokens \
    --save_dir checkpoints_self_novelty_score_based
>>>>>>> Stashed changes
    # "$@"

