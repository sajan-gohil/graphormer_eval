#!/usr/bin/env bash
set -eux pipefail
# Adversarial Distillation Ratchet — sample run script.
#
# Runs the full cyclic pipeline:
#   Cycle K: train G + finetune → distill M_base → weight swap → repeat
#
# By default runs 5 ratchet cycles with backbone pretraining in cycle 0.
# Checkpoints land in ./checkpoints_adv_distill/cycleN/.
#
# Override args on the command line:
#   ./run_adversarial_distillation.sh --num_cycles 3 --backbone hybrid
#
# Resume from a specific cycle (requires prior checkpoints):
#   python main_adversarial_distillation.py --resume_cycle 2 \
#       --model_path checkpoints_adv_distill/cycle1/swapped_model.pt

python main_adversarial_distillation.py \
    --num_cycles 5 \
    --run_stage1 \
    --backbone vanilla_gt \
    --generator graph_coarsening \
    --num_proxies 32 \
    --hidden_dim 256 \
    --num_layers 5 \
    --num_heads 8 \
    --num_gred_layers 8 \
    --num_transformer_layers 2 \
    --state_dim 88 \
    --use_lap_pe --lap_pe_dim 8 \
    --batch_size 64 \
    --num_workers 4 \
    \
    --s1_lr 1e-4 --s1_max_epochs 300 --s1_patience 50 \
    --s2_lr 1e-4 --s2_max_epochs 500 --s2_patience 50 \
    --s3_lr_gen 5e-4 --s3_lr_transformer 1e-4 --s3_max_epochs 500 --s3_patience 50 \
    \
    --novelty_temperature 1.0 \
    --novelty_alpha 1.0 \
    --novelty_alpha_node 1.0 \
    --diversity_weight 1.0 \
    \
    --distill_lr 5e-5 \
    --distill_max_epochs 300 \
    --distill_patience 40 \
    --distill_temperature 2.0 \
    --distill_kl_weight 1.0 \
    --distill_node_weight 1.0 \
    --distill_graph_weight 0.5 \
    --distill_task_weight 0.5 \
    \
    --gap_threshold 0.005 \
    --distill_gap_threshold 0.003 \
    \
    --save_dir checkpoints_adv_distill \
    "$@"
