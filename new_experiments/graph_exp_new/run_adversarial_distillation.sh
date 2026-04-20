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
    --generator flow_matching \
    --num_proxies 8 \
    --hidden_dim 512 \
    --num_layers 4 \
    --num_heads 8 \
    --num_gred_layers 8 \
    --num_transformer_layers 2 \
    --state_dim 88 \
    --batch_size 32 \
    --num_workers 4 \
    \
    --s1_lr 3e-5 --s1_max_epochs 300 --s1_patience 50 \
    --s2_lr 3e-5 --s2_max_epochs 500 --s2_patience 50 \
    --s3_lr_gen 3e-6 --s3_lr_transformer 1e-6 --s3_max_epochs 500 --s3_patience 50 \
    \
    --novelty_temperature 1.5 \
    --novelty_alpha 0.1 \
    --novelty_alpha_node 0.1 \
    --diversity_weight 0.5 \
    \
    --distill_lr 3e-6 \
    --distill_max_epochs 500 \
    --distill_patience 40 \
    --distill_temperature 1.5 \
    --distill_kl_weight 0.5 \
    --distill_node_weight 0.5 \
    --distill_graph_weight 0.5 \
    --distill_task_weight 1 \
    \
    --gap_threshold 0.005 \
    --distill_gap_threshold 0.003 \
    \
    --save_dir checkpoints_adv_distill_4

