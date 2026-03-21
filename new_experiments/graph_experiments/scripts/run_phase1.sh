#!/bin/bash
# Phase 1: Train base GPS Transformer on Peptides-func
# Run from the project root directory (graph_experiments/)

set -e

echo "============================================"
echo "Phase 1: Training Base GPS Transformer"
echo "============================================"

cd "$(dirname "$0")/.."

python -u training/pretrain.py \
    --hidden_dim 64 \
    --num_layers 5 \
    --num_heads 8 \
    --dropout 0.1 \
    --batch_size 64 \
    --lr 1e-3 \
    --max_epochs 300 \
    --patience 25 \
    --seed 42 \
    --device auto \
    "$@"

echo "============================================"
echo "Phase 1 complete. Check logs/ and checkpoints/"
echo "============================================"
