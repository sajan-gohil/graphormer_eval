#!/bin/bash
# Phase 2: Validation Gate — Optimize proxy embeddings on frozen model
# Prerequisite: Phase 1 must be completed (checkpoints/best_model.pt exists)
# Run from the project root directory (graph_experiments/)

set -e

echo "============================================"
echo "Phase 2: Validation Gate"
echo "============================================"

cd "$(dirname "$0")/.."

# Check prerequisite
if [ ! -f "./checkpoints/best_model.pt" ]; then
    echo "ERROR: Phase 1 checkpoint not found at ./checkpoints/best_model.pt"
    echo "Please run Phase 1 first: bash scripts/run_phase1.sh"
    exit 1
fi

# Run unit tests first
echo "Running unit tests..."
python -u tests/test_proxy_insertion.py
echo ""

# Run validation gate
python -u training/validate_premise.py \
    --checkpoint ./checkpoints/best_model.pt \
    --num_samples 500 \
    --num_diagnostic 50 \
    --proxy_lr 1e-2 \
    --proxy_iterations 500 \
    --mmd_lambda 0.05 \
    --num_restarts 5 \
    --gradient_clip 1.0 \
    --M_values 2 4 8 16 \
    --seed 42 \
    --device auto \
    "$@"

echo "============================================"
echo "Phase 2 complete. Check logs/ for results."
echo "============================================"
