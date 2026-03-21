#!/bin/bash
# Phase 3: Simple Proxy Baselines
# Prerequisite: Phase 1 must be completed (checkpoints/best_model.pt exists)
# Run from the project root directory (graph_experiments/)

set -e

echo "============================================"
echo "Phase 3: Simple Proxy Baselines"
echo "============================================"

cd "$(dirname "$0")/.."

# Check prerequisite
if [ ! -f "./checkpoints/best_model.pt" ]; then
    echo "ERROR: Phase 1 checkpoint not found at ./checkpoints/best_model.pt"
    echo "Please run Phase 1 first: bash scripts/run_phase1.sh"
    exit 1
fi

# =========================================================
# Part A: Trainable baselines (train from scratch)
# =========================================================

echo ""
echo "==============================="
echo "Part A: Trainable Baselines"
echo "==============================="

echo ""
echo "--- VN-Fixed ---"
python -u training/train_baselines.py --model vn_fixed "$@"

echo ""
echo "--- VN-Aggregated ---"
python -u training/train_baselines.py --model vn_aggregated "$@"

echo ""
echo "--- K-Fixed-VN (M=4) ---"
python -u training/train_baselines.py --model kvn --M 4 "$@"

echo ""
echo "--- K-Fixed-VN (M=8) ---"
python -u training/train_baselines.py --model kvn --M 8 "$@"

echo ""
echo "--- K-Fixed-VN (M=16) ---"
python -u training/train_baselines.py --model kvn --M 16 "$@"


# =========================================================
# Part B: Frozen-model baselines (no training)
# =========================================================

echo ""
echo "==============================="
echo "Part B: Frozen Model Baselines"
echo "==============================="

python -u training/eval_frozen_baselines.py \
    --checkpoint ./checkpoints/best_model.pt \
    --M 8 \
    --num_random_seeds 10 \
    --device auto

echo ""
echo "============================================"
echo "Phase 3 complete. Check logs/ for all results."
echo "============================================"
