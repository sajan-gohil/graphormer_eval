#!/bin/bash
# Phase 4: Ablation Studies on Proxy Optimization
# Prerequisite: Phase 1 must be completed (checkpoints/best_model.pt exists)
# Run from the project root directory (graph_experiments/)

set -e

echo "============================================"
echo "Phase 4: Ablation Studies"
echo "============================================"

cd "$(dirname "$0")/.."

# Check prerequisite
if [ ! -f "./checkpoints/best_model.pt" ]; then
    echo "ERROR: Phase 1 checkpoint not found at ./checkpoints/best_model.pt"
    echo "Please run Phase 1 first: bash scripts/run_phase1.sh"
    exit 1
fi

# =========================================================
# Run all ablations
# =========================================================

echo ""
echo "--- Running ablation sweeps ---"
python -u training/run_ablations.py \
    --checkpoint ./checkpoints/best_model.pt \
    --num_samples 100 \
    --seed 42 \
    --device auto \
    "$@"

# =========================================================
# Generate visualizations
# =========================================================

echo ""
echo "--- Generating t-SNE visualization ---"

# Find the most recent phase4 log
PHASE4_LOG=$(ls -t ./logs/phase4_*.json 2>/dev/null | head -1)

python -u evaluation/visualize.py \
    --checkpoint ./checkpoints/best_model.pt \
    --num_graphs 25 \
    --M 8 \
    --seed 42 \
    --device auto \
    ${PHASE4_LOG:+--phase4_log "$PHASE4_LOG"} \
    --output_dir ./logs/visualizations

echo ""
echo "============================================"
echo "Phase 4 complete."
echo "  Ablation results:  logs/phase4_*.json"
echo "  Visualizations:    logs/visualizations/"
echo "============================================"
