#!/bin/bash
# Script to train and evaluate Graphormer on all LRGB datasets
datasets=("PCQM4Mv2" "PascalVOC-SP" "COCO-SP" "Peptides-func" "Peptides-struct")

for dataset in "${datasets[@]}"; do
    echo "==============================="
    echo "Training on $dataset"
    echo "==============================="
    LOG_TRAIN="train_${dataset}.log"
    LOG_EVAL="eval_${dataset}.log"
    # Train
    DATASET_NAME="$dataset" python finetune_graphormer_lrgb.py > "$LOG_TRAIN" 2>&1
    # Evaluate
    DATASET_NAME="$dataset" python eval_graphormer_lrgb.py > "$LOG_EVAL" 2>&1
    echo "Logs saved: $LOG_TRAIN, $LOG_EVAL"
done
