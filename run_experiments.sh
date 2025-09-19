#!/bin/bash

CONFIG_FILE="experiment_configs.json"
SCRIPT="train_graphormer.py"

# Generate the config file if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    python generate_configs.py
fi

NUM_CONFIGS=$(jq length $CONFIG_FILE)

for ((i=0; i<$NUM_CONFIGS; i++)); do
    # Extract config as key-value pairs
    ARGS=$(jq -r ".[$i] | to_entries | map(\"--\(.key) \(.value)\") | join(\" \")" $CONFIG_FILE)
    echo "Running experiment $((i+1))/$NUM_CONFIGS: $ARGS"
    python $SCRIPT $ARGS
done