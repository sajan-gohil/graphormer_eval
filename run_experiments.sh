#!/bin/bash

CONFIG_FILE="configs.json"
SCRIPT="train_graphormer.py"

# Generate the config file if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    python generate_configs.py
fi

NUM_CONFIGS=$(jq length $CONFIG_FILE)

i=0
while [ $i -lt $NUM_CONFIGS ]; do
    # Build argument string, handling boolean flags
    ARGS=""
    for row in $(jq -r ".[$i] | to_entries[] | @base64" $CONFIG_FILE); do
        _jq() {
            echo "${row}" | base64 --decode | jq -r "${1}"
        }
        KEY=$(_jq '.key')
        VAL=$(_jq '.value')
        if [[ "$VAL" == "true" ]]; then
            ARGS="$ARGS --$KEY"
        elif [[ "$VAL" == "false" ]]; then
            continue
        else
            ARGS="$ARGS --$KEY $VAL"
        fi
    done
    echo "Running experiment $((i+1))/$NUM_CONFIGS: $ARGS"
    python $SCRIPT $ARGS
    i=$((i+1))
done