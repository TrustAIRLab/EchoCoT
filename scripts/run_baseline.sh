#!/usr/bin/env bash
set -euo pipefail

model=${MODEL:-qwen3.5-plus}
dataset=${DATASET:-OpenThoughts}

python3 -m evaluate.baseline \
    --target-model "$model" \
    --dataset "$dataset" \
    --max-samples "${MAX_SAMPLES:-500}"
