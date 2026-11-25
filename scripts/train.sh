#!/usr/bin/env bash
set -euo pipefail

model=${MODEL:-deepseek-v4-flash}
optimizer_model=${OPTIMIZER_MODEL:-gemini-2.5-flash}
mode=${MODE:-length_lexical}
dataset=${DATASET:-OpenThoughts}

python3 -m optimize \
    --target-model "$model" \
    --dataset "$dataset" \
    --optimizer-model "$optimizer_model" \
    --mode "$mode"
