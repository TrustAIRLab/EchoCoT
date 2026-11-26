#!/usr/bin/env bash
set -euo pipefail

model=qwen3.5-plus
dataset=OpenThoughts
max_samples=500

python3 -m evaluate.baseline \
    --target-model "$model" \
    --dataset "$dataset" \
    --max-samples "$max_samples"
