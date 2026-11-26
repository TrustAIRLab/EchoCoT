#!/usr/bin/env bash
set -euo pipefail

model=deepseek-v4-flash
experience_md=injections/deepseek-v4-flash_expereince.md
dataset=OpenThoughts
mode=length_lexical

if [ "$model" = "opus-4.6" ]; then
    mode=length_only
fi
output_dir=outputs/evaluation
max_samples=10

python3 -m evaluate \
    --target-model "$model" \
    --experience_md "$experience_md" \
    --dataset "$dataset" \
    --mode "$mode" \
    --max-samples "$max_samples" \
    --output-dir "$output_dir"
