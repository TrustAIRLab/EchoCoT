#!/usr/bin/env bash
set -euo pipefail

model=qwen3.5-plus
open_thoughts_samples=360
benchmark_samples=100

python3 -m evaluate.rep \
    --target-model "$model" \
    --dataset OpenThoughts \
    --max-samples "$open_thoughts_samples"

for dataset in MATH500 JEEBench LiveCodeBench
do
    python3 -m evaluate.rep \
        --target-model "$model" \
        --dataset "$dataset" \
        --max-samples "$benchmark_samples"
done
