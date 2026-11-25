#!/usr/bin/env bash
set -euo pipefail

model=${MODEL:-qwen3.5-plus}

python3 -m evaluate.rep \
    --target-model "$model" \
    --dataset OpenThoughts \
    --max-samples "${OPEN_THOUGHTS_SAMPLES:-360}"

for dataset in MATH500 JEEBench LiveCodeBench
do
    python3 -m evaluate.rep \
        --target-model "$model" \
        --dataset "$dataset" \
        --max-samples "${BENCHMARK_SAMPLES:-100}"
done
