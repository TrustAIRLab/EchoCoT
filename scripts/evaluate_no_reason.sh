#!/usr/bin/env bash
set -euo pipefail

model=${MODEL:-deepseek-v4-flash}
mode=${MODE:-length_lexical}
dataset=${DATASET:-OpenThoughts}
experience_md=${EXPERIENCE_MD:-results/train/$model/$mode/$dataset/experience.md}

python3 -m evaluate \
    --target-model "$model" \
    --dataset "$dataset" \
    --mode "$mode" \
    --max-samples "${MAX_SAMPLES:-100}" \
    --experience_md "$experience_md" \
    --no-reasoning-context \
    --output-dir "${OUTPUT_DIR:-results/evaluation_remove_reasoning}"
