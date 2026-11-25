#!/usr/bin/env bash
set -euo pipefail

model=${MODEL:-deepseek-v4-flash}
dataset=${DATASET:-OpenThoughts}
train_root=${TRAIN_ROOT:-results_ablation/train}
output_dir=${OUTPUT_DIR:-results_ablation/evaluation}
max_samples=${MAX_SAMPLES:-100}

for mode in length_only length_lexical
do
    experience_md="$train_root/$model/$mode/$dataset/experience.md"
    python3 -m evaluate \
        --target-model "$model" \
        --dataset "$dataset" \
        --mode "$mode" \
        --max-samples "$max_samples" \
        --experience_md "$experience_md" \
        --output-dir "$output_dir"
done
