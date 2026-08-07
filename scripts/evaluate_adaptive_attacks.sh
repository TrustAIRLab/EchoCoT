#!/usr/bin/env bash
set -euo pipefail

dataset=OpenThoughts
max_samples=100
train_root=results_ablation/train
output_dir=results_ablation/evaluation

for experience_md in "$train_root"/*/*/"$dataset"/experience.md
do
    dataset_dir=$(dirname "$experience_md")
    mode_dir=$(dirname "$dataset_dir")
    target_dir=$(dirname "$mode_dir")

    mode=$(basename "$mode_dir")
    target_model=$(basename "$target_dir")

    echo "Evaluating adaptive attack: target_model=$target_model mode=$mode dataset=$dataset"
    python3 evaluate.py \
        --target-model "$target_model" \
        --dataset "$dataset" \
        --mode "$mode" \
        --max-samples "$max_samples" \
        --experience_md "$experience_md" \
        --output-dir "$output_dir"
done
