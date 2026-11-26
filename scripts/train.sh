#!/usr/bin/env bash
set -euo pipefail

model=gemini-3.5
optimizer_model=gemini-2.5-flash
mode=length_lexical
dataset=OpenThoughts

python3 -m optimize \
    --target-model "$model" \
    --dataset "$dataset" \
    --optimizer-model "$optimizer_model" \
    --mode "$mode"