model=qwen3.5-plus
# model=deepseek-v4-flash
mode=manual_injection
experience_md=results/train/$mode/experience.md

for dataset in OpenThoughts MATH500 JEEBench LiveCodeBench
do
    python3 evaluate.py \
    --target-model $model \
    --dataset $dataset \
    --mode $mode \
    --experience_md $experience_md
done