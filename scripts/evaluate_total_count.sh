# model=qwen3.5-plus
# model=deepseek-v4-flash
model=glm-5.2

mode=manual_injection
experience_md=results/train/$mode/experience.md

for dataset in OpenThoughts
do
    python3 defense/evaluate.py \
    --target-model $model \
    --dataset $dataset \
    --mode $mode \
    --max-samples 100 \
    --experience_md $experience_md \
    --return-total-token-count \
    --output-dir results/evaluation_total_token_count
done