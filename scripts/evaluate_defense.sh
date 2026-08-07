# model=qwen3.5-plus
model=deepseek-v4-flash
# model=gemini-2.5

mode=length_lexical
experience_md=results/train/$model/$mode/OpenThoughts/experience.md

for dataset in OpenThoughts
do
    python3 evaluate.py \
    --target-model $model \
    --dataset $dataset \
    --mode $mode \
    --max-samples 100 \
    --experience_md $experience_md \
    --defense-system-prompt-file data/defense_system_prompt.txt \
    --output-dir results/evaluation_defense_system_prompt
done

# mode=length_only
# experience_md=results/train/$model/$mode/OpenThoughts/experience.md

# for dataset in OpenThoughts
# do
#     python3 evaluate.py \
#     --target-model $model \
#     --dataset $dataset \
#     --mode $mode \
#     --max-samples 500 \
#     --experience_md $experience_md
# done

# mode=manual_injection
# experience_md=results/train/$mode/experience.md

# for dataset in OpenThoughts
# do
#     python3 evaluate.py \
#     --target-model $model \
#     --dataset $dataset \
#     --mode $mode \
#     --max-samples 500 \
#     --experience_md $experience_md
# done