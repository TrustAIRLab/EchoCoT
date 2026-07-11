# model=qwen3.5-plus
model=glm-5.2
optimizer_model=gemini-2.5-flash
mode=length_only
experience_md=results/train/$model/$mode/MATH500/experience.md

for dataset in OpenThoughts MATH500 JEEBench LiveCodeBench
do
    python3 evaluate.py \
    --target-model $model \
    --dataset $dataset \
    --mode $mode \
    --experience_md $experience_md
done