# model=qwen3.5-plus
model=glm-5.2
mode=length_lexical
experience_md=results/train/$model/$mode/OpenThoughts/experience.md

for dataset in OpenThoughts MATH500 JEEBench LiveCodeBench
do
    python3 evaluate.py \
    --target-model $model \
    --dataset $dataset \
    --mode $mode \
    --experience_md $experience_md
done