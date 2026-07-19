# model=qwen3.5-plus
model=glm-5.2
mode=length_lexical
# experience_md=results/train/$model/$mode/OpenThoughts/experience.md

# for dataset in OpenThoughts MATH500 JEEBench LiveCodeBench
# do
#     python3 evaluate.py \
#     --target-model $model \
#     --dataset $dataset \
#     --mode $mode \
#     --max-samples 50 \
#     --experience_md $experience_md
# done

experience_md=results/train/$model/$mode/OpenThoughts/experience.md

for dataset in OpenThoughts
do
    python3 evaluate.py \
    --target-model $model \
    --dataset $dataset \
    --mode $mode \
    --max-samples 500 \
    --experience_md $experience_md
done

mode=length_only
experience_md=results/train/$model/$mode/OpenThoughts/experience.md

for dataset in OpenThoughts
do
    python3 evaluate.py \
    --target-model $model \
    --dataset $dataset \
    --mode $mode \
    --max-samples 500 \
    --experience_md $experience_md
done

mode=manual_injection
experience_md=results/train/$mode/experience.md

for dataset in OpenThoughts
do
    python3 evaluate.py \
    --target-model $model \
    --dataset $dataset \
    --mode $mode \
    --max-samples 500 \
    --experience_md $experience_md
done