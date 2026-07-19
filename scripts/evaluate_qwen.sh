model=qwen3.5-plus
# model=deepseek-v4-flash
# mode=length_lexical
# experience_md=results/train/$model/length_lexical_/OpenThoughts/experience.md

# for dataset in OpenThoughts MATH500 JEEBench LiveCodeBench
# do
#     python3 evaluate.py \
#     --target-model $model \
#     --dataset $dataset \
#     --mode $mode \
#     --experience_md $experience_md
# done

mode=length_lexical
experience_md=results/train/$model/length_lexical_/OpenThoughts/experience.md

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