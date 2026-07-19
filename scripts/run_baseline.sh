model=qwen3.5-plus
model=deepseek-v4-flash
# model=glm-5.2

# for dataset in OpenThoughts MATH500 JEEBench LiveCodeBench
# do
#     python3 baseline_evaluate.py --target-model $model --dataset $dataset
# done

for dataset in OpenThoughts
do
    python3 baseline_evaluate.py --target-model $model --dataset $dataset
done