DIR=/home/yitingqu/projects/CoTRecon
model=qwen3.5-plus
# model=deepseek-v4-flash


for dataset in OpenThoughts MATH500 JEEBench LiveCodeBench
do
    python3 $DIR/baseline_evaluate.py --target-model $model --dataset $dataset
done