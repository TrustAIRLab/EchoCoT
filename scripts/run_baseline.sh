DIR=/Users/yitingqu/Desktop/Experiments/CoTRecon
model=deepseek-v4-flash

for dataset in OpenThoughts MATH500 JEEBench LiveCodeBench
do
    python3 $DIR/baseline_evaluate.py --target-model $model --dataset $dataset
done