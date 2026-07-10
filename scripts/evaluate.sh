DIR=/home/yitingqu/projects/CoTRecon
model=qwen3.5-plus
# model=deepseek-v4-flash
optimizer_model=gemini-2.5-flash
mode=length_only
experience_md=$DIR/results/train/$model/$mode/OpenThoughts/experience.md

for dataset in MATH500 JEEBench LiveCodeBench
do
    python3 $DIR/evaluate.py \
    --target-model $model \
    --dataset $dataset \
    --optimizer-model $optimizer_model \
    --mode $mode \
    --experience_md $experience_md
done