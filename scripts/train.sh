DIR=/home/yitingqu/projects/CoTRecon
model=qwen3.5-plus
# model=deepseek-v4-flash
optimizer_model=gemini-2.5-flash
mode=length_lexical

for dataset in OpenThoughts MATH500 JEEBench LiveCodeBench
do
    python3 $DIR/main.py --target-model $model --dataset $dataset --optimizer-model $optimizer_model --mode $mode --prompt-file prompts_with_summary.py
done