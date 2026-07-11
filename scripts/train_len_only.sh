DIR=/Users/yitingqu/Documents/projects/CoTRecon
# model=qwen3.5-plus
# model=deepseek-v4-flash
# model=minimax-m3
model=glm-4.7
optimizer_model=gemini-2.5-flash
mode=length_only

for dataset in OpenThoughts LiveCodeBench
do
    python3 $DIR/main.py --target-model $model --dataset $dataset --optimizer-model $optimizer_model --mode $mode --prompt-file prompts.py
done