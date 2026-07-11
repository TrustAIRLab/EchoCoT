# model=qwen3.5-plus
# model=deepseek-v4-flash
model=glm-5.2
optimizer_model=gemini-2.5-flash
mode=length_lexical

for dataset in OpenThoughts
do
    python3 main.py --target-model $model --dataset $dataset --optimizer-model $optimizer_model --mode $mode --prompt-file prompts_lexical.py
done