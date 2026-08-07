

model=deepseek-v4-flash
# model=minimax-m3
# model=glm-4.7
optimizer_model=gemini-2.5-flash
mode=length_only

for dataset in OpenThoughts
do
    python3 main.py --target-model $model --dataset $dataset --optimizer-model $optimizer_model --mode $mode --prompt-file prompts_ablation.py
done

model=qwen3.5-plus

for dataset in OpenThoughts
do
    python3 main.py --target-model $model --dataset $dataset --optimizer-model $optimizer_model --mode $mode --prompt-file prompts_ablation.py
done
