import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

target_models = ["deepseek-v4-flash", "qwen3.5-plus", "glm-5.2"][:1]
train_dataset = "OpenThoughts"
mode = "length_lexical_old"

def get_data(target_model, mode,  train_dataset):
    data_path = f"../results/train/{target_model}/{mode}/{train_dataset}/artifacts.jsonl"

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

rows = []

for model in target_models:
    data = get_data(model, mode, train_dataset)
    for idx in range(len(data)):
        turns = data[idx]["trajectory_turns"]

        for turn_data in turns:
            rows.append({
                "model": model,
                "summary_recall": turn_data["summary_token_recall"],
                "token_f1": turn_data["token_f1"],
                "rouge_l": turn_data["rouge_l"],
            })

df = pd.DataFrame(rows).dropna()

for metric, label in [
    ("token_f1", "Token F1"),
    ("rouge_l", "ROUGE-L"),
]:
    x = df["summary_recall"]
    y = df[metric]

    r, p = pearsonr(x, y)
    print(f"Summary Recall vs. {label}: r={r:.3f}, p={p:.3e}, n={len(df)}")

    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, alpha=0.6)

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    plt.plot(x_line, slope * x_line + intercept)

    plt.xlabel("Summary Token Recall")
    plt.ylabel(label)
    plt.title(f"Pearson r={r:.3f}, p={p:.3e}")
    plt.tight_layout()
    plt.savefig(f"../results/figures/summary_recall_vs_{metric}.pdf")
    plt.show()