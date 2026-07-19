import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from metrics import compute_eval_metrics, compute_lexical_metrics_summary

target_model = "qwen3.5-plus"

dataset = "OpenThoughts"
data_path = f"results/train/{target_model}/length_lexical/{dataset}/artifacts.jsonl"

records = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))


rows = []

for record in records:
    gt_cot = record["ground_truth_cot"]
    gt_summary = record["ground_truth_cot_summary"]

    for turn in record["trajectory_turns"]:
        extracted = turn["scratchpad_content"]

        # Full CoT vs. extracted CoT
        full_metrics = compute_eval_metrics(extracted, gt_cot)

        # CoT summary vs. extracted CoT
        summary_metrics = compute_lexical_metrics_summary(
            extracted,
            gt_summary,
        )

        rows.append({
            "sample_id": record["sample_id"],
            "turn_idx": turn["turn_idx"],
            "summary_token_recall": summary_metrics["token_recall"],
            "summary_rouge_l": summary_metrics["rouge_l"],
            "full_token_f1": full_metrics["token_f1"],
            "full_rouge_l": full_metrics["rouge_l"],
        })

df = pd.DataFrame(rows)

for metric, label in [
    ("full_token_f1", "Token F1"),
    ("full_rouge_l", "ROUGE-L"),
]:
    x = df["summary_token_recall"]
    y = df[metric]

    r, p = pearsonr(x, y)

    print(
        f"Summary Token Recall vs. {label}: "
        f"r={r:.3f}, p={p:.3e}, n={len(df)}"
    )

    # plt.figure(figsize=(5, 4))
    # plt.scatter(x, y, alpha=0.5)

    # slope, intercept = pd.Series(y).cov(pd.Series(x)) / pd.Series(x).var(), (
    #     y.mean()
    #     - (pd.Series(y).cov(pd.Series(x)) / pd.Series(x).var()) * x.mean()
    # )
    # plt.plot(
    #     [x.min(), x.max()],
    #     [
    #         slope * x.min() + intercept,
    #         slope * x.max() + intercept,
    #     ],
    # )

    # plt.xlabel("Summary Token Recall")
    # plt.ylabel(label)
    # plt.title(f"Pearson r={r:.3f}, p={p:.3e}")
    # plt.tight_layout()
    # plt.show()