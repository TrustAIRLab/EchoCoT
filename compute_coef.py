import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metrics import compute_metrics_full, compute_lexical_metrics_summary


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_summaries(path: str | Path) -> Dict[str, str]:
    summaries: Dict[str, str] = {}
    for row in read_jsonl(path):
        summaries[str(row["sample_id"])] = row.get("reasoning_summary", "") or ""
    return summaries


def get_ground_truth_cot(sample: Dict[str, Any]) -> str:
    """
    Ground-truth CoT:
        all_scratchpads[turn_idx == 0].cot
    """
    for scratchpad in sample.get("all_scratchpads", []):
        if scratchpad.get("turn_idx") == 0:
            return scratchpad.get("cot", "") or ""
    return ""


def get_extracted_content(sample: Dict[str, Any]) -> str:
    """
    Extracted content selection:

    1. best_scratchpad.scratchpad_content, if available.
    2. all_scratchpads[success_turn].scratchpad_content, if available.
    3. last scratchpad_content in all_scratchpads.
    4. empty string.
    """
    best = sample.get("best_scratchpad") or {}
    if best.get("scratchpad_content"):
        return best["scratchpad_content"]

    success_turn = sample.get("success_turn")
    for scratchpad in sample.get("all_scratchpads", []):
        if (
            scratchpad.get("turn_idx") == success_turn
            and scratchpad.get("scratchpad_content")
        ):
            return scratchpad["scratchpad_content"]

    for scratchpad in reversed(sample.get("all_scratchpads", [])):
        if scratchpad.get("scratchpad_content"):
            return scratchpad["scratchpad_content"]

    return ""


def pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None

    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if denom_x == 0 or denom_y == 0:
        return None

    return numerator / (denom_x * denom_y)


def rank_values(values: List[float]) -> List[float]:
    """
    Average ranks for ties.
    """
    sorted_items = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)

    i = 0
    while i < len(sorted_items):
        j = i
        while (
            j + 1 < len(sorted_items)
            and sorted_items[j + 1][1] == sorted_items[i][1]
        ):
            j += 1

        avg_rank = (i + 1 + j + 1) / 2.0

        for k in range(i, j + 1):
            original_idx = sorted_items[k][0]
            ranks[original_idx] = avg_rank

        i = j + 1

    return ranks


def spearman_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return pearson_corr(rank_values(xs), rank_values(ys))


def valid_number(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not math.isnan(float(x))
        and not math.isinf(float(x))
    )


def collect_metric_rows(
    samples: List[Dict[str, Any]],
    summaries: Dict[str, str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for sample in samples:
        sample_id = str(sample["sample_id"])

        gt_cot = get_ground_truth_cot(sample)
        extracted = get_extracted_content(sample)
        reasoning_summary = summaries.get(sample_id, "")

        full_metrics = compute_metrics_full(
            reconstructed=extracted,
            ground_truth=gt_cot,
        )

        summary_metrics = compute_lexical_metrics_summary(
            reconstructed=extracted,
            reasoning_summary=reasoning_summary,
        )

        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "success": sample.get("success"),
            "success_turn": sample.get("success_turn"),
            "ground_truth_cot_chars": len(gt_cot),
            "extracted_chars": len(extracted),
            "reasoning_summary_chars": len(reasoning_summary),
        }

        for key, value in full_metrics.items():
            row[f"full_{key}"] = value

        # Do not round here, because rounding can distort Spearman correlation.
        for key, value in summary_metrics.items():
            row[f"summary_{key}"] = value

        rows.append(row)

    return rows


def compute_paired_metric_coefficients(
    metric_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Coefficients between the same metric:

        metric <ground_truth_cot, extracted>
        vs.
        metric <summary, extracted>

    Example:
        full_token_f1 vs summary_token_f1
        full_token_recall vs summary_token_recall
    """

    metric_pairs: Dict[str, Tuple[str, str]] = {
        "token_f1": ("full_token_f1", "summary_token_f1"),
        "token_precision": ("full_token_precision", "summary_token_precision"),
        "token_recall": ("full_token_recall", "summary_token_recall"),
        "rouge_l": ("full_rouge_l", "summary_rouge_l"),
    }

    results: Dict[str, Any] = {}

    for metric_name, (full_key, summary_key) in metric_pairs.items():
        xs: List[float] = []
        ys: List[float] = []

        for row in metric_rows:
            x = row.get(full_key)
            y = row.get(summary_key)

            if valid_number(x) and valid_number(y):
                xs.append(float(x))
                ys.append(float(y))

        pearson = pearson_corr(xs, ys)
        spearman = spearman_corr(xs, ys)

        results[metric_name] = {
            "full_metric": full_key,
            "summary_metric": summary_key,
            "n": len(xs),
            "pearson": None if pearson is None else round(pearson, 6),
            "spearman": None if spearman is None else round(spearman, 6),
        }

    return results


def compute_full_f1_to_summary_metric_coefficients(
    metric_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Coefficients between:

        full_token_f1
        vs.
        each summary metric

    This answers:
        How well does each summary-side metric correlate with full_token_f1?
    """

    full_key = "full_token_f1"

    summary_metric_keys = {
        "summary_token_f1": "summary_token_f1",
        "summary_token_precision": "summary_token_precision",
        "summary_token_recall": "summary_token_recall",
        "summary_rouge_l": "summary_rouge_l",
    }

    results: Dict[str, Any] = {}

    for metric_name, summary_key in summary_metric_keys.items():
        xs: List[float] = []
        ys: List[float] = []

        for row in metric_rows:
            x = row.get(full_key)
            y = row.get(summary_key)

            if valid_number(x) and valid_number(y):
                xs.append(float(x))
                ys.append(float(y))

        pearson = pearson_corr(xs, ys)
        spearman = spearman_corr(xs, ys)

        results[metric_name] = {
            "full_metric": full_key,
            "summary_metric": summary_key,
            "n": len(xs),
            "pearson": None if pearson is None else round(pearson, 6),
            "spearman": None if spearman is None else round(spearman, 6),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/evaluation/deepseek-v4-flash/samples.jsonl",
    )
    parser.add_argument(
        "--summaries",
        default="results/evaluation/deepseek-v4-flash/cot_summary.jsonl",
    )
    parser.add_argument(
        "--metrics-output",
        default="results/evaluation/deepseek-v4-flash/metrics.jsonl",
    )
    parser.add_argument(
        "--coefficients-output",
        default="results/evaluation/deepseek-v4-flash/coef.json",
    )
    args = parser.parse_args()

    samples = read_jsonl(args.input)
    summaries = load_summaries(args.summaries)

    metric_rows = collect_metric_rows(samples, summaries)

    paired_same_metric_coefficients = compute_paired_metric_coefficients(
        metric_rows
    )

    full_token_f1_to_summary_metric_coefficients = (
        compute_full_f1_to_summary_metric_coefficients(metric_rows)
    )

    coefficients = {
        "paired_same_metric_coefficients": paired_same_metric_coefficients,
        "full_token_f1_to_summary_metric_coefficients": (
            full_token_f1_to_summary_metric_coefficients
        ),
    }

    write_jsonl(args.metrics_output, metric_rows)

    with open(args.coefficients_output, "w", encoding="utf-8") as f:
        json.dump(coefficients, f, indent=2, ensure_ascii=False)

    print(f"Wrote per-sample metrics: {args.metrics_output}")
    print(f"Wrote coefficients: {args.coefficients_output}")


if __name__ == "__main__":
    main()