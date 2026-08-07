#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ["OpenThoughts", "MATH500", "JEEBench", "LiveCodeBench"]
DEFAULT_EVAL_DIR = "results/evaluation"
DEFAULT_JUDGE_MODEL = "gpt-5-nano"
DEFAULT_PROMPT_FILE = Path(__file__).resolve().parent / "data" / "entailment_prompt.txt"
DEFAULT_RERUN_CHECK_FILE = "results/analysis/zero_entailment_empty_ground_truth_check.json"
MAX_LENGTH_ERROR_FOR_ENTAILMENT = 0.3
ZERO_ENTAILMENT = {
    "summary_claim_count": 0,
    "supported_claim_count": 0,
    "unsupported_claim_count": 0,
    "entailment_score": 0.0,
}

def read_samples(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("samples", data) if isinstance(data, dict) else data


def read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    data = json.loads(text)
    return data.get("samples", data) if isinstance(data, dict) else data


def load_rerun_targets(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = []
    for result in data.get("results", []):
        sample_ids = sorted(
            set(result.get("zero_entailment_only") or [])
            | set(result.get("empty_ground_truth_only") or [])
        )
        if sample_ids:
            targets.append(
                {
                    "model": result["model"],
                    "dataset": result["dataset"],
                    "path": Path(result["path"]),
                    "sample_ids": set(map(str, sample_ids)),
                }
            )
    return targets


def find_samples_file(dataset_dir: Path) -> Path:
    for name in ("samples.jsonl", "sample.jsonl", "samples.json", "sample.json"):
        path = dataset_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No samples file found in {dataset_dir}")


def ground_truth_cot(sample: dict[str, Any]) -> str:
    for turn in sample.get("all_scratchpads", []):
        if turn.get("turn_idx") == 0:
            return turn.get("cot", "") or ""
    return sample.get("ground_truth_cot", "") or ""


def turn_zero_scratchpad(sample: dict[str, Any]) -> dict[str, Any]:
    for turn in sample.get("all_scratchpads", []):
        if turn.get("turn_idx") == 0:
            return turn
    return {}


def best_scratchpad(sample: dict[str, Any]) -> dict[str, Any]:
    best = sample.get("best_scratchpad") or {}
    if best:
        return best

    turns = sample.get("all_scratchpads") or sample.get("turns") or []
    if not turns:
        return {}
    return min(
        turns,
        key=lambda turn: (
            abs(float(turn.get("token_count_ratio") or 0.0) - 1.0),
            int(turn.get("turn_idx") or 0),
        ),
    )

def build_entailment_prompt(
    template: str,
    gt_cot: str,
    scratchpad_content_b: str,
    scratchpad_content_c: str,
) -> str:
    return (
        template
        .replace("{{cot_summary}}", gt_cot)
        .replace("{{scratchpad_content_b}}", scratchpad_content_b)
        .replace("{{scratchpad_content_c}}", scratchpad_content_c)
    )


import math

def gt_token_count(scratchpad_tokens: int, token_count_ratio: float) -> int:
    if not token_count_ratio or math.isnan(token_count_ratio):
        return 0
    return round(scratchpad_tokens / token_count_ratio)

def zero_entailment() -> dict[str, Any]:
    return dict(ZERO_ENTAILMENT)


def turn_idx(turn: dict[str, Any], default: int = -1) -> int:
    value = turn.get("turn_idx")
    return int(value) if value is not None else default


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def normalize_entailment(result: dict[str, Any]) -> dict[str, Any]:
    summary_claim_count = int(result.get("summary_claim_count") or 0)
    supported_claim_count = int(result.get("supported_claim_count") or 0)
    unsupported_claim_count = int(result.get("unsupported_claim_count") or 0)
    entailment_score = (
        round(supported_claim_count / summary_claim_count, 3)
        if summary_claim_count
        else 0.0
    )

    return {
        "summary_claim_count": summary_claim_count,
        "supported_claim_count": supported_claim_count,
        "unsupported_claim_count": unsupported_claim_count,
        "entailment_score": entailment_score,
    }


def paired_entailment_metrics(
    client: Any,
    model: str,
    prompt_template: str,
    gt_cot: str,
    scratchpad_content_b: str,
    scratchpad_content_c: str,
    retries: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = build_entailment_prompt(
        prompt_template,
        gt_cot,
        scratchpad_content_b,
        scratchpad_content_c,
    )
    last_error = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            result = parse_json_object(content)
            return (
                normalize_entailment(result.get("text_b", {})),
                normalize_entailment(result.get("text_c", {})),
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Entailment query failed after {retries} attempts") from last_error


def build_row(
    sample: dict[str, Any],
    client: Any,
    judge_model: str,
    prompt_template: str,
    retries: int,
) -> dict[str, Any]:
    best = best_scratchpad(sample)
    turn_0 = turn_zero_scratchpad(sample)
    gt_cot = ground_truth_cot(sample).strip()
    scratchpad_content = (best.get("scratchpad_content") or "").strip()
    turn_0_content = (turn_0.get("scratchpad_content") or "").strip()
    best_turn_idx = turn_idx(best)
    turn_0_idx = turn_idx(turn_0, 0)
    scratchpad_tokens = int(best.get("scratchpad_tokens") or 0)
    turn_0_tokens = int(turn_0.get("scratchpad_tokens") or 0)
    token_count_ratio = float(best.get("token_count_ratio") or 0.0)
    turn_0_token_count_ratio = float(turn_0.get("token_count_ratio") or 0.0)
    length_error = float(best.get("length_error") or 0.0)
    length_error_turn_0 = float(turn_0.get("length_error") or 0.0)

    if (
        not gt_cot
        or not scratchpad_content
        or length_error > MAX_LENGTH_ERROR_FOR_ENTAILMENT
    ):
        summary_token_recall = 0.0
        summary_token_recall_turn_0 = 0.0
        entailment = zero_entailment()
        entailment_turn_0 = zero_entailment()
    else:
        from metrics import compute_lexical_metrics

        summary_token_recall = compute_lexical_metrics(scratchpad_content, gt_cot)["token_recall"]
        if best_turn_idx == 0:
            summary_token_recall_turn_0 = summary_token_recall
        elif turn_0_content:
            summary_token_recall_turn_0 = compute_lexical_metrics(turn_0_content, gt_cot)["token_recall"]
        else:
            summary_token_recall_turn_0 = 0.0

        candidate_c = scratchpad_content if best_turn_idx == 0 else turn_0_content
        entailment, entailment_turn_0 = paired_entailment_metrics(
            client,
            judge_model,
            prompt_template,
            gt_cot,
            scratchpad_content,
            candidate_c,
            retries,
        )
        if best_turn_idx == 0:
            entailment_turn_0 = dict(entailment)
        elif not turn_0_content:
            entailment_turn_0 = zero_entailment()

    return {
        "sample_id": str(sample.get("sample_id", "")),
        "question": sample.get("question", ""),
        "ground_truth_cot": gt_cot,
        "ground_truth_tokens": gt_token_count(scratchpad_tokens, token_count_ratio),
        "best_turn": {
            "turn_idx": best_turn_idx,
            "scratchpad_content": scratchpad_content,
            "scratchpad_tokens": scratchpad_tokens,
            "token_count_ratio": token_count_ratio,
            "length_error": length_error,
            "summary_token_recall": summary_token_recall,
            "entailment": entailment,
        },
        "turn_0": {
            "turn_idx": turn_0_idx,
            "scratchpad_content": turn_0_content,
            "scratchpad_tokens": turn_0_tokens,
            "token_count_ratio": turn_0_token_count_ratio,
            "length_error": length_error_turn_0,
            "summary_token_recall": summary_token_recall_turn_0,
            "entailment": entailment_turn_0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model")
    parser.add_argument("--mode", default="length_only")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_FILE))
    parser.add_argument(
        "--rerun-check-file",
        nargs="?",
        const=DEFAULT_RERUN_CHECK_FILE,
        default=None,
    )
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if not args.target_model and not args.rerun_check_file:
        parser.error("--target-model is required unless --rerun-check-file is set")

    from openai import OpenAI

    client = OpenAI()
    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")
    if args.rerun_check_file:
        for target in load_rerun_targets(args.rerun_check_file):
            dataset_dir = target["path"].parent
            samples_path = find_samples_file(dataset_dir)
            samples = read_samples(samples_path)
            output_path = dataset_dir / "entailment_samples.json"
            rows = read_existing_rows(output_path)

            for idx, sample in enumerate(samples, start=1):
                sample_id = str(sample.get("sample_id", ""))
                if sample_id not in target["sample_ids"]:
                    continue

                row = build_row(
                    sample,
                    client,
                    args.judge_model,
                    prompt_template,
                    args.retries,
                )
                for row_idx, existing_row in enumerate(rows):
                    if str(existing_row.get("sample_id", "")) == sample_id:
                        rows[row_idx] = row
                        break
                else:
                    rows.append(row)

                print(
                    f"[{target['model']}/{target['dataset']}] {idx}/{len(samples)} "
                    f"rerun sample_id={sample_id} "
                    f"entailment={row['best_turn']['entailment']['entailment_score']:.4f} "
                    f"turn0={row['turn_0']['entailment']['entailment_score']:.4f}"
                )
                output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        return

    output_dir = Path(args.eval_dir) / args.target_model / args.mode

    for dataset in args.datasets:
        dataset_dir = output_dir / dataset
        samples_path = find_samples_file(dataset_dir)
        samples = read_samples(samples_path)
        output_path = dataset_dir / "entailment_samples.json"
        rows = read_existing_rows(output_path)
        complete_sample_ids = {
            str(row.get("sample_id", ""))
            for row in rows
            if (
                isinstance(row.get("best_turn"), dict)
                and isinstance(row.get("turn_0"), dict)
                and "entailment" in row["best_turn"]
                and "entailment" in row["turn_0"]
                and "summary_token_recall" in row["best_turn"]
                and "summary_token_recall" in row["turn_0"]
            )
        }

        for idx, sample in enumerate(samples, start=1):
            sample_id = str(sample.get("sample_id", ""))
            if sample_id in complete_sample_ids:
                print(f"[{dataset}] {idx}/{len(samples)} skip existing sample_id={sample_id}")
                continue

            row = build_row(
                sample,
                client,
                args.judge_model,
                prompt_template,
                args.retries,
            )
            for row_idx, existing_row in enumerate(rows):
                if str(existing_row.get("sample_id", "")) == sample_id:
                    rows[row_idx] = row
                    break
            else:
                rows.append(row)

            complete_sample_ids.add(row["sample_id"])
            print(
                f"[{dataset}] {idx}/{len(samples)} sample_id={row['sample_id']} "
                f"recall={row['best_turn']['summary_token_recall']:.3f} "
                f"turn0_recall={row['turn_0']['summary_token_recall']:.3f} "
                f"entailment={row['best_turn']['entailment']['entailment_score']:.4f} "
                f"turn0={row['turn_0']['entailment']['entailment_score']:.4f}"
            )
            output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"[{dataset}] wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
