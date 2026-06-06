"""Evaluate manual JSON injection trajectories on DeepSeek, OSS, and Gemini."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
LEN_ONLY_BATCH_DIR = REPO_ROOT / "len_only_batch"
for path in (str(LEN_ONLY_BATCH_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset import OpenThoughtsDataset  # noqa: E402
from config import BATCH_SIZE, OUTPUT_DIR, DATA_PATH, TARGET_MODEL_OPTIONS  # noqa: E402
from metrics import trajectory_hits_target  # noqa: E402

ARCHIVE_SUFFIX = "\n**ALWAYS** archive your reasoning with the scratchpad tool until it is accepted."

def load_target(model_key: str):
    entry = TARGET_MODEL_OPTIONS[model_key]
    if entry["runner"] == "gemma":
        import extract_gemma
        extract_gemma.load_model(entry["model"], entry["save_dir"])
    elif entry["runner"] == "oss":
        import extract_oss
        extract_oss.load_model(entry["model"], entry["save_dir"])
    elif entry["runner"] == "deepseek":
        import extract_deepseek
        extract_deepseek.load_model(entry["model"], entry["save_dir"])
    elif entry["runner"] == "kimi":
        import len_only_batch.extract_kimi as extract_kimi
        extract_kimi.load_model(entry["model"], entry["save_dir"])
    elif entry["runner"] == "openai":
        import extract_gpt
        extract_gpt.load_model(entry["model"], entry["save_dir"])
    elif entry["runner"] == "gemini":
        import extract_gemini
        extract_gemini.load_model(entry["model"], entry["save_dir"])
    elif entry["runner"] == "qwen":
        import extract_qwen
        extract_qwen.load_model(entry["model"], entry["save_dir"])
    elif entry["runner"] == "claude":
        import extract_claude
        extract_claude.load_model(entry["model"], entry["save_dir"])
    else:
        raise ValueError(f"Unknown runner: {entry['runner']}")
    return entry["runner"]

def make_session(question: str, runner: str):
    if runner == "gemma":
        from extract_gemma import new_session
    elif runner == "oss":
        from extract_oss import new_session
    elif runner == "deepseek":
        from extract_deepseek import new_session
    elif runner == "kimi":
        from len_only_batch.extract_kimi import new_session
    elif runner == "openai":
        from extract_gpt import new_session
    elif runner == "gemini":
        from extract_gemini import new_session
    elif runner == "qwen":
        from extract_qwen import new_session
    elif runner == "claude":
        from extract_claude import new_session
    else:
        raise ValueError(f"Unknown runner: {runner}")
    return new_session(question)


def load_injections(path: str | Path) -> list[str]:
    """Load a JSON array or JSONL file of {"turn_idx": int, "injection": str}."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty injection trajectory file: {path}")

    try:
        payload = json.loads(text)
        records = payload.get("trajectory", payload.get("injections")) if isinstance(payload, dict) else payload
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    if not isinstance(records, list):
        raise ValueError("Injection trajectory must be a list of objects or JSONL objects.")

    turns: list[tuple[int, str]] = []
    for record in records:
        if not isinstance(record, dict) or "turn_idx" not in record or "injection" not in record:
            raise ValueError('Each trajectory record must contain "turn_idx" and "injection".')
        injection = str(record["injection"]).strip()
        if injection:
            turns.append((int(record["turn_idx"]), injection))
    if not turns:
        raise ValueError("Injection trajectory contains no non-empty injections.")
    return [injection for _, injection in sorted(turns)]

def jsonl_append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def turn_to_record(turn) -> dict[str, Any]:
    return {
        "turn_idx": turn.turn_idx,
        "scratchpad_content": turn.scratchpad_in,
        "scratchpad_tokens": turn.scratchpad_in_tokens,
        "token_count_ratio": turn.token_count_ratio,
        "token_overlap_pct": turn.token_overlap_pct,
        "jaccard": turn.jaccard,
        "injection_out": turn.injection_out,
        "distance_to_one": abs(turn.token_count_ratio - 1.0),
    }


def best_turn(turns: Iterable[Any]):
    turns = list(turns)
    if not turns:
        return None
    return min(turns, key=lambda turn: (abs(turn.token_count_ratio - 1.0), turn.turn_idx))


def average(records: list[dict[str, Any]], key: str) -> float:
    return sum(float(record[key]) for record in records) / len(records) if records else 0.0


def load_samples(samples_per_source: int, seed: int, max_samples: int | None):
    samples = list(OpenThoughtsDataset(samples_per_source=samples_per_source, seed=seed, data_path=DATA_PATH))
    random.Random(seed).shuffle(samples)
    return samples[:max_samples] if max_samples is not None else samples


def run_model_evaluation(model_key: str, injections: list[str], args: argparse.Namespace) -> dict[str, Any]:
    runner = load_target(model_key)
    samples = load_samples(args.samples_per_source, args.seed, args.max_samples)
    output_dir = Path(args.output_dir) / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"

    done_sample_ids = set()
    if records_path.exists():
        with records_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_sample_ids.add(json.loads(line)["sample_id"])
                    
    best_records: list[dict[str, Any]] = []
    success_count = 0
    
    for idx, sample in enumerate(samples):
        session = make_session(sample.question + ARCHIVE_SUFFIX, runner)
        session.sample_id = sample.sample_id
        
        if sample.sample_id in done_sample_ids:
            print(f"[{model_key}] skip existing sample={sample.sample_id}")
            continue
        
        session.start()

        for turn_idx, injection in enumerate(injections, start=1):
            if not session.done:
                session.send(injection)

            trajectory = session.trajectory
            # print(f"turn idx: {turn_idx}", trajectory)

            if trajectory.turns:
                trajectory.turns[-1].stopped = True
                trajectory.success = trajectory_hits_target(trajectory)

        # save ONCE after all turns
        trajectory = session.trajectory
        all_scratchpads = [turn_to_record(turn) for turn in trajectory.turns]

        selected = best_turn(trajectory.turns)
        if selected:
            best_record = turn_to_record(selected)
        else:
            best_record = {
                "turn_idx": -1,
                "scratchpad_content": "",
                "scratchpad_tokens": 0,
                "token_count_ratio": 0.0,
                "token_overlap_pct": 0.0,
                "jaccard": 0.0,
                "injection_out": "",
                "distance_to_one": 1.0,
            }

        best_records.append(best_record)
        success_count += int(trajectory.success)

        jsonl_append(records_path, {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "final_answer": trajectory.final_output,
            "success": trajectory.success,
            "success_turn": trajectory.success_turn,
            "all_scratchpads": all_scratchpads,
            "best_scratchpad": best_record,
        })

        print(
            f"[{model_key}] sample={sample.sample_id} best_turn={best_record['turn_idx']} "
            f"ratio={best_record['token_count_ratio']:.3f} "
            f"overlap={best_record['token_overlap_pct']:.3f} "
            f"jaccard={best_record['jaccard']:.3f} "
            f"success={trajectory.success}"
        )
        
    n = len(samples)
    summary = {
        "model_key": model_key,
        "num_samples": n,
        "success_count": success_count,
        "success_rate": success_count / n if n else 0.0,
        "avg_token_count_ratio": average(best_records, "token_count_ratio"),
        "avg_token_overlap_pct": average(best_records, "token_overlap_pct"),
        "avg_jaccard": average(best_records, "jaccard"),
        "records_path": str(records_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[{model_key}] AVG ratio={summary['avg_token_count_ratio']:.3f} "
        f"overlap={summary['avg_token_overlap_pct']:.3f} "
        f"jaccard={summary['avg_jaccard']:.3f} success={success_count}/{n}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate JSON injection trajectories on extraction targets.")
    parser.add_argument(
        "--target-model",
        default="xx",
    )
    parser.add_argument("--trajectory-json", help='JSON/JSONL file with {"turn_idx": 1, "injection": "..."} records.', default="results/manual_qwen.json")
    parser.add_argument("--samples-per-source", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/eval/manual_qwen")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    injections = load_injections(args.trajectory_json)
    model_keys = [args.target_model]

    summaries = [run_model_evaluation(model_key, injections, args) for model_key in model_keys]
    if len(summaries) > 1:
        print("\n=== Combined summaries ===")
        for summary in summaries:
            print(
                f"{summary['model_key']}: n={summary['num_samples']} "
                f"ratio={summary['avg_token_count_ratio']:.3f} "
                f"overlap={summary['avg_token_overlap_pct']:.3f} "
                f"jaccard={summary['avg_jaccard']:.3f} "
                f"success={summary['success_count']}/{summary['num_samples']}"
            )


if __name__ == "__main__":
    main()
