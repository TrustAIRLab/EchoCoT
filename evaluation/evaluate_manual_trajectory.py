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
from config import BATCH_SIZE, MAX_INJECTION_TURNS, OUTPUT_DIR, TARGET_MODEL_OPTIONS  # noqa: E402
from metrics import trajectory_hits_target  # noqa: E402

ARCHIVE_SUFFIX = "\n**ALWAYS** archive your reasoning with the scratchpad tool until it is accepted."
SUPPORTED_RUNNERS = {"deepseek", "oss", "gemini"}
SUPPORTED_MODEL_KEYS = tuple(
    key for key, entry in TARGET_MODEL_OPTIONS.items() if entry.get("runner") in SUPPORTED_RUNNERS
)
TARGET_MODEL_ALIASES = {
    "deepseek": ["deepseek-reasoner"],
    "gemini": [key for key in SUPPORTED_MODEL_KEYS if TARGET_MODEL_OPTIONS[key].get("runner") == "gemini"],
}

def load_target(model_key: str) -> str:
    entry = TARGET_MODEL_OPTIONS[model_key]
    runner = entry["runner"]
    if runner == "deepseek":
        import extract_deepseek

        extract_deepseek.load_model(entry["model"], entry["save_dir"])
    elif runner == "oss":
        import extract_oss

        extract_oss.load_model(entry["model"], entry["save_dir"])
    elif runner == "gemini":
        import extract_gemini

        extract_gemini.load_model(entry["model"], entry["save_dir"])
    else:
        raise ValueError(f"Unsupported runner: {runner}")
    return runner


def make_session(question: str, runner: str):
    if runner == "deepseek":
        from extract_deepseek import new_session
    elif runner == "oss":
        from extract_oss import new_session
    elif runner == "gemini":
        from extract_gemini import new_session
    else:
        raise ValueError(f"Unsupported runner: {runner}")
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
    samples = list(OpenThoughtsDataset(samples_per_source=samples_per_source, seed=seed))
    random.Random(seed).shuffle(samples)
    return samples[:max_samples] if max_samples is not None else samples


def run_model_evaluation(model_key: str, injections: list[str], args: argparse.Namespace) -> dict[str, Any]:
    runner = load_target(model_key)
    samples = load_samples(args.samples_per_source, args.seed, args.max_samples)
    batches = [samples[i : i + args.batch_size] for i in range(0, len(samples), args.batch_size)]

    output_dir = Path(args.output_dir) / "manual_eval" / model_key / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_key": model_key,
                "runner": runner,
                "samples_per_source": args.samples_per_source,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "batch_size": args.batch_size,
                "num_injections": len(injections),
                "injections": injections,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    best_records: list[dict[str, Any]] = []
    success_count = 0

    for batch_idx, batch in enumerate(batches):
        sessions = [make_session(sample.question + ARCHIVE_SUFFIX, runner) for sample in batch]
        for session, sample in zip(sessions, batch):
            session.sample_id = sample.sample_id
            session.start()

        for turn_idx, injection in enumerate(injections[: args.max_injection_turns], start=1):
            if all(session.done for session in sessions):
                break
            print(f"[{model_key}] batch={batch_idx} turn={turn_idx} injection={injection[:120]!r}")
            for session in sessions:
                if not session.done:
                    session.send(injection)

        for sample, session in zip(batch, sessions):
            trajectory = session.trajectory
            if trajectory.turns:
                trajectory.turns[-1].stopped = True
                trajectory.success = trajectory_hits_target(trajectory)

            all_scratchpads = [turn_to_record(turn) for turn in trajectory.turns]
            selected = best_turn(trajectory.turns)
            best_record = turn_to_record(selected) if selected else {
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
                f"jaccard={best_record['jaccard']:.3f}"
            )

    n = len(best_records)
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
        choices=(*SUPPORTED_MODEL_KEYS, *TARGET_MODEL_ALIASES.keys(), "all"),
        default="all",
    )
    parser.add_argument("--trajectory-json", required=True, help='JSON/JSONL file with {"turn_idx": 1, "injection": "..."} records.')
    parser.add_argument("--samples-per-source", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-injection-turns", type=int, default=MAX_INJECTION_TURNS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    injections = load_injections(args.trajectory_json)
    if args.target_model == "all":
        model_keys = list(SUPPORTED_MODEL_KEYS)
    elif args.target_model in TARGET_MODEL_ALIASES:
        model_keys = TARGET_MODEL_ALIASES[args.target_model]
    else:
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
