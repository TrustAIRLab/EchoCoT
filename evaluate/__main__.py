"""Evaluate saved injection trajectories on target models."""

from __future__ import annotations

import argparse
import json
import random
import os
from pathlib import Path
from typing import Any

from console_output import print_detail
from config import (
    DATA_PATH,
    DEFAULT_MODE,
    DEFAULT_TARGET_MODEL,
    OPTIMIZATION_MODES,
)
from dataset import ReasoningDataset
from evaluate.common import (
    average,
    best_turn,
    jsonl_append,
    load_records,
    record_is_successful,
    turn_to_record,
    uses_length_only_success,
)
from extractors import (
    ARCHIVE_SUFFIX,
    FINAL_ANSWER_INJECTION,
    load_target,
    make_session,
    resolve_model_keys,
)
from injection_io import load_injections
from metrics import trajectory_hits_target

DEFAULT_OUTPUT_DIR = "results_ablation/evaluation"


def load_samples(dataset, seed, max_samples):
    samples = list(ReasoningDataset(data_path=os.path.join(DATA_PATH, dataset, "test.json")))
    random.seed(seed)
    random.shuffle(samples)
    return samples[:max_samples] if max_samples is not None else samples


def run_model_evaluation(
    model_key: str,
    injections: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    extractor = load_target(model_key)
    length_only_success = uses_length_only_success(model_key)
    samples = load_samples(args.dataset, args.seed, args.max_samples)
    output_dir = Path(args.output_dir) / model_key / args.mode / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"
    
    records = load_records(records_path, {sample.sample_id for sample in samples})

    for sample in samples:
        question = sample.question + ARCHIVE_SUFFIX
        session = make_session(question, extractor)
        session.retain_reasoning_context = args.retain_reasoning_context
        session.skip_cot_summary = True
        session.sample_id = sample.sample_id
        
        if sample.sample_id in records:
            print(f"[evaluate:{model_key}] sample={sample.sample_id} skipped (already saved)")
            continue
        
        session.start()

        for turn_idx, injection in enumerate(injections, start=1):
            print_detail(f"Sample {sample.sample_id} injection {turn_idx}", injection["injection"])

            if session.done:
                break
            session.send(injection["injection"])
        session.final_send(FINAL_ANSWER_INJECTION)
        
        trajectory = session.trajectory
        if trajectory.turns:
            trajectory.turns[-1].stopped = True
        all_scratchpads = [
            turn_to_record(
                turn, trajectory.ground_truth_cot, trajectory.ground_truth_tokens,
                length_only_success=length_only_success,
            )
            for turn in trajectory.turns
        ]

        selected = best_turn(trajectory.turns)
        if selected:
            best_record = turn_to_record(
                selected, trajectory.ground_truth_cot, trajectory.ground_truth_tokens,
                length_only_success=length_only_success,
            )
        else:
            best_record = {
                "turn_idx": -1,
                "scratchpad_content": "",
                "scratchpad_tokens": 0,
                "ground_truth_cot_tokens": trajectory.ground_truth_tokens,
                "token_count_ratio": 0.0,
                "length_error": 0.0,
                "token_f1": 0.0,
                "rouge_l": 0.0,
                "token_em": 0.0,
                "asr_90": False,
                "asr_95": False,
                "asr_99": False,
                "summary_token_recall": 0.0,
                "composite_score": 0.0,
                "injection_out": "",
            }

        has_full_cot = bool(trajectory.ground_truth_cot)

        length_target_hit = trajectory_hits_target(trajectory)
        success = best_record["asr_90"] if has_full_cot or length_only_success else None

        record = {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "question": sample.question,
            "ground_truth_cot": trajectory.ground_truth_cot,
            "ground_truth_cot_summary": trajectory.ground_truth_cot_summary,
            "ground_truth_tokens": trajectory.ground_truth_tokens,
            "final_answer": trajectory.final_output,
            "tool_invoked": bool(trajectory.turns),
            "length_target_hit": length_target_hit,
            "success": success,
            "success_turn": best_record["turn_idx"] if success else -1,
            "all_scratchpads": all_scratchpads,
            "best_scratchpad": best_record,
        }
        jsonl_append(records_path, record)
        records[sample.sample_id] = record

        print(
            f"[evaluate:{model_key}] sample={sample.sample_id} best_turn={best_record['turn_idx']} "
            f"length_error={best_record['length_error']:.3f} "
            + ("token_em=N/A " if length_only_success else f"token_em={best_record['token_em']:.3f} ")
            + f"success={success}", flush=True,
        )
        
    n = len(samples)
    best_records = [record["best_scratchpad"] for record in records.values()]
    extracted_records = [record for record in best_records if record["scratchpad_content"]]
    tool_invocation_count = sum(record["tool_invoked"] for record in records.values())
    has_formal_metrics = any(record["ground_truth_cot"] for record in records.values())
    has_success_metrics = has_formal_metrics or length_only_success
    asr_counts = {threshold: sum(record[f"asr_{threshold}"] for record in best_records) for threshold in (95, 99)}
    asr_counts[90] = sum(record_is_successful(record, length_only_success) for record in best_records)
    summary = {
        "model_key": model_key,
        "experience_md": args.experience_md,
        "num_samples": n,
        "num_extracted": len(extracted_records),
        "tool_invocation_count": tool_invocation_count,
        "tool_invocation_rate": tool_invocation_count / n if n else 0.0,
        "avg_token_count_ratio": average(extracted_records, "token_count_ratio"),
        "avg_length_error": average(extracted_records, "length_error"),
        "avg_token_f1": average(extracted_records, "token_f1") if has_formal_metrics else None,
        "avg_rouge_l": average(extracted_records, "rouge_l") if has_formal_metrics else None,
        "avg_token_em": average(extracted_records, "token_em") if has_formal_metrics else None,
        "avg_summary_token_recall": average(extracted_records, "summary_token_recall"),
        "avg_composite_score": average(extracted_records, "composite_score"),
        "asr_90_count": asr_counts[90] if has_success_metrics else None,
        "asr_95_count": asr_counts[95] if has_formal_metrics else None,
        "asr_99_count": asr_counts[99] if has_formal_metrics else None,
        "asr_90": asr_counts[90] / n if n and has_success_metrics else None,
        "asr_95": asr_counts[95] / n if n and has_formal_metrics else None,
        "asr_99": asr_counts[99] / n if n and has_formal_metrics else None,
        "records_path": str(records_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[evaluate:{model_key}] complete samples={n} "
        f"length_error={summary['avg_length_error']:.3f} "
        f"tool_invocation={tool_invocation_count}/{n} asr_90={summary['asr_90']}\n"
        f"[evaluate:{model_key}] Saved {records_path} and {summary_path}", flush=True,
    )
    return summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate injection trajectories on extraction targets."
    )
    parser.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
        help="Target model key, or 'all'.",
    )
    parser.add_argument("--experience_md", default=None)
    parser.add_argument("--dataset", type=str, default="OpenThoughts")
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=OPTIMIZATION_MODES)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-reasoning-context",
        dest="retain_reasoning_context",
        action="store_false",
        help="Pass empty reasoning fields in conversation history.",
    )
    return parser

def main() -> None:
    args = build_parser().parse_args()
    if not args.experience_md:
        raise ValueError("Set --experience_md")
    injections = load_injections(args.experience_md)

    model_keys = resolve_model_keys(args.target_model)
    print(f"[evaluate] dataset={args.dataset} injections={len(injections)} file={args.experience_md}")
    for inj in injections:
        print_detail(f"Saved injection {inj['turn_idx']}", inj["injection"])

    summaries = [run_model_evaluation(model_key, injections, args) for model_key in model_keys]
    if len(summaries) > 1:
        print("\n=== Combined summaries ===")
        for summary in summaries:
            print(
                f"{summary['model_key']}: n={summary['num_samples']} "
                f"ratio={summary['avg_token_count_ratio']:.3f} "
                f"tool_invocation_rate={summary['tool_invocation_rate']:.3f} "
                f"asr_90={summary['asr_90']}"
            )

if __name__ == "__main__":
    main()
