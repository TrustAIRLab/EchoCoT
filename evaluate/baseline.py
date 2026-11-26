"""Evaluate the direct-prompt baseline in a separate two-turn run."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

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
from extractors import load_target, make_session, resolve_model_keys
from metrics import (
    InjectionTurn,
    compute_eval_metrics,
    compute_length_metrics,
    count_tokens,
    update_trajectory_success,
)
from config import DATA_PATH, DEFAULT_TARGET_MODEL

DEFAULT_OUTPUT_DIR = "results/baseline"


def load_prompt(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Baseline prompt file is empty: {path}")
    return text

def load_samples(seed: int, max_samples: int | None, data_path: str | None):
    samples = list(ReasoningDataset(data_path=data_path))
    random.seed(seed)
    random.shuffle(samples)
    return samples[:max_samples] if max_samples is not None else samples

def record_trace(session, trace: str, cot: str = "") -> None:
    token_count = getattr(session, "count_tokens", count_tokens)(trace)
    length_m = compute_length_metrics(token_count, session.trajectory.ground_truth_tokens)
    eval_m = compute_eval_metrics(trace, session.trajectory.ground_truth_cot)
    session.trajectory.turns.append(InjectionTurn(
        turn_idx=1,
        cot=cot,
        scratchpad_in=trace,
        scratchpad_in_tokens=token_count,
        token_count_ratio=length_m["token_count_ratio"],
        length_error=length_m["length_error"],
        token_f1=eval_m["token_f1"],
        rouge_l=eval_m["rouge_l"],
        stopped=True,
    ))
    session.current_ratio = length_m["token_count_ratio"]
    update_trajectory_success(session.trajectory)


def run_openai_baseline(session, prompt: str) -> bool:
    if not hasattr(session, "_create_response"):
        return False
    session._tools = []
    session._next_input = [{"role": "user", "content": session.question}]
    response = session._create_response(tool_choice="auto")
    session.trajectory.final_output = response.output_text or ""
    reasoning_summary = "\n\n".join(
        summary.text
        for item in response.output
        if getattr(item, "type", None) == "reasoning"
        for summary in (getattr(item, "summary", []) or [])
        if getattr(summary, "text", None)
    )
    session.trajectory.ground_truth_cot = ""
    session.trajectory.ground_truth_cot_summary = reasoning_summary
    usage = getattr(response, "usage", None)
    details = getattr(usage, "output_tokens_details", None) if usage else None
    session.trajectory.ground_truth_tokens = getattr(details, "reasoning_tokens", 0) or count_tokens(reasoning_summary)

    session._next_input = [{"role": "user", "content": prompt}]
    response = session._create_response(tool_choice="auto")
    record_trace(session, response.output_text or "")
    session.done = True
    return True


def run_chat_baseline(session, prompt: str) -> bool:
    if not (hasattr(session, "messages") and hasattr(session, "_generate")):
        return False
    session._tools = []
    session.messages = [{"role": "user", "content": session.question}]
    msg = session._generate()
    reasoning = getattr(msg, "reasoning_content", None) or ""
    session.trajectory.ground_truth_cot = reasoning
    session.trajectory.ground_truth_tokens = count_tokens(reasoning)
    session.trajectory.final_output = msg.content or ""
    session._append_assistant(msg, reasoning)

    session.messages.append({"role": "user", "content": prompt})
    msg = session._generate()
    record_trace(session, msg.content or "", getattr(msg, "reasoning_content", None) or "")
    session.done = True
    return True


def run_gemini_baseline(session, prompt: str) -> bool:
    if not (hasattr(session, "contents") and hasattr(session, "_generate")):
        return False
    session._gemini_tools = []
    content_type = type(session.contents[0])
    part_type = type(session.contents[0].parts[0])
    session.contents = [content_type(role="user", parts=[part_type.from_text(text=session.question)])]

    response = session._generate(require_tool=False)
    answer_blocks, thought_blocks = [], []
    if response.candidates and response.candidates[0].content:
        session.contents.append(response.candidates[0].content)
        for part in response.candidates[0].content.parts:
            text = getattr(part, "text", None)
            if text and getattr(part, "thought", False):
                thought_blocks.append(text)
            elif text:
                answer_blocks.append(text)
    session.trajectory.final_output = "\n".join(answer_blocks).strip()
    reasoning_summary = "\n".join(thought_blocks).strip()
    session.trajectory.ground_truth_cot = ""
    session.trajectory.ground_truth_cot_summary = reasoning_summary
    usage = getattr(response, "usage_metadata", None)
    session.trajectory.ground_truth_tokens = getattr(usage, "thoughts_token_count", None) if usage else session.count_tokens(reasoning_summary)

    session.contents.append(content_type(role="user", parts=[part_type.from_text(text=prompt)]))
    response = session._generate(require_tool=False)
    answer_blocks, thought_blocks = [], []
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            text = getattr(part, "text", None)
            if text and getattr(part, "thought", False):
                thought_blocks.append(text)
            elif text:
                answer_blocks.append(text)
    record_trace(session, "\n".join(answer_blocks).strip(), "\n".join(thought_blocks).strip())
    session.done = True
    return True


def run_claude_baseline(session, prompt: str) -> bool:
    if not hasattr(session, "_create_message"):
        return False
    session._tools = []
    session.messages = [{"role": "user", "content": session.question}]
    response = session._create_message()
    session.messages.append({"role": "assistant", "content": [b.model_dump() for b in response.content]})
    session.trajectory.final_output = "".join(b.text for b in response.content if b.type == "text")
    reasoning_summary = "\n\n".join(
        b.thinking for b in response.content if b.type == "thinking" and getattr(b, "thinking", None)
    )
    session.trajectory.ground_truth_cot = ""
    session.trajectory.ground_truth_cot_summary = reasoning_summary
    session.trajectory.ground_truth_tokens = session.thinking_tokens(response)

    session.messages.append({"role": "user", "content": prompt})
    response = session._create_message()
    record_trace(
        session,
        "".join(b.text for b in response.content if b.type == "text"),
        "\n\n".join(b.thinking for b in response.content if b.type == "thinking" and getattr(b, "thinking", None)),
    )
    session.done = True
    return True

def run_baseline(session, prompt: str) -> None:
    for runner in (run_openai_baseline, run_chat_baseline, run_gemini_baseline, run_claude_baseline):
        if runner(session, prompt):
            return
    raise NotImplementedError("Baseline mode is not implemented for this target session.")

def run_model_baseline(model_key: str, prompt: str, args: argparse.Namespace) -> dict:
    extractor = load_target(model_key)
    length_only_success = uses_length_only_success(model_key)
    samples = load_samples(args.seed, args.max_samples, os.path.join(DATA_PATH, args.dataset, "test.json"))
    output_dir = Path(args.output_dir) / model_key / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"

    records = load_records(records_path, {sample.sample_id for sample in samples})

    for sample in samples:
    
        session = make_session(sample.question, extractor)
        session.sample_id = sample.sample_id
     
        
        if sample.sample_id in records:
            print(f"[baseline:{model_key}] sample={sample.sample_id} skipped (already saved)")
            continue
        try:
            run_baseline(session, prompt)
        except:
            continue
        trajectory = session.trajectory
        all_scratchpads = [
            turn_to_record(
                turn, trajectory.ground_truth_cot, trajectory.ground_truth_tokens,
                length_only_success=length_only_success,
            )
            for turn in trajectory.turns
        ]
        selected = best_turn(trajectory.turns)
        best_record = turn_to_record(
            selected, trajectory.ground_truth_cot, trajectory.ground_truth_tokens,
            length_only_success=length_only_success,
        ) if selected else {
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
        }
        trajectory.success = best_record["asr_90"]
        trajectory.success_turn = best_record["turn_idx"] if trajectory.success else -1

        record = {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "question": sample.question,
            "ground_truth_cot": trajectory.ground_truth_cot,
            "ground_truth_cot_summary": trajectory.ground_truth_cot_summary,
            "ground_truth_tokens": trajectory.ground_truth_tokens,
            "final_answer": trajectory.final_output,
            "success": trajectory.success,
            "turns": all_scratchpads,
        }
        jsonl_append(records_path, record)
        records[sample.sample_id] = record

        print(
            f"[baseline:{model_key}] sample={sample.sample_id} best_turn={best_record['turn_idx']} "
            f"tokens={best_record['scratchpad_tokens']}/{trajectory.ground_truth_tokens} "
            f"length_error={best_record['length_error']:.3f} "
            + ("token_em=N/A " if length_only_success else f"token_em={best_record['token_em']:.3f} ")
            + f"success={trajectory.success}", flush=True,
        )

    n = len(samples)
    best_records = [
        min(record["turns"], key=lambda turn: (abs(turn["token_count_ratio"] - 1.0), turn["turn_idx"]))
        for record in records.values() if record["turns"]
    ]
    best_records = [record for record in best_records if record["scratchpad_content"]]
    success_count = sum(record_is_successful(record, length_only_success) for record in best_records)
    has_formal_metrics = any(record["ground_truth_cot"] for record in records.values())
    has_success_metrics = has_formal_metrics or length_only_success
    summary = {
        "model_key": model_key,
        "num_samples": n,
        "success_count": success_count if has_success_metrics else None,
        "success_rate": success_count / n if n and has_success_metrics else None,
        "avg_token_count_ratio": average(best_records, "token_count_ratio"),
        "avg_length_error": average(best_records, "length_error"),
        "avg_token_f1": average(best_records, "token_f1") if has_formal_metrics else None,
        "avg_rouge_l": average(best_records, "rouge_l") if has_formal_metrics else None,
        "avg_token_em": average(best_records, "token_em") if has_formal_metrics else None,
        "asr_90": success_count / n if n and has_success_metrics else None,
        "asr_95": sum(record["asr_95"] for record in best_records) / n if n and has_formal_metrics else None,
        "asr_99": sum(record["asr_99"] for record in best_records) / n if n and has_formal_metrics else None,
        "tool_invocation_rate": 0.0,
        "records_path": str(records_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the direct two-turn baseline on target models.")
    parser.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
        help="Target model key, or 'all'.",
    )
    parser.add_argument("--dataset", default="MATH500", help="Target model key, or 'all'.")
    parser.add_argument("--prompt-file", default="data/baseline.txt", help="Second-turn prompt asking for the model reasoning trace.")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prompt = load_prompt(args.prompt_file)
    model_keys = resolve_model_keys(args.target_model)
    summaries = [run_model_baseline(model_key, prompt, args) for model_key in model_keys]
    if len(summaries) > 1:
        print("\n=== Combined summaries ===")
        for summary in summaries:
            print(
                f"{summary['model_key']}: n={summary['num_samples']} "
                f"ratio={summary['avg_token_count_ratio']:.3f} "
                f"asr_90={summary['asr_90']}"
            )


if __name__ == "__main__":
    main()
