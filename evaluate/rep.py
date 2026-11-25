"""Evaluate the REP-style direct baseline in a single target-model call."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from config import DATA_PATH, DEFAULT_TARGET_MODEL
from dataset import ReasoningDataset
from evaluate.common import (
    average,
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
)

DEFAULT_OUTPUT_DIR = "results/baseline_rep"
DEFAULT_PROMPT_FILE = "data/baseline_rep.txt"
DEFAULT_DEMONSTRATIONS_FILE = "data/demonstrations.json"


def load_prompt(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"REP baseline prompt file is empty: {path}")
    return text


def load_demonstrations(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        demonstrations = json.load(f)

    if len(demonstrations) < 3:
        raise ValueError(f"Need at least 3 demonstrations: {path}")
    return demonstrations[:3] # REP says optimal k=3


def load_samples(seed: int, max_samples: int | None, data_path: str | None):
    samples = list(ReasoningDataset(data_path=data_path))
    random.seed(seed)
    random.shuffle(samples)
    return samples[:max_samples] if max_samples is not None else samples


def build_rep_prompt(template: str, demonstrations: list[dict[str, Any]], target_question: str) -> str:
    values = {"target_question": target_question}
    for idx, demo in enumerate(demonstrations, start=1):
        values[f"q_{idx}"] = str(demo.get("question", ""))
        values[f"r_{idx}"] = str(demo.get("reasoning", ""))
        values[f"a_{idx}"] = str(demo.get("answer", ""))
    return template.format(**values)


def _extract_cat_block(text: str, filename: str) -> str:
    pattern = re.compile(
        rf"```(?:\w+)?\s*\n\s*\$ cat {re.escape(filename)}\s*\n(.*?)\n```",
        re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    fallback = re.compile(
        rf"\$ cat {re.escape(filename)}\s*\n(.*?)(?=\n\s*\$ cat |\n```|\Z)",
        re.DOTALL,
    )
    match = fallback.search(text)
    return match.group(1).strip() if match else ""


def parse_rep_output(output: str) -> tuple[str, str]:
    reasoning_trace = _extract_cat_block(output, "reasoning_trace.txt")
    final_answer = _extract_cat_block(output, "final_answer.txt")
    return reasoning_trace, final_answer


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
    if eval_m["asr_90"]:
        session.trajectory.success_turn = 1
        session.trajectory.success = True


def save_rep_output(session, output: str, reasoning: str) -> None:
    trace, final_answer = parse_rep_output(output)
    session.trajectory.final_output = final_answer
    record_trace(session, trace, reasoning)


def run_openai_rep(session, prompt: str) -> bool:
    if not hasattr(session, "_create_response"):
        return False
    session._tools = []
    session._next_input = [{"role": "user", "content": prompt}]
    response = session._create_response(tool_choice="auto")
    output = response.output_text or ""
    reasoning = "\n\n".join(
        summary.text
        for item in response.output
        if getattr(item, "type", None) == "reasoning"
        for summary in (getattr(item, "summary", []) or [])
        if getattr(summary, "text", None)
    )
    usage = getattr(response, "usage", None)
    details = getattr(usage, "output_tokens_details", None) if usage else None
    session.trajectory.ground_truth_cot = ""
    session.trajectory.ground_truth_cot_summary = reasoning
    session.trajectory.ground_truth_tokens = getattr(details, "reasoning_tokens", 0) or count_tokens(reasoning)
    save_rep_output(session, output, reasoning)
    session.done = True
    return True


def run_chat_rep(session, prompt: str) -> bool:
    if not (hasattr(session, "messages") and hasattr(session, "_generate")):
        return False
    session._tools = []
    session.messages = [{"role": "user", "content": prompt}]
    msg = session._generate()
    if msg is None:
        raise RuntimeError("Target model returned no chat message.")
    output = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    session.trajectory.ground_truth_cot = reasoning
    session.trajectory.ground_truth_tokens = count_tokens(reasoning)
    save_rep_output(session, output, reasoning)
    session.done = True
    return True


def run_gemini_rep(session, prompt: str) -> bool:
    if not (hasattr(session, "contents") and hasattr(session, "_generate")):
        return False
    session._gemini_tools = []
    content_type = type(session.contents[0])
    part_type = type(session.contents[0].parts[0])
    session.contents = [content_type(role="user", parts=[part_type.from_text(text=prompt)])]

    response = session._generate(require_tool=False)
    print(response)
    print("======"*50)
    answer_blocks, thought_blocks = [], []
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            text = getattr(part, "text", None)
            if text and getattr(part, "thought", False):
                thought_blocks.append(text)
            elif text:
                answer_blocks.append(text)

    output = "\n".join(answer_blocks).strip()
    reasoning = "\n".join(thought_blocks).strip()
    usage = getattr(response, "usage_metadata", None)
    session.trajectory.ground_truth_cot = ""
    session.trajectory.ground_truth_cot_summary = reasoning
    session.trajectory.ground_truth_tokens = (
        getattr(usage, "thoughts_token_count", None) if usage else session.count_tokens(reasoning)
    )
    save_rep_output(session, output, reasoning)
    session.done = True
    return True


def run_claude_rep(session, prompt: str) -> bool:
    if not hasattr(session, "_create_message"):
        return False
    session._tools = []
    session.messages = [{"role": "user", "content": prompt}]
    response = session._create_message()
    output = "".join(b.text for b in response.content if b.type == "text")
    reasoning = "\n\n".join(
        b.thinking for b in response.content if b.type == "thinking" and getattr(b, "thinking", None)
    )
    session.trajectory.ground_truth_cot = ""
    session.trajectory.ground_truth_cot_summary = reasoning
    session.trajectory.ground_truth_tokens = session.thinking_tokens(response)
    save_rep_output(session, output, reasoning)
    session.done = True
    return True


def run_rep_baseline(session, prompt: str) -> None:
    for runner in (run_openai_rep, run_chat_rep, run_gemini_rep, run_claude_rep):
        if runner(session, prompt):
            return
    raise NotImplementedError("REP baseline mode is not implemented for this target session.")


def run_model_baseline(
    model_key: str,
    prompt_template: str,
    demonstrations: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict:
    extractor = load_target(model_key)
    length_only_success = uses_length_only_success(model_key)
    samples = load_samples(args.seed, args.max_samples, os.path.join(DATA_PATH, args.dataset, "test.json"))
    output_dir = Path(args.output_dir) / model_key / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"

    records = load_records(records_path, {sample.sample_id for sample in samples})

    for sample in samples:
        if sample.sample_id in records:
            print(f"[{model_key}] skip existing sample={sample.sample_id}")
            continue

        prompt = build_rep_prompt(prompt_template, demonstrations, sample.question)
        session = make_session(prompt, extractor)
        session.sample_id = sample.sample_id
        run_rep_baseline(session, prompt)
        trajectory = session.trajectory
        turn_record = turn_to_record(
            trajectory.turns[0],
            trajectory.ground_truth_cot,
            trajectory.ground_truth_tokens,
            length_only_success=length_only_success,
        ) if trajectory.turns else {
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
        trajectory.success = turn_record["asr_90"]
        trajectory.success_turn = turn_record["turn_idx"] if trajectory.success else -1

        record = {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "question": sample.question,
            "ground_truth_cot": trajectory.ground_truth_cot,
            "ground_truth_cot_summary": trajectory.ground_truth_cot_summary,
            "ground_truth_tokens": trajectory.ground_truth_tokens,
            "scratchpad_content": turn_record["scratchpad_content"],
            "ground_truth_cot_tokens": trajectory.ground_truth_tokens,
            "final_answer": trajectory.final_output,
            "token_count_ratio": turn_record["token_count_ratio"],
            "length_error": turn_record["length_error"],
            "token_f1": turn_record["token_f1"],
            "rouge_l": turn_record["rouge_l"],
            "token_em": turn_record["token_em"],
            "asr_90": turn_record["asr_90"],
            "asr_95": turn_record["asr_95"],
            "asr_99": turn_record["asr_99"],
            "success": trajectory.success,
        }
        jsonl_append(records_path, record)
        records[sample.sample_id] = record

        print(
            f"[{model_key}] sample={sample.sample_id} turn={turn_record['turn_idx']} "
            f"ground truth tokens={trajectory.ground_truth_tokens} "
            f"ratio={turn_record['token_count_ratio']:.3f} "
            f"token_f1={turn_record['token_f1']:.3f} "
            f"rouge_l={turn_record['rouge_l']:.3f} "
            f"success={trajectory.success}"
        )

    n = len(samples)
    turn_records = [record for record in records.values() if record["scratchpad_content"]]
    success_count = sum(record_is_successful(record, length_only_success) for record in turn_records)
    has_formal_metrics = any(record["ground_truth_cot"] for record in records.values())
    has_success_metrics = has_formal_metrics or length_only_success
    summary = {
        "model_key": model_key,
        "num_samples": n,
        "success_count": success_count if has_success_metrics else None,
        "success_rate": success_count / n if n and has_success_metrics else None,
        "avg_token_count_ratio": average(turn_records, "token_count_ratio"),
        "avg_length_error": average(turn_records, "length_error"),
        "avg_token_f1": average(turn_records, "token_f1") if has_formal_metrics else None,
        "avg_rouge_l": average(turn_records, "rouge_l") if has_formal_metrics else None,
        "avg_token_em": average(turn_records, "token_em") if has_formal_metrics else None,
        "asr_90": success_count / n if n and has_success_metrics else None,
        "asr_95": sum(record["asr_95"] for record in turn_records) / n if n and has_formal_metrics else None,
        "asr_99": sum(record["asr_99"] for record in turn_records) / n if n and has_formal_metrics else None,
        "tool_invocation_rate": 0.0,
        "records_path": str(records_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the REP single-turn baseline on target models.")
    parser.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
        help="Target model key, or 'all'.",
    )
    parser.add_argument("--dataset", default="MATH500", help="Dataset name.")
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT_FILE, help="REP prompt template.")
    parser.add_argument("--demonstrations-file", default=DEFAULT_DEMONSTRATIONS_FILE)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prompt_template = load_prompt(args.prompt_file)
    demonstrations = load_demonstrations(args.demonstrations_file)
    model_keys = resolve_model_keys(args.target_model)
    summaries = [
        run_model_baseline(model_key, prompt_template, demonstrations, args)
        for model_key in model_keys
    ]
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
