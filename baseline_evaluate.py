"""Evaluate the direct-prompt baseline in a separate two-turn run."""

from __future__ import annotations

import argparse
import json, os
import random
from pathlib import Path

from dataset import ReasoningDataset
from evaluate import (
    average,
    best_turn,
    jsonl_append,
    load_target,
    make_session,
    resolve_model_keys,
    turn_to_record,
)
from metrics import (
    InjectionTurn,
    compute_eval_metrics,
    compute_length_metrics,
    count_tokens,
    token_count_ratio_passes_threshold,
)
from config import TEST_DATA_PATH

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
    token_count = count_tokens(trace)
    length_m = compute_length_metrics(token_count, session.trajectory.ground_truth_tokens)
    eval_m = compute_eval_metrics(trace, session.trajectory.ground_truth_cot)
    session.trajectory.turns.append(InjectionTurn(
        turn_idx=1,
        cot=cot,
        scratchpad_in=trace,
        scratchpad_in_tokens=token_count,
        token_count_ratio=length_m["token_count_ratio"],
        length_error=length_m["length_error"],
        # token_precision=eval_m["token_precision"],
        # token_recall=eval_m["token_recall"],
        token_f1=eval_m["token_f1"],
        rouge_l=eval_m["rouge_l"],
        stopped=True,
    ))
    session.current_ratio = length_m["token_count_ratio"]
    if token_count_ratio_passes_threshold(session.current_ratio):
        session.trajectory.success_turn = 1
        session.trajectory.success = True


def run_openai_baseline(session, prompt: str) -> bool:
    if not hasattr(session, "_create_response"):
        return False
    session._tools = []
    session._next_input = [{"role": "user", "content": session.question}]
    response = session._create_response(tool_choice="auto")
    session.trajectory.final_output = response.output_text or ""
    session.trajectory.ground_truth_cot = "\n\n".join(
        summary.text
        for item in response.output
        if getattr(item, "type", None) == "reasoning"
        for summary in (getattr(item, "summary", []) or [])
        if getattr(summary, "text", None)
    )
    usage = getattr(response, "usage", None)
    details = getattr(usage, "output_tokens_details", None) if usage else None
    session.trajectory.ground_truth_tokens = getattr(details, "reasoning_tokens", 0) or count_tokens(session.trajectory.ground_truth_cot)

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
    # if hasattr(session, "_append_assistant"):
    session._append_assistant(msg, reasoning)
    # else:
    #     session.messages.append({"role": "assistant", "content": msg.content or ""})

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
    session.trajectory.ground_truth_cot = "\n".join(thought_blocks).strip()
    usage = getattr(response, "usage_metadata", None)
    session.trajectory.ground_truth_tokens = getattr(usage, "thoughts_token_count", None) if usage else count_tokens(session.trajectory.ground_truth_cot)

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
    session.trajectory.ground_truth_cot = "\n\n".join(
        b.thinking for b in response.content if b.type == "thinking" and getattr(b, "thinking", None)
    )
    session.trajectory.ground_truth_tokens = count_tokens(session.trajectory.ground_truth_cot)

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
    runner = load_target(model_key)
    samples = load_samples(args.seed, args.max_samples, os.path.join(TEST_DATA_PATH, args.dataset, "test.json"))
    output_dir = Path(args.output_dir) / model_key / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "samples.jsonl"
    summary_path = output_dir / "summary.json"

    done_sample_ids = set()
    if records_path.exists():
        with records_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_sample_ids.add(json.loads(line)["sample_id"])
                    
    best_records = []
    success_count = 0

    for sample in samples:
        session = make_session(sample.question, runner)
        session.sample_id = sample.sample_id
        
        if sample.sample_id in done_sample_ids:
            print(f"[{model_key}] skip existing sample={sample.sample_id}")
            continue

        run_baseline(session, prompt)

        trajectory = session.trajectory
        all_scratchpads = [turn_to_record(turn) for turn in trajectory.turns]
        selected = best_turn(trajectory.turns)
        best_record = turn_to_record(selected) if selected else {
            "turn_idx": -1,
            "scratchpad_content": "",
            "scratchpad_tokens": 0,
            "token_count_ratio": 0.0,
            "length_error": 0.0,
            # "token_precision": 0.0,
            # "token_recall": 0.0,
            "token_f1": 0.0,
            "rouge_l": 0.0
        }
        best_records.append(best_record)
        success_count += int(trajectory.success)

        jsonl_append(records_path, {
            "sample_id": sample.sample_id,
            "source": sample.source,
            "question": sample.question,
            "ground_truth_cot": trajectory.ground_truth_cot,
            "final_answer": trajectory.final_output,
            "success": trajectory.success,
            "turns": all_scratchpads,
        })

        # print(all_scratchpads[0]["scratchpad_content"])
        print(
            f"[{model_key}] sample={sample.sample_id} best_turn={best_record['turn_idx']} "
            f"ground truth tokens={trajectory.ground_truth_tokens} "
            f"ratio={best_record['token_count_ratio']:.3f} "
            f"token_f1={best_record['token_f1']:.3f} "
            f"rouge_l={best_record['rouge_l']:.3f} "
            f"success={trajectory.success}"
        )

    n = len(samples)
    summary = {
        "model_key": model_key,
        "num_samples": n,
        "success_count": success_count,
        "success_rate": success_count / n if n else 0.0,
        "avg_token_count_ratio": average(best_records, "token_count_ratio"),
        "avg_length_error": average(best_records, "length_error"),
        # "avg_token_precision": average(best_records, "token_precision"),
        # "avg_token_recall": average(best_records, "token_recall"),
        "avg_token_f1": average(best_records, "token_f1"),
        "avg_rouge_l": average(best_records, "rouge_l"),
        "records_path": str(records_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the direct two-turn baseline on target models.")
    parser.add_argument("--target-model", default="xx", help="Target model key, or 'all'.")
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
                f"token_f1={summary['avg_token_f1']:.3f} "
                f"rouge_l={summary['avg_rouge_l']:.3f} "
                f"success={summary['success_count']}/{summary['num_samples']}"
            )


if __name__ == "__main__":
    main()
