"""Run a user-supplied injection trajectory against one target model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import DEFAULT_TARGET_MODEL, TARGET_MODEL_OPTIONS
from injection_io import load_injections
from extractors import (
    ARCHIVE_SUFFIX,
    FINAL_ANSWER_INJECTION,
    load_target,
    make_session,
)


def _turn_record(turn, ground_truth_cot_tokens: int) -> dict[str, Any]:
    return {
        "turn_idx": turn.turn_idx,
        "target_model_reaction": turn.cot,
        "scratchpad": turn.scratchpad_in,
        "injection_sent": turn.injection_out,
        "scratchpad_tokens": turn.scratchpad_in_tokens,
        "ground_truth_cot_tokens": ground_truth_cot_tokens,
    }


def run_inference(
    target_model: str,
    question: str,
    injections: list[dict[str, Any]],
    *,
    request_final_answer: bool = True,
) -> dict[str, Any]:
    """Apply injections in order using the same sessions as optimize/evaluate."""
    extractor = load_target(target_model)
    session = make_session(question + ARCHIVE_SUFFIX, extractor)
    session.mode = "length_only"
    session.skip_cot_summary = True
    session.start()

    applied_injections = []
    for item in injections:
        if session.done:
            break
        injection = item["injection"]
        if request_final_answer and injection == FINAL_ANSWER_INJECTION:
            session.final_send(injection)
        else:
            session.send(injection)
        applied_injections.append({
            "turn_idx": item["turn_idx"],
            "injection": injection,
        })
        if session.done:
            break

    if request_final_answer and not session.done:
        session.final_send(FINAL_ANSWER_INJECTION)

    return {
        "target_model": target_model,
        "question": question,
        "injections": applied_injections,
        "turns": [_turn_record(turn, session.trajectory.ground_truth_tokens) for turn in session.trajectory.turns],
        "final_answer": session.trajectory.final_output,
        "completed": session.done,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a custom injection trajectory against one target model."
    )
    parser.add_argument(
        "--target-model",
        default=DEFAULT_TARGET_MODEL,
        choices=list(TARGET_MODEL_OPTIONS),
    )

    question_source = parser.add_mutually_exclusive_group(required=True)
    question_source.add_argument("--question", help="Question sent to the target model.")
    question_source.add_argument(
        "--question-file",
        type=Path,
        help="UTF-8 text file containing the question.",
    )

    injection_source = parser.add_mutually_exclusive_group(required=True)
    injection_source.add_argument(
        "--injection",
        action="append",
        help="One injection turn; repeat this option for a multi-turn trajectory.",
    )
    injection_source.add_argument(
        "--trajectory-file",
        type=Path,
        help="JSON, experience.md, Turn-N Markdown, or one plain-text injection.",
    )

    parser.add_argument(
        "--no-final-answer",
        action="store_true",
        help="Stop after the last injection instead of requesting the final answer.",
    )
    parser.add_argument("--output", type=Path, help="Optionally save the result as JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    question = (
        args.question
        if args.question is not None
        else args.question_file.read_text(encoding="utf-8").strip()
    )
    if args.injection:
        injections = [
            {"turn_idx": index, "injection": text}
            for index, text in enumerate(args.injection, start=1)
        ]
    else:
        injections = load_injections(args.trajectory_file, allow_plain_text=True)

    if not question:
        raise ValueError("Question cannot be empty")
    if not injections:
        raise ValueError("Injection trajectory cannot be empty")

    result = run_inference(
        args.target_model,
        question,
        injections,
        request_final_answer=not args.no_final_answer,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
