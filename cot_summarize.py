#!/usr/bin/env python3
"""
summarize_cot_to_jsonl.py

Purpose:
    For each sample in an input JSONL file, take the ground-truth CoT stored in:
        all_scratchpads[turn_idx == 0].cot

    Then call GPT-4.1-mini to summarize it using summarize_prompt.txt.

Output:
    A JSONL file with one row per sample:
        {
          "sample_id": "...",
          "reasoning_summary": "..."
        }

Example:
    export OPENAI_API_KEY="..."
    python summarize_cot_to_jsonl.py \
        --input data.jsonl \
        --prompt summarize_prompt.txt \
        --output cot_summaries.jsonl
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


DEFAULT_MODEL = "gpt-4.1-mini"


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_summaries(path: str | Path) -> Dict[str, str]:
    """
    Allows resumable runs. If output already contains a sample_id,
    that sample is skipped.
    """
    path = Path(path)
    if not path.exists():
        return {}

    cache: Dict[str, str] = {}
    for row in read_jsonl(path):
        if "sample_id" in row and "reasoning_summary" in row:
            cache[str(row["sample_id"])] = row["reasoning_summary"]
    return cache


def get_ground_truth_cot(sample: Dict[str, Any]) -> str:
    """
    Ground-truth CoT is defined by the task spec as:
        all_scratchpads[turn_idx == 0].cot
    """
    for scratchpad in sample.get("all_scratchpads", []):
        if scratchpad.get("turn_idx") == 0:
            return scratchpad.get("cot", "") or ""
    return ""

def summarize_cot(
    client: OpenAI,
    model: str,
    summarize_prompt: str,
    cot: str,
    max_output_tokens: int,
    retries: int,
    sleep_base: float = 2.0,
) -> str:
    if not cot.strip():
        return ""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": (
                    f"{summarize_prompt.strip()}\n\n"
                    "Full reasoning trace:\n"
                    f"{cot}"
                ),
            },
        ],
        temperature=0,
        max_output_tokens=max_output_tokens,
    )
    
    output_text = response.output_text.strip()
    print(output_text)
    print(len(cot), len(output_text))
    
    return output_text
        
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/evaluation/deepseek-v4-flash/samples.jsonl")
    parser.add_argument("--prompt", default="data/summarize_prompt.txt")
    parser.add_argument("--output", default="results/evaluation/deepseek-v4-flash/cot_summary.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model for summarization.")
    parser.add_argument("--max-output-tokens", type=int, default=2048*16)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete existing output and regenerate all summaries.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    prompt_path = Path(args.prompt)
    output_path = Path(args.output)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    samples = read_jsonl(input_path)

    with open(prompt_path, "r", encoding="utf-8") as f:
        summarize_prompt = f.read()

    existing = load_existing_summaries(output_path)

    client = OpenAI()

    total = len(samples)
    done = 0
    skipped = 0

    for idx, sample in enumerate(samples, start=1):
        sample_id = str(sample["sample_id"])

        if len(sample["all_scratchpads"]) == 0:
                continue
            
        if sample_id in existing:
            skipped += 1
            continue

        gt_cot = get_ground_truth_cot(sample)

        reasoning_summary = summarize_cot(
            client=client,
            model=args.model,
            summarize_prompt=summarize_prompt,
            cot=gt_cot,
            max_output_tokens=args.max_output_tokens,
            retries=args.retries,
        )

        append_jsonl(
            output_path,
            {
                "sample_id": sample_id,
                "reasoning_summary": reasoning_summary,
            },
        )

        done += 1
        print(f"[{idx}/{total}] summarized sample_id={sample_id}")

    print(f"Finished. New summaries: {done}. Skipped existing: {skipped}. Output: {output_path}")


if __name__ == "__main__":
    main()