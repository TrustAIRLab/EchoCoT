import json
import os
import random
from pathlib import Path
import sys
from pathlib import Path
from openai import OpenAI


DATA_PATH = Path("data/OpenThoughts/train.json")
OUTPUT_PATH = Path("data/demonstrations.json")

MODEL = "qwen3-14b"
SEED = 7
NUM_DEMOS = 50
WorkspaceId = "ws-tizfsmm37pwxnz4p"

client = OpenAI(
    api_key=os.environ["SP_DASHSCOPE_API_KEY"],
    base_url=os.getenv(
        "DASHSCOPE_BASE_URL",
        f"https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    ),
)

def generate_demo(question: str) -> tuple[str, str]:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        stream=True,
        extra_body={"enable_thinking": True},
    )

    reasoning_parts = []
    answer_parts = []

    for chunk in stream:
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_parts.append(reasoning)

        if delta.content:
            answer_parts.append(delta.content)

    reasoning = "".join(reasoning_parts).strip()
    answer = "".join(answer_parts).strip()

    reasoning = (
        reasoning.replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )

    if not reasoning or not answer:
        raise RuntimeError("Qwen returned empty reasoning or answer.")

    return reasoning, answer


def save_demonstrations(demonstrations: list[dict]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(demonstrations, f, ensure_ascii=False, indent=2)


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as f:
        pool = json.load(f)

    if len(pool) < 50:
        raise ValueError("OpenThoughts must contain at least 50 samples.")

    # Map sample_id to the original input record.
    pool_by_id = {
        str(sample["sample_id"]): sample
        for sample in pool
    }

    selected = random.Random(SEED).sample(pool, 50)[:NUM_DEMOS]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(encoding="utf-8") as f:
            demonstrations = json.load(f)
    else:
        demonstrations = []

    # Add or update source for all existing demonstrations.
    for demo in demonstrations:
        sample_id = str(demo["sample_id"])
        source_sample = pool_by_id.get(sample_id)

        if source_sample is None:
            print(f"Warning: sample {sample_id} not found in {DATA_PATH}")
            continue

        demo["source"] = source_sample.get("source", "")
        demo["reasoning_length"] = len(demo["reasoning"])

    save_demonstrations(demonstrations)

    existing_ids = {
        str(demo["sample_id"])
        for demo in demonstrations
    }

    for i, sample in enumerate(selected, start=1):
        sample_id = str(sample["sample_id"])

        if sample_id in existing_ids:
            print(f"[{i}/{NUM_DEMOS}] Skip existing sample {sample_id}")
            continue

        print(f"[{i}/{NUM_DEMOS}] Generating sample {sample_id}...")

        reasoning, answer = generate_demo(sample["question"])

        demonstrations.append(
            {
                "sample_id": sample_id,
                "question": sample["question"],
                "reasoning": reasoning,
                "answer": answer,
                "source": sample.get("source", ""),
                "shadow_model": MODEL,
                "length": len(reasoning)
            }
        )

        existing_ids.add(sample_id)

        # Save after every successful API call.
        save_demonstrations(demonstrations)

    print(
        f"Saved {len(demonstrations)} total demonstrations "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()