"""CLI entry point for the GEPA-style trajectory optimization."""

import argparse
import json
import os
import sys

from openai import OpenAI

from config import (
    BUDGET,
    DEFAULT_OPTIMIZER_MODEL,
    DEFAULT_TARGET_MODEL,
    EPSILON,
    GPT_OPTIMIZER_KEYS,
    L,
    MINIBATCH_SIZE,
    OPTIMIZER_MODEL_OPTIONS,
    OUTPUT_DIR,
    SAMPLES_PER_SOURCE,
    TAU,
)
from data_split import build_splits
from gepa_loop import optimize_gepa_cot
import extract_deepseek
import extract_oss


# Register a target extractor here. Each module must expose `make_client()`
# and `chat(client, model_id, messages, tools)`.
TARGET_EXTRACTORS = {
    "deepseek-reasoner": extract_deepseek,
    "oss": extract_oss,
}


def _make_optimizer_client(opt_key: str):
    if opt_key not in GPT_OPTIMIZER_KEYS:
        raise ValueError(
            f"Optimizer must be a GPT model: {GPT_OPTIMIZER_KEYS}; got {opt_key}"
        )
    entry = OPTIMIZER_MODEL_OPTIONS[opt_key]
    api_key = os.getenv(entry["api_key_env"])
    if not api_key:
        raise ValueError(f"Set {entry['api_key_env']} for {entry['provider']}")
    kwargs = {"api_key": api_key}
    if entry["base_url"]:
        kwargs["base_url"] = entry["base_url"]
    return OpenAI(**kwargs), entry["model"], entry["api_type"]


def main():
    p = argparse.ArgumentParser(description="GEPA-style CoT extraction optimizer")
    p.add_argument("--target-model", default=DEFAULT_TARGET_MODEL,
                   choices=list(TARGET_EXTRACTORS.keys()))
    p.add_argument("--optimizer-model", default=DEFAULT_OPTIMIZER_MODEL,
                   choices=GPT_OPTIMIZER_KEYS)
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--minibatch-size", type=int, default=MINIBATCH_SIZE)
    p.add_argument("--test-size", type=int, default=10)
    p.add_argument("--samples-per-source", type=int, default=SAMPLES_PER_SOURCE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--epsilon", type=float, default=EPSILON)
    p.add_argument("--L", type=int, default=L,
                   help="Number of injection turns (must match T0)")
    args = p.parse_args()

    extractor = TARGET_EXTRACTORS[args.target_model]
    target_client, target_model_id = extractor.make_client()
    opt_client, opt_model, opt_api_type = _make_optimizer_client(args.optimizer_model)

    D_pareto, D_feedback, D_test = build_splits(
        samples_per_source=args.samples_per_source,
        seed=args.seed,
        test_size=args.test_size,
    )
    print(f"[gepa] target = {args.target_model} ({target_model_id})")
    print(f"[gepa] optimizer = {args.optimizer_model} ({opt_model}/{opt_api_type})")
    print(f"[gepa] |D_pareto|={len(D_pareto)} |D_feedback|={len(D_feedback)} "
          f"|D_test|={len(D_test)}")
    print(f"[gepa] budget={args.budget} minibatch={args.minibatch_size} "
          f"L={args.L} tau={args.tau} epsilon={args.epsilon}")

    output_dir = os.path.join(OUTPUT_DIR, args.target_model)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"D_pareto": D_pareto, "D_feedback": D_feedback, "D_test": D_test},
            f, ensure_ascii=False, indent=2, default=str,
        )
    optimize_gepa_cot(
        extractor=extractor,
        target_client=target_client,
        target_model_id=target_model_id,
        opt_client=opt_client,
        opt_model=opt_model,
        opt_api_type=opt_api_type,
        D_feedback=D_feedback,
        D_pareto=D_pareto,
        D_test=D_test,
        output_dir=output_dir,
        budget=args.budget,
        minibatch_size=args.minibatch_size,
        tau=args.tau,
        epsilon=args.epsilon,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
