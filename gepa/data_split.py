"""Split OpenThoughts samples into D_pareto / D_feedback / D_test."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import OpenThoughtsDataset  # noqa: E402

from config import DATA_PATH, PARETO_SET_SIZE


def _to_sample(ds_sample) -> dict:
    return {
        "id": ds_sample.sample_id,
        "input": ds_sample.question,
        "target_token_count": None,   # filled by first rollout
        "ground_truth_cot": None,     # filled by first rollout
    }


def build_splits(
    samples_per_source: int,
    seed: int = 42,
    test_size: int = 10,
    pareto_size: int = PARETO_SET_SIZE,
):
    ds = OpenThoughtsDataset(
        samples_per_source=samples_per_source,
        seed=seed,
        data_path=DATA_PATH,
    )
    raw = [_to_sample(s) for s in ds]
    rng = random.Random(seed)
    rng.shuffle(raw)

    if len(raw) < pareto_size + test_size + 1:
        raise ValueError(
            f"Need at least {pareto_size + test_size + 1} samples; got {len(raw)}"
        )

    D_pareto = raw[:pareto_size]
    D_test = raw[pareto_size:pareto_size + test_size]
    D_feedback = raw[pareto_size + test_size:]
    return D_pareto, D_feedback, D_test
