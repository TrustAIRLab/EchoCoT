"""Split OpenThoughts samples into D_pareto / D_feedback / D_test.

D_pareto = one sample from each source.
The remaining samples are shuffled and split into D_test (first `test_size`)
and D_feedback (the rest).
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import OpenThoughtsDataset  # noqa: E402

from config import DATA_PATH


def _to_sample(ds_sample) -> dict:
    return {
        "id": ds_sample.sample_id,
        "source": ds_sample.source,
        "input": ds_sample.question,
        "target_token_count": None,   # filled by first rollout
        "ground_truth_cot": None,     # filled by first rollout
    }


def build_splits(
    samples_per_source: int,
    seed: int = 42,
    test_size: int = 10,
):
    ds = OpenThoughtsDataset(
        samples_per_source=samples_per_source,
        seed=seed,
        data_path=DATA_PATH,
    )
    raw = [_to_sample(s) for s in ds]
    rng = random.Random(seed)

    by_source: dict[str, list] = {}
    for s in raw:
        by_source.setdefault(s["source"], []).append(s)

    D_pareto: list = []
    pool: list = []
    for source in sorted(by_source.keys()):
        bucket = by_source[source]
        rng.shuffle(bucket)
        D_pareto.append(bucket[0])
        pool.extend(bucket[1:])

    if len(pool) < test_size + 1:
        raise ValueError(
            f"Need at least {test_size + 1} non-pareto samples; got {len(pool)}"
        )

    rng.shuffle(pool)
    D_test = pool[:test_size]
    D_feedback = pool[test_size:]
    return D_pareto, D_feedback, D_test
