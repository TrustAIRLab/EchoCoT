"""GEPA-style Pareto parent selection (readme §9)."""

import random


def weighted_sample(probs: dict):
    keys = list(probs.keys())
    weights = [probs[k] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def select_parent_by_pareto(S: dict, active_ids, tau: float = 0.05):
    """Pick a parent from `active_ids` using instance-wise Pareto winners.

    S[candidate_id][sample_id] = token_count_ratio.
    Returns (parent_id, info_dict).
    """
    active_ids = list(active_ids)
    sample_ids = list(next(iter(S.values())).keys())

    winners_by_sample = {}
    for sample_id in sample_ids:
        distances = {k: abs(S[k][sample_id] - 1.0) for k in active_ids}
        best_distance = min(distances.values())
        winners = {k for k, d in distances.items() if d <= best_distance + tau}
        winners_by_sample[sample_id] = winners

    C = set()
    for winners in winners_by_sample.values():
        C.update(winners)

    winset = {
        k: {sid for sid, winners in winners_by_sample.items() if k in winners}
        for k in C
    }

    dominated = set()
    for k in C:
        for l in C:
            if k == l:
                continue
            if winset[k] < winset[l]:
                dominated.add(k)
                break

    frontier = C - dominated
    f = {k: len(winset[k]) for k in frontier}
    total = sum(f.values())
    probs = {k: f[k] / total for k in frontier} if total > 0 else {
        k: 1.0 / len(frontier) for k in frontier
    }

    parent = weighted_sample(probs)
    return parent, {
        "winners_by_sample": {sid: sorted(w) for sid, w in winners_by_sample.items()},
        "winset": {k: sorted(v) for k, v in winset.items()},
        "dominated": sorted(dominated),
        "frontier": sorted(frontier),
        "f": f,
        "probs": probs,
    }
