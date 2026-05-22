"""GEPA optimization loop (readme §13)."""

import copy
import json
import os
import random
import statistics

from initial_trajectory import T0
from mutator import mutate_turn
from pareto import select_parent_by_pareto
from rollout import run_trajectory


def _jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _mean(xs):
    return statistics.mean(xs) if xs else 0.0


def _run_batch(client, model_id, candidate, samples):
    return [run_trajectory(client, model_id, candidate, s) for s in samples]


def evaluate_on_pareto(client, model_id, candidate, D_pareto):
    results = _run_batch(client, model_id, candidate, D_pareto)
    scores = {r["sample_id"]: r["token_count_ratio"] for r in results}
    return scores, results


def _copy_candidate(c: dict) -> dict:
    return {"id": c["id"], "turns": dict(c["turns"])}


def _batch_avg(rollouts: list) -> dict:
    return {
        "avg_token_count_ratio": _mean([r["token_count_ratio"] for r in rollouts]),
        "avg_distance_to_1": _mean([r["distance_to_1"] for r in rollouts]),
        "avg_token_overlap_pct": _mean([r["token_overlap_pct"] for r in rollouts]),
        "avg_jaccard": _mean([r["jaccard"] for r in rollouts]),
    }


def optimize_gepa_cot(
    target_client,
    target_model_id: str,
    opt_client,
    opt_model: str,
    opt_api_type: str,
    D_feedback,
    D_pareto,
    D_test,
    output_dir: str,
    budget: int,
    minibatch_size: int,
    tau: float,
    epsilon: float,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    batch_log = os.path.join(output_dir, "batch_log.jsonl")
    history_log = os.path.join(output_dir, "history.jsonl")
    open(batch_log, "w").close()
    open(history_log, "w").close()

    rng = random.Random(seed)

    P = {0: copy.deepcopy(T0)}
    active_ids = {0}
    next_id = 1

    print(f"[gepa] evaluating T0 on D_pareto ({len(D_pareto)} samples)...")
    S0, _ = evaluate_on_pareto(target_client, target_model_id, P[0], D_pareto)
    S = {0: S0}
    print(f"[gepa] T0 pareto ratios: "
          f"{ {k: round(v, 3) for k, v in S0.items()} }")
    print(f"[gepa] T0 avg distance_to_1 on D_pareto = "
          f"{_mean([abs(r - 1.0) for r in S0.values()]):.4f}")

    _jsonl(history_log, {
        "id": 0, "parent": None, "changed_turn": None,
        "mutation_reason": "initial trajectory",
        "pareto_scores": S0,
    })

    L = len(P[0]["turns"])

    for it in range(1, budget + 1):
        print(f"\n[gepa] === iter {it}/{budget} ===")
        parent_id, sel_info = select_parent_by_pareto(S, active_ids, tau=tau)
        print(f"[gepa]   parent_id={parent_id}  frontier={sel_info['frontier']}")
        T_parent = P[parent_id]
        target_turn = ((it - 1) % L) + 1

        minibatch = rng.sample(D_feedback, k=min(minibatch_size, len(D_feedback)))
        sample_ids = [s["id"] for s in minibatch]
        print(f"[gepa]   target_turn={target_turn}  batch={sample_ids}")

        parent_rollouts = _run_batch(
            target_client, target_model_id, T_parent, minibatch
        )
        parent_stats = _batch_avg(parent_rollouts)
        print(f"[gepa]   parent batch avg: "
              f"ratio={parent_stats['avg_token_count_ratio']:.3f} "
              f"dist={parent_stats['avg_distance_to_1']:.3f} "
              f"overlap={parent_stats['avg_token_overlap_pct']:.3f} "
              f"jaccard={parent_stats['avg_jaccard']:.3f}")

        mutation = mutate_turn(
            opt_client=opt_client,
            opt_model=opt_model,
            opt_api_type=opt_api_type,
            parent_trajectory=T_parent,
            target_turn=target_turn,
            feedback_batch=parent_rollouts,
        )
        print(f"[gepa]   mutation_reason: {mutation['mutation_reason'][:160]}")

        T_child = _copy_candidate(T_parent)
        T_child["id"] = next_id
        T_child["turns"][target_turn] = mutation["mutated_turn"]

        child_rollouts = _run_batch(
            target_client, target_model_id, T_child, minibatch
        )
        child_stats = _batch_avg(child_rollouts)
        print(f"[gepa]   child  batch avg: "
              f"ratio={child_stats['avg_token_count_ratio']:.3f} "
              f"dist={child_stats['avg_distance_to_1']:.3f} "
              f"overlap={child_stats['avg_token_overlap_pct']:.3f} "
              f"jaccard={child_stats['avg_jaccard']:.3f}")

        accepted = child_stats["avg_distance_to_1"] <= (
            parent_stats["avg_distance_to_1"] - epsilon
        )

        _jsonl(batch_log, {
            "iter": it,
            "parent_id": parent_id,
            "candidate_child_id": next_id,
            "target_turn": target_turn,
            "batch_sample_ids": sample_ids,
            "parent": parent_stats,
            "child": child_stats,
            "accepted": accepted,
            "mutation_reason": mutation["mutation_reason"],
        })

        if accepted:
            P[next_id] = T_child
            active_ids.add(next_id)
            scores, _ = evaluate_on_pareto(
                target_client, target_model_id, T_child, D_pareto
            )
            S[next_id] = scores
            _jsonl(history_log, {
                "id": next_id,
                "parent": parent_id,
                "changed_turn": target_turn,
                "mutation_reason": mutation["mutation_reason"],
                "mutated_turn": mutation["mutated_turn"],
                "parent_avg_distance": parent_stats["avg_distance_to_1"],
                "child_avg_distance": child_stats["avg_distance_to_1"],
                "pareto_scores": scores,
                "selection_info": sel_info,
            })
            print(f"[gepa]   ACCEPTED -> id={next_id}  "
                  f"pareto avg distance = "
                  f"{_mean([abs(r - 1.0) for r in scores.values()]):.4f}")
            next_id += 1
        else:
            print(f"[gepa]   REJECTED  (delta = "
                  f"{parent_stats['avg_distance_to_1'] - child_stats['avg_distance_to_1']:+.4f})")

    # --- Final selection ---
    best_id = min(
        active_ids,
        key=lambda k: _mean([abs(r - 1.0) for r in S[k].values()]),
    )
    T_best = P[best_id]
    print(f"\n[gepa] best_id = {best_id}  "
          f"pareto avg distance = "
          f"{_mean([abs(r - 1.0) for r in S[best_id].values()]):.4f}")

    # Persist pool + score table
    with open(os.path.join(output_dir, "pool.json"), "w", encoding="utf-8") as f:
        json.dump(
            {str(k): {"id": v["id"], "turns": {str(i): t for i, t in v["turns"].items()}}
             for k, v in P.items()},
            f, ensure_ascii=False, indent=2,
        )
    with open(os.path.join(output_dir, "score_table.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in S.items()}, f, ensure_ascii=False, indent=2)

    # --- D_test evaluation ---
    print(f"\n[gepa] evaluating best on D_test ({len(D_test)} samples)...")
    test_rollouts = _run_batch(target_client, target_model_id, T_best, D_test)
    test_stats = _batch_avg(test_rollouts)
    per_sample = [
        {
            "sample_id": r["sample_id"],
            "token_count_ratio": r["token_count_ratio"],
            "distance_to_1": r["distance_to_1"],
            "token_overlap_pct": r["token_overlap_pct"],
            "jaccard": r["jaccard"],
            "final_output_tokens": r["final_output_tokens"],
            "target_token_count": r["target_token_count"],
        }
        for r in test_rollouts
    ]
    test_report = {
        "best_id": best_id,
        "best_trajectory": {
            "id": T_best["id"],
            "turns": {str(i): t for i, t in T_best["turns"].items()},
        },
        "averages": test_stats,
        "per_sample": per_sample,
    }
    with open(os.path.join(output_dir, "test_result.json"), "w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)

    print(f"[gepa] D_test avg: "
          f"ratio={test_stats['avg_token_count_ratio']:.3f} "
          f"dist={test_stats['avg_distance_to_1']:.3f} "
          f"overlap={test_stats['avg_token_overlap_pct']:.3f} "
          f"jaccard={test_stats['avg_jaccard']:.3f}")

    return {
        "best_id": best_id,
        "best_trajectory": T_best,
        "pool": P,
        "score_table": S,
        "test_report": test_report,
    }
