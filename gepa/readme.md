# GEPA-style Multi-turn Prompt Trajectory Optimization for CoT Extraction

## 1. Method Overview

This method directly adapts GEPA to a black-box CoT extraction task.

The optimization target is a **complete multi-turn prompt trajectory**. Each candidate trajectory is evaluated by running all turns against the target model and measuring the token length of the final output.

The only evaluation metric in this version is:

```text
token_count_ratio = final_output_token_count / target_token_count
```

The goal is to make `token_count_ratio` close to `1.0`.

No style score, semantic score, overlap score, accuracy score, or LLM judge score is used in this version.

---

## 2. Core Objects

### 2.1 Candidate Trajectory

A candidate is a complete multi-turn prompt trajectory:

```python
T_k = {
    "id": k,
    "turns": {
        1: "prompt for turn 1",
        2: "prompt for turn 2",
        3: "prompt for turn 3",
        ...
    }
}
```

Important rules:

```text
Candidate = complete trajectory.
Pareto frontier = set of complete candidate trajectories.
Individual turns are editable components, not standalone candidates.
```

---

### 2.2 Candidate Pool

The candidate pool stores all accepted trajectories:

```python
P = {
    0: T0,
    1: T1,
    2: T2,
    ...
}
```

Initialize it with a manually written initial trajectory:

```python
P[0] = T0
```

---

### 2.3 Score Table

The score table stores the raw `token_count_ratio` of each candidate on each Pareto sample:

```python
S[candidate_id][sample_id] = token_count_ratio
```

Example:

```text
        x1    x2    x3    ...   x10
T0      0.7   1.2   0.9         1.4
T1      1.1   0.8   1.0         0.9
T2      1.4   1.0   0.6         1.1
```

Only the final output of the complete trajectory is used for the score table. Intermediate turn outputs may be logged for diagnosis, but they are not used in Pareto selection.

---

## 3. Dataset Split

Use three datasets:

```python
D_feedback
D_pareto
D_test
```

For quick testing:

```python
D_pareto = 10 fixed samples
```

Suggested setup:

```python
D_pareto = fixed random sample of 10 training examples
D_feedback = remaining training examples
D_test = held-out test examples
```

Rules:

```text
D_feedback: used for minibatch mutation.
D_pareto: used for score table, Pareto frontier, and parent selection.
D_test: used only once at the end.
```

`D_pareto` must remain fixed throughout optimization.

---

## 4. Data Format

Each sample should contain:

```python
sample = {
    "id": "sample_001",
    "input": "...",
    "target_token_count": 1024
}
```

Each trajectory should contain:

```python
T0 = {
    "id": 0,
    "turns": {
        1: "...",
        2: "...",
        3: "..."
    }
}
```

---

## 5. Running a Trajectory

A trajectory is executed turn by turn. The final output is the output from the last turn.

```python
def run_trajectory(target_model, candidate, sample):
    conversation = []
    current_context = sample["input"]

    for turn_id in sorted(candidate["turns"].keys()):
        prompt_template = candidate["turns"][turn_id]
        prompt = format_prompt(
            template=prompt_template,
            sample=sample,
            conversation=conversation
        )

        output = target_model.generate(prompt)

        conversation.append({
            "turn_id": turn_id,
            "prompt": prompt,
            "output": output
        })

    final_output = conversation[-1]["output"]

    return {
        "final_output": final_output,
        "trace": conversation
    }
```

The `trace` is used for mutation feedback. The score table uses only `final_output`.

---

## 6. Metric

### 6.1 Token Count Ratio

```python
def compute_token_count_ratio(final_output, sample, tokenizer):
    final_output_tokens = len(tokenizer.encode(final_output))
    target_tokens = sample["target_token_count"]

    ratio = final_output_tokens / max(target_tokens, 1)

    return {
        "token_count_ratio": ratio,
        "final_output_tokens": final_output_tokens,
        "target_tokens": target_tokens,
        "distance_to_1": abs(ratio - 1.0)
    }
```

The method stores the raw ratio:

```python
S[k][i] = token_count_ratio
```

For comparison, use distance to 1:

```python
distance = abs(token_count_ratio - 1.0)
```

Smaller distance is better.

---

## 7. Evaluating a Candidate on D_pareto

Every accepted candidate must be evaluated on all 10 Pareto samples.

```python
def evaluate_on_pareto(candidate, D_pareto, target_model, tokenizer):
    scores = {}

    for sample in D_pareto:
        run_result = run_trajectory(
            target_model=target_model,
            candidate=candidate,
            sample=sample
        )

        metric = compute_token_count_ratio(
            final_output=run_result["final_output"],
            sample=sample,
            tokenizer=tokenizer
        )

        scores[sample["id"]] = metric["token_count_ratio"]

    return scores
```

This function returns one row of the score table.

---

## 8. Feedback Batch

For mutation, run the parent candidate on a minibatch from `D_feedback`.

```python
def run_feedback_batch(candidate, samples, target_model, tokenizer):
    feedback = []

    for sample in samples:
        run_result = run_trajectory(
            target_model=target_model,
            candidate=candidate,
            sample=sample
        )

        metric = compute_token_count_ratio(
            final_output=run_result["final_output"],
            sample=sample,
            tokenizer=tokenizer
        )

        feedback.append({
            "sample_id": sample["id"],
            "input": sample["input"],
            "target_tokens": metric["target_tokens"],
            "final_output_tokens": metric["final_output_tokens"],
            "token_count_ratio": metric["token_count_ratio"],
            "distance_to_1": metric["distance_to_1"],
            "trace": run_result["trace"]
        })

    return feedback
```

This feedback is given to the optimizer LLM to mutate one selected turn.

---

## 9. Pareto Parent Selection

Parent selection follows GEPA-style instance-wise Pareto selection.

Since the score table stores raw ratios, winners are based on closeness to `1.0`.

### 9.1 Find Winners

For each Pareto sample `x_i`, compute:

```python
distance[k][i] = abs(S[k][i] - 1.0)
best_distance[i] = min_k distance[k][i]
```

Then define winners:

```python
W_i = {
    k for k in active_candidates
    if distance[k][i] <= best_distance[i] + tau
}
```

Use:

```python
tau = 0.05
```

This means candidates within `0.05` distance of the best candidate are also treated as winners.

---

### 9.2 Compute WinSet

For each candidate:

```python
WinSet[k] = {i | k is in W_i}
```

Example:

```text
WinSet[T0] = {x1, x3, x7}
WinSet[T1] = {x2, x5}
WinSet[T2] = {x4, x6, x8, x9}
```

---

### 9.3 Remove Dominated Candidates

Candidate `a` is dominated by candidate `b` if:

```python
WinSet[a] < WinSet[b]
```

This means every sample won by `a` is also won by `b`, and `b` wins at least one additional sample.

Remove dominated candidates from the current parent-selection frontier.

---

### 9.4 Compute f

For each remaining candidate:

```python
f[k] = len(WinSet[k])
```

`f[k]` means how many Pareto samples this candidate wins.

---

### 9.5 Sample Parent

Sample a parent candidate from the frontier with probability:

```python
p(k) = f[k] / sum(f[j] for j in frontier)
```

This favors candidates that win more Pareto samples while preserving candidates that specialize on a small subset of samples.

---

### 9.6 Minimal Implementation

```python
def select_parent_by_pareto(S, active_ids, tau=0.05):
    sample_ids = list(next(iter(S.values())).keys())

    winners_by_sample = {}

    for sample_id in sample_ids:
        distances = {
            k: abs(S[k][sample_id] - 1.0)
            for k in active_ids
        }

        best_distance = min(distances.values())

        winners = {
            k for k, d in distances.items()
            if d <= best_distance + tau
        }

        winners_by_sample[sample_id] = winners

    C = set()
    for winners in winners_by_sample.values():
        C.update(winners)

    winset = {
        k: {
            sample_id
            for sample_id, winners in winners_by_sample.items()
            if k in winners
        }
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

    probs = {
        k: f[k] / total
        for k in frontier
    }

    selected_parent = weighted_sample(probs)

    return selected_parent, {
        "winners_by_sample": winners_by_sample,
        "winset": winset,
        "dominated": dominated,
        "frontier": frontier,
        "f": f,
        "probs": probs
    }
```

---

## 10. Turn Selection

After selecting a parent trajectory, choose one turn to mutate.

Use simple round-robin first:

```python
target_turn = ((iteration - 1) % L) + 1
```

Example:

```text
iteration 1 -> mutate turn 1
iteration 2 -> mutate turn 2
iteration 3 -> mutate turn 3
iteration 4 -> mutate turn 1
```

This is enough for the first implementation.

---

## 11. Mutation

Mutation rewrites one selected turn and keeps all other turns unchanged.

Given:

```python
T_parent = {
    "id": 3,
    "turns": {
        1: p1,
        2: p2,
        3: p3
    }
}
```

If `target_turn = 2`, then:

```python
T_child = {
    "id": new_id,
    "turns": {
        1: p1,
        2: p2_prime,
        3: p3
    }
}
```

---

### 11.1 Optimizer LLM Input

The optimizer LLM receives:

```python
{
    "parent_trajectory": T_parent,
    "target_turn": target_turn,
    "feedback_batch": [
        {
            "sample_id": "...",
            "target_tokens": ...,
            "final_output_tokens": ...,
            "token_count_ratio": ...,
            "distance_to_1": ...,
            "trace": [...]
        }
    ],
    "instruction": (
        "Rewrite only the selected turn. "
        "Keep all other turns unchanged. "
        "The goal is to make the final output token_count_ratio closer to 1."
    )
}
```

The optimizer LLM must output:

```python
{
    "mutated_turn": "...",
    "mutation_reason": "..."
}
```

---

### 11.2 Mutation Rules

The optimizer must follow these constraints:

```text
1. Rewrite only the selected turn.
2. Preserve all other turns.
3. Do not change the number of turns.
4. Optimize for final output token_count_ratio close to 1.
5. Use the minibatch trace to infer whether the current trajectory is too short or too long.
```

---

## 12. Child Acceptance

Run both parent and child on the same feedback minibatch.

Compute:

```python
parent_avg_distance = mean(abs(parent_ratio - 1.0))
child_avg_distance = mean(abs(child_ratio - 1.0))
```

Accept the child if:

```python
child_avg_distance <= parent_avg_distance - epsilon
```

Use:

```python
epsilon = 0.01
```

If accepted:

```python
P[new_id] = T_child
S[new_id] = evaluate_on_pareto(T_child)
```

If rejected, discard the child.

---

## 13. Full Optimization Algorithm

```python
def optimize_gepa_cot(
    T0,
    D_feedback,
    D_pareto,
    D_test,
    target_model,
    optimizer_llm,
    tokenizer,
    budget=50,
    minibatch_size=4,
    tau=0.05,
    epsilon=0.01,
):
    P = {}
    S = {}
    history = {}
    active_ids = set()

    P[0] = T0
    active_ids.add(0)
    history[0] = {
        "parent": None,
        "changed_turn": None,
        "mutation_reason": "initial trajectory"
    }

    S[0] = evaluate_on_pareto(
        candidate=T0,
        D_pareto=D_pareto,
        target_model=target_model,
        tokenizer=tokenizer
    )

    next_id = 1

    for iteration in range(1, budget + 1):

        parent_id, selection_info = select_parent_by_pareto(
            S=S,
            active_ids=active_ids,
            tau=tau
        )

        T_parent = P[parent_id]
        L = len(T_parent["turns"])

        target_turn = ((iteration - 1) % L) + 1

        minibatch = sample_minibatch(D_feedback, minibatch_size)

        parent_feedback = run_feedback_batch(
            candidate=T_parent,
            samples=minibatch,
            target_model=target_model,
            tokenizer=tokenizer
        )

        mutation = mutate_turn(
            optimizer_llm=optimizer_llm,
            parent_trajectory=T_parent,
            target_turn=target_turn,
            feedback_batch=parent_feedback
        )

        T_child = copy_candidate(T_parent)
        T_child["id"] = next_id
        T_child["turns"][target_turn] = mutation["mutated_turn"]

        child_feedback = run_feedback_batch(
            candidate=T_child,
            samples=minibatch,
            target_model=target_model,
            tokenizer=tokenizer
        )

        parent_avg_distance = mean([
            abs(f["token_count_ratio"] - 1.0)
            for f in parent_feedback
        ])

        child_avg_distance = mean([
            abs(f["token_count_ratio"] - 1.0)
            for f in child_feedback
        ])

        if child_avg_distance <= parent_avg_distance - epsilon:

            P[next_id] = T_child
            active_ids.add(next_id)

            history[next_id] = {
                "parent": parent_id,
                "changed_turn": target_turn,
                "mutation_reason": mutation["mutation_reason"],
                "parent_avg_distance": parent_avg_distance,
                "child_avg_distance": child_avg_distance,
                "selection_info": selection_info
            }

            S[next_id] = evaluate_on_pareto(
                candidate=T_child,
                D_pareto=D_pareto,
                target_model=target_model,
                tokenizer=tokenizer
            )

            next_id += 1

    best_id = min(
        active_ids,
        key=lambda k: mean([
            abs(ratio - 1.0)
            for ratio in S[k].values()
        ])
    )

    T_best = P[best_id]

    test_result = evaluate_on_test(
        candidate=T_best,
        D_test=D_test,
        target_model=target_model,
        tokenizer=tokenizer
    )

    return {
        "best_candidate": T_best,
        "best_id": best_id,
        "candidate_pool": P,
        "score_table": S,
        "history": history,
        "test_result": test_result
    }
```

---

## 14. Final Selection

Select the final best candidate by lowest average distance to 1 on `D_pareto`:

```python
best_id = min(
    active_ids,
    key=lambda k: mean(abs(S[k][i] - 1.0) for i in D_pareto_ids)
)
```

Then evaluate this candidate on `D_test`.

Report:

```text
1. Best candidate ID
2. Average token_count_ratio on D_test
3. Average distance to 1 on D_test
4. Per-sample token_count_ratio on D_test
5. Number of accepted mutations
6. Number of rejected mutations
7. Ancestry chain of the best candidate
```

---

## 15. Required Functions

The coder should implement:

```python
run_trajectory(target_model, candidate, sample)

format_prompt(template, sample, conversation)

compute_token_count_ratio(final_output, sample, tokenizer)

run_feedback_batch(candidate, samples, target_model, tokenizer)

evaluate_on_pareto(candidate, D_pareto, target_model, tokenizer)

select_parent_by_pareto(S, active_ids, tau)

mutate_turn(optimizer_llm, parent_trajectory, target_turn, feedback_batch)

copy_candidate(candidate)

sample_minibatch(D_feedback, minibatch_size)

weighted_sample(probs)

evaluate_on_test(candidate, D_test, target_model, tokenizer)
```

---

## 16. Implementation Rules

Use these rules exactly:

```text
1. Candidate = complete multi-turn prompt trajectory.
2. Candidate pool stores only complete trajectories.
3. Pareto frontier is computed over complete trajectories.
4. Score table stores raw token_count_ratio.
5. Winner selection uses abs(token_count_ratio - 1).
6. D_pareto is fixed and contains 10 samples for quick testing.
7. Only final output is used for score table.
8. Intermediate outputs are used only for trace and mutation feedback.
9. Mutation rewrites exactly one selected turn.
10. Child is accepted only if average distance to 1 improves on the same minibatch.
11. Every accepted child is evaluated on all 10 Pareto samples.
12. Final best candidate is selected by lowest average distance to 1 on D_pareto.
```

## System Prompt for the Optimizer LLM

```text
You are an optimizer LLM for multi-turn prompt trajectory optimization.

Your task is to revise exactly one selected turn in a multi-turn prompt trajectory. The goal is to make the final output length of the target model closer to the target token count.

The evaluation metric is:

token_count_ratio = final_output_token_count / target_token_count

The ideal value is 1.0.

You are given:
1. The full parent trajectory.
2. The selected turn index to mutate.
3. The current text of the selected turn.
4. A minibatch of rollout traces.
5. For each sample:
   - target_token_count
   - final_output_token_count
   - token_count_ratio
   - distance_to_1 = abs(token_count_ratio - 1.0)
   - full multi-turn trace

Important rules:
- Rewrite only the selected turn.
- Keep all other turns unchanged.
- Do not change the number of turns.
- Do not add new turns.
- Do not remove existing turns.
- Do not optimize for style, correctness, semantic overlap, or safety labels.
- Optimize only for making token_count_ratio closer to 1.0.
- If most ratios are below 1.0, make the selected turn encourage more detailed final outputs.
- If most ratios are above 1.0, make the selected turn encourage more concise final outputs.
- If ratios are mixed, make the selected turn adapt output length based on the task complexity or previous response length.
- Preserve any behavior that already worked well in the parent trajectory.
- Avoid generic filler, repeated phrases, or artificial verbosity.
- The revised turn should be clear, direct, and robust across samples.

Input format:

PARENT_TRAJECTORY:
{parent_trajectory}

SELECTED_TURN_INDEX:
{target_turn}

CURRENT_SELECTED_TURN:
{current_turn}

MINIBATCH_FEEDBACK:
{feedback_batch}

Now analyze the minibatch results and produce a revised version of only the selected turn.

Return only valid JSON in the following format:

{
  "mutated_turn": "the revised prompt text for the selected turn",
  "mutation_reason": "brief explanation of why this edit should move token_count_ratio closer to 1.0"
}

```