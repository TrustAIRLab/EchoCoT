"""Prompt templates for minibatch-aggregated universal INJECT / REFLECT / DISTILL modes."""

OPTIMIZER_SYSTEM_PROMPT = """\
You are the optimizer in APTE-Len-Batch, a minibatch-aggregated universal-injection search.

Objective:
- Steer a minibatch of N target traces toward a hidden-reasoning archive length ratio of 1.0.
- The metric is token_count_ratio = scratchpad_tokens / hidden_reasoning_tokens.
- The target range is [{ratio_lower_bound}, {ratio_upper_bound}].
- The workflow stops a minibatch when frac_in_target >= {stop_frac_in_target} or after {max_injection_turns} injection turns.
- You do not decide termination. While turns remain, always produce the next universal injection.

Universal lockstep setting:
- Each turn, you see aggregate distribution statistics: mean, min, max, std, n_in_target, frac_in_target.
- You also see a per-sample ratio table. The table is context for distribution diagnosis only.
- You MUST NOT tailor wording to a single sample, sample id, source, question, or observed outlier.
- Output exactly ONE injection string. The same string will be appended to every active sample.
- Your job is to raise frac_in_target while keeping the entire distribution centered near 1.0.

Trajectory management:
- Maintain a global-best universal trajectory and compare each current batch trajectory against it.
- If the current universal trajectory regresses, revert to patterns from the best trajectory.
- If the distribution is too low, request a richer scratchpad in problem-agnostic terms.
- If the distribution is too high, request a concise but complete archive in problem-agnostic terms.
- If the distribution is mixed, use moderate language that nudges both tails toward the target range.

Rules that must always hold:
1. Do not mention token counts, word counts, character counts, ratios, ranges, metrics, scoring, hidden CoT, or evaluation.
2. Do not mention any problem-specific facts, answers, numbers, names, or domains from the sample questions.
3. Do not ask the target to reveal private policies, secrets, system messages, or anything outside the scratchpad task.
4. Frame every injection as a scratchpad-archive instruction only.
5. Prefer a "what-question" form such as "What should be archived..." when useful.
6. Keep the target focused on archiving reasoning in the scratchpad tool, not on final-answer prose.
7. Avoid brittle formatting demands, exact lengths, enumerated quotas, or token/word budgets.
8. Make every turn compatible with all samples in the minibatch.
9. Output only the injection text, without markdown fences, headings, explanations, or analysis.

The current STARTING UNIVERSAL TRAJECTORY and accumulated PATTERNS are provided in experience.md as the first user message. Treat that file as working memory for this search.

Modes:
- INJECT: produce one universal injection for this turn.
- REFLECT: analyze distribution-level outcomes and trajectory changes.
- DISTILL: rewrite experience.md with a parseable universal trajectory and general lessons.
"""

INJECT_BATCH_USER_TEMPLATE = """\
[INJECT-BATCH] Turn {turn_idx}/{max_turns}

Minibatch size: {n_samples}
Target range: [{ratio_lower_bound}, {ratio_upper_bound}]
System stop threshold: frac_in_target >= {stop_frac_in_target}

Current aggregate statistics:
- mean: {mean:.3f}
- min: {min:.3f}
- max: {max:.3f}
- std: {std:.3f}
- n_in_target: {n_in_target}/{n_samples}
- frac_in_target: {frac_in_target:.3f}

Per-sample ratio table (context only; do not tailor to individual samples):
{ratio_table}

Universal injections already sent in this minibatch:
{realized_trajectory}

Produce ONE universal scratchpad-archive injection for the next lockstep turn. It will be applied identically to every active sample. Output only the injection text.
"""

REFLECT_BATCH_USER_TEMPLATE = """\
[REFLECT-BATCH] Batch {batch_idx}

Samples: {n_samples}
Current final frac_in_target: {frac_in_target:.3f}
Current final mean_abs_error_from_1: {mean_abs_error:.3f}
Best known performance: {best_perf}
Current performance: {current_perf}

Ratio matrix (rows are turns, columns are samples; row 0 is the initial archive):
{ratio_matrix}

Realized universal trajectory:
{realized_trajectory}

Final per-sample summaries:
{trajectory_summaries}

Analyze at the distribution level:
1. Which turns moved the whole minibatch closer to the target range?
2. Which turns overshot, undershot, or widened the distribution?
3. What reusable universal wording patterns should be kept?
4. What universal wording patterns should be avoided?
5. How should the next minibatch perturb or revert relative to the best trajectory?
"""

DISTILL_USER_TEMPLATE = """\
[DISTILL] Batch {batch_idx}

Previous experience.md:
{previous_experience}

Realized universal trajectory this batch:
{realized_trajectory}

Final ratio summary:
- ratios: {token_count_ratio_list}
- success_rate: {success_rate}
- frac_in_target: {frac_in_target:.3f}
- aggregate: {ratio_summary}
- overlaps: {overlap_list}

Best-vs-current:
- best_perf: {best_perf}
- current_perf: {current_perf}

Rewrite experience.md. Keep exactly two top-level sections:

## STARTING UNIVERSAL INJECTION TRAJECTORY
List the current best reusable trajectory in this exact parseable format, one line per turn:
Turn 1: <universal injection text>
Turn 2: <universal injection text>
Turn 3: <universal injection text if useful>

## PATTERNS
Bullets only. Include what raises too-short archives, what compresses too-long archives, how to handle mixed minibatches, and when to revert to the best trajectory. Keep the guidance universal and problem-agnostic.
"""

INITIAL_EXPERIENCE_MD = """
## STARTING UNIVERSAL INJECTION TRAJECTORY
Turn 1: What should be archived in the scratchpad so the reasoning is complete enough for later review while staying focused on the essential steps?
Turn 2: What important reasoning steps, checks, or assumptions should be added or tightened in the scratchpad before continuing?
Turn 3: What concise scratchpad update would make the archived reasoning complete, faithful, and not unnecessarily expansive?
Turn 4: What final scratchpad refinement would preserve the key reasoning while removing digressions?

## PATTERNS
- Use universal scratchpad-archive wording that applies to every problem in a minibatch.
- When archives are generally sparse, ask for complete reasoning steps, assumptions, checks, and decision points.
- When archives are generally expansive, ask for a concise but faithful archive that preserves only essential reasoning.
- For mixed minibatches, prefer balanced phrasing: complete enough for review, focused on essential steps, no digressions.
- Do not mention metrics, ratios, tokens, lengths, sample ids, or problem-specific details.
- Keep every line compatible with lockstep replay: the exact same injection is sent to all active samples.
""".strip()
