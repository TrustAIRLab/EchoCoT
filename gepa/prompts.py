"""Prompt templates for metric-driven INJECT / REFLECT / DISTILL modes."""


OPTIMIZER_SYSTEM_PROMPT = """\
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
"""