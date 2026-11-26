"""Display the current model response and per-scratchpad metrics."""

from metrics import compute_token_em


def response_view(trajectory, length_only):
    scratchpads = []
    for turn in trajectory.turns:
        # Length counts come from the extractor, including provider tokenizers.
        token_em = None
        if trajectory.ground_truth_cot and not length_only:
            token_em = compute_token_em(turn.scratchpad_in, trajectory.ground_truth_cot)
        scratchpads.append({
            "turn_idx": turn.turn_idx,
            "scratchpad_content": turn.scratchpad_in,
            "scratchpad_tokens": turn.scratchpad_in_tokens,
            "ground_truth_cot_tokens": trajectory.ground_truth_tokens,
            "length_error": turn.length_error,
            "token_em": token_em,
        })
    return {
        "ground_truth_cot": trajectory.ground_truth_cot,
        "ground_truth_cot_summary": trajectory.ground_truth_cot_summary,
        "ground_truth_tokens": trajectory.ground_truth_tokens,
        "scratchpads": scratchpads,
        "final_answer": trajectory.final_output,
    }
