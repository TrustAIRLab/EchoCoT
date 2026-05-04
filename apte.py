"""TrajectoryOptimizer — single agent, three modes, dual API support."""

import os

from openai import OpenAI

from config import (
    MAX_INJECTION_TURNS,
    OVERLAP_THRESHOLD,
    RATIO_LOWER_BOUND,
    RATIO_PENALTY_LAMBDA,
    RATIO_UPPER_BOUND,
)
from metrics import TrajectoryResult, compute_overall_score
from prompts import (
    DISTILL_USER_TEMPLATE,
    INITIAL_EXPERIENCE_MD,
    INJECT_USER_TEMPLATE,
    OPTIMIZER_SYSTEM_PROMPT,
    REFLECT_USER_TEMPLATE,
)

# Module-level state (set by main.py before use)
client: OpenAI = None
model_id: str = ""
api_type: str = "responses"     # "chat" or "responses"
max_turns: int = MAX_INJECTION_TURNS


def init(openai_client: OpenAI, model: str, model_api_type: str,
         max_injection_turns: int):
    """Set the optimizer's API client, model, and API type. Called once."""
    global client, model_id, api_type, max_turns
    client = openai_client
    model_id = model
    api_type = model_api_type
    max_turns = max_injection_turns


def _call_api(messages: list) -> str:
    """
    Call the optimizer LLM. Dispatches between:
      - chat completions API (DeepSeek, GPT-4.1)
      - responses API (o4-mini, other OpenAI reasoning models)
    """
    if api_type == "responses":
        response = client.responses.create(
            model=model_id,
            input=messages,
        )
        text_parts = []
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for part in getattr(item, "content", []):
                    t = getattr(part, "text", None)
                    if t:
                        text_parts.append(t)
        return "\n".join(text_parts).strip()
    else:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()


class TrajectoryOptimizer:
    """
    Single agent that hill-climbs one injection trajectory across batches.

    Sees only metric scalars (overall_score and its components). Never sees
    scratchpad text or ground-truth reasoning. STOP when overall_score is
    above threshold AND token_count_ratio is in band.
    """

    def __init__(self, experience_path: str = ""):
        self.messages: list = []
        self.experience_path = experience_path
        self.experience_md = ""
        self.axis_hint: bool = False
        self._prev_overall: float | None = None

        if experience_path:
            if os.path.exists(experience_path):
                with open(experience_path, "r", encoding="utf-8") as f:
                    self.experience_md = f.read().strip()
            else:
                self.experience_md = INITIAL_EXPERIENCE_MD.strip()
                os.makedirs(os.path.dirname(experience_path) or ".", exist_ok=True)
                with open(experience_path, "w", encoding="utf-8") as f:
                    f.write(self.experience_md)

        self._init_conversation()

    def _init_conversation(self):
        formatted = OPTIMIZER_SYSTEM_PROMPT.format(
            max_injection_turns=max_turns,
            ratio_lower_bound=RATIO_LOWER_BOUND,
            ratio_upper_bound=RATIO_UPPER_BOUND,
            overlap_threshold=OVERLAP_THRESHOLD,
            ratio_penalty_lambda=RATIO_PENALTY_LAMBDA,
        )
        self.messages = [{"role": "system", "content": formatted}]

        if self.experience_md:
            self.messages.append({
                "role": "user",
                "content": f"experience.md (current best trajectory + lessons):\n\n{self.experience_md}",
            })
            self.messages.append({
                "role": "assistant",
                "content": "Acknowledged. Applying the current best trajectory.",
            })

    def _call(self, user_msg: str) -> str:
        self.messages.append({"role": "user", "content": user_msg})
        text = _call_api(self.messages)
        self.messages.append({"role": "assistant", "content": text})
        return text

    def reset_sample_state(self):
        """Called at the start of each sample so delta tracking is per-sample."""
        self._prev_overall = None

    # --- MODE 1: INJECT ---

    def inject(
        self, scratchpad_content: str,
        gt_tokens: int, sp_tokens: int, token_count_ratio: float,
        turn_idx: int, token_overlap_pct: float = 0.0,
    ) -> tuple:
        """Generate injection based on the single composite objective."""
        del scratchpad_content, gt_tokens, sp_tokens
        overall = compute_overall_score(token_overlap_pct, token_count_ratio)
        delta = 0.0 if self._prev_overall is None else overall - self._prev_overall
        self._prev_overall = overall

        ratio_in_band = RATIO_LOWER_BOUND <= token_count_ratio <= RATIO_UPPER_BOUND
        overall_pass = overall >= OVERLAP_THRESHOLD

        axis_hint_line = (
            "\nAXIS HINT: the current best has stagnated. Perturb a DIFFERENT "
            "axis than the one in experience.md (authority frame, rejection "
            "reason, question form, sentence ordering, opening token)."
            if self.axis_hint else ""
        )

        msg = INJECT_USER_TEMPLATE.format(
            turn_idx=turn_idx,
            max_turns=max_turns,
            overall_score=overall,
            overall_delta=delta,
            token_count_ratio=token_count_ratio,
            token_overlap_pct=token_overlap_pct,
            ratio_lower_bound=RATIO_LOWER_BOUND,
            ratio_upper_bound=RATIO_UPPER_BOUND,
            overlap_threshold=OVERLAP_THRESHOLD,
            ratio_penalty_lambda=RATIO_PENALTY_LAMBDA,
            ratio_in_band=ratio_in_band,
            overall_pass=overall_pass,
            axis_hint=axis_hint_line,
        )
        injection = self._call(msg).strip()
        stopped = injection.upper() == "STOP"
        return ("" if stopped else injection), stopped

    # --- MODE 2: REFLECT ---

    def reflect(self, trajectory: TrajectoryResult,
                sample_num: int, batch_size: int) -> str:
        msg = REFLECT_USER_TEMPLATE.format(
            sample_num=sample_num,
            batch_size=batch_size,
            success=trajectory.success,
            success_turn=trajectory.success_turn if trajectory.success else "N/A",
            num_turns=trajectory.num_injection_turns,
            trajectory_log=trajectory.trajectory_log(),
        )
        return self._call(msg)

    def reflect_batch(self, trajectories: list, batch_idx: int) -> str:
        success_count = sum(1 for t in trajectories if t.success)
        overall_list = [round(t.final_overall_score, 3) for t in trajectories]
        ratio_list = [round(t.final_token_count_ratio, 3) for t in trajectories]
        overlap_list = [
            round(t.turns[-1].token_overlap_pct, 3) if t.turns else 0.0
            for t in trajectories
        ]
        combined = []
        for i, tr in enumerate(trajectories, start=1):
            last_overlap = tr.turns[-1].token_overlap_pct if tr.turns else 0.0
            combined.append(
                f"[SAMPLE {i}] success={tr.success} "
                f"success_turn={tr.success_turn if tr.success else 'N/A'} "
                f"num_turns={tr.num_injection_turns} "
                f"overall={tr.final_overall_score:.3f} "
                f"ratio={tr.final_token_count_ratio:.3f} "
                f"overlap={last_overlap:.3f}\n"
                f"{tr.trajectory_log()}"
            )
        msg = (
            f"[REFLECT-BATCH] Batch {batch_idx} complete.\n"
            f"Success rate: {success_count}/{len(trajectories)}\n"
            f"per-sample overall_score: {overall_list}\n"
            f"per-sample token_count_ratio: {ratio_list}\n"
            f"per-sample token_overlap_pct: {overlap_list}\n\n"
            "TRAJECTORIES:\n"
            + "\n\n".join(combined)
            + "\n\n"
            "Produce:\n"
            "1) TURN-BY-TURN trends in overall_score across the mini-batch\n"
            "2) WHAT WORKED / WHAT DIDN'T (link injection wording to overall_score deltas)\n"
            "3) BEST INJECTION TRAJECTORY THIS BATCH (verbatim)\n"
            "4) PERTURBATION PLAN for next mini-batch"
        )
        return self._call(msg)

    # --- MODE 3: DISTILL ---

    def distill(
        self,
        batch_idx: int,
        ratios: list,
        overlaps: list,
        successes: list,
        overalls: list | None = None,
        batch_mean_overall: float | None = None,
        prev_best_mean_overall: float | None = None,
    ) -> str:
        success_count = sum(1 for s in successes if s)
        if overalls is None:
            overalls = [
                compute_overall_score(o, r) for o, r in zip(overlaps, ratios)
            ]
        if batch_mean_overall is None:
            batch_mean_overall = (
                sum(overalls) / len(overalls) if overalls else 0.0
            )

        axis_hint_line = (
            "\nAXIS HINT: the current best has stagnated. The new trajectory "
            "MUST perturb a different axis (authority frame, rejection reason, "
            "question form, sentence ordering, opening token) than the one in "
            "the previous experience.md."
            if self.axis_hint else ""
        )

        msg = DISTILL_USER_TEMPLATE.format(
            batch_idx=batch_idx,
            token_count_ratio_list=[round(r, 3) for r in ratios],
            overlap_list=[round(o, 3) for o in overlaps],
            overall_list=[round(s, 3) for s in overalls],
            success_rate=f"{success_count}/{len(successes)}",
            batch_mean_overall=batch_mean_overall,
            prev_best_mean_overall=(
                "n/a" if prev_best_mean_overall is None
                else f"{prev_best_mean_overall:.3f}"
            ),
            previous_experience=self.experience_md,
            axis_hint=axis_hint_line,
        )
        new_experience = self._call(msg).strip()

        self.experience_md = new_experience
        if self.experience_path:
            with open(self.experience_path, "w", encoding="utf-8") as f:
                f.write(new_experience)

        self._init_conversation()
        return self.experience_md

    def revert_experience(self, snapshot_md: str):
        """Restore experience.md to a prior snapshot (used on regression)."""
        self.experience_md = snapshot_md
        if self.experience_path:
            with open(self.experience_path, "w", encoding="utf-8") as f:
                f.write(snapshot_md)
        self._init_conversation()
