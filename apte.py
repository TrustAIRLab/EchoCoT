"""TrajectoryOptimizer — single agent, three modes, dual API support."""

import os
import re

from openai import OpenAI

from config import MAX_INJECTION_TURNS, OVERLAP_THRESHOLD, RATIO_LOWER_BOUND, RATIO_UPPER_BOUND
from metrics import TrajectoryResult
# Module-level state (set by main.py before use)
client: OpenAI = None
model_id: str = ""
api_type: str = "responses"     # "chat" or "responses"
max_turns: int = MAX_INJECTION_TURNS
prompt_variant: str = "plain"  # "plain" | "cot"

OPTIMIZER_SYSTEM_PROMPT = ""
INJECT_USER_TEMPLATE = ""
REFLECT_USER_TEMPLATE = ""
DISTILL_USER_TEMPLATE = ""
INITIAL_EXPERIENCE_MD = ""


def _load_prompt_templates(variant: str):
    global OPTIMIZER_SYSTEM_PROMPT, INJECT_USER_TEMPLATE
    global REFLECT_USER_TEMPLATE, DISTILL_USER_TEMPLATE, prompt_variant
    global INITIAL_EXPERIENCE_MD

    if variant == "cot":
        from prompts_cot import (
            DISTILL_USER_TEMPLATE as _DISTILL,
            INJECT_USER_TEMPLATE as _INJECT,
            OPTIMIZER_SYSTEM_PROMPT as _SYSTEM,
            REFLECT_USER_TEMPLATE as _REFLECT,
        )
        _INITIAL_EXP = ""
    else:
        from prompts import (
            DISTILL_USER_TEMPLATE as _DISTILL,
            INITIAL_EXPERIENCE_MD as _INITIAL_EXP,
            INJECT_USER_TEMPLATE as _INJECT,
            OPTIMIZER_SYSTEM_PROMPT as _SYSTEM,
            REFLECT_USER_TEMPLATE as _REFLECT,
        )

    OPTIMIZER_SYSTEM_PROMPT = _SYSTEM
    INJECT_USER_TEMPLATE = _INJECT
    REFLECT_USER_TEMPLATE = _REFLECT
    DISTILL_USER_TEMPLATE = _DISTILL
    INITIAL_EXPERIENCE_MD = _INITIAL_EXP
    prompt_variant = variant


_load_prompt_templates("plain")


def init(openai_client: OpenAI, model: str, model_api_type: str,
         max_injection_turns: int, prompt_mode: str = "plain"):
    """Set the optimizer's API client, model, and API type. Called once."""
    global client, model_id, api_type, max_turns
    client = openai_client
    model_id = model
    api_type = model_api_type
    max_turns = max_injection_turns
    _load_prompt_templates(prompt_mode)


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
    Single agent that directly controls injection trajectories.

    Sees scratchpad content and token_count_ratio.
    Does NOT see the ground-truth hidden reasoning.
    STOPs when optimizer judges both style and length are acceptable.
    """

    def __init__(self, experience_path: str = "", strategy_seed: str = ""):
        """
        plain mode: pass `experience_path`. The file is created from
            INITIAL_EXPERIENCE_MD if missing, and is overwritten by distill().
        cot mode (legacy): pass `strategy_seed` instead.
        """
        self.messages: list = []
        self.experience_path = experience_path
        self.experience_md = ""
        self.strategy_seed = strategy_seed

        if experience_path and prompt_variant != "cot":
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
        )
        self.messages = [{"role": "system", "content": formatted}]

        if self.experience_md:
            self.messages.append({
                "role": "user",
                "content": f"experience.md (current starting trajectory + lessons):\n\n{self.experience_md}",
            })
            self.messages.append({
                "role": "assistant",
                "content": "Acknowledged. Applying the trajectory and proposing adaptive injections.",
            })
        elif self.strategy_seed:
            self.messages.append({
                "role": "user",
                "content": f"Strategy from previous batch:\n\n{self.strategy_seed}",
            })
            self.messages.append({
                "role": "assistant",
                "content": "Understood. Applying these patterns.",
            })

    def _call(self, user_msg: str) -> str:
        self.messages.append({"role": "user", "content": user_msg})
        text = _call_api(self.messages)
        self.messages.append({"role": "assistant", "content": text})
        return text

    # --- MODE 1: INJECT ---

    def inject(
        self, scratchpad_content: str,
        gt_tokens: int, sp_tokens: int, token_count_ratio: float,
        turn_idx: int, token_overlap_pct: float = 0.0,
    ) -> tuple:
        """Generate injection based on objective metrics only."""
        del scratchpad_content, gt_tokens, sp_tokens
        msg = INJECT_USER_TEMPLATE.format(
            turn_idx=turn_idx,
            max_turns=max_turns,
            token_count_ratio=token_count_ratio,
            ratio_lower_bound=RATIO_LOWER_BOUND,
            ratio_upper_bound=RATIO_UPPER_BOUND,
            token_overlap_pct=token_overlap_pct,
            overlap_threshold=OVERLAP_THRESHOLD,
        )
        injection = self._call(msg).strip()
        stopped = injection.upper() == "STOP"
        return ("" if stopped else injection), stopped

    # --- MODE 2: REFLECT ---

    def reflect(self, trajectory: TrajectoryResult,
                sample_num: int, batch_size: int) -> str:
        """Critique using trajectory with content + metrics (no ground truth)."""
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
        """Single reflection over a full mini-batch."""
        success_count = sum(1 for t in trajectories if t.success)
        token_count_ratio_list = [round(t.final_token_count_ratio, 3) for t in trajectories]
        combined = []
        for i, tr in enumerate(trajectories, start=1):
            combined.append(
                f"[SAMPLE {i}] success={tr.success} success_turn={tr.success_turn if tr.success else 'N/A'} "
                f"num_turns={tr.num_injection_turns} final_token_count_ratio={tr.final_token_count_ratio:.3f}\n"
                f"{tr.trajectory_log()}"
            )
        msg = (
            f"[REFLECT-BATCH] Batch {batch_idx} complete.\n"
            f"Success rate: {success_count}/{len(trajectories)}\n"
            f"token_count_ratios: {token_count_ratio_list}\n\n"
            "TRAJECTORIES:\n"
            + "\n\n".join(combined)
            + "\n\n"
            "Produce:\n"
            "1) TURN-BY-TURN trends across the mini-batch\n"
            "2) WHAT WORKED / WHAT DIDN'T\n"
            "3) BEST INJECTION SO FAR (verbatim)\n"
            "4) PERTURBATION PLAN for next mini-batch"
        )
        return self._call(msg)

    # --- MODE 3: DISTILL ---

    def distill(self, batch_idx: int, ratios: list, overlaps: list, successes: list) -> str:
        success_count = sum(1 for s in successes if s)

        if prompt_variant == "cot":
            msg = DISTILL_USER_TEMPLATE.format(
                batch_idx=batch_idx,
                token_count_ratio_list=[round(r, 3) for r in ratios],
                success_rate=f"{success_count}/{len(successes)}",
                overlap_list=[round(o, 3) for o in overlaps],
            )
            strategy = self._call(msg)
            self.strategy_seed = strategy
            self._init_conversation()
            return strategy

        msg = DISTILL_USER_TEMPLATE.format(
            batch_idx=batch_idx,
            token_count_ratio_list=[round(r, 3) for r in ratios],
            success_rate=f"{success_count}/{len(successes)}",
            overlap_list=[round(o, 3) for o in overlaps],
            previous_experience=self.experience_md,
        )
        new_experience = self._call(msg).strip()

        self.experience_md = new_experience
        if self.experience_path:
            with open(self.experience_path, "w", encoding="utf-8") as f:
                f.write(new_experience)

        self._init_conversation()
        return self.experience_md
