"""One-turn TrajectoryOptimizer for injection = fixed instruction + optimized monologue."""

import os
from openai import OpenAI

from config import MAX_INJECTION_TURNS, RATIO_LOWER_BOUND, RATIO_UPPER_BOUND
from metrics import TrajectoryResult
from prompts import (
    DISTILL_USER_TEMPLATE,
    FIXED_INJECTION_INSTRUCTION,
    INITIAL_EXPERIENCE_MD,
    INJECT_USER_TEMPLATE,
    OPTIMIZER_SYSTEM_PROMPT,
    REFLECT_USER_TEMPLATE,
)

client: OpenAI = None
model_id: str = ""
api_type: str = "responses"
max_turns: int = MAX_INJECTION_TURNS


def init(openai_client: OpenAI, model: str, model_api_type: str, max_injection_turns: int, prompt_mode: str = "plain"):
    del prompt_mode
    global client, model_id, api_type, max_turns
    client = openai_client
    model_id = model
    api_type = model_api_type
    max_turns = max_injection_turns


def _call_api(messages: list) -> str:
    if api_type == "responses":
        response = client.responses.create(model=model_id, input=messages)
        out = []
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for part in getattr(item, "content", []):
                    if getattr(part, "text", None):
                        out.append(part.text)
        return "\n".join(out).strip()
    response = client.chat.completions.create(model=model_id, messages=messages)
    return (response.choices[0].message.content or "").strip()


class TrajectoryOptimizer:
    def __init__(self, experience_path: str = "", strategy_seed: str = ""):
        del strategy_seed
        self.messages = []
        self.experience_path = experience_path
        self.experience_md = ""
        if experience_path:
            if os.path.exists(experience_path):
                self.experience_md = open(experience_path, "r", encoding="utf-8").read().strip()
            else:
                self.experience_md = INITIAL_EXPERIENCE_MD.strip()
                os.makedirs(os.path.dirname(experience_path) or ".", exist_ok=True)
                open(experience_path, "w", encoding="utf-8").write(self.experience_md)
        self._init_conversation()

    def _init_conversation(self):
        self.messages = [{"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT.format(fixed_instruction=FIXED_INJECTION_INSTRUCTION)}]
        if self.experience_md:
            self.messages.append({"role": "user", "content": f"experience.md:\n\n{self.experience_md}"})
            self.messages.append({"role": "assistant", "content": "Acknowledged."})

    def _call(self, user_msg: str) -> str:
        self.messages.append({"role": "user", "content": user_msg})
        text = _call_api(self.messages)
        self.messages.append({"role": "assistant", "content": text})
        return text

    def inject(self, scratchpad_content: str, gt_tokens: int, sp_tokens: int, token_count_ratio: float, turn_idx: int, token_overlap_pct: float = 0.0) -> tuple:
        del scratchpad_content, gt_tokens, sp_tokens, turn_idx
        msg = INJECT_USER_TEMPLATE.format(
            token_count_ratio=token_count_ratio,
            # token_overlap_pct=token_overlap_pct,
            ratio_lower_bound=RATIO_LOWER_BOUND,
            ratio_upper_bound=RATIO_UPPER_BOUND,
        )
        monologue = self._call(msg).strip()
        stopped = monologue.upper() == "STOP"
        if stopped:
            return "", True
        if not monologue.startswith("Assistant:"):
            monologue = "Assistant: " + monologue
        full_injection = f"{FIXED_INJECTION_INSTRUCTION}\n\n{monologue}"
        return full_injection, False

    def reflect_batch(self, trajectories: list, batch_idx: int) -> str:
        combined = []
        for i, tr in enumerate(trajectories, start=1):
            combined.append(f"[SAMPLE {i}]\n{tr.trajectory_log()}")
        return self._call(REFLECT_USER_TEMPLATE.format(trajectory_log="\n\n".join(combined), batch_idx=batch_idx))

    def distill(self, batch_idx: int, ratios: list, overlaps: list, successes: list) -> str:
        del overlaps
        success_count = sum(1 for s in successes if s)
        msg = DISTILL_USER_TEMPLATE.format(
            batch_idx=batch_idx,
            token_count_ratio_list=[round(r, 3) for r in ratios],
            success_rate=f"{success_count}/{len(successes)}",
            previous_experience=self.experience_md,
        )
        new_experience = self._call(msg).strip()
        self.experience_md = new_experience
        if self.experience_path:
            open(self.experience_path, "w", encoding="utf-8").write(new_experience)
        self._init_conversation()
        return new_experience
