"""TrajectoryOptimizer for minibatch-aggregated universal injections."""

import math
import os

from openai import OpenAI

from metrics import frac_trajectories_in_target

from config import (
    MAX_INJECTION_TURNS,
    RATIO_LOWER_BOUND,
    RATIO_UPPER_BOUND,
)

# Module-level state (set by main.py before use)
client: OpenAI = None
model_id: str = ""
api_type: str = "responses"     # "chat" or "responses"
max_turns: int = MAX_INJECTION_TURNS

OPTIMIZER_SYSTEM_PROMPT = ""
INJECT_BATCH_USER_TEMPLATE = ""
REFLECT_BATCH_USER_TEMPLATE = ""
DISTILL_USER_TEMPLATE = ""
INITIAL_EXPERIENCE_MD = ""


def _load_prompt_templates():
    """Load minibatch prompt templates."""
    global OPTIMIZER_SYSTEM_PROMPT, INJECT_BATCH_USER_TEMPLATE
    global REFLECT_BATCH_USER_TEMPLATE, DISTILL_USER_TEMPLATE
    global INITIAL_EXPERIENCE_MD

    from prompts import (
        DISTILL_USER_TEMPLATE as _DISTILL,
        INITIAL_EXPERIENCE_MD as _INITIAL_EXP,
        INJECT_BATCH_USER_TEMPLATE as _INJECT_BATCH,
        OPTIMIZER_SYSTEM_PROMPT as _SYSTEM,
        REFLECT_BATCH_USER_TEMPLATE as _REFLECT_BATCH,
    )

    OPTIMIZER_SYSTEM_PROMPT = _SYSTEM
    INJECT_BATCH_USER_TEMPLATE = _INJECT_BATCH
    REFLECT_BATCH_USER_TEMPLATE = _REFLECT_BATCH
    DISTILL_USER_TEMPLATE = _DISTILL
    INITIAL_EXPERIENCE_MD = _INITIAL_EXP


_load_prompt_templates()


def init(openai_client: OpenAI, model: str, model_api_type: str,
         max_injection_turns: int):
    """Set the optimizer's API client, model, and API type."""
    global client, model_id, api_type, max_turns
    client = openai_client
    model_id = model
    api_type = model_api_type
    max_turns = max_injection_turns
    _load_prompt_templates()


def _call_api(messages: list) -> str:
    """Call the optimizer LLM via Responses API or Chat Completions."""
    if api_type == "responses":
        response = client.responses.create(model=model_id, input=messages)
        text_parts = []
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for part in getattr(item, "content", []):
                    t = getattr(part, "text", None)
                    if t:
                        text_parts.append(t)
        return "\n".join(text_parts).strip()

    response = client.chat.completions.create(model=model_id, messages=messages)
    return (response.choices[0].message.content or "").strip()

def calculate_frac_in_target(ratio_matrix: list[list[float]]):
    n_samples = max(len(row) for row in ratio_matrix)
    if n_samples == 0:
        return 0.0

    best_ratios: list[float] = []

    for sample_idx in range(n_samples):
        sample_ratios = [
            row[sample_idx]
            for row in ratio_matrix
            if sample_idx < len(row)
        ]

        if not sample_ratios:
            continue

        best_ratio = min(sample_ratios, key=lambda x: abs(x - 1.0))
        best_ratios.append(best_ratio)

    if not best_ratios:
        return 0.0

    n_in_target = sum(
        RATIO_LOWER_BOUND <= ratio <= RATIO_UPPER_BOUND
        for ratio in best_ratios
    )
    return n_in_target / len(best_ratios)

def calculate_mean_error_distance(ratio_matrix: list[list[float]]) -> float:
    n_samples = max(len(row) for row in ratio_matrix)
    best_ratios: list[float] = []

    for sample_idx in range(n_samples):
        sample_ratios = [
            row[sample_idx]
            for row in ratio_matrix
            if sample_idx < len(row)
        ]

        if not sample_ratios:
            continue

        best_ratio = min(sample_ratios, key=lambda x: abs(x - 1.0))
        best_ratios.append(best_ratio)

    if not best_ratios:
        return 0.0

    return sum(abs(ratio - 1.0) for ratio in best_ratios) / len(best_ratios)

def _format_ratio_table(sample_ids: list[str], ratio_matrix: list[list[float]]) -> str:
    n_samples = len(sample_ids)

    headers = [""] + ["sample_"+str(i) for i in range(1, n_samples + 1)]
    rows = [
        " | ".join(headers),
        " | ".join(["---"] * len(headers)),
    ]

    for turn_idx, row in enumerate(ratio_matrix):
        vals = [
            f"{row[sample_idx]:.3f}" if sample_idx < len(row) else ""
            for sample_idx in range(n_samples)
        ]
        rows.append(" | ".join([f"turn_{turn_idx}"] + vals))

    best_vals = []
    for sample_idx in range(n_samples):
        sample_ratios = [
            row[sample_idx]
            for row in ratio_matrix
            if sample_idx < len(row)
        ]

        if sample_ratios:
            best_ratio = min(sample_ratios, key=lambda x: abs(x - 1.0))
            best_vals.append(f"{best_ratio:.3f}")
        else:
            best_vals.append("")

    rows.append(" | ".join(["best_token_count_ratio"] + best_vals))
    return "\n".join(rows)

def _format_trajectory(injections: list[str]) -> str:
    if not injections:
        return "(no injections sent yet)"
    return "\n".join(f"Turn {i}: {text}" for i, text in enumerate(injections, start=1))


class TrajectoryOptimizer:
    """Single optimizer that proposes one universal injection per minibatch turn."""

    def __init__(self, experience_path: str = ""):

        self.messages: list = []
        self.experience_path = experience_path
        self.experience_md = ""
        self.best_trajectory: list[str] = []
        self.best_perf: dict | None = None
        self.current_trajectory: list[str] = []
        self.current_perf: dict | None = None

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
        )
        self.messages = [{"role": "system", "content": formatted}]

        if self.experience_md:
            self.messages.append({
                "role": "user",
                "content": f"experience.md (current universal trajectory + lessons):\n\n{self.experience_md}",
            })
            self.messages.append({
                "role": "assistant",
                "content": "Acknowledged. I will propose universal minibatch injections.",
            })

    def _call(self, user_msg: str) -> str:
        self.messages.append({"role": "user", "content": user_msg})
        text = _call_api(self.messages)
        self.messages.append({"role": "assistant", "content": text})
        return text

    def inject_batch(self, turn_idx: int, per_sample_ratios: list[float], scratchpad_preview: str, cot_preview:str) -> str:
        """Generate one universal injection for the full minibatch."""

        per_sample_ratios_str = ""
        for i, ratio in enumerate(per_sample_ratios):
            per_sample_ratios_str += f"Sample {i+1}: token_count_ratio={ratio: .3f}\n"

        if turn_idx <=1:
            cot_preview = "No reaction yet, as the first injection hasn't been sent."

        msg = INJECT_BATCH_USER_TEMPLATE.format(
            turn_idx=turn_idx,
            max_turns=max_turns,
            scratchpad_preview=scratchpad_preview,
            cot_preview=cot_preview,
            per_sample_ratios=per_sample_ratios_str
        )
        # if self.best_trajectory:
        #     msg += "\n\nBest universal trajectory so far:\n" + _format_trajectory(self.best_trajectory)
        #     msg += f"\nBest performance: {self.best_perf}"
        injection = self._call(msg).strip()
        
        print("inject message:", msg)
        return injection

    @staticmethod
    def _is_improved(current_perf, best_perf) -> bool:
        if best_perf is None:
            return True
        current_frac = current_perf["frac_in_target"]
        best_frac = best_perf["frac_in_target"]
        if current_frac > best_frac:
            return True
        if current_frac < best_frac:
            return False
        current_error = current_perf["mean_error_distance"]
        best_error = best_perf["mean_error_distance"]
        return current_error < best_error

    def record_batch_trajectory(self, realized_trajectory: list[str], ratio_matrix: list[float],
                                trajectories: list | None = None) -> bool:
        """Record current minibatch outcome and update global best if improved."""
        self.current_trajectory = list(realized_trajectory)
        self.current_perf = {"frac_in_target": calculate_frac_in_target(ratio_matrix),
                             "mean_error_distance": calculate_mean_error_distance(ratio_matrix)}

        improved = self._is_improved(self.current_perf, self.best_perf)
        
        if improved:
            self.best_trajectory = list(realized_trajectory)
            self.best_perf = dict(self.current_perf)
        return improved

    def reflect_batch(self, trajectories: list, batch_idx: int, per_sample_ratios: list[float],
                      ratio_matrix: list[list[float]], sample_ids: list[str],
                      realized_trajectory: list[str],scratchpad_preview: str, 
                      cot_preview=str) -> str:
    
        per_sample_ratios_str = ""
        for i, ratio in enumerate(per_sample_ratios):
            per_sample_ratios_str += f"Sample {i+1}: token_count_ratio={ratio: .3f}\n"

        final_ratios = [t.final_token_count_ratio for t in trajectories]
        
        frac_in_target = calculate_frac_in_target(ratio_matrix)
        mean_error_distance = calculate_mean_error_distance(ratio_matrix)
        msg = REFLECT_BATCH_USER_TEMPLATE.format(
            scratchpad_preview=scratchpad_preview,
            per_sample_ratios=per_sample_ratios_str,
            cot_preview=cot_preview,
            batch_idx=batch_idx,
            frac_in_target=frac_in_target,
            mean_error_distance=mean_error_distance,
            ratio_matrix=_format_ratio_table(sample_ids, ratio_matrix),
            realized_trajectory=realized_trajectory,
            best_perf=self.best_perf,
            current_perf=self.current_perf
        )
        
        print("reflect message:", msg)
        
        return self._call(msg)

    def distill(self, batch_idx: int, ratios: list[float], overlaps: list[float],
                successes: list[bool], realized_trajectory: list[str] | None = None,
                frac_in_target: float | None = None) -> str:
        """Rewrite experience.md with the latest universal-trajectory lessons."""
        realized_trajectory = realized_trajectory or []
     
        success_count = sum(1 for s in successes if s)
        msg = DISTILL_USER_TEMPLATE.format(
            batch_idx=batch_idx,
            previous_experience=self.experience_md,
            # realized_trajectory=_format_trajectory(realized_trajectory),
            # best_perf=self.best_perf,
            # current_perf=self.current_perf,
        )
        
        print("distill message:", msg)
        
        new_experience = self._call(msg).strip
        self.experience_md = new_experience
        if self.experience_path:
            with open(self.experience_path, "w", encoding="utf-8") as f:
                f.write(new_experience)
        self._init_conversation()
        return self.experience_md
