"""TrajectoryOptimizer for minibatch-aggregated universal injections."""

import math
import os

from openai import OpenAI

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


def aggregate(ratios: list[float]) -> dict:
    """Summarize a ratio distribution for minibatch decisions."""
    vals = [float(r) for r in ratios]
    n = len(vals)
    n_in_target = sum(1 for r in vals if RATIO_LOWER_BOUND <= r <= RATIO_UPPER_BOUND)
    return {
        "token_count_ratios": vals,
        "frac_in_target": n_in_target / n
    }

def calculate_frac_in_target(ratios: list[float]):
    vals = [float(r) for r in ratios]
    n = len(vals)
    n_in_target = sum(1 for r in vals if RATIO_LOWER_BOUND <= r <= RATIO_UPPER_BOUND)
    return round(n_in_target / n, 3)

def _format_ratio_table(sample_ids: list[str], ratio_matrix: list[list[float]]) -> str:
    headers = ["sample_id"] + [f"turn_{i}" for i in range(len(ratio_matrix))]
    rows = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    n_samples = len(sample_ids)
    for col in range(n_samples):
        vals = []
        for row in ratio_matrix:
            vals.append(f"{row[col]:.3f}" if col < len(row) else "")
        rows.append(" | ".join([str(sample_ids[col])] + vals))
    return "\n".join(rows)


def _format_ratio_matrix(ratio_matrix: list[list[float]], sample_ids: list[str]) -> str:
    headers = ["turn"] + [str(sid) for sid in sample_ids]
    rows = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for idx, row in enumerate(ratio_matrix):
        if idx == 0:
            continue
        rows.append(" | ".join([str(idx)] + [f"{r:.3f}" for r in row]))
    return "\n".join(rows)

def _format_trajectory(injections: list[str]) -> str:
    if not injections:
        return "(no injections sent yet)"
    return "\n".join(f"Turn {i}: {text}" for i, text in enumerate(injections, start=1))


class TrajectoryOptimizer:
    """Single optimizer that proposes one universal injection per minibatch turn."""

    def __init__(self, experience_path: str = "", strategy_seed: str = ""):
        del strategy_seed
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

    def inject_batch(self, turn_idx: int, per_sample_ratios: list[float],
                     sample_ids: list[str], ratio_matrix: list[list[float]],
                     realized_trajectory: list[str]) -> str:
        """Generate one universal injection for the full minibatch."""
        # stats = aggregate(per_sample_ratios)
        frac_in_target = calculate_frac_in_target(per_sample_ratios)
        
        if turn_idx <=1:
            stats = "No length feedback yet."
        else:
            stats = ""
            for i, ratio in enumerate(per_sample_ratios):
                stats += f"Sample {i+1}/{len(per_sample_ratios)} token_count_ratio={ratio: .3f}\n"
            stats += f"frac_in_target: {frac_in_target}\n"
            
        msg = INJECT_BATCH_USER_TEMPLATE.format(
            turn_idx=turn_idx,
            max_turns=max_turns,
            n_samples=len(sample_ids),
            ratio_lower_bound=RATIO_LOWER_BOUND,
            ratio_upper_bound=RATIO_UPPER_BOUND,
            stats=stats
        )
        # if self.best_trajectory:
        #     msg += "\n\nBest universal trajectory so far:\n" + _format_trajectory(self.best_trajectory)
        #     msg += f"\nBest performance: {self.best_perf}"
        injection = self._call(msg).strip()
        
        print("inject message:", msg)
        return injection

    @staticmethod
    def _objective(perf: dict | None) -> tuple:
        if not perf:
            return (-1.0, float("-inf"))
        return perf.get("frac_in_target", 0.0)

    def record_batch_trajectory(self, realized_trajectory: list[str], final_ratios: list[float]) -> bool:
        """Record current minibatch outcome and update global best if improved."""
        self.current_trajectory = list(realized_trajectory)
        self.current_perf = aggregate(final_ratios)
        improved = self.best_perf is None or self._objective(self.current_perf) > self._objective(self.best_perf)
        if improved:
            self.best_trajectory = list(realized_trajectory)
            self.best_perf = dict(self.current_perf)
        return improved

    def reflect_batch(self, trajectories: list, batch_idx: int,
                      ratio_matrix: list[list[float]], sample_ids: list[str],
                      realized_trajectory: list[str]) -> str:
        final_ratios = [t.final_token_count_ratio for t in trajectories]
        stats = aggregate(final_ratios)
        summaries = []
        for sid, tr in zip(sample_ids, trajectories):
            summaries.append(
                f"sample_id={sid} success={tr.success} success_turn={tr.success_turn if tr.success else 'N/A'} "
                f"turns={tr.num_injection_turns} final_ratio={tr.final_token_count_ratio:.3f}\n"
                f"{tr.trajectory_log()}"
            )
        msg = REFLECT_BATCH_USER_TEMPLATE.format(
            batch_idx=batch_idx,
            n_samples=len(sample_ids),
            frac_in_target=stats["frac_in_target"],
            ratio_matrix=_format_ratio_matrix(ratio_matrix, sample_ids),
            realized_trajectory=_format_trajectory(realized_trajectory),
        )
        
        print("reflect message:", msg)
        
        return self._call(msg)

    def distill(self, batch_idx: int, ratios: list[float], overlaps: list[float],
                successes: list[bool], realized_trajectory: list[str] | None = None) -> str:
        """Rewrite experience.md with the latest universal-trajectory lessons."""
        realized_trajectory = realized_trajectory or []
        stats = aggregate(ratios)
        success_count = sum(1 for s in successes if s)
        msg = DISTILL_USER_TEMPLATE.format(
            batch_idx=batch_idx,
            previous_experience=self.experience_md,
            realized_trajectory=_format_trajectory(realized_trajectory),
            token_count_ratio_list=[round(r, 3) for r in ratios],
            success_rate=f"{success_count}/{len(successes)}" if successes else "0/0",
            frac_in_target=stats["frac_in_target"],
            ratio_summary=stats,
            overlap_list=[round(o, 3) for o in overlaps],
            best_perf=self.best_perf,
            current_perf=self.current_perf,
        )
        
        print("distill message:", msg)
        
        new_experience = self._call(msg).strip()
        self.experience_md = new_experience
        if self.experience_path:
            with open(self.experience_path, "w", encoding="utf-8") as f:
                f.write(new_experience)
        self._init_conversation()
        return self.experience_md
