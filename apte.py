"""TrajectoryOptimizer for minibatch-aggregated universal injections."""

import importlib
import os

from openai import OpenAI

from config import (
    MAX_INJECTION_TURNS,
    RATIO_LOWER_BOUND,
    RATIO_UPPER_BOUND,
    WORKFLOW_MODE,
)

# Module-level state (set by main.py before use)
client: OpenAI = None
model_id: str = ""
api_type: str = "responses"     # "chat" or "responses"
max_turns: int = MAX_INJECTION_TURNS
prompt_module_name: str = "prompts"
workflow_mode: str = WORKFLOW_MODE


OPTIMIZER_SYSTEM_PROMPT = ""
INJECT_BATCH_USER_TEMPLATE = ""
REFLECT_BATCH_USER_TEMPLATE = ""
DISTILL_USER_TEMPLATE = ""
INITIAL_EXPERIENCE_MD = ""


def _normalize_prompt_module(name: str) -> str:
    """Return the importable prompt module name for supported aliases."""
    name = os.path.splitext(os.path.basename(name))[0]
    return name
    # aliases = {
    #     "prompts": "prompts",
    #     "prompts_cot": "prompts_cot",
    #     "prompts_cot_gpt": "prompts_cot_gpt",
    #     "prompts_with_summary": "prompts_with_summary",
    # }
    # try:
    #     return aliases[name]
    # except KeyError as exc:
    #     raise ValueError("prompt module must be one of: prompts, prompts_cot") from exc


def _load_prompt_templates(module_name: str | None = None):
    """Load minibatch prompt templates."""
    global OPTIMIZER_SYSTEM_PROMPT, INJECT_BATCH_USER_TEMPLATE
    global REFLECT_BATCH_USER_TEMPLATE, DISTILL_USER_TEMPLATE
    global INITIAL_EXPERIENCE_MD, prompt_module_name

    if module_name is not None:
        prompt_module_name = _normalize_prompt_module(module_name)

    prompts_module = importlib.import_module(prompt_module_name)

    OPTIMIZER_SYSTEM_PROMPT = prompts_module.OPTIMIZER_SYSTEM_PROMPT
    INJECT_BATCH_USER_TEMPLATE = prompts_module.INJECT_BATCH_USER_TEMPLATE
    REFLECT_BATCH_USER_TEMPLATE = prompts_module.REFLECT_BATCH_USER_TEMPLATE
    DISTILL_USER_TEMPLATE = prompts_module.DISTILL_USER_TEMPLATE
    INITIAL_EXPERIENCE_MD = prompts_module.INITIAL_EXPERIENCE_MD

_load_prompt_templates()


def init(openai_client: OpenAI, model: str, model_api_type: str,
         max_injection_turns: int, prompt_module: str = "prompts", mode: str = "length_only"):
    """Set the optimizer's API client, model, API type, and prompt module."""
    global client, model_id, api_type, max_turns, workflow_mode
    client = openai_client
    model_id = model
    api_type = model_api_type
    max_turns = max_injection_turns
    workflow_mode = mode
    _load_prompt_templates(prompt_module)


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

    def inject_batch(self, turn_idx: int, per_sample_ratios: list[float], scratchpad_preview: str, cot_preview:str, trajectories: list | None = None) -> str:
        """Generate one universal injection for the full minibatch."""

        per_sample_ratios_str = ""
        for i, score in enumerate(per_sample_ratios):
            if workflow_mode == "length_only":
                per_sample_ratios_str += f"Sample {i+1}: length_error={score[0]: .3f}\n"
            else:
                length_error, summary_token_recall, composite_score = score
                if summary_token_recall == 0.0 and composite_score == 0.0:
                    per_sample_ratios_str += f"Sample {i+1}: length_error={length_error: .3f} summary_token_recall=N/A composite_score=N/A\n"
                else:
                    per_sample_ratios_str += f"Sample {i+1}: length_error={length_error: .3f} summary_token_recall={summary_token_recall: .3f} composite_score={composite_score: .3f}\n"

        if turn_idx <=1:
            cot_preview = "No reaction yet, as the first injection hasn't been sent."

        msg = INJECT_BATCH_USER_TEMPLATE.format(
            turn_idx=turn_idx,
            max_turns=max_turns,
            scratchpad_preview=scratchpad_preview,
            cot_preview=cot_preview,
            per_sample_ratios=per_sample_ratios_str
        )
        injection = self._call(msg).strip()
        
        print("inject message:", msg)
        return injection

    @staticmethod
    def _is_improved(current_perf, best_perf) -> bool:
        if best_perf is None:
            return True

        if workflow_mode == "length_only":
            current_frac = current_perf["frac_in_target"]
            best_frac = best_perf["frac_in_target"]

            if current_frac > best_frac:
                return True
            if current_frac < best_frac:
                return False

            return current_perf["mean_length_error"] < best_perf["mean_length_error"]

        elif workflow_mode == "length_lexical":
            current_frac = current_perf["frac_in_target"]
            best_frac = best_perf["frac_in_target"]

            if current_frac > best_frac:
                return True
            if current_frac < best_frac:
                return False

            return current_perf["mean_composite_score"] > best_perf["mean_composite_score"]

        else:
            current_score = current_perf["mean_composite_score"]
            best_score = best_perf["mean_composite_score"]

            if current_score > best_score:
                return True
            if current_score < best_score:
                return False

            return current_perf["mean_length_error"] < best_perf["mean_length_error"]

    def record_batch_trajectory(self, realized_trajectory: list[str], mean_ratio_matrix, frac_in_target) -> bool:
        """Record current minibatch outcome and update global best if improved."""
        self.current_trajectory = list(realized_trajectory)
        if workflow_mode == "length_only":
            self.current_perf = {"mean_length_error": mean_ratio_matrix[0], "frac_in_target": frac_in_target}

        # elif workflow_mode == "length_lexical":
        #     self.current_perf = {"mean_composite_score": mean_ratio_matrix[2],
        #                          "mean_length_error": mean_ratio_matrix[0],
        #                          "mean_summary_token_recall": mean_ratio_matrix[1]}
            
        elif workflow_mode == "length_lexical":
            self.current_perf = {"frac_in_target": frac_in_target,
                "mean_composite_score": mean_ratio_matrix[2]}

        improved = self._is_improved(self.current_perf, self.best_perf)
        
        if improved:
            self.best_trajectory = list(realized_trajectory)
            self.best_perf = dict(self.current_perf)
        return improved

    def reflect_batch(self, trajectories: list, batch_idx: int, per_sample_ratios: list[float],
                      ratio_matrix_table: str,
                      ratio_matrix_best_turn_table: str,
                      mean_ratio_matrix: tuple,
                      sample_ids: list[str],
                      realized_trajectory: list[str],
                      scratchpad_preview: str, 
                      cot_preview=str,
                      frac_in_target: float | None = None) -> str:
    
        per_sample_ratios_str = ""
        for i, score in enumerate(per_sample_ratios):
            if workflow_mode == "length_only":
                per_sample_ratios_str += f"Sample {i+1}: length_error={score[0]: .3f}\n"
                # frac_in_target = 
            else:
                length_error, summary_token_recall, composite_score = score
                if summary_token_recall == 0.0 and composite_score == 0.0:
                    per_sample_ratios_str += f"Sample {i+1}: length_error={length_error: .3f} summary_token_recall=N/A composite_score=N/A\n"
                else:
                    per_sample_ratios_str += f"Sample {i+1}: length_error={length_error: .3f} summary_token_recall={summary_token_recall: .3f} composite_score={composite_score: .3f}\n"

            mean_ratio_matrix_for_prompt = mean_ratio_matrix
            if workflow_mode == "length_lexical":
                mean_ratio_matrix_for_prompt = mean_ratio_matrix[2]
        
        msg = REFLECT_BATCH_USER_TEMPLATE.format(
            scratchpad_preview=scratchpad_preview,
            per_sample_ratios=per_sample_ratios_str,
            cot_preview=cot_preview,
            batch_idx=batch_idx,
            ratio_matrix_table=ratio_matrix_table,
            ratio_matrix_best_turn_table=ratio_matrix_best_turn_table,
            mean_ratio_matrix=mean_ratio_matrix_for_prompt,
            frac_in_target=frac_in_target,
            realized_trajectory=realized_trajectory,
            best_perf=self.best_perf,
            current_perf=self.current_perf
        )
        print("reflect message:", msg)
        
        return self._call(msg)

    def distill(self, batch_idx: int,
                successes: list[bool], realized_trajectory: list[str] | None = None,
                frac_in_target: float | None = None) -> str:
        """Rewrite experience.md with the latest universal-trajectory lessons."""
        realized_trajectory = realized_trajectory or []
     
        msg = DISTILL_USER_TEMPLATE.format(
            batch_idx=batch_idx,
            previous_experience=self.experience_md,
            realized_trajectory=_format_trajectory(realized_trajectory),
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
