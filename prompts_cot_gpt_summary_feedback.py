"""GPT-CoT prompt templates for CoT-summary lexical-feedback mode."""

from prompts_cot_gpt import *  # noqa: F401,F403
from prompts_cot_gpt import OPTIMIZER_SYSTEM_PROMPT as _BASE_SYSTEM

OPTIMIZER_SYSTEM_PROMPT = _BASE_SYSTEM.replace(
    "token_count_ratio = scratchpad_tokens / hidden_reasoning_tokens.",
    "token_count_ratio = scratchpad_tokens / hidden_reasoning_tokens.\n"
    "token_recall = fraction of the concise CoT-summary vocabulary covered by the scratchpad content. Use token_recall to identify missing key reasoning content, but improve it without adding filler or moving token_count_ratio outside the target range.",
)
