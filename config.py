"""Model registries, constants, thresholds."""

import os

# ---------------------------------------------------------------------------
# Target models
# ---------------------------------------------------------------------------

TARGET_MODEL_OPTIONS = {
    "deepseek-v4-flash": {
        "model": "deepseek-v4-flash",
        "extractor": "extractors.extract_deepseek",
        "prompts": {
            "length_only": "prompts.standard_length_only",
            "length_lexical": "prompts.standard_length_lexical",
        },
        "notes": "DeepSeek 2026-04 target hosted by OpenRouter",
    },
    "deepseek-v4-pro": {
        "model": "deepseek-v4-pro",
        "extractor": "extractors.extract_deepseek_official",
        "prompts": {
            "length_only": "prompts.deepseek_length_only",
            "length_lexical": "prompts.deepseek_length_lexical",
        },
        "notes": "DeepSeek 2026-07 target hosted by the official API",
    },
    "glm-5.2": {
        "model": "glm-5.2",
        "extractor": "extractors.extract_glm",
        "prompts": {
            "length_only": "prompts.standard_length_only",
            "length_lexical": "prompts.standard_length_lexical",
        },
        "notes": "GLM 5.2",
    },
    "qwen3.5-plus": {
        "model": "qwen3.5-plus",
        "extractor": "extractors.extract_qwen",
        "prompts": {
            "length_only": "prompts.standard_length_only",
            "length_lexical": "prompts.standard_length_lexical",
        },
        "notes": "Qwen 3.5 Plus",
    },
    "gemini-2.5": {
        "model": "gemini-2.5-flash",
        "extractor": "extractors.extract_gemini",
        "prompts": {
            "length_only": "prompts.frontier_length_only",
            "length_lexical": "prompts.frontier_length_lexical",
        },
        "notes": "Gemini 2.5 Flash",
    },
    "gemini-3.1-flash-lite": {
        "model": "gemini-3.1-flash-lite",
        "extractor": "extractors.extract_gemini",
        "prompts": {
            "length_only": "prompts.frontier_length_only",
            "length_lexical": "prompts.frontier_length_lexical",
        },
        "notes": "Gemini 3.1 Flash Lite",
    },
    "gemini-3.5": {
        "model": "gemini-3.5-flash",
        "extractor": "extractors.extract_gemini",
        "prompts": {
            "length_only": "prompts.frontier_length_only",
            "length_lexical": "prompts.frontier_length_lexical",
        },
        "notes": "Gemini 3.5 Flash",
    },
    "claude-4.6": {
        "model": "claude-opus-4-6",
        "extractor": "extractors.extract_claude",
        "prompts": {
            "length_only": "prompts.frontier_length_only",
            "length_lexical": "prompts.frontier_length_lexical",
        },
        "notes": "Claude Opus 4.6",
    },
    "sonnet-4.6": {
        "model": "claude-sonnet-4-6",
        "extractor": "extractors.extract_claude",
        "prompts": {
            "length_only": "prompts.frontier_length_only",
            "length_lexical": "prompts.frontier_length_lexical",
        },
        "notes": "Claude Sonnet 4.6",
    },
    "o4-mini": {
        "model": "o4-mini",
        "extractor": "extractors.extract_gpt",
        "prompts": {
            "length_only": "prompts.frontier_length_only",
            "length_lexical": "prompts.frontier_length_lexical",
        },
        "notes": "OpenAI o4-mini",
    },
}

OPTIMIZATION_MODES = ("length_only", "length_lexical")


def get_prompt_module(target_model_key: str, mode: str) -> str:
    if mode not in OPTIMIZATION_MODES:
        raise ValueError(f"Unknown optimization mode: {mode}")
    try:
        return TARGET_MODEL_OPTIONS[target_model_key]["prompts"][mode]
    except KeyError as exc:
        raise ValueError(f"Unknown target model: {target_model_key}") from exc

# ---------------------------------------------------------------------------
# Optimizer models (API-based)
#   api_type: "chat"      → client.chat.completions.create  (DeepSeek, GPT-4.1)
#             "responses"  → client.responses.create          (o4-mini and other reasoning models)
# ---------------------------------------------------------------------------

OPTIMIZER_MODEL_OPTIONS = {
    "deepseek-reasoner": {
        "provider": "DeepSeek", "model": "deepseek-reasoner",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY", "api_type": "chat",
        "notes": "DeepSeek R1 reasoning — default optimizer",
    },
    "deepseek-chat": {
        "provider": "DeepSeek", "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY", "api_type": "chat",
        "notes": "DeepSeek V3.2 chat — faster optimizer",
    },
    "o4-mini": {
        "provider": "OpenAI", "model": "o4-mini",
        "base_url": None, "api_key_env": "OPENAI_API_KEY",
        "api_type": "responses",
        "notes": "OpenAI o4-mini — reasoning optimizer",
    },
    "gpt-4.1-mini": {
        "provider": "OpenAI", "model": "gpt-4.1-mini",
        "base_url": None, "api_key_env": "OPENAI_API_KEY",
        "api_type": "chat",
        "notes": "OpenAI GPT-4.1 Mini — fast optimizer",
    },
    "gpt-4.1": {
        "provider": "OpenAI", "model": "gpt-4.1",
        "base_url": None, "api_key_env": "OPENAI_API_KEY",
        "api_type": "chat",
        "notes": "OpenAI GPT-4.1 — strong optimizer",
    },
    "gemini-2.5-flash": {
        "provider": "Google", "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY", "api_type": "chat",
        "notes": "Google Gemini 2.5 Flash — fast optimizer",
    },
    "gemini-2.5-pro": {
        "provider": "Google", "model": "gemini-2.5-pro",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY", "api_type": "chat",
        "notes": "Google Gemini 2.5 Pro — strong optimizer",
    },
}

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

BATCH_SIZE = 8
MAX_BATCHES = 100
EARLY_STOP_PATIENCE = 10
MAX_INJECTION_TURNS = 3

# Training inner-loop target range for token_count_ratio
RATIO_LOWER_BOUND = 0.90
RATIO_UPPER_BOUND = 1.10

DEFAULT_MODE = "length_lexical"
SUMMARY_MODEL = "gpt-4.1-mini"

DEFAULT_TARGET_MODEL = "deepseek-v4-flash"
DEFAULT_OPTIMIZER_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "data")
OUTPUT_DIR = os.path.join(_HERE, "results_ablation")
SUMMARY_PROMPT_FILE = os.path.join(_HERE, "data/summarize_prompt.txt")
