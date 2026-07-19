"""Model registries, constants, thresholds."""

import os

# ---------------------------------------------------------------------------
# Target models (open-source, local HF inference)
# ---------------------------------------------------------------------------

TARGET_MODEL_OPTIONS = {
    "deepseek-v4-flash": {
        "provider": "DeepSeek API",
        "model": "deepseek-v4-flash", # deepseek-reasoner
        "save_dir": "",
        "runner": "deepseek",
        "notes": "DeepSeek Reasoner (commercial API target)",
    },
    "deepseek-v4-pro": {
        "provider": "DeepSeek API",
        "model": "deepseek-v4-pro", # deepseek-reasoner
        "save_dir": "",
        "runner": "deepseek",
        "notes": "DeepSeek Reasoner (commercial API target)",
    },
    "minimax-m3": {
        "provider": "MiniMax API",
        "model": "MiniMax-M3",
        "save_dir": "",
        "runner": "minimax",
        "notes": "MiniMax M3 (commercial API target)",
    },
    "glm-5.2": {
        "provider": "GLM API",
        "model": "glm-5.2",
        "save_dir": "",
        "runner": "glm",
        "notes": "GLM 5.2 (commercial API target)",
    },
    "glm-4.7": {
        "provider": "GLM API",
        "model": "glm-4.7-flashx",
        "save_dir": "",
        "runner": "glm",
        "notes": "GLM 4.7 (commercial API target)",
    },
    "o4-mini": {
        "provider": "OpenAI API",
        "model": "o4-mini",
        "save_dir": "",
        "runner": "openai",
        "notes": "OpenAI o4-mini (commercial API target)",
    },
    "o3": {
        "provider": "OpenAI API",
        "model": "o3-2025-04-16",
        "save_dir": "",
        "runner": "openai",
        "notes": "OpenAI o4 (commercial API target)",
    },
    "gpt-5-nano": {
        "provider": "OpenAI API",
        "model": "gpt-5-nano-2025-08-07",
        "save_dir": "",
        "runner": "openai",
        "notes": "OpenAI gpt-5-mini (commercial API target)",
    },
    "gemini-3": {
        "provider": "Google API",
        "model": "gemini-3.1-pro-preview",
        "save_dir": "",
        "runner": "gemini",
        "notes": "Google Gemini 3.1 Pro (commercial API target)",
    },
    "gemini-2.5": {
        "provider": "Google API",
        "model": "gemini-2.5-flash",
        "save_dir": "",
        "runner": "gemini",
        "notes": "Google Gemini 2.5 flash (commercial API target)",
    },
    "gemini-3.1": {
        "provider": "Google API",
        "model": "gemini-3.1-flash-lite", # gemini-3.5-flash
        "save_dir": "",
        "runner": "gemini",
        "notes": "Google Gemini 3.5 Pro (commercial API target)",
    },
    "qwen3.5-plus": {
        "provider": "DashScope API",
        "model": "qwen3.5-plus",
        "save_dir": "",
        "runner": "qwen",
        "notes": "Qwen3.5 Plus — hybrid thinking, on by default (commercial)",
    },
    "qwen3.7-plus": {
        "provider": "DashScope API",
        "model": "qwen3.7-plus",
        "save_dir": "",
        "runner": "qwen",
        "notes": "Qwen3 Max",
    },
    "sonnet-4.6": {
        "provider": "Claude",
        "model": "claude-sonnet-4-6",
        "save_dir": "",
        "runner": "claude",
        "notes": "Claude Sonnet 4.6",
    },
    "opus-4.7": {
        "provider": "Claude",
        "model": "claude-opus-4-7",
        "save_dir": "",
        "runner": "claude",
        "notes": "Claude Sonnet 4.6",
    },
    "opus-4.6": {
        "provider": "Claude",
        "model": "claude-opus-4-6", # haiku can be doable, so is opus-4-6
        "save_dir": "",
        "runner": "claude",
        "notes": "Claude Sonnet 4.6",
    },
    "grok-4-1": {
        "provider": "Vertex AI (xAI)",
        "model": "grok-4-1-fast-reasoning",   # <-- add the xai/ prefix
        "save_dir": "",
        "runner": "grok",
        "notes": "Grok 4.1 Fast Reasoning — Vertex relay strips reasoning_content",
    },
    "grok-4-20": {
        "provider": "Vertex AI (xAI)",
        "model": "grok-4.20-reasoning",   # <-- add the xai/ prefix
        "save_dir": "",
        "runner": "grok",
        "notes": "Grok 4.1 Fast Reasoning — Vertex relay strips reasoning_content",
    },
    "grok-4-3": {
        "provider": "Vertex AI (xAI)",
        "model": "grok-4.3",   # <-- add the xai/ prefix
        "save_dir": "",
        "runner": "grok",
        "notes": "Grok 4.1 Fast Reasoning — Vertex relay strips reasoning_content",
    },
    # Plug for future API-based target models:
    # "o4-mini": {"provider":"openai_api", "model":"o4-mini", "runner":"openai_responses"},
}

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
MAX_INJECTION_TURNS = 2

# Training inner-loop target range for token_count_ratio
RATIO_LOWER_BOUND = 0.90
RATIO_UPPER_BOUND = 1.10

WORKFLOW_MODE = "length_lexical" # or length_only
SUMMARY_MODEL = "gpt-4.1-mini"

# After first success, allow this many extra turns for improvement
EXTRA_TURNS_AFTER_SUCCESS = 0

DEFAULT_TARGET_MODEL = "glm-4.6"
DEFAULT_OPTIMIZER_MODEL = "deepseek-reasoner"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "data")
TEST_DATA_PATH = os.path.join(_HERE, "data")
OUTPUT_DIR = os.path.join(_HERE, "results")
SUMMARY_PROMPT_FILE = os.path.join(_HERE, "data/summarize_prompt.txt")
