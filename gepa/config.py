"""Model registries, constants, thresholds."""

import os

# ---------------------------------------------------------------------------
# Target models (open-source, local HF inference)
# ---------------------------------------------------------------------------

TARGET_MODEL_OPTIONS = {
    "gemma": {
        "provider": "local",
        "model": "google/gemma-4-26B-A4B-it",
        "save_dir": "/home/c01yiqu/CISPA-az6/safe_vlm-2024/Data/LLMs/gemma-4-26B-A4B-it",
        "runner": "gemma",
        "notes": "Google Gemma 4 26B-A4B-it (local HF inference)",
    },
    "oss": {
        "provider": "local",
        "model": "openai/gpt-oss-20b",
        "save_dir": "/home/c01yiqu/CISPA-az6/safe_vlm-2024/Data/LLMs/gpt-oss-20b",
        "runner": "oss",
        "notes": "OpenAI GPT-OSS 20B (local HF, openai_harmony)",
    },
    "deepseek-reasoner": {
        "provider": "DeepSeek API",
        "model": "deepseek-reasoner",
        "save_dir": "",
        "runner": "deepseek",
        "notes": "DeepSeek Reasoner (commercial API target)",
    },
    "o4-mini": {
        "provider": "OpenAI API",
        "model": "o4-mini",
        "save_dir": "",
        "runner": "gpt",
        "notes": "OpenAI o4-mini (commercial API target)",
    },
    "gemini-3": {
        "provider": "Google API",
        "model": "gemini-3-flash-preview",
        "save_dir": "",
        "runner": "gemini",
        "notes": "Google Gemini 3.1 Pro (commercial API target)",
    },
    "gemini-2.5": {
        "provider": "Google API",
        "model": "gemini-2.5-pro",
        "save_dir": "",
        "runner": "gemini",
        "notes": "Google Gemini 2.5 Pro (commercial API target)",
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
# GEPA training loop
# ---------------------------------------------------------------------------

# Trajectory shape
L = 2                       # number of injection turns per trajectory
PARETO_SET_SIZE = 10        # fixed D_pareto size

# GEPA loop
BUDGET = 50                 # outer iterations
MINIBATCH_SIZE = 4          # samples drawn from D_feedback per iteration
TAU = 0.05                  # winner-set tolerance (readme §9.1)
EPSILON = 0.01              # child acceptance margin (readme §12)

DEFAULT_TARGET_MODEL = "deepseek-reasoner"
DEFAULT_OPTIMIZER_MODEL = "gpt-4.1"

# Optimizer choices restricted to GPT entries (user requirement)
GPT_OPTIMIZER_KEYS = ["gpt-4.1", "gpt-4.1-mini", "o4-mini"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "../data/openthoughts_sample.json")
SAMPLES_PER_SOURCE = 5
OUTPUT_DIR = os.path.join(_HERE, "results")
