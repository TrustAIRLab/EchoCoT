"""DeepSeek target: API client + one chat round trip.

Adding a new target = drop in `extract_<model>.py` with the same two functions
and register it in `run_gepa.TARGET_EXTRACTORS`.
"""

import json
import os
import re

from openai import OpenAI


NAME = "deepseek-reasoner"
BASE_URL = "https://api.deepseek.com"


def _safe_json_loads(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    prefix = '{"content":"'
    if fixed.startswith(prefix):
        return {"content": fixed[len(prefix):]}
    return {"content": fixed}


def make_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY for the DeepSeek target")
    return OpenAI(api_key=api_key, base_url=BASE_URL), NAME


def chat(client, model_id: str, messages: list, tools: list):
    """One round trip. Returns (reasoning, tool_calls, assistant_echo).

    - reasoning: the model's hidden CoT text (DeepSeek's `reasoning_content`)
    - tool_calls: list of {id, name, args} with args already parsed to dict
    - assistant_echo: the message dict to append back into `messages` so
      reasoning_content stays in context for the next turn
    """
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=81920,  # max new tokens
        extra_body={"thinking": {"type": "enabled"}},
    )
    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    raw_tool_calls = msg.tool_calls or []

    tool_calls = []
    for tc in raw_tool_calls:
        args = _safe_json_loads(tc.function.arguments or "{}")
        if not isinstance(args, dict):
            args = {"content": str(args)}
        tool_calls.append({"id": tc.id, "name": tc.function.name, "args": args})

    assistant_echo = {
        "role": "assistant",
        "content": msg.content or "",
        "reasoning_content": reasoning,
        "tool_calls": [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in raw_tool_calls
        ] if raw_tool_calls else None,
    }
    return reasoning, tool_calls, assistant_echo
