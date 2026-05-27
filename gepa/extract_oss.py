"""OSS (gpt-oss-20b) target: local HF inference + one chat round trip.

Mirrors the contract of `extract_deepseek.py` so `gepa/rollout.py` can drive
this target without modification. Internally translates the chat-format
message list it receives into a Harmony `Conversation` each turn.
"""

import json
import os
import re
from pathlib import Path

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message as HarmonyMessage,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)

from config import TARGET_MODEL_OPTIONS

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from custom_tools import get_harmony_tools  # noqa: E402


NAME = "oss"

_model = None
_tokenizer = None
_encoding = None

DEVELOPER_INSTRUCTIONS = (
    "You are a helpful assistant that can answer questions and help with user's tasks.\n\n"
    "Before sending any final answer, **ALWAYS** use the scratchpad tool to "
    "archive your reasoning until it is accepted."
)


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


def load_model(model_name: str, save_dir: str):
    global _model, _tokenizer, _encoding

    from transformers import AutoModelForCausalLM, AutoTokenizer

    save_path = Path(save_dir) if save_dir else None
    model_path = model_name
    if save_path is not None and save_path.exists() and any(save_path.iterdir()):
        model_path = save_path

    print(f"[oss] Loading tokenizer from {model_path} ...")
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"[oss] Loading model from {model_path} ...")
    _model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto",
    )

    if save_path is not None and str(model_path) == model_name:
        save_path.mkdir(parents=True, exist_ok=True)
        _tokenizer.save_pretrained(save_path)
        _model.save_pretrained(save_path)

    _encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    print("[oss] Harmony encoding loaded.")


def make_client():
    if _model is None:
        cfg = TARGET_MODEL_OPTIONS["oss"]
        load_model(cfg["model"], cfg["save_dir"])
    return None, NAME


def _generate_turn(harmony_messages: list):
    """Run one assistant completion and split it by Harmony channel.

    Returns (parsed, raw_cot, raw_tool_calls, final_text):
      - parsed: list of HarmonyMessage emitted by the model
      - raw_cot: concatenated analysis-channel text
      - raw_tool_calls: [{"recipient", "args"}] for commentary-channel calls
        to `functions.scratchpad`
      - final_text: concatenated final-channel text
    """
    import torch

    convo = Conversation.from_messages(harmony_messages)
    input_ids = _encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    input_ids_t = torch.tensor([input_ids], device=_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids=input_ids_t,
            max_new_tokens=2048 * 4,
            do_sample=True,
            use_cache=True,
            temperature=1.0,
            eos_token_id=list(_encoding.stop_tokens_for_assistant_actions()),
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_token_ids = output_ids[0, input_ids_t.shape[-1]:].tolist()
    parsed = _encoding.parse_messages_from_completion_tokens(
        new_token_ids, role=Role.ASSISTANT, strict=False,
    )

    raw_cot_parts, final_parts, raw_tool_calls = [], [], []
    for msg in parsed:
        text_parts = []
        for part in getattr(msg, "content", []) or []:
            t = getattr(part, "text", None)
            if t:
                text_parts.append(t)
        text = "".join(text_parts)

        channel = getattr(msg, "channel", None)
        if channel == "analysis":
            raw_cot_parts.append(text)
        elif channel == "final":
            final_parts.append(text)

        recipient = getattr(msg, "recipient", None) or getattr(
            getattr(msg, "author", None), "name", None,
        )
        if (
            getattr(getattr(msg, "author", None), "role", None) == Role.ASSISTANT
            and channel == "commentary"
            and recipient == "functions.scratchpad"
        ):
            raw_tool_calls.append({
                "recipient": recipient,
                "args": _safe_json_loads(text.strip() or "{}"),
            })

    return (
        parsed,
        "\n\n".join(raw_cot_parts),
        raw_tool_calls,
        "\n".join(final_parts).strip(),
    )


def _to_harmony_conversation(messages: list, tools: list) -> list:
    """Translate chat-format `messages` (as passed by rollout.py) into a
    list of HarmonyMessage suitable for `Conversation.from_messages`.

    The chat-format `tools` argument is ignored; we always render the
    scratchpad spec via `get_harmony_tools`, matching how the deepseek
    extractor relies on the externally-fixed tool set.
    """
    del tools
    harmony_msgs = [
        HarmonyMessage.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_reasoning_effort(ReasoningEffort.MEDIUM),
        ),
    ]

    system_text = DEVELOPER_INSTRUCTIONS
    for m in messages:
        if m.get("role") == "system":
            system_text = m.get("content") or DEVELOPER_INSTRUCTIONS
            break

    harmony_tools = get_harmony_tools(ToolDescription, ["scratchpad"])
    harmony_msgs.append(
        HarmonyMessage.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new()
            .with_instructions(system_text)
            .with_function_tools(harmony_tools),
        )
    )

    recipients = {}
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            harmony_msgs.append(
                HarmonyMessage.from_role_and_content(Role.USER, m.get("content") or "")
            )
        elif role == "assistant":
            parsed = m.get("_harmony_parsed")
            if parsed:
                harmony_msgs.extend(parsed)
                recipients.update(m.get("_harmony_recipients") or {})
            else:
                reasoning = m.get("reasoning_content") or ""
                if reasoning:
                    harmony_msgs.append(
                        HarmonyMessage.from_role_and_content(Role.ASSISTANT, reasoning)
                        .with_channel("analysis")
                    )
                content = m.get("content") or ""
                if content:
                    harmony_msgs.append(
                        HarmonyMessage.from_role_and_content(Role.ASSISTANT, content)
                        .with_channel("final")
                    )
        elif role == "tool":
            tool_call_id = m.get("tool_call_id")
            recipient = recipients.get(tool_call_id, "functions.scratchpad")
            harmony_msgs.append(
                HarmonyMessage.from_author_and_content(
                    Author.new(Role.TOOL, recipient),
                    m.get("content") or "",
                ).with_channel("commentary")
            )

    return harmony_msgs


def chat(client, model_id: str, messages: list, tools: list):
    """One round trip. Returns (reasoning, tool_calls, assistant_echo).

    - reasoning: the model's analysis-channel CoT text
    - tool_calls: list of {id, name, args} with args already parsed to dict;
      `name` is the bare tool name (e.g. "scratchpad")
    - assistant_echo: a dict appended to `messages`; carries private
      `_harmony_parsed` / `_harmony_recipients` keys so the next turn can
      rebuild the Harmony conversation faithfully
    """
    del client, model_id

    harmony_messages = _to_harmony_conversation(messages, tools)
    parsed, raw_cot, raw_tool_calls, final_text = _generate_turn(harmony_messages)

    tool_calls = []
    recipients = {}
    for i, tc in enumerate(raw_tool_calls):
        args = tc["args"]
        if not isinstance(args, dict):
            args = {"content": str(args)}
        call_id = f"oss_tc_{i}"
        recipient = tc.get("recipient") or "functions.scratchpad"
        # Strip the Harmony "functions." prefix so rollout sees the bare name.
        name = recipient.split(".", 1)[1] if recipient.startswith("functions.") else recipient
        tool_calls.append({"id": call_id, "name": name, "args": args})
        recipients[call_id] = recipient

    assistant_echo = {
        "role": "assistant",
        "content": final_text,
        "reasoning_content": raw_cot,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"], ensure_ascii=False),
                },
            }
            for tc in tool_calls
        ] if tool_calls else None,
        "_harmony_parsed": parsed,
        "_harmony_recipients": recipients,
    }
    return raw_cot, tool_calls, assistant_echo
