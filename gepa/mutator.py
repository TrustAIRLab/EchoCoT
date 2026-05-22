"""Call the optimizer LLM (GPT) to rewrite one selected turn of a trajectory."""

import json
import re

from optimizer_prompts import OPTIMIZER_SYSTEM_PROMPT, USER_TEMPLATE


def _safe_json_loads(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Strip code fences if present
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    # Find the first {...} block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    return {}


def _call_api(client, model_id: str, api_type: str, messages: list) -> str:
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


def _format_feedback(feedback_batch: list) -> str:
    """Strip diagnostic-only fields before showing the batch to the optimizer."""
    cleaned = []
    for entry in feedback_batch:
        cleaned.append({
            "sample_id": entry["sample_id"],
            "target_tokens": entry["target_token_count"],
            "final_output_tokens": entry["final_output_tokens"],
            "token_count_ratio": round(entry["token_count_ratio"], 4),
            "distance_to_1": round(entry["distance_to_1"], 4),
            "trace": [
                {
                    "scratchpad_call_idx": t["scratchpad_call_idx"],
                    "scratchpad_in": t["scratchpad_in"],
                    "scratchpad_in_tokens": t["scratchpad_in_tokens"],
                    "injection_sent": t["injection_sent"],
                }
                for t in entry["trace"]
            ],
        })
    return json.dumps(cleaned, ensure_ascii=False, indent=2)


def mutate_turn(
    opt_client,
    opt_model: str,
    opt_api_type: str,
    parent_trajectory: dict,
    target_turn: int,
    feedback_batch: list,
) -> dict:
    parent_turns_json = json.dumps(
        {str(k): v for k, v in parent_trajectory["turns"].items()},
        ensure_ascii=False,
        indent=2,
    )
    user_msg = USER_TEMPLATE.format(
        parent_trajectory=parent_turns_json,
        target_turn=target_turn,
        current_turn=parent_trajectory["turns"][target_turn],
        feedback_batch=_format_feedback(feedback_batch),
    )
    
    print()
    print("user message:", user_msg)
    print()
    
    messages = [
        {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    raw = _call_api(opt_client, opt_model, opt_api_type, messages)
    parsed = _safe_json_loads(raw)
    mutated = (parsed.get("mutated_turn") or "").strip()
    reason = (parsed.get("mutation_reason") or "").strip()
    
    print("optimizer raw output:", raw)
    print()
    
    if not mutated:
        # Fallback: keep parent unchanged so the acceptance gate will reject
        mutated = parent_trajectory["turns"][target_turn]
        reason = reason or "optimizer returned no mutated_turn; reusing parent"
    return {"mutated_turn": mutated, "mutation_reason": reason, "raw": raw}
