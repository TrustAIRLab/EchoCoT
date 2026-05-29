"""Call the optimizer LLM to rewrite one selected turn of a trajectory."""

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
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
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


def _format_trajectory(parent_trajectory: dict) -> str:
    turns = parent_trajectory["turns"]
    lines = []
    for k in sorted(turns.keys(), key=lambda x: int(x)):
        lines.append(f"turn{k}: {turns[k]}")
    return "\n\n".join(lines)


def _fmt_ratio(x) -> str:
    if x is None:
        return "  -  "
    return f"{x:.2f}"


def _format_feedback_table(feedback_batch: list, num_turns: int) -> str:
    """Render per-sample per-turn ratios as a markdown table.

    Columns: sample_id, source, target_tokens, initial, turn1 .. turnN.
    `initial` is the model's first scratchpad archive before any injection
    (scratchpad_call_idx == 0). `turnK` is the archive after injection
    turn K (scratchpad_call_idx == K). Cells are token_count_ratio rounded
    to 2 decimals.
    """
    turn_cols = [f"turn{i + 1}" for i in range(num_turns)]
    header = ["sample_id", "source", "target_tokens", "initial"] + turn_cols
    sep_align = ["---", "---", "---:", "---:"] + ["---:"] * num_turns

    rows = []
    rows.append("| " + " | ".join(header) + " |")
    rows.append("| " + " | ".join(sep_align) + " |")

    for entry in feedback_batch:
        trace = entry.get("trace") or []
        per_turn = {t["scratchpad_call_idx"]: t.get("token_count_ratio_at_turn")
                    for t in trace}
        cells = [_fmt_ratio(per_turn.get(0))]
        cells += [_fmt_ratio(per_turn.get(k)) for k in range(1, num_turns + 1)]
        row = [
            str(entry.get("sample_id", "")),
            str(entry.get("source", "")),
            str(entry.get("target_token_count", "")),
        ] + cells
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)


def mutate_turn(
    opt_client,
    opt_model: str,
    opt_api_type: str,
    parent_trajectory: dict,
    target_turn: int,
    feedback_batch: list,
) -> dict:
    num_turns = len(parent_trajectory["turns"])
    trajectory_display = _format_trajectory(parent_trajectory)
    feedback_table = _format_feedback_table(feedback_batch, num_turns)

    user_msg = USER_TEMPLATE.format(
        trajectory_display=trajectory_display,
        target_turn=target_turn,
        current_turn=parent_trajectory["turns"][target_turn],
        feedback_table=feedback_table,
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
    diagnosis = (parsed.get("diagnosis") or "").strip()

    print("optimizer raw output:", raw)
    print()

    if not mutated:
        mutated = parent_trajectory["turns"][target_turn]
        reason = reason or "optimizer returned no mutated_turn; reusing parent"

    return {
        "mutated_turn": mutated,
        "mutation_reason": reason,
        "diagnosis": diagnosis,
        "raw": raw,
    }
