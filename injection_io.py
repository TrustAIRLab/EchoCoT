"""Read reusable injection trajectories for evaluation and inference."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_TURN_PATTERN = re.compile(
    r"^Turn(?:\s+|-)(\d+):\s*\n?"
    r"(.*?)"
    r"(?=^Turn(?:\s+|-)\d+:\s*|^#{1,6}\s*Evaluation Metrics\b|"
    r"^Evaluation Metrics\b|^---\s*$|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _normalize_json_injections(data: Any) -> list[dict]:
    if isinstance(data, dict):
        data = data.get("injections", data.get("trajectory"))
    if not isinstance(data, list):
        raise ValueError("Injection JSON must be a list or contain an 'injections' list")

    injections = []
    for index, item in enumerate(data, start=1):
        if isinstance(item, str):
            text = item.strip()
            turn_idx = index
        elif isinstance(item, dict):
            text = str(item.get("injection", item.get("content", ""))).strip()
            turn_idx = int(item.get("turn_idx", index))
        else:
            raise ValueError("Each injection must be a string or object")
        if text:
            injections.append({"turn_idx": turn_idx, "injection": text})
    return injections


def parse_markdown_injections(text: str) -> list[dict]:
    """Parse both ``Turn 1:`` and ``Turn-1:`` trajectory headings."""
    return [
        {"turn_idx": int(match.group(1)), "injection": match.group(2).strip()}
        for match in _TURN_PATTERN.finditer(text)
        if match.group(2).strip()
    ]


def load_injections(
    path: str | Path,
    *,
    allow_plain_text: bool = False,
) -> list[dict]:
    """Load JSON trajectories or the Turn 1:/Turn-1: Markdown format."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json":
        return _normalize_json_injections(json.loads(text))

    injections = parse_markdown_injections(text)
    if injections:
        return injections
    if allow_plain_text and text.strip():
        return [{"turn_idx": 1, "injection": text.strip()}]
    return []
