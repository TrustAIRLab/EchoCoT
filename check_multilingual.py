#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator


# 至少出现多少个汉字才报告。
# 设置为 1 表示只要出现一个汉字就报告。
MIN_CHINESE_CHARS = 1

# 输出片段中，在中文前后各保留多少字符。
CONTEXT_LENGTH = 150


# CJK Unified Ideographs:
#   \u3400-\u4DBF   Extension A
#   \u4E00-\u9FFF   Basic CJK
#   \uF900-\uFAFF   Compatibility Ideographs
#   U+20000 以上    CJK Extensions B–F 等
CHINESE_PATTERN = re.compile(
    "["
    "\u3400-\u4DBF"
    "\u4E00-\u9FFF"
    "\uF900-\uFAFF"
    "\U00020000-\U0002EBEF"
    "]"
)


def load_records(path: Path) -> Iterator[tuple[int, Any]]:
    """逐条读取 JSON 或 JSONL 文件。"""

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    print(
                        f"[WARN] Invalid JSONL: "
                        f"{path}:{line_number}: {exc}"
                    )

    elif path.suffix.lower() == ".json":
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
            return

        if isinstance(data, list):
            for index, record in enumerate(data):
                yield index, record
        else:
            yield 0, data


def iter_strings(
    value: Any,
    json_path: str = "$",
) -> Iterator[tuple[str, str]]:
    """递归提取 JSON 中的所有字符串及其 JSON 路径。"""

    if isinstance(value, str):
        yield json_path, value

    elif isinstance(value, dict):
        for key, child in value.items():
            yield from iter_strings(
                child,
                f"{json_path}.{key}",
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_strings(
                child,
                f"{json_path}[{index}]",
            )


def get_sample_id(record: Any) -> Any:
    if not isinstance(record, dict):
        return None

    return record.get("sample_id", record.get("id"))


def should_exclude(path: Path, record: Any) -> bool:
    """
    排除 OpenThoughts 数据集中的 sample_id == 99399。
    同时兼容整数 99399 和字符串 "99399"。
    """

    if not isinstance(record, dict):
        return False

    if "openthoughts" not in str(path).lower():
        return False

    return str(record.get("sample_id")) == "99399"


def find_chinese(text: str) -> list[re.Match]:
    return list(CHINESE_PATTERN.finditer(text))


def build_snippet(
    text: str,
    first_match: re.Match,
) -> str:
    """提取第一处汉字附近的文本。"""

    start = max(0, first_match.start() - CONTEXT_LENGTH)
    end = min(len(text), first_match.end() + CONTEXT_LENGTH)

    snippet = text[start:end]
    snippet = re.sub(r"\s+", " ", snippet).strip()

    if start > 0:
        snippet = "..." + snippet

    if end < len(text):
        snippet += "..."

    return snippet


def scan_file(path: Path) -> list[dict[str, Any]]:
    findings = []

    for record_index, record in load_records(path):
        if should_exclude(path, record):
            continue

        sample_id = get_sample_id(record)

        for json_path, text in iter_strings(record):
            matches = find_chinese(text)

            if len(matches) < MIN_CHINESE_CHARS:
                continue

            chinese_text = "".join(
                match.group(0)
                for match in matches
            )

            findings.append(
                {
                    "file": str(path),
                    "record_index": record_index,
                    "sample_id": sample_id,
                    "json_path": json_path,
                    "chinese_char_count": len(matches),
                    "chinese_text": chinese_text,
                    "snippet": build_snippet(text, matches[0]),
                }
            )

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find Chinese characters "
            "in JSON and JSONL files."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to scan.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chinese_report.jsonl"),
        help="Output JSONL report.",
    )

    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {root}"
        )

    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl"}
        and path.resolve() != output_path
    )

    print(f"Found {len(files)} JSON/JSONL files.")
    print(f"Scanning: {root}")

    total_findings = 0
    affected_samples: set[tuple[str, str]] = set()
    affected_files: set[str] = set()

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, path in enumerate(files, start=1):
            findings = scan_file(path)

            if not findings:
                print(f"[{index}/{len(files)}] OK: {path}")
                continue

            print(
                f"[{index}/{len(files)}] CHINESE: "
                f"{path} ({len(findings)} fields)"
            )

            affected_files.add(str(path))
            total_findings += len(findings)

            for finding in findings:
                affected_samples.add(
                    (
                        finding["file"],
                        str(finding["sample_id"]),
                    )
                )

                output_file.write(
                    json.dumps(
                        finding,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print("\nDone.")
    print(f"Files scanned: {len(files)}")
    print(f"Files containing Chinese: {len(affected_files)}")
    print(f"Samples containing Chinese: {len(affected_samples)}")
    print(f"Fields containing Chinese: {total_findings}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()