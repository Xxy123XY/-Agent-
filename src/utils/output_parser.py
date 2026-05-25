"""Robust parsers for structured LLM output.

The agents ask models for JSON, but real model output often contains markdown
fences, explanatory text, full-width wrappers, or a missing final bracket.  This
module keeps that cleanup in one conservative place instead of scattering
fragile ``json.loads`` calls across business logic.
"""

from __future__ import annotations

import json
import re
from typing import Any


class OutputParseError(ValueError):
    """Raised when model output cannot be parsed into the expected structure."""


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown code fence if the whole text is fenced."""
    value = str(text or "").strip()
    match = _FENCE_RE.match(value)
    if match:
        return match.group(1).strip()

    if value.startswith("```"):
        parts = value.split("```")
        if len(parts) >= 3:
            fenced = parts[1].strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].strip()
            return fenced
    return value


def normalize_json_text(text: str) -> str:
    """Normalize common wrapper characters around JSON without changing content."""
    value = strip_code_fence(text)
    value = value.strip().strip("\ufeff")

    # Models sometimes wrap JSON in Chinese book-title style brackets.
    wrappers = {
        "【": "】",
        "「": "」",
        "『": "』",
        "（": "）",
        "(": ")",
    }
    changed = True
    while changed and value:
        changed = False
        value = value.strip()
        for left, right in wrappers.items():
            if value.startswith(left) and value.endswith(right):
                value = value[1:-1].strip()
                changed = True
                break

    # Extra trailing full-width wrappers are common after an otherwise valid JSON.
    while value.endswith(("】", "」", "』", "）")) and not value.startswith(("【", "「", "『", "（")):
        value = value[:-1].strip()

    return value


def extract_balanced_json(text: str, open_char: str, close_char: str) -> str:
    """Extract the first balanced JSON object/array candidate from text."""
    value = normalize_json_text(text)
    start = value.find(open_char)
    if start < 0:
        raise OutputParseError(f"没有找到 JSON 起始符 {open_char!r}")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return value[start : index + 1]
            if depth < 0:
                break

    candidate = maybe_close_json(value[start:])
    if candidate != value[start:]:
        return candidate

    raise OutputParseError(f"JSON 结构不完整，无法匹配 {open_char!r}{close_char!r}")


def maybe_close_json(text: str) -> str:
    """Append missing closing brackets only when the remaining structure is clear."""
    value = normalize_json_text(text)
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}
    closing = set(pairs.values())

    for char in value:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif char in closing:
            if not stack or stack[-1] != char:
                return value
            stack.pop()

    if in_string:
        return value
    if 0 < len(stack) <= 3:
        return value + "".join(reversed(stack))
    return value


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output."""
    return _parse_expected(text, dict, "{", "}")


def extract_json_array(text: str) -> list[Any]:
    """Parse a JSON array from model output."""
    return _parse_expected(text, list, "[", "]")


def parse_json_object(
    text: str,
    defaults: dict[str, Any] | None = None,
    required_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Parse an object, merge defaults, and validate required keys."""
    data = extract_json_object(text)
    result = {**(defaults or {}), **data}

    missing = [key for key in (required_keys or []) if key not in result]
    if missing:
        raise OutputParseError(f"JSON 缺少必要字段：{', '.join(missing)}")
    return result


def parse_json_array(text: str, min_items: int = 0) -> list[Any]:
    """Parse an array and optionally validate its minimum length."""
    data = extract_json_array(text)
    if len(data) < min_items:
        raise OutputParseError(f"JSON 数组元素不足：至少需要 {min_items} 个")
    return data


def _parse_expected(text: str, expected_type: type, open_char: str, close_char: str):
    attempts = [
        normalize_json_text(text),
        maybe_close_json(normalize_json_text(text)),
    ]

    try:
        attempts.append(extract_balanced_json(text, open_char, close_char))
    except OutputParseError:
        pass

    last_error: Exception | None = None
    for candidate in _dedupe(attempts):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, expected_type):
                return parsed
            raise OutputParseError(f"期望 JSON {expected_type.__name__}，实际为 {type(parsed).__name__}")
        except (json.JSONDecodeError, OutputParseError) as exc:
            last_error = exc

    raise OutputParseError(f"无法解析模型 JSON 输出：{last_error}")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
