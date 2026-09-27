"""Detect two retrieved passages that disagree on the fact the question asks.

The check is a rule on the passages already retrieved. It does not call a model.
One passage, or two passages that name the same value, is not a contradiction.
"""

from __future__ import annotations

import re
from typing import Any

_WHAT_IS = re.compile(r"(?i)^what\s+is\s+the\s+(.+?)\s*\??$")
_CN_WHAT = re.compile(r"^([\u4e00-\u9fffA-Za-z0-9]{1,16})是(?:多少|什么)")


def passages_contradict(query: str, hits: list[dict[str, Any]]) -> bool:
    attribute = asked_attribute(query)
    if not attribute or len(hits) < 2:
        return False
    stated: list[str] = []
    for hit in hits:
        values = {_norm(value) for value in extract_values(attribute, str(hit.get("text") or ""))}
        values.discard("")
        if len(values) != 1:
            continue
        stated.append(next(iter(values)))
    return len(stated) >= 2 and len(set(stated)) >= 2


def asked_attribute(query: str) -> str | None:
    text = " ".join((query or "").strip().split())
    match = _WHAT_IS.match(text)
    if match:
        attribute = re.sub(r"\s+", " ", match.group(1)).strip(" ?？")
        return attribute or None
    match = _CN_WHAT.match(text)
    if match:
        return match.group(1)
    return None


def extract_values(attribute: str, text: str) -> list[str]:
    escaped = re.escape(attribute)
    patterns = [
        rf"(?i)(?:\bthe\s+)?\b{escaped}\s+is\s+([^.\n;]+)",
        rf"(?i)\b{escaped}\s*[:：]\s*([^.\n;]+)",
        rf"{escaped}(?:是|为)\s*([^。\n；]+)",
        rf"{escaped}\s*[:：]\s*([^。\n；]+)",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(match.group(1).strip() for match in re.finditer(pattern, text))
    return found


def _norm(value: str) -> str:
    text = value.strip().strip("\"'“”").rstrip(".。")
    return re.sub(r"\s+", " ", text).casefold()
