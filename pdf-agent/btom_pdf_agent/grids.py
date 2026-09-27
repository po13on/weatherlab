"""Turn OCR lines of a three-candidate grid figure into integer matrices.

Color names are mapped only when the retrieved rule passage states the id.
Separator value 10 is dropped and never copied into an answer grid.
"""

from __future__ import annotations

import json
import re
from typing import Any

_SECTION_RE = re.compile(
    r"^(Candidate\s+[ABC]|Query|Output)\s*:?\s*$",
    re.IGNORECASE,
)
_SECTION_KEYS = {
    "A": "candidate_a",
    "B": "candidate_b",
    "C": "candidate_c",
    "QUERY": "query",
    "OUTPUT": "output",
}
_COLOR_RE = re.compile(
    r"(\d+)\s+(black|blue|red|green|yellow|grey|gray|pink|orange|teal|maroon)",
    re.IGNORECASE,
)


def parse_color_ids(rule_text: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for match in _COLOR_RE.finditer(rule_text or ""):
        name = match.group(2).lower()
        if name == "gray":
            name = "grey"
        mapping[name] = int(match.group(1))
    return mapping


def parse_grid_lines(lines: list[str], color_ids: dict[str, int] | None = None) -> dict[str, Any]:
    colors = color_ids or {}
    sections: dict[str, list[list[int]]] = {}
    current: str | None = None
    dropped_separator = 0
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        header = _SECTION_RE.match(line)
        if header:
            label = header.group(1).upper().replace("  ", " ")
            if label.startswith("CANDIDATE"):
                letter = label.split()[-1]
                current = _SECTION_KEYS[letter]
            else:
                current = _SECTION_KEYS[label.split()[0]]
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        cells: list[int] = []
        for part in line.split():
            if part.isdigit():
                value = int(part)
            else:
                value = colors.get(part.lower())
                if value is None:
                    cells = []
                    break
            if value == 10:
                dropped_separator += 1
                continue
            if value < 0 or value > 9:
                cells = []
                break
            cells.append(value)
        if cells:
            sections[current].append(cells)
    return {"grids": sections, "dropped_separator": dropped_separator}


def grids_to_json(parsed: dict[str, Any], citations: list[dict[str, Any]]) -> str:
    payload = {
        "candidates": {
            "A": parsed["grids"].get("candidate_a") or [],
            "B": parsed["grids"].get("candidate_b") or [],
            "C": parsed["grids"].get("candidate_c") or [],
        },
        "query": parsed["grids"].get("query") or [],
        "output": parsed["grids"].get("output") or [],
        "dropped_separator": parsed.get("dropped_separator") or 0,
        "citations": citations,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
