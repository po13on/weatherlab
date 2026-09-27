"""Rule-based heading and paragraph chunks. No model call.

A parent is one heading section or paragraph, after neighbors that are too
short have been merged. A parent is not merged across a heading. Children are
smaller slices of that same parent and exist only so retrieval can hit a
sentence without throwing away the surrounding paragraph.
"""

from __future__ import annotations

import re
from typing import Any

MIN_PARENT_CHARS = 80
CHILD_TARGET_CHARS = 180
MIN_CHILD_CHARS = 40

_MARKDOWN_HEADING = re.compile(r"#{1,6}\s+\S.*")
_COLON_HEADING = re.compile(r"[\w\u4e00-\u9fff][\w\u4e00-\u9fff \-_/]{0,40}[:：]")
_SENTENCE_BREAK = re.compile(r"[.!?。！？](?:\s+|$)")


def is_heading_line(content: str) -> bool:
    text = content.strip()
    if not text or len(text) > 80:
        return False
    if _MARKDOWN_HEADING.fullmatch(text):
        return True
    return _COLON_HEADING.fullmatch(text) is not None


def heading_title(content: str) -> str:
    text = content.strip()
    if text.startswith("#"):
        return text.lstrip("#").strip()
    return text.rstrip(":：").strip()


def page_title(text: str) -> str | None:
    """Title of a page when its first non-empty line is a heading."""

    for line in text.splitlines():
        content = line.strip()
        if not content:
            continue
        if is_heading_line(content):
            title = heading_title(content)
            return title or None
        return None
    return None


def split_parents(text: str) -> list[dict[str, Any]]:
    """Split ``text`` into parents. Offsets refer to ``text`` and cover each slice."""

    blocks = _merge_short_blocks(text, _raw_blocks(text), MIN_PARENT_CHARS)
    parents: list[dict[str, Any]] = []
    for start, end in blocks:
        slice_text = text[start:end]
        if not slice_text.strip():
            continue
        children = _child_slices(text, start, end)
        if not children:
            children = [{"text": slice_text.strip(), "start": start, "end": end}]
        parents.append(
            {
                "text": slice_text.strip(),
                "start": start,
                "end": end,
                "children": children,
            }
        )
    return parents


def build_page_chunks(
    page_no: int,
    kind: str,
    origin: str,
    text: str,
    title: str | None = None,
) -> list[dict[str, Any]]:
    """Parents for one page, with child slices kept for retrieval."""

    chunks: list[dict[str, Any]] = []
    for index, parent in enumerate(split_parents(text)):
        parent_id = f"p{page_no}-{kind}-{index}"
        children: list[dict[str, Any]] = []
        for child_index, child in enumerate(parent["children"]):
            child_text = str(child["text"] or "").strip()
            if not child_text:
                continue
            children.append(
                {
                    "chunk_id": f"{parent_id}-c{child_index}",
                    "text": child_text,
                    "start": int(child["start"]),
                    "end": int(child["end"]),
                }
            )
        if not children:
            continue
        metadata: dict[str, Any] = {
            "page": page_no,
            "kind": kind,
            "source": "pdf",
            "origin": origin,
        }
        if title:
            metadata["title"] = title
        chunks.append(
            {
                "chunk_id": parent_id,
                "text": parent["text"],
                "start": int(parent["start"]),
                "end": int(parent["end"]),
                "metadata": metadata,
                "children": children,
            }
        )
    return chunks


def _raw_blocks(text: str) -> list[tuple[int, int]]:
    lines = _lines(text)
    blocks: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for start, end, line in lines:
        content = line.strip()
        if not content:
            if current is not None:
                blocks.append(current)
                current = None
            continue
        if is_heading_line(content) and current is not None:
            blocks.append(current)
            current = (start, end)
            continue
        if current is None:
            current = (start, end)
        else:
            current = (current[0], end)
    if current is not None:
        blocks.append(current)
    return blocks


def _merge_short_blocks(
    text: str,
    blocks: list[tuple[int, int]],
    min_chars: int,
) -> list[tuple[int, int]]:
    if not blocks:
        return []
    merged: list[tuple[int, int]] = [blocks[0]]
    for start, end in blocks[1:]:
        prev_start, prev_end = merged[-1]
        prev_short = len(text[prev_start:prev_end].strip()) < min_chars
        if prev_short and not _starts_with_heading(text, start, end):
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    if len(merged) >= 2:
        start, end = merged[-1]
        if len(text[start:end].strip()) < min_chars and not _starts_with_heading(text, start, end):
            prev_start, _prev_end = merged[-2]
            merged[-2] = (prev_start, end)
            merged.pop()
    return merged


def _starts_with_heading(text: str, start: int, end: int) -> bool:
    for line in text[start:end].splitlines():
        if line.strip():
            return is_heading_line(line)
    return False


def _lines(text: str) -> list[tuple[int, int, str]]:
    rows: list[tuple[int, int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        rows.append((offset, offset + len(line), line))
        offset += len(line)
    if text and not text.endswith(("\n", "\r")) and rows:
        return rows
    return rows


def _child_slices(text: str, start: int, end: int) -> list[dict[str, Any]]:
    segment = text[start:end]
    relative = _sentence_spans(segment)
    pieces: list[tuple[int, int]] = []
    for left, right in relative:
        pieces.extend(_split_long(segment, left, right, CHILD_TARGET_CHARS))
    pieces = _merge_short_spans(segment, pieces, MIN_CHILD_CHARS)
    children: list[dict[str, Any]] = []
    for left, right in pieces:
        snippet = segment[left:right]
        if not snippet.strip():
            continue
        children.append(
            {
                "text": snippet.strip(),
                "start": start + left,
                "end": start + right,
            }
        )
    return children


def _sentence_spans(segment: str) -> list[tuple[int, int]]:
    if not segment:
        return []
    spans: list[tuple[int, int]] = []
    last = 0
    for match in _SENTENCE_BREAK.finditer(segment):
        spans.append((last, match.end()))
        last = match.end()
    if last < len(segment):
        spans.append((last, len(segment)))
    if not spans:
        return [(0, len(segment))]
    return spans


def _split_long(segment: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > limit:
        window_end = min(cursor + limit, end)
        space = segment.rfind(" ", cursor, window_end)
        if space <= cursor + limit // 4:
            split_at = window_end
        else:
            split_at = space + 1
        if split_at <= cursor:
            split_at = min(cursor + limit, end)
        spans.append((cursor, split_at))
        cursor = split_at
    if cursor < end:
        spans.append((cursor, end))
    return spans


def _merge_short_spans(
    segment: str,
    spans: list[tuple[int, int]],
    min_chars: int,
) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if len(segment[prev_start:prev_end].strip()) < min_chars:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    if len(merged) >= 2:
        start, end = merged[-1]
        if len(segment[start:end].strip()) < min_chars:
            prev_start, _prev_end = merged[-2]
            merged[-2] = (prev_start, end)
            merged.pop()
    return merged
