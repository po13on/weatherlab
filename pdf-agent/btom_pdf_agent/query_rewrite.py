"""Offline query rewrite: pronoun expansion, two-question split, small synonyms.

No network and no API key. A pronoun is expanded only when this same query
contains one clear referent before it. The caller still searches the original
string; the list returned here is extra text, never a replacement of the input.
"""

from __future__ import annotations

import re

SYNONYMS = {
    "document": "文档",
    "文档": "document",
    "date": "日期",
    "日期": "date",
    "amount": "金额",
    "金额": "amount",
    "title": "标题",
    "标题": "title",
    "total": "合计",
    "合计": "total",
}

_PRONOUN = re.compile(
    r"\b(it|its|they|them|he|him|she|her|his|their|this|that|these|those)\b",
    re.IGNORECASE,
)
_ALWAYS_STANDALONE = {"it", "its", "they", "them", "he", "him", "she"}
_BARE_ONLY = {"her", "his", "their", "this", "that", "these", "those"}
_OBJECT = {"it", "its", "they", "them", "this", "that", "these", "those"}
_PERSON = {"he", "him", "his", "she", "her", "their"}
_NP = re.compile(r"\b(?:the|a|an)\s+([A-Za-z][A-Za-z0-9_\-]*)\b", re.IGNORECASE)
_NAME = re.compile(r"\b([A-Z][a-z]{2,})\b")
_NAME_STOP = {
    "Where",
    "What",
    "Who",
    "When",
    "Why",
    "How",
    "Does",
    "Did",
    "The",
    "This",
    "That",
    "There",
    "These",
    "Those",
    "She",
    "Her",
    "His",
    "They",
    "Them",
    "And",
}
_CN_PRONOUN = re.compile(r"它|他们|她们|它们|他|她")
_CN_REF = re.compile(r"^([\u4e00-\u9fff]{2,8})(?=(?:在|是|有|的))")
_CN_SPLIT = re.compile(r"[，,。！？?!]")


def rewrite_query(query: str) -> list[str]:
    """Extra queries. The original string is not included."""

    text = query or ""
    extras: list[str] = []
    expanded = expand_pronouns(text)
    if expanded and expanded != text:
        extras.append(expanded)
    extras.extend(part for part in split_questions(text) if part != text)
    synonym_bases = [text, *extras]
    for base in synonym_bases:
        variant = synonym_variant(base)
        if variant and variant != base:
            extras.append(variant)
    seen = {text}
    unique: list[str] = []
    for item in extras:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def expand_pronouns(query: str) -> str | None:
    """Replace a pronoun only when the query itself names one referent before it."""

    if not query:
        return None
    updated = _expand_english(query)
    updated = _expand_chinese(updated)
    if updated == query:
        return None
    return updated


def split_questions(query: str) -> list[str]:
    """Split when the query contains two question marks. One question stays whole."""

    text = query or ""
    if text.count("?") + text.count("？") < 2:
        return []
    parts = [part.strip() for part in re.split(r"[?？]", text) if part.strip()]
    if len(parts) < 2:
        return []
    mark = "？" if "？" in text else "?"
    return [part + mark for part in parts]


def synonym_variant(query: str) -> str | None:
    """One alternate string using the fixed synonym table. No model and no network."""

    if not query:
        return None
    pieces: list[str] = []
    index = 0
    changed = False
    keys = sorted(SYNONYMS, key=len, reverse=True)
    lowered = query.lower()
    while index < len(query):
        matched = None
        for key in keys:
            if re.search(r"[A-Za-z]", key):
                size = len(key)
                if lowered.startswith(key.lower(), index) and _english_bounds(query, index, index + size):
                    matched = key
                    break
            elif query.startswith(key, index):
                matched = key
                break
        if matched:
            pieces.append(SYNONYMS[matched])
            index += len(matched)
            changed = True
        else:
            pieces.append(query[index])
            index += 1
    if not changed:
        return None
    return "".join(pieces)


def _expand_english(query: str) -> str:
    replacements: list[tuple[int, int, str]] = []
    for match in _PRONOUN.finditer(query):
        if not _standalone(query, match):
            continue
        word = match.group(0).lower()
        before = query[: match.start()]
        if word in _OBJECT:
            referent = _single_noun(before)
        elif word in _PERSON:
            referent = _single_name(before)
        else:
            referent = None
        if not referent:
            continue
        replacements.append((match.start(), match.end(), referent))
    if not replacements:
        return query
    updated = query
    for start, end, referent in reversed(replacements):
        updated = updated[:start] + referent + updated[end:]
    return updated


def _expand_chinese(query: str) -> str:
    raw_clauses = [part.strip() for part in _CN_SPLIT.split(query) if part.strip()]
    if len(raw_clauses) < 2:
        return query
    updated = query
    seen: list[str] = []
    for clause in raw_clauses:
        pronoun = _CN_PRONOUN.match(clause)
        if pronoun:
            unique = []
            for item in seen:
                if item not in unique:
                    unique.append(item)
            if len(unique) == 1:
                replacement = clause.replace(pronoun.group(0), unique[0], 1)
                if replacement != clause:
                    updated = _replace_once(updated, clause, replacement)
            continue
        found = _cn_referent(clause)
        if found:
            seen.append(found)
    return updated


def _cn_referent(clause: str) -> str | None:
    text = clause.strip()
    if not text or any(mark in text for mark in ("和", "与", "或")):
        return None
    match = _CN_REF.match(text)
    if not match:
        return None
    return match.group(1)


def _replace_once(text: str, source: str, replacement: str) -> str:
    index = text.find(source)
    if index < 0:
        return text
    return text[:index] + replacement + text[index + len(source) :]


def _standalone(text: str, match: re.Match[str]) -> bool:
    word = match.group(0).lower()
    if word in _ALWAYS_STANDALONE:
        return True
    if word not in _BARE_ONLY:
        return False
    return re.match(r"\s+[A-Za-z]", text[match.end() :]) is None


def _single_noun(before: str) -> str | None:
    nouns = [match.group(1) for match in _NP.finditer(before)]
    unique = []
    for noun in nouns:
        if noun.casefold() not in {item.casefold() for item in unique}:
            unique.append(noun)
    if len(unique) != 1:
        return None
    return "the " + unique[0]


def _single_name(before: str) -> str | None:
    names = []
    for match in _NAME.finditer(before):
        name = match.group(1)
        if name in _NAME_STOP:
            continue
        if name not in names:
            names.append(name)
    if len(names) != 1:
        return None
    return names[0]


def _english_bounds(text: str, start: int, end: int) -> bool:
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True
