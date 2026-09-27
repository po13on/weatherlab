"""Ledger passages and a judgement that only reads retrieved chunks."""

from __future__ import annotations

import re
from typing import Any

_HEAD_RE = re.compile(r"L\s*(\d+)\s*H\s*(\d+)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"token\s*[:=]?\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
_ROLE_RE = re.compile(r"role\s*[:=]\s*(.+)$", re.IGNORECASE)

_PATTERN_TOKENS = {
    "A-->V": ["A-", "V"],
    "A-->A-": ["A-"],
    "Answer->Answer": ["A-"],
    "V->V": ["V"],
    "AnswerSpan->AnswerSpan": ["V"],
    "V->VK_C": ["V", "VK_C"],
    "V->VK_I": ["V", "VK_I"],
    "A-->QK_C": ["A-", "QK_C"],
    "A-->QK_I": ["A-", "QK_I"],
}


def pattern_tokens(pattern: str) -> list[str]:
    if pattern in _PATTERN_TOKENS:
        return list(_PATTERN_TOKENS[pattern])
    for separator in ("-->", "->"):
        if separator not in pattern:
            continue
        left, right = (part.strip() for part in pattern.split(separator, 1))
        if separator == "-->" and left == "A":
            left = "A-"
        tokens: list[str] = []
        for part in (left, right):
            if part and part not in tokens:
                tokens.append(part)
        return tokens
    if pattern:
        return [pattern]
    return []


def ledger_passage(row: dict[str, Any]) -> str:
    note = str(row.get("note") or "")
    frontier = "、".join(str(item) for item in (row.get("active_frontier") or []))
    tokens = "、".join(pattern_tokens(str(row.get("pattern") or "")))
    sentence = (
        f"层 {row.get('layer')} 头 {row.get('head')} 的角色是「{row.get('role') or ''}」，"
        f"模式 {row.get('pattern') or ''}，"
        f"节点类型 {row.get('node_type') or ''}，"
        f"当时前沿为 {frontier}，"
        f"涉及 token：{tokens}。"
    )
    return sentence + note


def ledger_documents(state: dict[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for row in state.get("role_ledger") or []:
        tokens = pattern_tokens(str(row.get("pattern") or ""))
        header = "\n".join(
            [
                f"layer: {int(row['layer'])}",
                f"head: {int(row['head'])}",
                "kind: role",
                *[f"token: {token}" for token in tokens],
            ]
        )
        documents.append(
            {
                "text": header + "\n" + ledger_passage(row),
                "metadata": {
                    "kind": "role",
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "role": str(row.get("role") or ""),
                    "tokens": "|".join(tokens),
                    "page": 0,
                    "source": "ledger",
                },
            }
        )
    frontier = [str(item) for item in (state.get("active_frontier") or [])]
    history_bits = [
        f"{item.get('after')} {item.get('frontier')}"
        for item in (state.get("frontier_history") or [])
    ]
    history = "；".join(history_bits) if history_bits else "（无）"
    documents.append(
        {
            "text": (
                "kind: frontier\n"
                f"当前前沿：{'、'.join(frontier) if frontier else '（无）'}。前沿演变：{history}。"
            ),
            "metadata": {"kind": "frontier", "page": 0, "source": "ledger"},
        }
    )
    stop_reason = str(state.get("stop_reason") or "")
    stop_text = str((state.get("report_fields") or {}).get("stop_text") or "")
    documents.append(
        {
            "text": f"kind: stop\n停机原因：{stop_reason}。{stop_text}",
            "metadata": {"kind": "stop", "page": 0, "source": "ledger", "stop_reason": stop_reason},
        }
    )
    report = state.get("report")
    if isinstance(report, str) and report.strip():
        documents.append(
            {
                "text": "kind: report\n" + report.strip(),
                "metadata": {"kind": "report", "page": 0, "source": "ledger"},
            }
        )
    report_line = (state.get("report_fields") or {}).get("frontier_line")
    if report_line:
        documents.append(
            {
                "text": f"kind: report\n报告中的前沿演变：{report_line}",
                "metadata": {"kind": "report", "page": 0, "source": "ledger"},
            }
        )
    return documents


def parse_claims(lines: list[str]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for line in lines:
        head = _HEAD_RE.search(line)
        if not head:
            continue
        token_match = _TOKEN_RE.search(line)
        role_match = _ROLE_RE.search(line)
        claims.append(
            {
                "layer": int(head.group(1)),
                "head": int(head.group(2)),
                "token": token_match.group(1) if token_match else "",
                "figure_role": role_match.group(1).strip() if role_match else None,
                "source_line": line,
            }
        )
    return claims


def claim_query(claim: dict[str, Any]) -> str:
    return (
        f"layer: {claim['layer']}\n"
        f"head: {claim['head']}\n"
        "kind: role\n"
        f"token: {claim['token']}\n"
        "角色账本"
    )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def judge_claim(claim: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Decide from retrieved hits only. The raw ledger list is not an argument."""

    retrieved = [_preview(hit) for hit in hits]
    matched = None
    for hit in hits:
        meta = hit.get("metadata") or {}
        if meta.get("kind") != "role":
            continue
        if _as_int(meta.get("layer")) == claim["layer"] and _as_int(meta.get("head")) == claim["head"]:
            matched = hit
            break
    if matched is None:
        fallback = next((hit for hit in hits if (hit.get("metadata") or {}).get("kind") == "role"), None)
        quote_hit = fallback or (hits[0] if hits else None)
        return {
            "layer": claim["layer"],
            "head": claim["head"],
            "token": claim["token"] or None,
            "figure_role": claim["figure_role"],
            "consistent": False,
            "matched_role": None,
            "ledger_quote": (quote_hit or {}).get("text") or "",
            "chunk_id": (quote_hit or {}).get("chunk_id"),
            "reason": "检索结果里没有该层该头的角色条目，不能只按图上的字认定角色。",
            "retrieved": retrieved,
        }
    meta = matched.get("metadata") or {}
    listed = [item for item in str(meta.get("tokens") or "").split("|") if item]
    token = claim["token"]
    token_ok = bool(token) and token in listed
    role_ok = True
    if claim["figure_role"]:
        role_ok = claim["figure_role"] == str(meta.get("role") or "")
    consistent = token_ok and role_ok
    if consistent:
        reason = f"与账本角色「{meta.get('role') or ''}」一致。"
    elif not token:
        reason = "图上没有读出 token，账本原文如下。"
    elif not token_ok:
        reason = "图上的 token 不在该账本条目列出的模式 token 中。"
    else:
        reason = "图上写的角色与账本条目不一致。"
    return {
        "layer": claim["layer"],
        "head": claim["head"],
        "token": token or None,
        "figure_role": claim["figure_role"],
        "consistent": consistent,
        "matched_role": meta.get("role") or None,
        "ledger_quote": matched.get("text") or "",
        "chunk_id": matched.get("chunk_id"),
        "reason": reason,
        "retrieved": retrieved,
    }


def _preview(hit: dict[str, Any]) -> dict[str, Any]:
    meta = hit.get("metadata") or {}
    return {
        "kind": meta.get("kind"),
        "layer": meta.get("layer"),
        "head": meta.get("head"),
        "chunk_id": hit.get("chunk_id"),
        "score": hit.get("score"),
        "text": hit.get("text"),
    }
