"""Tools closed over one ingested PDF session."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from btom_pdf_agent.contradict import passages_contradict
from btom_pdf_agent.grids import grids_to_json, parse_color_ids, parse_grid_lines
from btom_pdf_agent.ledger import claim_query, judge_claim, parse_claims


def _lines(session: dict[str, Any], page: int) -> list[str]:
    raw = session["ocr_lines"]
    return list(raw.get(str(page)) or raw.get(page) or [])


def _cite(hit: dict[str, Any]) -> str:
    meta = hit.get("metadata") or {}
    page = int(meta.get("page") or 0)
    chunk_id = hit.get("chunk_id")
    if page:
        where = f"页 {page} 块 {chunk_id}"
    else:
        where = f"账本块 {chunk_id}"
    return f"出处：{where}\n{hit.get('text') or ''}"


def format_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "证据不足"
    return "\n\n".join(_cite(hit) for hit in hits)


def search_text(session: dict[str, Any], query: str) -> str:
    hits = session["index"].search(query, k=4)
    if passages_contradict(query, hits):
        return "证据不足"
    return format_hits(hits)


def ledger_text(session: dict[str, Any], page: int) -> str:
    lines = _lines(session, page)
    claims = parse_claims(lines)
    if not claims:
        return f"证据不足。页 {page} 没有读出层号和头号。"
    blocks: list[str] = []
    findings = []
    for claim in claims:
        hits = session["index"].search(claim_query(claim), k=4)
        finding = judge_claim(claim, hits)
        findings.append(finding)
        quote = finding.get("ledger_quote") or ""
        chunk_id = finding.get("chunk_id") or ""
        blocks.append(
            "\n".join(
                [
                    f"页 {page} L{finding['layer']}H{finding['head']} token:{finding.get('token') or ''}",
                    finding["reason"],
                    f"出处：账本块 {chunk_id}" if chunk_id else "出处：检索为空",
                    quote,
                ]
            )
        )
    consistent = [item for item in findings if item["consistent"]]
    if consistent and len(consistent) == len(findings):
        head = "；".join(
            f"L{item['layer']}H{item['head']} 与账本角色「{item['matched_role']}」一致"
            for item in findings
        )
    else:
        head = "核对未全部一致，以下只引用检索结果。"
    return head + "\n\n" + "\n\n".join(blocks)


def grid_text(session: dict[str, Any], page: int) -> str:
    lines = _lines(session, page)
    if not lines:
        return f"证据不足。页 {page} 没有 OCR 文本。"
    rule_hits = session["index"].search(
        "Color ids 0 black 1 blue separator 10 never appears in the answer",
        k=2,
    )
    rule_text = "\n".join(hit["text"] for hit in rule_hits)
    parsed = parse_grid_lines(lines, parse_color_ids(rule_text))
    if not any(parsed["grids"].values()):
        return f"证据不足。页 {page} 的 OCR 没有读出候选网格。"
    citations = [{"page": page, "chunk_id": f"p{page}-ocr-0", "quote": "\n".join(lines)}]
    citations.extend(
        {
            "page": int((hit.get("metadata") or {}).get("page") or 0),
            "chunk_id": hit.get("chunk_id"),
            "quote": hit.get("text"),
        }
        for hit in rule_hits
    )
    body = grids_to_json(parsed, citations)
    cites = format_hits(rule_hits)
    if cites == "证据不足":
        return f"色号规则未检索到，证据不足。\n页 {page} 的整数网格如下：\n{body}"
    return f"页 {page} 网格 JSON：\n{body}\n\n{cites}"


def ocr_text(session: dict[str, Any], page: int) -> str:
    lines = _lines(session, page)
    if not lines:
        return f"证据不足。页 {page} 没有 OCR 文本。"
    return f"页 {page} OCR：\n" + "\n".join(lines)


def build_tools(session: dict[str, Any]):
    @tool
    def search_chunks(query: str) -> str:
        """在已索引的页面、OCR 和账本块中检索。没有命中时返回证据不足。"""

        return search_text(session, query)

    @tool
    def check_ledger(page: int) -> str:
        """核对这一页注意力图的 OCR 与检索到的角色账本，并引用账本原句。"""

        return ledger_text(session, page)

    @tool
    def grid_to_json(page: int) -> str:
        """把这一页网格图写成候选 A/B/C、Query、Output 的整数 JSON，并引用色号规则。"""

        return grid_text(session, page)

    @tool
    def ocr_page(page: int) -> str:
        """返回这一页在入库时读到的 OCR 文本。"""

        return ocr_text(session, page)

    return [search_chunks, check_ledger, grid_to_json, ocr_page]
