from __future__ import annotations

from functools import lru_cache
from typing import Any

from rag.ocr import ocr_png, render_png
from store import FIXTURE_PATH, runbooks

RUNBOOK_DIR = FIXTURE_PATH.parent / "runbooks"


def _parse_pages(text: str) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    body: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("--- page ") and line.endswith("---"):
            if current is not None:
                current["body"] = "\n".join(body).strip()
                pages.append(current)
            page_no = int(line[len("--- page ") : -len("---")].strip())
            current = {"page": page_no, "source": "text"}
            body = []
            continue
        if current is None:
            continue
        if line.startswith("source:"):
            current["source"] = line.split(":", 1)[1].strip()
            continue
        body.append(raw.rstrip())
    if current is not None:
        current["body"] = "\n".join(body).strip()
        pages.append(current)
    return pages


def _chunk(doc: dict[str, Any], page: dict[str, Any], text: str, source: str) -> dict[str, Any]:
    page_no = int(page["page"])
    return {
        "doc_id": doc["doc_id"],
        "chunk_id": f"{doc['doc_id']}:p{page_no}",
        "page": page_no,
        "section": doc["section"],
        "tag": doc.get("tag"),
        "title": doc["title"],
        "span": "page",
        "quote": text,
        "source": source,
        "gold_page": doc.get("gold_page"),
    }


@lru_cache(maxsize=1)
def load_corpus() -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    scans: dict[tuple[str, int], bytes] = {}
    for doc in runbooks():
        path = RUNBOOK_DIR / f"{doc['doc_id']}.md"
        if not path.exists():
            chunks.append(
                _chunk(
                    doc,
                    {"page": doc.get("gold_page") or 1},
                    doc["quote"],
                    "text",
                )
            )
            continue
        for page in _parse_pages(path.read_text(encoding="utf-8")):
            if page["source"] == "scan":
                png = render_png(page["body"])
                recognized = ocr_png(png)
                if not recognized.get("ok"):
                    raise RuntimeError(
                        f"ocr failed for {doc['doc_id']} page {page['page']}: {recognized.get('why')}"
                    )
                text = str(recognized["text"])
                scans[(doc["doc_id"], int(page["page"]))] = png
                chunks.append(_chunk(doc, page, text, "ocr"))
                continue
            chunks.append(_chunk(doc, page, page["body"], "text"))
        gold_page = doc.get("gold_page")
        if gold_page is not None:
            gold_text = " ".join(
                item["quote"]
                for item in chunks
                if item["doc_id"] == doc["doc_id"] and item["page"] == gold_page
            )
            if doc["quote"] not in gold_text:
                raise RuntimeError(f"gold quote missing on {doc['doc_id']} page {gold_page}")
    return {"chunks": chunks, "scans": scans}


def chunks() -> list[dict[str, Any]]:
    return list(load_corpus()["chunks"])


def ocr_page(doc_id: str, page: int) -> dict[str, Any]:
    png = load_corpus()["scans"].get((doc_id, int(page)))
    if png is None:
        return {"error": "not_a_scan", "doc_id": doc_id, "page": page}
    recognized = ocr_png(png)
    if not recognized.get("ok"):
        return {
            "error": "ocr_failed",
            "doc_id": doc_id,
            "page": page,
            "why": recognized.get("why"),
            "engine": recognized.get("engine"),
        }
    return {
        "ok": True,
        "payload": {
            "doc_id": doc_id,
            "page": int(page),
            "text": recognized["text"],
            "engine": recognized["engine"],
        },
        "facts": [f"ocr {doc_id} p.{page}: {recognized['text']}"],
    }


def corpus_stats() -> dict[str, int]:
    data = load_corpus()
    return {
        "chunks": len(data["chunks"]),
        "ocr_pages": len(data["scans"]),
    }
