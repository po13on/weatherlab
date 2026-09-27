"""DocVQA text retrieval Recall@K on the first 100 rows already saved locally.

The page text is the dataset word sequence, not RapidOCR. Questions are not
written into the PDF. A gold parent is one whose child slice overlaps the
annotated evidence span. This does not compute ANLS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from btom_pdf_agent.ingest import build_ingest_chain
from btom_pdf_agent.pdf_io import split_pdf
from btom_pdf_agent.retrieval_eval import gold_passage_split, retrieval_k

DEFAULT_ROWS = Path("/tmp/docvqa/rows_0_100.json")
SOURCE_URL = "https://huggingface.co/datasets/nielsr/docvqa_1200_examples"
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=nielsr/docvqa_1200_examples"
    "&config=default&split=train&offset=0&length=100"
)
PREVIOUS_RECALL = "此前同口径 Recall@4 59/100（上一次运行，不是本次）"


def run_docvqa_recall(
    directory: str | Path,
    rows_path: str | Path = DEFAULT_ROWS,
) -> dict[str, Any]:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(rows_path)
    k = retrieval_k()
    if not source.is_file():
        lines = [
            f"Recall@{k} 0/0",
            f"未找到 {source}，没有编造 DocVQA 金标。",
            "数据集自身指标：ANLS（DocVQA）。本次没有计算 ANLS。",
            "检索的是该集自带文本，没有跑 RapidOCR。",
        ]
        return {
            "k": k,
            "hits": 0,
            "checked": 0,
            "gold_split_count": 0,
            "pages": 0,
            "lines": lines,
        }

    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = [item["row"] for item in payload["rows"]]
    if len(rows) != 100:
        raise SystemExit(f"expected 100 rows, got {len(rows)}")
    total = int(payload["num_rows_total"])
    documents, questions = _pack_rows(rows)

    pdf_path = folder / "docvqa_sample.pdf"
    _write_pdf(documents, pdf_path)
    extracted_pages = split_pdf(pdf_path)
    if len(extracted_pages) != len(documents):
        raise SystemExit(f"pdf pages {len(extracted_pages)} != documents {len(documents)}")
    for doc, page in zip(documents, extracted_pages):
        norm, raw_index = _whitespace_map(page["text"])
        if norm != doc["text"]:
            raise SystemExit(
                "indexed page text does not match the dataset word sequence "
                f"on page {page['page']}"
            )
        doc["page_text"] = page["text"]
        doc["raw_index"] = raw_index

    state_path = folder / "empty_state.json"
    state_path.write_text("{}", encoding="utf-8")
    session = build_ingest_chain().invoke(
        {
            "pdf_path": str(pdf_path),
            "state_path": str(state_path),
            "persist_dir": str(folder / "index"),
            "ocr_kind": "fake",
        }
    )
    by_page: dict[int, list[dict[str, Any]]] = {}
    for chunk in session["chunks"]:
        if (chunk.get("metadata") or {}).get("source") != "pdf":
            continue
        by_page.setdefault(int(chunk["metadata"]["page"]), []).append(chunk)
    for doc, page in zip(documents, extracted_pages):
        chunks = by_page.get(int(page["page"])) or []
        for chunk in chunks:
            raw = page["text"]
            if raw[int(chunk["start"]) : int(chunk["end"])].strip() != chunk["text"]:
                raise SystemExit(f"parent offset diverged on page {page['page']}")
            for child in chunk.get("children") or []:
                if raw[int(child["start"]) : int(child["end"])].strip() != child["text"]:
                    raise SystemExit(f"child offset diverged on page {page['page']}")
        doc["chunks"] = chunks

    index = session["index"]
    hits = 0
    checked = 0
    dropped_question = 0
    dropped_span = 0
    gold_split_count = 0
    for case in questions:
        doc = documents[case["page"] - 1]
        gold_ids, split, covered = _gold(doc, case["start"], case["span"])
        if not covered or not gold_ids:
            dropped_span += 1
            continue
        gold_chunks = [chunk for chunk in doc["chunks"] if chunk["chunk_id"] in set(gold_ids)]
        overlapping = _overlapping_children(doc, case["start"], case["span"])
        if any(case["question"] and case["question"] in str(child.get("text") or "") for child in overlapping):
            dropped_question += 1
            continue
        found = [hit["chunk_id"] for hit in index.search(case["question"], k=k)]
        checked += 1
        if set(gold_ids) <= set(found):
            hits += 1
        if split:
            gold_split_count += 1

    lines = [
        f"Recall@{k} {hits}/{checked}",
        PREVIOUS_RECALL,
        f"金标被切散 {gold_split_count}/{checked}（金标文本跨了多个检索小块）",
        f"上限：100 问（nielsr/docvqa_1200_examples 训练集共 {total} 问，本次取前 100 问）",
        f"索引页面：{len(documents)} 页（这 100 问里不重复的文档页）",
        f"计入：{checked}",
        f"因问句出现在金标块而丢弃 {dropped_question}",
        f"因证据片段不在索引文本中而丢弃 {dropped_span}",
        "数据集自身指标：ANLS（DocVQA）。本次没有计算 ANLS。",
        "检索的是该集自带文本，没有跑 RapidOCR。",
        "问句没有写入被索引的页面。",
        "本次比例只属于这一次运行。",
        f"数据：{SOURCE_URL}",
        f"下载：{ROWS_URL}",
    ]
    summary = {
        "k": k,
        "hits": hits,
        "checked": checked,
        "gold_split_count": gold_split_count,
        "pages": len(documents),
        "dropped_question": dropped_question,
        "dropped_span": dropped_span,
        "lines": lines,
    }
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _pack_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents: list[dict[str, Any]] = []
    page_of: dict[str, int] = {}
    questions: list[dict[str, Any]] = []
    for row in rows:
        words = [str(word) for word in row["words"]]
        joined = " ".join(words)
        answer = row["answer"] or {}
        start = answer.get("start")
        span = str(answer.get("matched_text") or "")
        question = str((row.get("query") or {}).get("en") or "").strip()
        if joined not in page_of:
            page_of[joined] = len(documents) + 1
            documents.append({"text": joined, "spans": []})
        doc = documents[page_of[joined] - 1]
        if isinstance(start, int) and span and joined[start : start + len(span)] == span:
            doc["spans"].append((start, start + len(span)))
        questions.append(
            {
                "id": row.get("id"),
                "question": question,
                "start": start,
                "span": span,
                "page": page_of[joined],
            }
        )
    return documents, questions


def _gold(doc: dict[str, Any], start: object, span: str) -> tuple[list[str], bool, bool]:
    raw = _raw_span(doc, start, span)
    if raw is None:
        return [], False, False
    raw_start, raw_end = raw
    overlapping = _overlapping_children(doc, start, span)
    if not overlapping or not _covers(raw_start, raw_end, [(item["start"], item["end"]) for item in overlapping]):
        return [], False, False
    parent_ids: list[str] = []
    for child in overlapping:
        parent_id = _parent_of(doc, child)
        if parent_id and parent_id not in parent_ids:
            parent_ids.append(parent_id)
    return parent_ids, gold_passage_split(len(overlapping)), True


def _overlapping_children(doc: dict[str, Any], start: object, span: str) -> list[dict[str, Any]]:
    raw = _raw_span(doc, start, span)
    if raw is None:
        return []
    raw_start, raw_end = raw
    found: list[dict[str, Any]] = []
    for chunk in doc.get("chunks") or []:
        for child in chunk.get("children") or []:
            left = int(child["start"])
            right = int(child["end"])
            if left < raw_end and raw_start < right:
                found.append(child)
    return found


def _parent_of(doc: dict[str, Any], child: dict[str, Any]) -> str:
    child_id = str(child.get("chunk_id") or "")
    for chunk in doc.get("chunks") or []:
        for item in chunk.get("children") or []:
            if item.get("chunk_id") == child_id:
                return str(chunk["chunk_id"])
    return ""


def _raw_span(doc: dict[str, Any], start: object, span: str) -> tuple[int, int] | None:
    if not isinstance(start, int) or not span:
        return None
    joined = doc["text"]
    if joined[start : start + len(span)] != span:
        return None
    raw_index = doc["raw_index"]
    if start + len(span) - 1 >= len(raw_index):
        return None
    raw_start = raw_index[start]
    raw_end = raw_index[start + len(span) - 1] + 1
    raw_slice = doc["page_text"][raw_start:raw_end]
    if " ".join(raw_slice.split()) != " ".join(span.split()):
        return None
    return raw_start, raw_end


def _covers(start: int, end: int, pieces: list[tuple[int, int]]) -> bool:
    cursor = start
    for left, right in sorted(pieces):
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return cursor >= end


def _whitespace_map(text: str) -> tuple[str, list[int]]:
    raw: list[int] = []
    out: list[str] = []
    in_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if out and not in_space:
                out.append(" ")
                raw.append(index)
            in_space = True
        else:
            out.append(char)
            raw.append(index)
            in_space = False
    while out and out[0] == " ":
        out.pop(0)
        raw.pop(0)
    while out and out[-1] == " ":
        out.pop()
        raw.pop()
    return "".join(out), raw


def _write_pdf(documents: list[dict[str, Any]], pdf_path: Path) -> None:
    writer = canvas.Canvas(str(pdf_path), pagesize=A4)
    for doc in documents:
        lines = _wrap(doc["text"], _merge_spans(doc["spans"]), width=90)
        if len(lines) > 70:
            raise SystemExit(f"page would overflow: {len(lines)} lines")
        writer.setFont("Helvetica", 8)
        block = writer.beginText(36, 800)
        block.setLeading(10)
        for line in lines:
            block.textLine(line)
        writer.drawText(block)
        writer.showPage()
    writer.save()


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _wrap(text: str, protected: list[tuple[int, int]], width: int) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(text):
        guard = next((item for item in protected if item[0] <= index < item[1]), None)
        if guard and index == guard[0]:
            tokens.append(text[guard[0] : guard[1]])
            index = guard[1]
            continue
        if text[index] == " ":
            tokens.append(" ")
            index += 1
            continue
        limit = min((item[0] for item in protected if item[0] > index), default=len(text))
        end = index
        while end < limit and text[end] != " ":
            end += 1
        tokens.append(text[index:end])
        index = end
    lines: list[str] = []
    buf = ""
    prev_was_space = True
    for token in tokens:
        glued = bool(buf) and token != " " and not prev_was_space
        if buf and not glued and len(buf) + len(token) > width:
            lines.append(buf.rstrip())
            buf = "" if token == " " else token
        else:
            if not buf and token == " ":
                prev_was_space = True
                continue
            buf += token
        prev_was_space = token == " "
    if buf.strip():
        lines.append(buf.rstrip())
    return lines


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DocVQA 文本检索 Recall@K")
    parser.add_argument("--rows", default=str(DEFAULT_ROWS))
    parser.add_argument("--out", default="/tmp/docvqa/run_rules")
    args = parser.parse_args()
    result = run_docvqa_recall(args.out, args.rows)
    for line in result["lines"]:
        print(line)


if __name__ == "__main__":
    main()
