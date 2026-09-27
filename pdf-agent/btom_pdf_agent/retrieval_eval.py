"""Recall@K on Hi-ToM story pages ingested by the existing PDF chain.

Each CSV row becomes one PDF page of story text only. The question, answer,
and choices are not written into the page. The query is the CSV question.
Gold chunk ids are the chunks of that row's story page. A row is dropped when
the question string occurs inside any of those gold chunks, so a hit cannot
come from searching for a string that was already indexed.

K is the default width of ``HybridIndex.search``. A hit means every gold
chunk id for that question is inside the top K. IOI and ARC files are not in
the clone, and ``export_mtm_csv.py`` cannot write grids without a challenges
file plus a local tokenizer, so those sets are skipped.
"""

from __future__ import annotations

import csv
import inspect
import re
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from btom_pdf_agent.agent import ask
from btom_pdf_agent.ingest import build_ingest_chain
from btom_pdf_agent.retrieve import HybridIndex

DEFAULT_HITOM = Path("/home/ubuntu/BTOM-Transformerlens/data_uniform/Hi_ToM_order_1.csv")

_SKIPPED_OTHER = {
    "IOI": "BTOM 克隆里没有 IOI 数据文件。",
    "ARC": "没有 challenges/solutions JSON。export_mtm_csv.py 需要该文件和本地 tokenizer，本次不下载权重，也不导出网格。",
}

_NAME_RE = re.compile(r"\b[A-Z][a-zA-Z]+\b")
_NAME_STOP = {"Where", "What", "Who", "When", "Why", "How", "Does", "Do", "Did", "Is", "Are", "The"}


def retrieval_k() -> int:
    return int(inspect.signature(HybridIndex.search).parameters["k"].default)


def gold_passage_split(piece_count: int) -> bool:
    """True when the gold passage overlaps more than one retrieved child chunk."""

    return piece_count > 1


def index_pieces(chunks: list[dict[str, Any]]) -> int:
    total = 0
    for chunk in chunks:
        children = [child for child in (chunk.get("children") or []) if str(child.get("text") or "").strip()]
        total += len(children) if children else 1
    return total


def run_retrieval_eval(
    directory: str | Path,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(csv_path) if csv_path is not None else DEFAULT_HITOM
    k = retrieval_k()
    skipped = dict(_SKIPPED_OTHER)
    if not source.is_file():
        skipped["Hi-ToM"] = f"未找到 {source}，没有金标可建。"
        return _empty(k, skipped)

    rows = _load_rows(source)
    pdf_path = folder / "hitom.pdf"
    _write_pdf(rows, pdf_path)
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
    chunks = session["chunks"]
    cases, dropped = _cases(rows, chunks)
    recall_hits = 0
    citation_hits = 0
    index = session["index"]
    for case in cases:
        found = [hit["chunk_id"] for hit in index.search(case["query"], k=k)]
        case["retrieved"] = found
        if set(case["gold_chunk_ids"]) <= set(found):
            recall_hits += 1
        answer = ask(session, case["query"], thread_id=f"hitom-{case['page']}", force_fake=True)
        case["answer_cites"] = all(chunk_id in answer for chunk_id in case["gold_chunk_ids"])
        citation_hits += int(case["answer_cites"])
    checked = len(cases)
    gold_split_count = sum(1 for case in cases if case["gold_split"])
    summary_recall = f"Recall@{k} {recall_hits}/{checked}"
    summary_citation = f"引用命中 {citation_hits}/{checked}"
    fixture_note = _fixture_note(cases, chunks, recall_hits, checked)
    dataset_files = [str(source)]
    previous = None
    if source.resolve() == DEFAULT_HITOM.resolve():
        previous = "此前同口径 Recall@4 99/100（上一次运行，不是本次）"
    lines = _lines(
        summary_recall,
        summary_citation,
        fixture_note,
        dataset_files,
        skipped,
        dropped,
        gold_split_count,
        checked,
        previous,
    )
    return {
        "k": k,
        "checked": checked,
        "recall_hits": recall_hits,
        "citation_hits": citation_hits,
        "gold_split_count": gold_split_count,
        "rows": len(rows),
        "dropped_question_in_story": dropped,
        "summary_recall": summary_recall,
        "summary_citation": summary_citation,
        "fixture_note": fixture_note,
        "dataset_files": dataset_files,
        "skipped": skipped,
        "cases": cases,
        "chunks": chunks,
        "lines": lines,
    }


def _empty(k: int, skipped: dict[str, str]) -> dict[str, Any]:
    summary_recall = f"Recall@{k} 0/0"
    summary_citation = "引用命中 0/0"
    fixture_note = "没有读到 Hi-ToM 文件，没有编造金标。这不是模型准确率。"
    lines = _lines(summary_recall, summary_citation, fixture_note, [], skipped, 0, 0, 0, None)
    return {
        "k": k,
        "checked": 0,
        "recall_hits": 0,
        "citation_hits": 0,
        "gold_split_count": 0,
        "rows": 0,
        "dropped_question_in_story": 0,
        "summary_recall": summary_recall,
        "summary_citation": summary_citation,
        "fixture_note": fixture_note,
        "dataset_files": [],
        "skipped": skipped,
        "cases": [],
        "chunks": [],
        "lines": lines,
    }


def _fixture_note(
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    recall_hits: int,
    checked: int,
) -> str:
    note = "本次数只统计这一份 Hi-ToM 夹具，不是模型准确率。问句不在被索引的页面里。"
    overlap = _lexical_overlap_sentence(cases, chunks, recall_hits, checked)
    if overlap:
        note = f"{note}{overlap}"
    return note


def _lexical_overlap_sentence(
    cases: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    recall_hits: int,
    checked: int,
) -> str:
    """Describe a perfect score that BM25 can get from unique pages and shared names."""

    if checked == 0 or recall_hits != checked:
        return ""
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    texts: list[str] = []
    for case in cases:
        gold_text = "\n".join(str(by_id[chunk_id]["text"]) for chunk_id in case["gold_chunk_ids"])
        texts.append(gold_text)
        names = {name for name in _NAME_RE.findall(case["query"]) if name not in _NAME_STOP}
        if not names or not any(name in gold_text for name in names):
            return ""
    if len(texts) != len(set(texts)):
        return ""
    return (
        "每一页故事互不相同，问句里的人名也出现在该页正文里。"
        "这个满分只说明词面重叠能把该页排进前 K，不是模型准确率。"
    )


def _lines(
    summary_recall: str,
    summary_citation: str,
    fixture_note: str,
    dataset_files: list[str],
    skipped: dict[str, str],
    dropped: int,
    gold_split_count: int,
    checked: int,
    previous: str | None,
) -> list[str]:
    files = "、".join(dataset_files) if dataset_files else "无"
    lines = [
        summary_recall,
        summary_citation,
        f"金标被切散 {gold_split_count}/{checked}（金标文本跨了多个检索小块）",
        f"因问句出现在故事页而丢弃 {dropped}",
        fixture_note,
        f"数据文件：{files}",
    ]
    if previous:
        lines.append(previous)
    for name in ("Hi-ToM", "IOI", "ARC"):
        if name in skipped:
            lines.append(f"跳过 {name}：{skipped[name]}")
    return lines


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _story_lines(row: dict[str, str]) -> list[str]:
    return [line.strip() for line in str(row.get("story") or "").splitlines() if line.strip()]


def _write_pdf(rows: list[dict[str, str]], pdf_path: Path) -> None:
    writer = canvas.Canvas(str(pdf_path), pagesize=A4)
    for row in rows:
        writer.setFont("Helvetica", 9)
        block = writer.beginText(36, 800)
        for line in _story_lines(row):
            block.textLine(line)
        writer.drawText(block)
        writer.showPage()
    writer.save()


def _cases(
    rows: list[dict[str, str]],
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    by_page: dict[int, list[dict[str, Any]]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        if metadata.get("source") != "pdf":
            continue
        page = int(metadata.get("page") or 0)
        by_page.setdefault(page, []).append(chunk)
    cases: list[dict[str, Any]] = []
    dropped = 0
    for index, row in enumerate(rows, start=1):
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        gold_chunks = by_page.get(index) or []
        if not gold_chunks:
            continue
        if any(question in str(chunk.get("text") or "") for chunk in gold_chunks):
            dropped += 1
            continue
        cases.append(
            {
                "query": question,
                "page": index,
                "gold_chunk_ids": [chunk["chunk_id"] for chunk in gold_chunks],
                "gold_split": gold_passage_split(index_pieces(gold_chunks)),
            }
        )
    return cases, dropped
