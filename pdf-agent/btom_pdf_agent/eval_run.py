"""Fixture questions. The report is a hit count for this run, not a résumé metric."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from btom_pdf_agent.agent import ask
from btom_pdf_agent.fixtures import build_demo_pdf
from btom_pdf_agent.ingest import build_ingest_chain

CASES = [
    {
        "question": "请核对第 2 页注意力图和账本",
        "need": ["Answer value reader", "页 2"],
    },
    {
        "question": "请把第 3 页的网格写成 JSON",
        "need": ['"candidates"', "Color ids", "页 3"],
    },
    {
        "question": "文档里有没有火星基地的坐标",
        "need": ["证据不足"],
    },
]


def run_eval(directory: str | Path, state_path: str | Path) -> dict[str, Any]:
    folder = Path(directory)
    pdf_path = build_demo_pdf(folder)
    persist = folder / "index"
    session = build_ingest_chain().invoke(
        {
            "pdf_path": str(pdf_path),
            "state_path": str(state_path),
            "persist_dir": str(persist),
            "ocr_kind": "fake",
        }
    )
    hits = 0
    details = []
    for case in CASES:
        answer = ask(session, case["question"], thread_id=case["question"], force_fake=True)
        ok = all(piece in answer for piece in case["need"])
        hits += int(ok)
        details.append({"question": case["question"], "ok": ok, "answer": answer})
    summary = f"夹具引用命中 {hits}/{len(CASES)}"
    return {"cited": hits, "checked": len(CASES), "summary": summary, "details": details}
