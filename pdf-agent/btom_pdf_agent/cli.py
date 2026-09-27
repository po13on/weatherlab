"""Command line for ingest, ask, and the fixture citation check."""

from __future__ import annotations

import argparse
from pathlib import Path

from btom_pdf_agent.agent import ask
from btom_pdf_agent.consistency_audit import audit_pdf, score_inconsistent
from btom_pdf_agent.eval_run import run_eval
from btom_pdf_agent.ingest import build_ingest_chain, load_index
from btom_pdf_agent.docvqa_eval import run_docvqa_recall
from btom_pdf_agent.retrieval_eval import run_retrieval_eval


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="多模态 PDF Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("pdf")
    ingest.add_argument("--state", required=True)
    ingest.add_argument("--out", required=True)
    ingest.add_argument("--ocr", default="fake", choices=["fake", "rapidocr"])

    ask_parser = sub.add_parser("ask")
    ask_parser.add_argument("persist")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--fake", action="store_true")

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--dir", default="examples/eval")
    evaluate.add_argument("--state", default="examples/run_state.json")

    docvqa = sub.add_parser("recall-docvqa")
    docvqa.add_argument("--rows", default="/tmp/docvqa/rows_0_100.json")
    docvqa.add_argument("--out", default="/tmp/docvqa/run_rules")

    audit = sub.add_parser("audit-consistency")
    audit.add_argument("pdf")
    audit.add_argument(
        "--gold",
        default=None,
        help="打分时才读取的金标 id 文件。预测本身不读这个文件。",
    )

    args = parser.parse_args(argv)
    if args.command == "ingest":
        build_ingest_chain().invoke(
            {
                "pdf_path": args.pdf,
                "state_path": args.state,
                "persist_dir": args.out,
                "ocr_kind": args.ocr,
            }
        )
        print(args.out)
        return
    if args.command == "ask":
        loaded = load_index(args.persist)
        session = {
            "index": loaded["index"],
            "ocr_lines": loaded["manifest"]["ocr_lines"],
        }
        print(ask(session, args.question, force_fake=args.fake or not _has_key()))
        return
    if args.command == "recall-docvqa":
        recalled = run_docvqa_recall(args.out, args.rows)
        for line in recalled["lines"]:
            print(line)
        return
    if args.command == "audit-consistency":
        predicted = audit_pdf(args.pdf)
        print(f"页数 {predicted['page_count']}")
        print(f"预测不一致页数 {len(predicted['inconsistent_ids'])}")
        print("预测不一致页 " + " ".join(predicted["inconsistent_ids"]))
        if predicted["tied"]:
            tied = " ".join(
                f"{item['page_id']}={item['key_indices']}" for item in predicted["tied"]
            )
            print(f"并列页 {tied}")
        else:
            print("并列页 无")
        if args.gold:
            scored = score_inconsistent(predicted["inconsistent_ids"], args.gold)
            print(f"召回 {scored['recall']}")
            print(f"误报页数 {scored['false_positive_count']}")
            print("误报页 " + (" ".join(scored["false_positive_ids"]) or "无"))
            print("漏报页 " + (" ".join(scored["missed_ids"]) or "无"))
        return
    root = Path(__file__).resolve().parents[1]
    result = run_eval(root / args.dir, root / args.state)
    print(result["summary"])
    retrieval = run_retrieval_eval(root / args.dir / "hitom")
    for line in retrieval["lines"]:
        print(line)


def _has_key() -> bool:
    import os

    return bool(os.environ.get("OPENAI_API_KEY"))
