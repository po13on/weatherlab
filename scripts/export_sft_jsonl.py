from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import JobService
from prompts.loader import prompt_template


NOTE = (
    "SFT export only. This project did not fine-tune a model. "
    "Do not cite accuracy gains from this file."
)


def result_to_record(result: dict, system_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {"question": result.get("question"), "scope": result.get("scope") or {}},
                ensure_ascii=False,
            ),
        },
    ]
    for step in result.get("tool_trace") or []:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": step.get("name"),
                            "arguments": json.dumps(step.get("args") or {}, ensure_ascii=False),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "name": step.get("name"),
                "content": json.dumps(step.get("observation") or {}, ensure_ascii=False),
            }
        )
    return {
        "messages": messages,
        "verdict": result.get("verdict"),
        "not_trained": True,
        "note": NOTE,
    }


def export_jobs(db_path: Path | None, out_path: Path, limit: int = 50) -> int:
    svc = JobService(db_path)
    system_prompt = prompt_template("system")
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"note": NOTE, "format": "openai_chat_sft"}, ensure_ascii=False) + "\n")
        for job in svc.list_jobs(limit=limit):
            result = job.get("result")
            if not isinstance(result, dict) or not result.get("tool_trace"):
                continue
            handle.write(json.dumps(result_to_record(result, system_prompt), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export analysis traces to SFT jsonl. Does not train.")
    parser.add_argument("--db", default=None)
    parser.add_argument("--out", default=str(ROOT / "data" / "sft.jsonl"))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    db_path = Path(args.db) if args.db else None
    n = export_jobs(db_path, Path(args.out), args.limit)
    print(f"wrote {n} traces to {args.out}. not trained.")


if __name__ == "__main__":
    main()
