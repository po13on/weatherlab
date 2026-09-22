import json

from scripts.export_sft_jsonl import export_jobs, result_to_record
from jobs import JobService
from prompts.loader import prompt_template


def test_sft_record_marks_not_trained():
    record = result_to_record(
        {
            "question": "why oom?",
            "scope": {"run_id": "run-fail-oom"},
            "verdict": "supported",
            "tool_trace": [
                {
                    "name": "extract_log_error",
                    "args": {"run_id": "run-fail-oom"},
                    "observation": {"ok": True},
                }
            ],
        },
        prompt_template("system"),
    )
    assert record["not_trained"] is True
    assert "did not fine-tune" in record["note"].lower()
    assert record["messages"][0]["role"] == "system"


def test_export_jobs_writes_jsonl(tmp_path):
    db = tmp_path / "jobs.sqlite3"
    svc = JobService(db)
    job = svc.create_job("why oom?", {"run_id": "run-fail-oom"})
    svc.complete_cas(
        job["job_id"],
        0,
        "succeeded",
        {
            "question": "why oom?",
            "scope": {"run_id": "run-fail-oom"},
            "tool_trace": [{"name": "extract_log_error", "args": {}, "observation": {}}],
        },
    )
    out = tmp_path / "sft.jsonl"
    n = export_jobs(db, out, limit=10)
    assert n == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    header = json.loads(lines[0])
    assert "did not fine-tune" in header["note"].lower()
