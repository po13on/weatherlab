from jobs import JobService


def test_idempotency_and_cas(tmp_path):
    svc = JobService(tmp_path / "jobs.sqlite3")
    assert svc.lookup_idempotent("q", {"run_id": "a"}) is None
    job = svc.create_job("q", {"run_id": "a"})
    svc.remember_idempotent("q", {"run_id": "a"}, job["job_id"])
    assert svc.lookup_idempotent("q", {"run_id": "a"}) == job["job_id"]
    svc.remember_idempotent("q", {"run_id": "a"}, "job-guest", role="guest")
    assert svc.lookup_idempotent("q", {"run_id": "a"}, role="guest") == "job-guest"
    assert svc.lookup_idempotent("q", {"run_id": "a"}, role="analyst") == job["job_id"]
    assert svc.complete_cas(job["job_id"], 0, "succeeded", {"verdict": "supported"}) is True
    assert svc.complete_cas(job["job_id"], 0, "failed", {"verdict": "x"}) is False
    stored = svc.get_job(job["job_id"])
    assert stored["result_version"] == 1
    assert stored["status"] == "succeeded"


def test_rate_limit(tmp_path):
    svc = JobService(tmp_path / "rl.sqlite3")
    assert all(svc.check_rate_limit() for _ in range(8))
    assert svc.check_rate_limit() is False


def test_replay_restores_graph_state_not_gpu(tmp_path):
    svc = JobService(tmp_path / "replay.sqlite3")
    job = svc.create_job("why oom?", {"run_id": "run-fail-oom"})
    result = {
        "question": "why oom?",
        "scope": {"run_id": "run-fail-oom"},
        "facts": [{"text": "CUDA OOM"}],
        "evidence_ids": ["ev-1"],
        "claims": [{"kind": "fact", "text": "OOM", "evidence_ids": ["ev-1"]}],
        "verdict": "supported",
        "tool_trace": [{"name": "extract_log_error"}],
        "missing": [],
    }
    assert svc.complete_cas(job["job_id"], 0, "succeeded", result) is True
    snap = svc.replay(job["job_id"])
    assert snap["state"]["facts"][0]["text"] == "CUDA OOM"
    assert snap["state"]["evidence_ids"] == ["ev-1"]
    assert "does not rerun GPU" in snap["note"]


def test_save_and_latest_eval(tmp_path):
    svc = JobService(tmp_path / "eval.sqlite3")
    report = {
        "eval_id": "eval-test1",
        "kind": "system",
        "created_at": "2026-09-20T00:00:00Z",
        "summary": {"n": 10, "verdict_accuracy": 1.0},
        "items": [{"case_id": "Q1", "verdict_ok": True}],
    }
    svc.save_eval(report)
    latest = svc.latest_evals()
    assert latest["system"]["eval_id"] == "eval-test1"
    assert latest["baseline"] is None


def test_publish_and_listen_terminal(tmp_path):
    svc = JobService(tmp_path / "ev.sqlite3")
    job = svc.create_job("q", {"run_id": "a"})
    svc.publish(job["job_id"], "tool.end", {"name": "list_runs"})
    svc.complete_cas(job["job_id"], 0, "insufficient", {"verdict": "insufficient"})
    events = list(svc.listen(job["job_id"], after_seq=0, timeout=0.2))
    names = [item["event"] for item in events]
    assert "queued" in names
    assert "tool.end" in names
    assert "final" in names
