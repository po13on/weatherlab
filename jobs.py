from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator

from cache import MemoryCache, build_cache
from db import connect, init_db

ISO = "%Y-%m-%dT%H:%M:%SZ"


def _now() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


def idem_key(question: str, scope: dict[str, Any], role: str = "analyst") -> str:
    raw = json.dumps(
        {"question": question, "scope": scope, "role": role},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class JobService:
    def __init__(self, db_path=None, cache=None) -> None:
        self.conn = connect(db_path)
        init_db(self.conn)
        self.cache = cache or MemoryCache()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._seq: dict[str, int] = {}
        self._hitl: dict[str, dict[str, Any]] = {}

    def check_rate_limit(self) -> bool:
        with self._lock:
            return self.cache.check_rate_limit("demo")

    def lookup_idempotent(
        self, question: str, scope: dict[str, Any], role: str = "analyst"
    ) -> str | None:
        return self.cache.lookup_idempotent(idem_key(question, scope, role))

    def remember_idempotent(
        self, question: str, scope: dict[str, Any], job_id: str, role: str = "analyst"
    ) -> None:
        self.cache.remember_idempotent(idem_key(question, scope, role), job_id)

    def create_job(self, question: str, scope: dict[str, Any]) -> dict[str, Any]:
        job_id = "job-" + uuid.uuid4().hex[:12]
        now = _now()
        self.conn.execute(
            """
            INSERT INTO jobs (job_id, question, scope_json, status, result_version, created_at, updated_at)
            VALUES (?, ?, ?, 'queued', 0, ?, ?)
            """,
            (job_id, question, json.dumps(scope, ensure_ascii=False), now, now),
        )
        self.conn.commit()
        self.publish(job_id, "queued", {"job_id": job_id})
        return self.get_job(job_id)

    def mark_running(self, job_id: str) -> None:
        self.conn.execute(
            "UPDATE jobs SET status='running', updated_at=? WHERE job_id=?",
            (_now(), job_id),
        )
        self.conn.commit()
        self.publish(job_id, "running", {"job_id": job_id})

    def complete_cas(self, job_id: str, expected_version: int, status: str, result: dict[str, Any] | None, error: str | None = None) -> bool:
        cur = self.conn.execute(
            """
            UPDATE jobs
            SET status=?, result_json=?, error=?, result_version=result_version+1, updated_at=?
            WHERE job_id=? AND result_version=?
            """,
            (
                status,
                json.dumps(result, ensure_ascii=False) if result is not None else None,
                error,
                _now(),
                job_id,
                expected_version,
            ),
        )
        self.conn.commit()
        if cur.rowcount != 1:
            return False
        self.publish(job_id, "final", {"job_id": job_id, "status": status})
        return True

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._job_row(row)

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._job_row(row) for row in rows]

    def traces(self, job_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT seq, event_type, payload_json, created_at FROM traces WHERE job_id=? ORDER BY seq",
            (job_id,),
        ).fetchall()
        return [
            {
                "seq": row["seq"],
                "event": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def publish(self, job_id: str, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._cv:
            seq = self._seq.get(job_id, 0) + 1
            self._seq[job_id] = seq
            event = {
                "seq": seq,
                "event": event_type,
                "payload": payload,
                "created_at": _now(),
            }
            self.conn.execute(
                """
                INSERT INTO traces (job_id, seq, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, seq, event_type, json.dumps(payload, ensure_ascii=False), event["created_at"]),
            )
            self.conn.commit()
            self.cache.set_checkpoint(job_id, seq)
            self.cache.set_job_status(job_id, event_type)
            self._cv.notify_all()
            return event

    def listen(self, job_id: str, after_seq: int = 0, timeout: float = 15.0) -> Iterator[dict[str, Any]]:
        last = after_seq
        while True:
            with self._cv:
                rows = self.conn.execute(
                    """
                    SELECT seq, event_type, payload_json, created_at
                    FROM traces WHERE job_id=? AND seq>? ORDER BY seq
                    """,
                    (job_id, last),
                ).fetchall()
                if not rows:
                    job = self.get_job(job_id)
                    if job and job["status"] in {"succeeded", "failed", "insufficient"}:
                        return
                    # awaiting_review keeps the stream open
                    self._cv.wait(timeout=timeout)
                    continue
            for row in rows:
                last = row["seq"]
                yield {
                    "seq": row["seq"],
                    "event": row["event_type"],
                    "payload": json.loads(row["payload_json"]),
                    "created_at": row["created_at"],
                }

    def set_status(self, job_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE jobs SET status=?, updated_at=? WHERE job_id=?",
            (status, _now(), job_id),
        )
        self.conn.commit()
        self.publish(job_id, status, {"job_id": job_id, "status": status})

    def replay(self, job_id: str) -> dict[str, Any] | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        result = job.get("result") or {}
        return {
            "job_id": job_id,
            "status": job["status"],
            "result_version": job["result_version"],
            "state": {
                "question": result.get("question") or job["question"],
                "scope": result.get("scope") or job["scope"],
                "facts": result.get("facts") or [],
                "evidence_ids": result.get("evidence_ids") or [],
                "claims": result.get("claims") or [],
                "verdict": result.get("verdict"),
                "tool_trace": result.get("tool_trace") or [],
                "missing": result.get("missing") or [],
                "run_metrics": result.get("run_metrics") or {},
                "result_status": result.get("result_status"),
            },
            "note": "Replay restores task/graph state only. It does not rerun GPU jobs or restore object storage.",
        }

    def wait_hitl(self, job_id: str, claims: list[dict[str, Any]], timeout: float = 300.0) -> dict[str, Any]:
        event = threading.Event()
        with self._lock:
            self._hitl[job_id] = {
                "event": event,
                "claims": claims,
                "decision": None,
            }
        self.set_status(job_id, "awaiting_review")
        self.publish(job_id, "hitl", {"job_id": job_id, "claims": claims})
        event.wait(timeout=timeout)
        with self._lock:
            payload = self._hitl.pop(job_id, {})
        decision = payload.get("decision") or {"action": "timeout"}
        if self.get_job(job_id) and self.get_job(job_id)["status"] == "awaiting_review":
            self.set_status(job_id, "running")
        return decision

    def resolve_hitl(self, job_id: str, action: str, claims: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        with self._lock:
            pending = self._hitl.get(job_id)
            if pending is None:
                return {"ok": False, "error": "no_pending_hitl"}
            pending["decision"] = {"action": action, "claims": claims}
            pending["event"].set()
        return {"ok": True, "job_id": job_id, "action": action}

    def save_eval(self, report: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO eval_runs (eval_id, kind, summary_json, created_at) VALUES (?, ?, ?, ?)",
            (
                report["eval_id"],
                report["kind"],
                json.dumps(report["summary"], ensure_ascii=False),
                report["created_at"],
            ),
        )
        for item in report["items"]:
            self.conn.execute(
                "INSERT INTO eval_items (eval_id, case_id, score_json) VALUES (?, ?, ?)",
                (report["eval_id"], item["case_id"], json.dumps(item, ensure_ascii=False)),
            )
        self.conn.commit()

    def latest_eval(self, kind: str | None = None) -> dict[str, Any] | None:
        if kind:
            row = self.conn.execute(
                "SELECT * FROM eval_runs WHERE kind=? ORDER BY created_at DESC LIMIT 1",
                (kind,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        items = self.conn.execute(
            "SELECT case_id, score_json FROM eval_items WHERE eval_id=? ORDER BY id",
            (row["eval_id"],),
        ).fetchall()
        return {
            "eval_id": row["eval_id"],
            "kind": row["kind"],
            "created_at": row["created_at"],
            "summary": json.loads(row["summary_json"]),
            "items": [json.loads(item["score_json"]) for item in items],
            "not_production": True,
        }

    def latest_evals(self) -> dict[str, Any]:
        return {
            "system": self.latest_eval("system"),
            "baseline": self.latest_eval("baseline"),
            "langgraph": self.latest_eval("langgraph"),
            "retrieval": self.latest_eval("retrieval"),
            "cite_rule": "Only cite numbers from these eval_runs. Do not cite designed 87.5%/37.5%.",
            "not_production": True,
        }

    def list_evals(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT eval_id, kind, summary_json, created_at FROM eval_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "eval_id": row["eval_id"],
                "kind": row["kind"],
                "created_at": row["created_at"],
                "summary": json.loads(row["summary_json"]),
            }
            for row in rows
        ]

    def _job_row(self, row: Any) -> dict[str, Any]:
        result = json.loads(row["result_json"]) if row["result_json"] else None
        return {
            "job_id": row["job_id"],
            "question": row["question"],
            "scope": json.loads(row["scope_json"]),
            "status": row["status"],
            "result_version": row["result_version"],
            "result": result,
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


def build_service() -> JobService:
    return JobService(cache=build_cache())


SERVICE = build_service()
