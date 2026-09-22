from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from policy import PROMPT_VERSION
from agent import run_agent
from db import storage_kind
from eval_cases import EVAL_CASES
from eval_retrieval import run_retrieval_eval
from eval_suite import run_baseline_case, run_eval
from interview_bugs import BUGS
from jobs import SERVICE
from llm import complete, has_api_key, settings
from presets import PRESETS
from rag.ingest import corpus_stats
from rag.ocr import ocr_engine_name
from rag.runbook_index import rag_backend
from runners.langgraph_baseline import HAS_LANGGRAPH, run_langgraph_agent
from store import FIXTURE_PATH, experiments, load_fixture, runs

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="气象实验复核", version="0.5.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_eval_lock = threading.Lock()
_eval_status: dict[str, Any] = {"running": False, "error": None}


async def _call_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=2)
    scope: dict = Field(default_factory=dict)
    hitl: bool = False
    role: Literal["analyst", "guest"] = "analyst"


class ReviewRequest(BaseModel):
    action: Literal["approve", "reject", "edit_claims"]
    claims: Optional[List[dict]] = None


class EvalRequest(BaseModel):
    kinds: List[Literal["system", "baseline", "langgraph", "retrieval"]] = Field(
        default_factory=lambda: ["system", "baseline"]
    )


def hitl_mode() -> str:
    mode = os.getenv("HITL_MODE", "eval_skip").strip().lower()
    return mode if mode in {"off", "manual", "eval_skip"} else "eval_skip"


def _use_hitl(requested: bool) -> bool:
    mode = hitl_mode()
    if mode == "off":
        return False
    if mode == "manual":
        return True
    return bool(requested)


async def _run_job(job_id: str, question: str, scope: dict, use_hitl: bool, role: str = "analyst") -> None:
    SERVICE.mark_running(job_id)

    def gate(claims: list[dict[str, Any]]) -> dict[str, Any]:
        return SERVICE.wait_hitl(job_id, claims)

    try:
        result = await _call_in_thread(
            lambda: run_agent(
                question,
                scope,
                complete,
                on_event=lambda event, payload: SERVICE.publish(job_id, event, payload),
                hitl_gate=gate if use_hitl else None,
                role=role,
            )
        )
        status = result.get("verdict") or "failed"
        if status not in {"supported", "insufficient", "contradictory", "failed"}:
            status = "failed"
        mapped = {
            "supported": "succeeded",
            "insufficient": "insufficient",
            "contradictory": "insufficient",
            "failed": "failed",
        }[status]
        SERVICE.complete_cas(job_id, 0, mapped, result)
    except Exception:
        SERVICE.complete_cas(job_id, 0, "failed", None, error="llm_or_runtime_error")


def _system_runner(case: dict[str, Any]) -> dict[str, Any]:
    return run_agent(
        case["question"],
        case.get("scope") or {},
        complete,
        role=case.get("role") or "analyst",
    )


def _baseline_runner(case: dict[str, Any]) -> dict[str, Any]:
    return run_baseline_case(case, complete)


def _langgraph_runner(case: dict[str, Any]) -> dict[str, Any]:
    return run_langgraph_agent(
        case["question"],
        case.get("scope") or {},
        complete,
        role=case.get("role") or "analyst",
    )


async def _run_eval_job(kinds: list[str]) -> None:
    try:
        if "system" in kinds:
            report = await _call_in_thread(run_eval, _system_runner, EVAL_CASES, "system")
            SERVICE.save_eval(report)
        if "baseline" in kinds:
            report = await _call_in_thread(run_eval, _baseline_runner, EVAL_CASES, "baseline")
            SERVICE.save_eval(report)
        if "langgraph" in kinds:
            if not HAS_LANGGRAPH:
                raise RuntimeError("langgraph extra requires Python 3.10+ and pip install -r requirements.txt")
            report = await _call_in_thread(run_eval, _langgraph_runner, EVAL_CASES, "langgraph")
            SERVICE.save_eval(report)
        if "retrieval" in kinds:
            report = await _call_in_thread(run_retrieval_eval)
            SERVICE.save_eval(report)
        with _eval_lock:
            _eval_status["error"] = None
    except Exception as exc:
        with _eval_lock:
            _eval_status["error"] = str(exc)
    finally:
        with _eval_lock:
            _eval_status["running"] = False


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    fixture_ok = FIXTURE_PATH.exists()
    cfg = settings()
    return {
        "ok": True,
        "has_api_key": has_api_key(),
        "model": cfg["model"] if has_api_key() else None,
        "requested_model": cfg.get("requested_model") if has_api_key() else None,
        "base_url": cfg["base_url"] if has_api_key() else None,
        "fixture_loaded": fixture_ok,
        "experiment_count": len(load_fixture()["experiments"]) if fixture_ok else 0,
        "storage": storage_kind(),
        "cache": getattr(SERVICE.cache, "kind", "memory"),
        "hitl_mode": hitl_mode(),
        "prompt_version": PROMPT_VERSION,
        "eval_n": len(EVAL_CASES),
        "article_type": "business_assistant_agent",
        "python": sys.version.split()[0],
        "min_python_core": "3.8",
        "min_python_extras": "3.10",
        "rag_backend": rag_backend(),
        "rag_unit": "page",
        "ocr_engine": ocr_engine_name(),
        "runbook_chunks": corpus_stats()["chunks"],
        "ocr_pages": corpus_stats()["ocr_pages"],
        "langgraph_available": HAS_LANGGRAPH,
        "not_a_microservice": True,
        "persistence": storage_kind(),
        "job_semantics": ["status", "checkpoint_events", "idempotency", "rate_limit"],
        "not_production": True,
        "cite_rule": "Only cite metrics from the latest eval_runs. Unrun numbers are forbidden.",
    }


@app.get("/presets")
def presets() -> dict:
    return {"presets": PRESETS}


@app.get("/experiments")
def list_experiments() -> dict:
    return {"experiments": list(experiments().values())}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = runs().get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown_run")
    return run


@app.get("/bugs")
def bugs() -> dict:
    return {
        "bugs": BUGS,
        "template": ["symptom", "wrong_guess", "root_cause", "fix", "next_question"],
        "note": "Record the incident before changing code.",
    }


@app.get("/jobs")
def list_jobs() -> dict:
    return {"jobs": SERVICE.list_jobs()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = SERVICE.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown_job")
    job["traces"] = SERVICE.traces(job_id)
    return job


@app.get("/jobs/{job_id}/replay")
def replay_job(job_id: str) -> dict:
    snapshot = SERVICE.replay(job_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="unknown_job")
    return snapshot


@app.get("/jobs/{job_id}/events")
def job_events(job_id: str, after: int = 0) -> StreamingResponse:
    if SERVICE.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="unknown_job")

    def stream():
        for event in SERVICE.listen(job_id, after_seq=after):
            yield f"event: {event['event']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/jobs/{job_id}/review")
def review_job(job_id: str, req: ReviewRequest) -> dict:
    if SERVICE.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="unknown_job")
    result = SERVICE.resolve_hitl(job_id, req.action, req.claims)
    if not result.get("ok"):
        raise HTTPException(status_code=409, detail=result.get("error") or "no_pending_hitl")
    return result


@app.get("/eval/runs")
def get_eval_runs() -> dict:
    payload = SERVICE.latest_evals()
    payload["history"] = SERVICE.list_evals()
    with _eval_lock:
        payload["running"] = _eval_status["running"]
        payload["error"] = _eval_status["error"]
    return payload


@app.post("/eval/runs")
async def start_eval(req: EvalRequest) -> dict:
    if not has_api_key():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is missing. Copy .env.example to .env and fill the key.",
        )
    kinds = req.kinds or ["system", "baseline"]
    with _eval_lock:
        if _eval_status["running"]:
            raise HTTPException(status_code=409, detail="eval_already_running")
        _eval_status["running"] = True
        _eval_status["error"] = None
    asyncio.create_task(_run_eval_job(list(kinds)))
    return {
        "status": "running",
        "kinds": kinds,
        "n_cases": len(EVAL_CASES),
        "cite_rule": "Wait for this run. Do not cite older designed percentages.",
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> dict:
    if not has_api_key():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is missing. Copy .env.example to .env and fill the key.",
        )
    if not req.scope:
        raise HTTPException(status_code=400, detail="scope is required")
    if not SERVICE.check_rate_limit():
        raise HTTPException(status_code=429, detail="rate_limited")
    existing = SERVICE.lookup_idempotent(req.question, req.scope, req.role)
    if existing:
        job = SERVICE.get_job(existing)
        if job:
            return {"job_id": existing, "status": job["status"], "idempotent": True}
    job = SERVICE.create_job(req.question, req.scope)
    SERVICE.remember_idempotent(req.question, req.scope, job["job_id"], req.role)
    asyncio.create_task(
        _run_job(job["job_id"], req.question, req.scope, _use_hitl(req.hitl), req.role)
    )
    return {"job_id": job["job_id"], "status": job["status"], "idempotent": False}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)
