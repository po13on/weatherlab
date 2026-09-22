from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from eval_cases import EVAL_CASES
from store import load_fixture


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((p / 100.0) * (len(ordered) - 1)))
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def score_result(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    tools = [step.get("name") for step in result.get("tool_trace") or []]
    must_ok = all(name in tools for name in case.get("must_tools") or [])
    verdict = result.get("verdict")
    verdict_ok = verdict == case["expected_verdict"]
    claims = [
        claim
        for claim in result.get("claims") or []
        if isinstance(claim, dict) and claim.get("kind") in {"fact", "inference"}
    ]
    cited = [
        claim
        for claim in claims
        if claim.get("evidence_ids")
        and all(item in (result.get("evidence_ids") or []) for item in claim["evidence_ids"])
    ]
    citation_ok = (len(cited) / len(claims)) if claims else None
    blob = " ".join(str(claim.get("text") or "") for claim in claims)
    forbidden_hit = [
        token for token in case.get("forbidden_substrings") or [] if token and token in blob
    ]
    expected_status = case.get("expected_result_status")
    status_ok = (
        result.get("result_status") == expected_status if expected_status else None
    )
    q9_incomparable = False
    if case["id"] == "Q9":
        q9_incomparable = any(
            (step.get("observation") or {}).get("error") == "incomparable_metric"
            for step in result.get("tool_trace") or []
        )
    return {
        "case_id": case["id"],
        "verdict": verdict,
        "expected_verdict": case["expected_verdict"],
        "verdict_ok": verdict_ok,
        "must_tools_ok": must_ok,
        "tools": tools,
        "claim_count": len(claims),
        "cited_count": len(cited),
        "citation_support": citation_ok,
        "forbidden_hit": forbidden_hit,
        "status_ok": status_ok,
        "q9_incomparable": q9_incomparable,
        "steps_used": result.get("steps_used"),
        "category": case.get("category") or "normal",
        "result_status": result.get("result_status"),
        "wall_ms": (result.get("run_metrics") or {}).get("wall_ms"),
    }


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(items)
    verdict_hits = sum(1 for item in items if item.get("verdict_ok"))
    tool_hits = sum(1 for item in items if item.get("must_tools_ok"))
    refusal_items = [item for item in items if item.get("expected_verdict") == "insufficient"]
    refusal_hits = sum(1 for item in refusal_items if item.get("verdict") == "insufficient")
    cited = sum(item.get("cited_count") or 0 for item in items)
    claims = sum(item.get("claim_count") or 0 for item in items)
    steps = [item.get("steps_used") for item in items if item.get("steps_used") is not None]
    walls = [item.get("wall_ms") for item in items if item.get("wall_ms") is not None]
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_cat.setdefault(item.get("category") or "normal", []).append(item)
    return {
        "n": n,
        "verdict_accuracy": (verdict_hits / n) if n else None,
        "tool_selection_accuracy": (tool_hits / n) if n else None,
        "refusal_rate": (refusal_hits / len(refusal_items)) if refusal_items else None,
        "citation_support": (cited / claims) if claims else None,
        "avg_steps": (sum(steps) / len(steps)) if steps else None,
        "p50_wall_ms": _percentile(walls, 50),
        "p95_wall_ms": _percentile(walls, 95),
        "by_category": {
            cat: {
                "n": len(rows),
                "verdict_accuracy": (sum(1 for row in rows if row.get("verdict_ok")) / len(rows)),
            }
            for cat, rows in by_cat.items()
        },
        "not_production": True,
    }


def fixture_slice_for_case(case: dict[str, Any]) -> dict[str, Any]:
    data = load_fixture()
    scope = case.get("scope") or {}
    if scope.get("run_id"):
        run = next((item for item in data["runs"] if item["run_id"] == scope["run_id"]), None)
        return {"run": run}
    exp_ids = scope.get("exp_ids") or []
    return {
        "experiments": [item for item in data["experiments"] if item["exp_id"] in exp_ids],
        "runs": [item for item in data["runs"] if item["exp_id"] in exp_ids],
    }


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def run_baseline_case(
    case: dict[str, Any],
    completer: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Any],
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You may only use the JSON fixture slice. Tools are disabled. "
                "Reply with JSON: {verdict, claims:[{kind,text,evidence_ids}], missing}. "
                "verdict must be supported, insufficient, or contradictory."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": case["question"],
                    "scope": case.get("scope") or {},
                    "fixture": fixture_slice_for_case(case),
                },
                ensure_ascii=False,
            ),
        },
    ]
    response = completer(messages, [])
    text = ""
    try:
        text = response.choices[0].message.content or ""
    except Exception:
        text = ""
    parsed = extract_json_object(text)
    verdict = parsed.get("verdict")
    if verdict not in {"supported", "insufficient", "contradictory"}:
        verdict = "failed"
    return {
        "verdict": verdict,
        "question": case["question"],
        "scope": case.get("scope") or {},
        "claims": parsed.get("claims") or [],
        "facts": [],
        "evidence_ids": [],
        "tool_trace": [],
        "missing": parsed.get("missing") or [],
        "steps_used": 0,
        "synthetic_fixture": True,
        "not_production": True,
        "kind": "baseline",
    }


def run_eval(
    runner: Callable[[dict[str, Any]], dict[str, Any]],
    cases: list[dict[str, Any]] | None = None,
    kind: str = "system",
) -> dict[str, Any]:
    selected = cases or EVAL_CASES
    items = []
    for case in selected:
        result = runner(case)
        scored = score_result(case, result)
        scored["kind"] = kind
        items.append(scored)
    return {
        "eval_id": "eval-" + uuid.uuid4().hex[:10],
        "kind": kind,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summarize(items),
        "items": items,
        "synthetic_fixture": True,
    }
