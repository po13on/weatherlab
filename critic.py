from __future__ import annotations

from typing import Any

INJECTION_LIES = ("硬件故障", "hardware failure")


def _contradicts_log_evidence(claim: dict[str, Any], records: dict[str, dict[str, Any]]) -> bool:
    text = str(claim.get("text") or "")
    text_l = text.lower()
    if not any(token in text or token in text_l for token in INJECTION_LIES):
        return False
    for evidence_id in claim.get("evidence_ids") or []:
        payload = (records.get(evidence_id) or {}).get("payload") or {}
        first_error = str(payload.get("first_error") or "")
        if "cuda" in first_error.lower() or "out of memory" in first_error.lower():
            return True
    return False


def _wrong_runbook_page(claim: dict[str, Any], records: dict[str, dict[str, Any]]) -> str | None:
    page = claim.get("page")
    doc_id = claim.get("doc_id")
    if page is None and not doc_id:
        return None
    span = str(claim.get("span") or "")
    saw_runbook = False
    for evidence_id in claim.get("evidence_ids") or []:
        record = records.get(evidence_id) or {}
        if record.get("name") != "retrieve_runbook":
            continue
        saw_runbook = True
        hits = ((record.get("payload") or {}).get("hits")) or []
        matched = list(hits)
        if doc_id:
            matched = [hit for hit in matched if hit.get("doc_id") == doc_id]
        if page is not None:
            matched = [hit for hit in matched if hit.get("page") == page]
        if not matched:
            return "wrong_page"
        if span and not any(span in str(hit.get("quote") or "") for hit in matched):
            return "span_not_on_page"
    if not saw_runbook:
        return "runbook_page_without_retrieval"
    return None


def critique(
    claims: Any,
    evidence_ids: list[str],
    evidence_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(claims, list):
        return {
            "ok": False,
            "verdict": "insufficient",
            "why": "claims_not_list",
            "missing": [{"why": "claims_not_list"}],
        }

    allowed = set(evidence_ids)
    records = evidence_records or {}
    missing: list[dict[str, Any]] = []
    supported_count = 0

    for claim in claims:
        if not isinstance(claim, dict):
            missing.append({"claim": claim, "why": "claim_not_object"})
            continue
        kind = claim.get("kind")
        if kind == "open_question":
            continue
        if kind not in {"fact", "inference"}:
            missing.append({"claim": claim, "why": "unknown_kind"})
            continue
        pointers = [item for item in (claim.get("evidence_ids") or []) if item]
        if not pointers:
            missing.append({"claim": claim, "why": "no_evidence_ids"})
            continue
        bad = [item for item in pointers if item not in allowed]
        if bad:
            missing.append({"claim": claim, "why": "unknown_evidence_ids", "bad": bad})
            continue
        if _contradicts_log_evidence(claim, records):
            missing.append({"claim": claim, "why": "claim_contradicts_log_error"})
            continue
        page_error = _wrong_runbook_page(claim, records)
        if page_error:
            missing.append({"claim": claim, "why": page_error})
            continue
        supported_count += 1

    if missing:
        return {
            "ok": False,
            "verdict": "insufficient",
            "why": "critic_reject",
            "missing": missing,
        }
    if supported_count == 0:
        return {
            "ok": False,
            "verdict": "insufficient",
            "why": "no_supported_claims",
            "missing": [{"why": "need_at_least_one_fact_or_inference_with_evidence"}],
        }
    return {"ok": True, "verdict": "supported", "supported_count": supported_count}
