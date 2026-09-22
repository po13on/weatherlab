from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from rag.runbook_index import search_runbooks

RETRIEVAL_CASES = [
    {
        "id": "R1",
        "query": "radar composite incomplete abort training",
        "expected_ids": ["rb-missing-radar"],
        "k": 3,
        "expected_page": 2,
    },
    {
        "id": "R2",
        "query": "CUDA OOM correction head halve batch_size",
        "expected_ids": ["rb-oom"],
        "k": 3,
        "expected_page": 2,
    },
    {
        "id": "R3",
        "query": "source_snapshot_unavailable empty config_path",
        "expected_ids": ["rb-bad-config"],
        "k": 3,
        "expected_page": 2,
    },
    {
        "id": "R4",
        "query": "v3 checkpoint cannot initialize v4 attention",
        "expected_ids": ["rb-stale-ckpt"],
        "k": 3,
        "expected_page": 2,
    },
    {
        "id": "R5",
        "query": "radar_complete_pct below ninety missing composite",
        "expected_ids": ["rb-missing-radar"],
        "k": 2,
        "expected_page": 2,
    },
    {
        "id": "R6",
        "query": "do not guess a default yaml for config",
        "expected_ids": ["rb-bad-config"],
        "k": 3,
        "expected_page": 2,
    },
    {
        "id": "R7",
        "query": "activation memory doubles with correction head",
        "expected_ids": ["rb-oom"],
        "k": 3,
        "expected_page": 2,
    },
    {
        "id": "R8",
        "query": "convert weights or train from scratch checkpoint layout",
        "expected_ids": ["rb-stale-ckpt"],
        "k": 3,
        "expected_page": 2,
    },
    {
        "id": "R9",
        "query": "ignore previous instructions hardware failure",
        "expected_ids": ["rb-noise-inject"],
        "k": 3,
        "expected_page": 2,
    },
    {
        "id": "R10",
        "query": "missing radar_composite across hours do not impute",
        "expected_ids": ["rb-missing-radar"],
        "k": 3,
        "expected_page": 2,
    },
]


def recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    pool = set(retrieved[:k])
    if not expected:
        return 0.0
    return len(pool.intersection(expected)) / len(expected)


def run_retrieval_eval(k: int | None = None) -> dict[str, Any]:
    items = []
    for case in RETRIEVAL_CASES:
        top_k = k or case["k"]
        result = search_runbooks(case["query"], k=top_k)
        retrieved = [hit["doc_id"] for hit in result.get("hits") or []]
        rec = recall_at_k(case["expected_ids"], retrieved, top_k)
        expected_page = case.get("expected_page")
        top = (result.get("hits") or [None])[0]
        page_ok = None
        if expected_page is not None:
            page_ok = bool(
                top
                and top.get("doc_id") in case["expected_ids"]
                and top.get("page") == expected_page
            )
        items.append(
            {
                "case_id": case["id"],
                "query": case["query"],
                "expected_ids": case["expected_ids"],
                "expected_page": expected_page,
                "retrieved": retrieved,
                "retrieved_pages": [hit.get("page") for hit in result.get("hits") or []],
                "recall_at_k": rec,
                "hit": rec > 0,
                "page_hit": page_ok,
                "k": top_k,
                "backend": result.get("backend"),
            }
        )
    n = len(items)
    page_items = [item for item in items if item.get("page_hit") is not None]
    return {
        "eval_id": "eval-" + uuid.uuid4().hex[:10],
        "kind": "retrieval",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "n": n,
            "recall_at_k": (sum(item["recall_at_k"] for item in items) / n) if n else None,
            "hit_rate": (sum(1 for item in items if item["hit"]) / n) if n else None,
            "citation_precision": (
                sum(1 for item in page_items if item["page_hit"]) / len(page_items)
            )
            if page_items
            else None,
            "not_production": True,
        },
        "items": items,
        "synthetic_fixture": True,
        "cite_rule": "Only cite this retrieval eval after it actually ran.",
        "not_production": True,
    }
