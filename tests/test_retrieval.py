from eval_retrieval import RETRIEVAL_CASES, recall_at_k, run_retrieval_eval
from rag.runbook_index import rag_backend, search_runbooks


def test_hash_backend_default_in_tests(monkeypatch):
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    assert rag_backend() == "hash"


def test_missing_chroma_reports_hash(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "chroma")
    monkeypatch.setattr("rag.runbook_index._chroma_available", lambda: False)
    assert rag_backend() == "hash"


def test_oom_query_ranks_oom_runbook():
    result = search_runbooks("CUDA OOM correction head halve batch_size", k=3)
    ids = [hit["doc_id"] for hit in result["hits"]]
    assert "rb-oom" in ids


def test_retrieval_eval_has_ten_cases_and_positive_recall():
    assert len(RETRIEVAL_CASES) == 10
    report = run_retrieval_eval()
    assert report["kind"] == "retrieval"
    assert report["summary"]["n"] == 10
    assert report["summary"]["recall_at_k"] is not None
    assert report["summary"]["recall_at_k"] >= 0.6
    assert report["summary"]["citation_precision"] == 1.0
    oom = search_runbooks("CUDA OOM correction head halve batch_size", k=1)
    assert oom["hits"][0]["doc_id"] == "rb-oom"
    assert oom["hits"][0]["page"] == 2
    assert recall_at_k(["rb-oom"], ["rb-oom", "rb-missing-radar"], 3) == 1.0
