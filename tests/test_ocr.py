from rag.ingest import chunks, ocr_page
from rag.ocr import decode_gray_png, ocr_png, render_png
from rag.runbook_index import search_runbooks
from tools import dispatch


def test_template_ocr_roundtrip_png():
    blob = render_png("CUDA OOM HALVE BATCH")
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")
    assert decode_gray_png(blob)
    assert ocr_png(blob)["text"] == "CUDA OOM HALVE BATCH"


def test_blank_scan_fails():
    from rag.ocr import encode_gray_png

    blank = encode_gray_png([[255, 255], [255, 255]])
    assert ocr_png(blank)["ok"] is False


def test_scanned_runbook_page_is_indexed_from_ocr():
    scanned = [
        chunk
        for chunk in chunks()
        if chunk["doc_id"] == "rb-oom" and chunk["page"] == 3
    ]
    assert scanned
    assert scanned[0]["source"] == "ocr"
    assert scanned[0]["quote"] == "CUDA OOM HALVE BATCH"
    found = search_runbooks("CUDA OOM HALVE BATCH", k=3)
    top = found["hits"][0]
    assert top["doc_id"] == "rb-oom"
    assert top["page"] == 3
    assert top["source"] == "ocr"


def test_scan_index_uses_ocr_output_not_markdown(monkeypatch):
    import rag.ingest as ingest

    monkeypatch.setattr(
        ingest,
        "ocr_png",
        lambda blob: {"ok": True, "text": "DIFFERENT OCR TEXT", "engine": "template"},
    )
    ingest.load_corpus.cache_clear()
    try:
        scanned = [chunk for chunk in ingest.chunks() if chunk["source"] == "ocr"]
        assert scanned
        assert {chunk["quote"] for chunk in scanned} == {"DIFFERENT OCR TEXT"}
    finally:
        ingest.load_corpus.cache_clear()


def test_failed_ocr_does_not_index_markdown(monkeypatch):
    import rag.ingest as ingest

    monkeypatch.setattr(
        ingest,
        "ocr_png",
        lambda blob: {"ok": False, "text": "", "engine": "template", "why": "forced"},
    )
    ingest.load_corpus.cache_clear()
    try:
        try:
            ingest.chunks()
            raised = False
        except RuntimeError as exc:
            raised = "ocr failed" in str(exc)
        assert raised is True
    finally:
        ingest.load_corpus.cache_clear()


def test_ocr_page_tool_rereads_scan_and_rejects_text_page():
    scanned = ocr_page("rb-oom", 3)
    assert scanned["ok"] is True
    assert scanned["payload"]["text"] == "CUDA OOM HALVE BATCH"
    assert scanned["payload"]["engine"] == "template"
    denied = ocr_page("rb-oom", 2)
    assert denied["error"] == "not_a_scan"
    guest = dispatch("ocr_page", {"doc_id": "rb-oom", "page": 3}, role="guest")
    assert guest["error"] == "tool_denied"
