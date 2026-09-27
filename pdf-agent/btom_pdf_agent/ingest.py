"""Fixed ingest chain: split pages, OCR images, chunk, write Chroma.

The model does not choose this order. ``build_ingest_chain`` is a LangChain
runnable sequence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda

from btom_pdf_agent.chunking import build_page_chunks, page_title
from btom_pdf_agent.embeddings import HashEmbeddings
from btom_pdf_agent.ledger import ledger_documents
from btom_pdf_agent.ocr import build_ocr_engine, ocr_lines
from btom_pdf_agent.pdf_io import split_pdf
from btom_pdf_agent.retrieve import HybridIndex, documents_from_chunks


def build_ingest_chain():
    return (
        RunnableLambda(prepare)
        | RunnableLambda(attach_ocr)
        | RunnableLambda(make_chunks)
        | RunnableLambda(write_index)
    )


def prepare(payload: dict[str, Any]) -> dict[str, Any]:
    pdf_path = Path(payload["pdf_path"])
    pdf_bytes = pdf_path.read_bytes()
    state = payload.get("state")
    if state is None and payload.get("state_path"):
        state = json.loads(Path(payload["state_path"]).read_text(encoding="utf-8"))
    return {
        "pdf_path": str(pdf_path),
        "pdf_hash": hashlib.sha256(pdf_bytes).hexdigest(),
        "persist_dir": str(payload["persist_dir"]),
        "ocr_kind": payload.get("ocr_kind") or "fake",
        "pages": split_pdf(pdf_path),
        "state": state or {},
    }


def attach_ocr(payload: dict[str, Any]) -> dict[str, Any]:
    engine = build_ocr_engine(payload["ocr_kind"])
    ocr_by_page: dict[int, list[str]] = {}
    for page in payload["pages"]:
        lines: list[str] = []
        for image in page.get("images") or []:
            lines.extend(ocr_lines(engine, image))
        if lines:
            ocr_by_page[int(page["page"])] = lines
            page["ocr_text"] = "\n".join(lines)
        else:
            page["ocr_text"] = ""
    payload["ocr_lines"] = {str(key): value for key, value in ocr_by_page.items()}
    return payload


def _page_title(page: dict[str, Any]) -> str | None:
    title = page_title(page.get("text") or "")
    if title:
        return title
    if not (page.get("text") or "").strip():
        return page_title(page.get("ocr_text") or "")
    return None


def make_chunks(payload: dict[str, Any]) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    for page in payload["pages"]:
        page_no = int(page["page"])
        title = _page_title(page)
        blocks: list[tuple[str, str, str]] = []
        if page.get("text"):
            blocks.append(("text", "body", page["text"]))
        if page.get("ocr_text"):
            blocks.append(("ocr", "ocr", page["ocr_text"]))
        for kind, origin, text in blocks:
            chunks.extend(build_page_chunks(page_no, kind, origin, text, title=title))
    for index, document in enumerate(ledger_documents(payload.get("state") or {})):
        metadata = dict(document["metadata"])
        chunk_id = f"ledger-{metadata.get('kind')}-{index}"
        metadata["chunk_id"] = chunk_id
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": document["text"],
                "metadata": metadata,
            }
        )
    payload["chunks"] = chunks
    return payload


def write_index(payload: dict[str, Any]) -> dict[str, Any]:
    persist = Path(payload["persist_dir"])
    persist.mkdir(parents=True, exist_ok=True)
    cache_path = persist / f"embed-{payload['pdf_hash'][:16]}.json"
    embeddings = HashEmbeddings(cache_path=cache_path)
    documents = documents_from_chunks(payload["chunks"])
    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
        collection_name=f"pdf_{payload['pdf_hash'][:12]}",
        persist_directory=str(persist / "chroma"),
    )
    index = HybridIndex(payload["chunks"], embeddings, vectorstore)
    manifest = {
        "pdf_path": payload["pdf_path"],
        "pdf_hash": payload["pdf_hash"],
        "ocr_lines": payload["ocr_lines"],
        "chunks": payload["chunks"],
        "state": payload.get("state") or {},
    }
    (persist / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    payload["index"] = index
    payload["embeddings"] = embeddings
    return payload


def load_index(persist_dir: str | Path) -> dict[str, Any]:
    persist = Path(persist_dir)
    manifest = json.loads((persist / "manifest.json").read_text(encoding="utf-8"))
    cache_path = persist / f"embed-{manifest['pdf_hash'][:16]}.json"
    embeddings = HashEmbeddings(cache_path=cache_path)
    vectorstore = Chroma(
        collection_name=f"pdf_{manifest['pdf_hash'][:12]}",
        persist_directory=str(persist / "chroma"),
        embedding_function=embeddings,
    )
    index = HybridIndex(manifest["chunks"], embeddings, vectorstore)
    return {"manifest": manifest, "index": index, "embeddings": embeddings}
