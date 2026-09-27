"""Hybrid retrieval: Chroma vectors plus BM25, merged by reciprocal rank.

A hit is kept only when BM25 scores it above zero or the query names the same
layer/head/token as the chunk. Otherwise the result is empty, and the caller
must say the evidence is insufficient.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from btom_pdf_agent.embeddings import HashEmbeddings
from btom_pdf_agent.query_rewrite import rewrite_query

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_\-]+")
_STOP = set("的了和在是与或及中不也把被给对而从")
_LAYER_RE = re.compile(r"layer:\s*(\d+)")
_HEAD_RE = re.compile(r"head:\s*(\d+)")
_TOKEN_FIELD_RE = re.compile(r"token:\s*([A-Za-z0-9_\-]+)")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in _STOP}


class HybridIndex:
    def __init__(self, chunks: list[dict[str, Any]], embeddings: HashEmbeddings, vectorstore: Any) -> None:
        self.chunks = chunks
        self.by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.units = _units(chunks)
        self.unit_by_id = {unit["chunk_id"]: unit for unit in self.units}
        corpus = [tokenize(unit["text"]) or [""] for unit in self.units] or [[""]]
        self.bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        k: int = 4,
        *,
        page: int | None = None,
        origin: str | None = None,
        title: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search child text and return the parent passage.

        ``page``, ``origin`` (``body`` or ``ocr``), and ``title`` drop parents
        that do not carry that stored field. The original query is always
        ranked; ``rewrite_query`` only adds extra rankings.
        """

        if not self.units:
            return []
        variants = [query, *rewrite_query(query)]
        heads: list[list[str]] = []
        tails: list[list[str]] = []
        for variant in variants:
            head, tail = self._head_and_tail(variant, k=k, page=page, origin=origin, title=title)
            heads.append(head)
            tails.append(tail)
        allowed = {unit_id for ranking in heads for unit_id in ranking}
        fused_scores = _rrf(heads) if allowed else []
        fused = [unit_id for unit_id, _score in fused_scores if unit_id in allowed]
        score_of = {unit_id: score for unit_id, score in fused_scores if unit_id in allowed}
        ordered = list(fused)
        seen = set(ordered)
        for tail in tails:
            for unit_id in tail:
                if unit_id not in seen:
                    ordered.append(unit_id)
                    seen.add(unit_id)
        hits: list[dict[str, Any]] = []
        seen_parents: set[str] = set()
        for unit_id in ordered:
            parent_id = self.unit_by_id[unit_id]["parent_id"]
            if parent_id in seen_parents:
                continue
            if not self._passes(parent_id, page, origin, title):
                continue
            seen_parents.add(parent_id)
            parent = self.by_id[parent_id]
            hits.append(
                {
                    "chunk_id": parent_id,
                    "text": parent["text"],
                    "metadata": parent["metadata"],
                    "score": score_of.get(unit_id, 0.0),
                }
            )
            if len(hits) >= k:
                break
        return hits

    def _head_and_tail(
        self,
        query: str,
        k: int,
        page: int | None,
        origin: str | None,
        title: str | None,
    ) -> tuple[list[str], list[str]]:
        width = max(k * 2, 4)
        vector_ids = self._vector_ids(query, k=width, page=page, origin=origin, title=title)
        ranked = self._bm25_ranked(query, page=page, origin=origin, title=title)
        head_bm25 = ranked[:width]
        allowed = set(vector_ids) | set(head_bm25)
        fused = [unit_id for unit_id, _score in _rrf([vector_ids, head_bm25]) if unit_id in allowed]
        seen = set(fused)
        tail = [unit_id for unit_id in ranked if unit_id not in seen]
        return fused, tail

    def _passes(self, parent_id: str, page: int | None, origin: str | None, title: str | None) -> bool:
        meta = self.by_id[parent_id]["metadata"]
        if page is not None and int(meta.get("page") or -1) != int(page):
            return False
        if origin is not None and str(meta.get("origin") or "") != origin:
            return False
        if title is not None and str(meta.get("title") or "") != title:
            return False
        return True

    def _vector_ids(
        self,
        query: str,
        k: int,
        page: int | None,
        origin: str | None,
        title: str | None,
    ) -> list[str]:
        if not self.units:
            return []
        try:
            pairs = self.vectorstore.similarity_search_with_score(query, k=min(k, len(self.units)))
        except Exception:
            return []
        ids: list[str] = []
        for doc, _score in pairs:
            unit_id = str((doc.metadata or {}).get("chunk_id") or "")
            if not unit_id or unit_id not in self.unit_by_id:
                continue
            if not self._vector_ok(query, unit_id):
                continue
            parent_id = self.unit_by_id[unit_id]["parent_id"]
            if not self._passes(parent_id, page, origin, title):
                continue
            ids.append(unit_id)
        return ids

    def _vector_ok(self, query: str, unit_id: str) -> bool:
        """Keep a vector hit only when it shares a layer, head, or token field."""

        parent_id = self.unit_by_id[unit_id]["parent_id"]
        meta = self.by_id[parent_id]["metadata"]
        layers = {int(value) for value in _LAYER_RE.findall(query)}
        heads = {int(value) for value in _HEAD_RE.findall(query)}
        tokens = {value for value in _TOKEN_FIELD_RE.findall(query)}
        if layers or heads or tokens:
            if layers and int(meta.get("layer") or -1) not in layers:
                return False
            if heads and int(meta.get("head") or -1) not in heads:
                return False
            if tokens:
                listed = {item for item in str(meta.get("tokens") or "").split("|") if item}
                if not listed & tokens:
                    return False
            return True
        return False

    def _bm25_ranked(self, query: str, page: int | None, origin: str | None, title: str | None) -> list[str]:
        tokens = tokenize(query)
        needed = content_tokens(query)
        if not tokens or not needed or not self.units:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        ids: list[str] = []
        for index in ranked:
            if scores[index] <= 0:
                break
            unit = self.units[index]
            if not needed & content_tokens(unit["text"]):
                continue
            if not self._passes(unit["parent_id"], page, origin, title):
                continue
            ids.append(unit["chunk_id"])
        return ids


def _rrf(rankings: list[list[str]], constant: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _units(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for chunk in chunks:
        raw_children = chunk.get("children") or [{"chunk_id": chunk["chunk_id"], "text": chunk["text"]}]
        for child in raw_children:
            text = str(child.get("text") or "").strip()
            if not text:
                continue
            units.append(
                {
                    "chunk_id": str(child.get("chunk_id") or chunk["chunk_id"]),
                    "text": text,
                    "parent_id": str(chunk["chunk_id"]),
                }
            )
    return units


def _scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and not isinstance(value, bool)


def documents_from_chunks(chunks: list[dict[str, Any]]) -> list[Document]:
    """One Chroma document per child. The stored id is the child; search returns the parent."""

    documents: list[Document] = []
    for chunk in chunks:
        raw_children = chunk.get("children") or [{"chunk_id": chunk["chunk_id"], "text": chunk["text"]}]
        for child in raw_children:
            text = str(child.get("text") or "").strip()
            if not text:
                continue
            metadata = {
                key: value
                for key, value in (chunk.get("metadata") or {}).items()
                if _scalar(value)
            }
            metadata["chunk_id"] = str(child.get("chunk_id") or chunk["chunk_id"])
            metadata["parent_id"] = str(chunk["chunk_id"])
            documents.append(Document(page_content=text, metadata=metadata))
    return documents
