"""Hash embeddings with an on-disk cache keyed by the source file hash.

No model weights are downloaded. Layer and head numbers occupy fixed axes so a
query for one attention head stays near that ledger chunk.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from langchain_core.embeddings import Embeddings

_LAYER_SLOTS = 64
_HEAD_SLOTS = 64
_HASH_SLOTS = 32
DIM = _LAYER_SLOTS + _HEAD_SLOTS + _HASH_SLOTS
_LAYER_RE = re.compile(r"layer:\s*(\d+)")
_HEAD_RE = re.compile(r"head:\s*(\d+)")
_TOKEN_RE = re.compile(r"token:\s*([A-Za-z0-9_\-]+)")
_WORD_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_\-]{2,}")


def _embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for match in _LAYER_RE.finditer(text):
        layer = int(match.group(1))
        vec[min(layer, _LAYER_SLOTS - 1)] += 12.0
    for match in _HEAD_RE.finditer(text):
        head = int(match.group(1))
        vec[_LAYER_SLOTS + min(head, _HEAD_SLOTS - 1)] += 12.0
    for match in _TOKEN_RE.finditer(text):
        _add_hash(vec, "token:" + match.group(1), 1.2)
    for token in _WORD_RE.findall(text):
        _add_hash(vec, token, 0.35)
    return _normalize(vec)


def _add_hash(vec: list[float], key: str, weight: float) -> None:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    base = _LAYER_SLOTS + _HEAD_SLOTS
    for offset in range(4):
        vec[base + digest[offset] % _HASH_SLOTS] += weight


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm == 0.0:
        vec = list(vec)
        vec[-1] = 1.0
        norm = 1.0
    return [value / norm for value in vec]


class HashEmbeddings(Embeddings):
    """Deterministic embeddings. ``calls`` counts uncached computations."""

    def __init__(self, cache_path: str | Path | None = None) -> None:
        self.cache_path = Path(cache_path) if cache_path else None
        self.calls = 0
        self._memory: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            loaded = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._memory = {key: list(value) for key, value in loaded.items()}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = [self._one(text) for text in texts]
        self._flush()
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self._one(text)
        self._flush()
        return vector

    def _one(self, text: str) -> list[float]:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._memory.get(key)
        if cached is not None:
            return list(cached)
        self.calls += 1
        vector = _embed(text)
        self._memory[key] = vector
        return list(vector)

    def _flush(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._memory, ensure_ascii=False),
            encoding="utf-8",
        )
