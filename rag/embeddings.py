from __future__ import annotations

import hashlib
import math


def hash_embed(text: str, dim: int = 64) -> list[float]:
    vec = [0.0] * dim
    for token in (text or "").lower().split():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        vec[int(digest, 16) % dim] += 1.0
    norm = math.sqrt(sum(item * item for item in vec))
    if not norm:
        return vec
    return [item / norm for item in vec]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
