from __future__ import annotations

import time
from typing import Any

IDEM_TTL_SEC = 10 * 60
RATE_LIMIT = 8
RATE_WINDOW_SEC = 10 * 60


class MemoryCache:
    kind = "memory"

    def __init__(self) -> None:
        self._idem: dict[str, tuple[str, float]] = {}
        self._hits: list[float] = []

    def check_rate_limit(self, user: str = "demo") -> bool:
        del user
        now = time.time()
        self._hits = [item for item in self._hits if now - item < RATE_WINDOW_SEC]
        if len(self._hits) >= RATE_LIMIT:
            return False
        self._hits.append(now)
        return True

    def lookup_idempotent(self, key: str) -> str | None:
        now = time.time()
        expired = [item for item, (_, exp) in self._idem.items() if exp <= now]
        for item in expired:
            self._idem.pop(item, None)
        hit = self._idem.get(key)
        return hit[0] if hit else None

    def remember_idempotent(self, key: str, job_id: str) -> None:
        self._idem[key] = (job_id, time.time() + IDEM_TTL_SEC)

    def set_job_status(self, job_id: str, status: str) -> None:
        return None

    def set_checkpoint(self, job_id: str, seq: int) -> None:
        return None


class RedisCache:
    kind = "redis"

    def __init__(self, url: str) -> None:
        import redis

        self.client = redis.Redis.from_url(url, decode_responses=True)

    def check_rate_limit(self, user: str = "demo") -> bool:
        key = f"rl:{user}"
        now = time.time()
        pipe = self.client.pipeline()
        pipe.zremrangebyscore(key, 0, now - RATE_WINDOW_SEC)
        pipe.zcard(key)
        pipe.expire(key, RATE_WINDOW_SEC)
        _, count, _ = pipe.execute()
        if int(count) >= RATE_LIMIT:
            return False
        self.client.zadd(key, {str(now): now})
        self.client.expire(key, RATE_WINDOW_SEC)
        return True

    def lookup_idempotent(self, key: str) -> str | None:
        return self.client.get(f"idem:{key}")

    def remember_idempotent(self, key: str, job_id: str) -> None:
        self.client.setex(f"idem:{key}", IDEM_TTL_SEC, job_id)

    def set_job_status(self, job_id: str, status: str) -> None:
        self.client.setex(f"job:{job_id}", IDEM_TTL_SEC, status)

    def set_checkpoint(self, job_id: str, seq: int) -> None:
        self.client.set(f"ckpt:{job_id}", str(seq))


def build_cache() -> MemoryCache | RedisCache:
    import os

    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return MemoryCache()
    try:
        cache = RedisCache(url)
        cache.client.ping()
        return cache
    except Exception:
        return MemoryCache()
