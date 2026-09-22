from cache import MemoryCache, build_cache


def test_build_cache_defaults_to_memory(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    cache = build_cache()
    assert isinstance(cache, MemoryCache)
    assert cache.kind == "memory"
    assert cache.check_rate_limit("demo") is True
    cache.remember_idempotent("k", "job-1")
    assert cache.lookup_idempotent("k") == "job-1"
