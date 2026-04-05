from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=128)
def cached_text_response(cache_key: str) -> str:
    """Simple placeholder cache API for future extension.

    This function intentionally returns the cache key itself so the service can
    use Python's LRU machinery for deterministic unit testing and easy future
    replacement with Redis or diskcache.
    """
    return cache_key


@lru_cache(maxsize=128)
def cached_object(cache_key: str, obj: Any = None) -> Any:
    return obj
