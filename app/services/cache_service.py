# Small cache-related helpers used by the retrieval layer.
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any


@dataclass(frozen=True)
# Simple serializable view of functools.lru_cache statistics.
class CacheStats:
    hits: int
    misses: int
    maxsize: int | None
    currsize: int


_normalize_whitespace = re.compile(r"\s+")


# Normalize whitespace and casing so equivalent queries share the same cache key.
def normalize_query(query: str) -> str:
    """Normalize free-form user queries so semantically identical inputs reuse cache.

    Example:
        "  Analyze   NVDA risk  " -> "analyze nvda risk"
    """
    collapsed = _normalize_whitespace.sub(" ", query.strip())
    return collapsed.lower()


@lru_cache(maxsize=256)
def cached_text_response(cache_key: str, value: str = "") -> str:
    """Small helper cache for deterministic unit tests and simple string memoization."""
    return value


@lru_cache(maxsize=256)
def cached_object(cache_key: str, obj: Any = None) -> Any:
    """Generic object cache wrapper for future extension or testing."""
    return obj


# Read cache metrics from a function decorated with functools.lru_cache.
def get_lru_stats(func: Any) -> CacheStats:
    info = func.cache_info()
    return CacheStats(
        hits=info.hits,
        misses=info.misses,
        maxsize=info.maxsize,
        currsize=info.currsize,
    )
