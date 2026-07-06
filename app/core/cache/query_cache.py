from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time

from app.core.config.settings import AppSettings


logger = logging.getLogger(__name__)


class QueryCache:
    """
    In-memory or Redis-backed cache for Pinecone vector search results.
    Prevents repeated expensive Pinecone calls for frequent identical queries.
    """

    def __init__(self, settings: AppSettings) -> None:
        self.enabled = settings.query_cache_enabled
        self.ttl = settings.query_cache_ttl_seconds
        self._redis = None
        self._local_cache: dict[str, tuple[float, list]] = {}
        self._lock = threading.Lock()

        if self.enabled and settings.redis_url:
            try:
                import redis  # type: ignore

                self._redis = redis.from_url(settings.redis_url, decode_responses=True)
                logger.info("QueryCache: using Redis backend")
            except ImportError:
                logger.warning(
                    "QueryCache: redis package not installed; falling back to in-memory cache."
                )
        elif self.enabled:
            logger.info("QueryCache: using in-memory cache (no REDIS_URL provided)")

    def _normalize_query(self, query: str) -> str:
        """Normalize semantically similar queries to the same cache key input."""
        lowered = query.lower().strip()
        collapsed = re.sub(r"[^a-z0-9]+", " ", lowered)
        return re.sub(r"\s+", " ", collapsed).strip()

    def _hash_query(self, query: str, scope: str | None = None) -> str:
        """Normalize and hash the query plus scope to build a cache key."""
        normalized = self._normalize_query(query)
        normalized_scope = (scope or "default").strip().lower()
        raw_key = f"{normalized_scope}::{normalized}"
        return f"qc:{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()}"

    def get(self, query: str, scope: str | None = None) -> list | None:
        if not self.enabled:
            return None

        key = self._hash_query(query, scope)

        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    # In Redis we store the list of documents natively as JSON
                    logger.debug("QueryCache HIT (Redis): %s", query)
                    return json.loads(data)
            except Exception as e:
                logger.warning("QueryCache Redis GET failed: %s", e)
            return None

        # In-memory fallback
        now = time.monotonic()
        with self._lock:
            cached = self._local_cache.get(key)
            if cached:
                expires_at, data = cached
                if expires_at > now:
                    logger.debug("QueryCache HIT (Memory): %s", query)
                    return data
                else:
                    del self._local_cache[key]
        return None

    def set(self, query: str, results: list, scope: str | None = None) -> None:
        if not self.enabled or not results:
            return

        key = self._hash_query(query, scope)

        if self._redis:
            try:
                self._redis.setex(key, self.ttl, json.dumps(results))
            except Exception as e:
                logger.warning("QueryCache Redis SET failed: %s", e)
            return

        # In-memory fallback
        now = time.monotonic()
        with self._lock:
            # Simple eviction to prevent memory explosion if used heavily
            if len(self._local_cache) > 1000:
                # Evict the oldest 200 entries (20%) to avoid a thundering herd
                keys_to_evict = list(self._local_cache.keys())[:200]
                for k in keys_to_evict:
                    self._local_cache.pop(k, None)
            
            # Since dicts maintain insertion order, pop and re-insert to update LRU position
            if key in self._local_cache:
                self._local_cache.pop(key)
            self._local_cache[key] = (now + self.ttl, results)
