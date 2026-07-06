from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from app.core.security.exceptions import ConfigurationError


logger = logging.getLogger(__name__)


@dataclass
class _RateLimitRecord:
    count: int
    expires_at: float


class RateLimiter:
    def __init__(self, redis_url: str | None = None, strict_mode: bool = False) -> None:
        self._redis = None
        self._records: dict[str, _RateLimitRecord] = {}
        self._lock = threading.Lock()

        if redis_url:
            try:
                import redis  # type: ignore

                self._redis = redis.from_url(redis_url, decode_responses=True)
                logger.info("RateLimiter: using Redis backend at %s", redis_url)
            except ImportError:
                if strict_mode:
                    raise ConfigurationError(
                        "RateLimiter: redis package is required but not installed."
                    )
                logger.warning(
                    "RateLimiter: redis package not installed; falling back to "
                    "in-process memory store. This is NOT safe for multi-worker deployments."
                )
                self._redis = None
        else:
            if strict_mode:
                raise ConfigurationError(
                    "RateLimiter: REDIS_URL is required in strict mode (production)."
                )
            logger.warning(
                "RateLimiter: REDIS_URL is not set; using in-process memory store. "
                "Rate-limit counts will NOT be shared across workers. "
                "Set REDIS_URL or run with a single worker (--workers 1)."
            )

    def check(self, key: str, limit: int, window_seconds: int) -> bool:
        if self._redis is not None:
            current = self._redis.incr(key)
            if current == 1:
                self._redis.expire(key, window_seconds)
            return current <= limit

        now = time.monotonic()
        with self._lock:
            expired_keys = [record_key for record_key, record in self._records.items() if record.expires_at <= now]
            for expired_key in expired_keys:
                self._records.pop(expired_key, None)
            record = self._records.get(key)
            if record is None or record.expires_at <= now:
                self._records[key] = _RateLimitRecord(count=1, expires_at=now + window_seconds)
                return True

            record.count += 1
            return record.count <= limit
