"""Price-snapshot cache and rate-limit handling (Task 6.1).

Implements Requirements 16.1-16.4:

- **16.1** Flight/hotel prices are cached as a :class:`PriceSnapshot` with a TTL.
- **16.2** An unexpired cached snapshot is served *without* calling the provider.
- **16.3** Prices are always presented as point-in-time snapshots (every
  snapshot carries ``captured_at`` and ``is_snapshot=True``), never live values.
- **16.4** Provider rate-limit (HTTP 429) responses trigger bounded exponential
  backoff with jitter and, when retries are exhausted, fall back to the last
  known snapshot.

The Redis client is **injectable**: :class:`PriceCache` depends only on the
small :class:`RedisLike` structural interface, so tests pass an in-memory double
(:class:`InMemoryRedis`) and never require a live Redis server. The ``sleep``
and ``rng`` callables are likewise injectable for deterministic tests.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Rate-limit signalling
# =============================================================================


class RateLimitError(Exception):
    """Raised by a provider fetch callable when the upstream returns 429.

    Carries an optional ``retry_after`` (seconds) hint from the provider.
    """

    def __init__(self, message: str = "rate limited", *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class CacheUnavailable(Exception):
    """Raised when no snapshot is available and the provider cannot be reached."""


# =============================================================================
# Snapshot model
# =============================================================================


class PriceSnapshot(BaseModel):
    """A point-in-time price snapshot (Requirements 16.1, 16.3).

    ``is_snapshot`` is always ``True`` so consumers can never mistake a cached
    value for a live price; ``captured_at`` records when it was taken and
    ``ttl_seconds`` how long it stays fresh.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    value: dict[str, Any]
    captured_at: datetime
    ttl_seconds: int = Field(..., gt=0)
    is_snapshot: bool = True

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """True when the snapshot is older than its TTL."""
        now = now or datetime.now(timezone.utc)
        captured = self.captured_at
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=timezone.utc)
        age = (now - captured).total_seconds()
        return age >= self.ttl_seconds


# =============================================================================
# Injectable Redis interface + in-memory double
# =============================================================================


class RedisLike(Protocol):
    """The minimal Redis surface :class:`PriceCache` depends on."""

    def get(self, key: str) -> Any: ...

    def set(self, key: str, value: str) -> Any: ...

    def setex(self, key: str, ttl_seconds: int, value: str) -> Any: ...


class InMemoryRedis:
    """A tiny in-memory Redis double for tests (no live server required).

    Honours TTL via an injectable ``clock`` so expiry can be driven
    deterministically in tests.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        # key -> (value, expiry_epoch | None)
        self._store: dict[str, tuple[str, float | None]] = {}

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if expiry is not None and self._clock() >= expiry:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str) -> bool:
        self._store[key] = (value, None)
        return True

    def setex(self, key: str, ttl_seconds: int, value: str) -> bool:
        self._store[key] = (value, self._clock() + ttl_seconds)
        return True


# =============================================================================
# Price cache
# =============================================================================

_LAST_PREFIX = "last:"


class PriceCache:
    """Caches flight/hotel price snapshots with backoff/retry on rate limits."""

    def __init__(
        self,
        client: RedisLike,
        *,
        default_ttl: int = 900,
        max_retries: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 30.0,
        jitter: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        self._client = client
        self._default_ttl = default_ttl
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._now = now or (lambda: datetime.now(timezone.utc))

    # --- snapshot storage / retrieval ------------------------------------

    def store_snapshot(
        self, key: str, value: dict[str, Any], *, ttl: int | None = None
    ) -> PriceSnapshot:
        """Persist a value as a point-in-time snapshot with TTL (Req 16.1)."""
        ttl_seconds = ttl if ttl is not None else self._default_ttl
        snapshot = PriceSnapshot(
            key=key,
            value=value,
            captured_at=self._now(),
            ttl_seconds=ttl_seconds,
        )
        payload = snapshot.model_dump_json()
        # TTL-managed copy for "serve unexpired" + indefinite copy for fallback.
        self._client.setex(key, ttl_seconds, payload)
        self._client.set(_LAST_PREFIX + key, payload)
        return snapshot

    def get_snapshot(self, key: str) -> PriceSnapshot | None:
        """Return the unexpired snapshot for ``key``, or ``None`` (Req 16.2)."""
        raw = self._client.get(key)
        if raw is None:
            return None
        snapshot = self._deserialize(raw)
        if snapshot is None or snapshot.is_expired(now=self._now()):
            return None
        return snapshot

    def get_last_snapshot(self, key: str) -> PriceSnapshot | None:
        """Return the last known snapshot ignoring expiry (Req 16.4 fallback)."""
        raw = self._client.get(_LAST_PREFIX + key)
        if raw is None:
            return None
        return self._deserialize(raw)

    # --- high-level fetch ------------------------------------------------

    def get_or_fetch(
        self,
        key: str,
        fetch: Callable[[], dict[str, Any]],
        *,
        ttl: int | None = None,
    ) -> PriceSnapshot:
        """Serve an unexpired snapshot or fetch+cache a fresh one.

        - If an unexpired snapshot exists, it is returned and ``fetch`` is NOT
          called (Requirement 16.2).
        - Otherwise ``fetch`` is invoked with bounded exponential backoff +
          jitter on :class:`RateLimitError` (Requirement 16.4). When retries are
          exhausted, the last known snapshot is returned if one exists; only if
          none exists is :class:`CacheUnavailable` raised.
        """
        cached = self.get_snapshot(key)
        if cached is not None:
            return cached
        value = self._fetch_with_backoff(key, fetch)
        if value is None:
            # Retries exhausted: fall back to the last known snapshot.
            fallback = self.get_last_snapshot(key)
            if fallback is not None:
                return fallback
            raise CacheUnavailable(
                f"provider rate-limited and no snapshot available for {key!r}"
            )
        return self.store_snapshot(key, value, ttl=ttl)

    # --- internals -------------------------------------------------------

    def _fetch_with_backoff(
        self, key: str, fetch: Callable[[], dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Call ``fetch`` retrying on rate limits; return ``None`` if exhausted."""
        attempt = 0
        while True:
            try:
                return fetch()
            except RateLimitError as exc:
                if attempt >= self._max_retries:
                    return None
                self._sleep(self._backoff_delay(attempt, exc.retry_after))
                attempt += 1

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        """Exponential backoff capped at ``max_delay``, with optional jitter."""
        if retry_after is not None:
            base = min(retry_after, self._max_delay)
        else:
            base = min(self._base_delay * (2**attempt), self._max_delay)
        if self._jitter:
            # Full jitter in [0, base], so delays stay bounded by max_delay.
            return self._rng.uniform(0.0, base)
        return base

    @staticmethod
    def _deserialize(raw: Any) -> PriceSnapshot | None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return PriceSnapshot.model_validate(json.loads(raw))
        except (ValueError, TypeError):
            return None


__all__ = [
    "RateLimitError",
    "CacheUnavailable",
    "PriceSnapshot",
    "RedisLike",
    "InMemoryRedis",
    "PriceCache",
]
