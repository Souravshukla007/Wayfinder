"""Price-snapshot cache tests (Tasks 6.2 and 6.3).

Covers:
- **Property 27: Cache snapshot behavior** (Task 6.2) - price retrievals are
  cached as point-in-time snapshots, unexpired snapshots are served without
  calling the provider, and presented prices are snapshots not live values.
  Validates: Requirements 16.1, 16.2, 16.3.
  Tagged: Feature: wayfinder-travel-planner, Property 27.
- **Rate-limit backoff/retry** (Task 6.3) - simulated 429s produce bounded
  retries with backoff and fall back to the last snapshot.
  Validates: Requirement 16.4.

All tests use the in-memory Redis double and injected sleep/rng/clock, so no
live Redis server is required and timing is deterministic.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.cache.redis_cache import (
    CacheUnavailable,
    InMemoryRedis,
    PriceCache,
    PriceSnapshot,
    RateLimitError,
)


class _Clock:
    """A manually advanced clock for deterministic expiry tests."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def epoch(self) -> float:
        return self.t

    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.t, tz=timezone.utc)

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _make_cache(clock: _Clock | None = None, **kwargs) -> PriceCache:
    clock = clock or _Clock()
    client = InMemoryRedis(clock=clock.epoch)
    return PriceCache(
        client,
        sleep=lambda _d: None,  # no real waiting
        rng=random.Random(0),  # deterministic jitter
        now=clock.dt,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Property 27: Cache snapshot behavior
# Feature: wayfinder-travel-planner, Property 27
# Validates: Requirements 16.1, 16.2, 16.3
# ---------------------------------------------------------------------------

# JSON-serializable price payloads.
_price_value = st.dictionaries(
    keys=st.text(min_size=1, max_size=12),
    values=st.one_of(
        st.text(max_size=20),
        st.integers(min_value=0, max_value=10**9),
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        st.booleans(),
    ),
    max_size=6,
)


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    key=st.text(min_size=1, max_size=20).filter(lambda s: not s.startswith("last:")),
    value=_price_value,
    ttl=st.integers(min_value=1, max_value=86_400),
)
def test_property_27_cache_snapshot_behavior(
    key: str, value: dict, ttl: int
) -> None:
    clock = _Clock()
    cache = _make_cache(clock)

    calls = {"n": 0}

    def fetch() -> dict:
        calls["n"] += 1
        return value

    # First retrieval: provider is called once and the value is cached as a
    # point-in-time snapshot (Requirement 16.1, 16.3).
    snap = cache.get_or_fetch(key, fetch, ttl=ttl)
    assert isinstance(snap, PriceSnapshot)
    assert snap.is_snapshot is True            # presented as a snapshot (16.3)
    assert snap.captured_at is not None        # point-in-time stamp
    assert snap.value == value
    assert calls["n"] == 1

    # Second retrieval while unexpired: served from cache WITHOUT calling the
    # provider (Requirement 16.2).
    again = cache.get_or_fetch(key, fetch, ttl=ttl)
    assert again.value == value
    assert again.is_snapshot is True
    assert calls["n"] == 1, "unexpired snapshot must be served without provider call"

    # get_snapshot also returns the unexpired snapshot directly.
    direct = cache.get_snapshot(key)
    assert direct is not None and direct.value == value


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(ttl=st.integers(min_value=1, max_value=3600))
def test_snapshot_expires_after_ttl_and_refetches(ttl: int) -> None:
    clock = _Clock()
    cache = _make_cache(clock)
    calls = {"n": 0}

    def fetch() -> dict:
        calls["n"] += 1
        return {"price": calls["n"]}

    cache.get_or_fetch("k", fetch, ttl=ttl)
    assert calls["n"] == 1
    # Advance past the TTL: snapshot is expired and the provider is called again.
    clock.advance(ttl + 1)
    assert cache.get_snapshot("k") is None
    cache.get_or_fetch("k", fetch, ttl=ttl)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# Task 6.3: rate-limit backoff + retry, fallback to last snapshot
# Validates: Requirement 16.4
# ---------------------------------------------------------------------------


def test_429_bounded_retries_then_fallback_to_last_snapshot() -> None:
    clock = _Clock()
    delays: list[float] = []
    client = InMemoryRedis(clock=clock.epoch)
    cache = PriceCache(
        client,
        max_retries=4,
        base_delay=1.0,
        max_delay=100.0,
        jitter=False,  # deterministic exponential schedule
        sleep=delays.append,
        now=clock.dt,
    )

    # Seed a prior snapshot, then let it expire so a refetch is attempted.
    cache.store_snapshot("flight:DEL-KIX", {"price": "₹62,200"}, ttl=10)
    clock.advance(11)

    attempts = {"n": 0}

    def always_rate_limited() -> dict:
        attempts["n"] += 1
        raise RateLimitError(retry_after=None)

    result = cache.get_or_fetch("flight:DEL-KIX", always_rate_limited)

    # Bounded retries: initial attempt + max_retries.
    assert attempts["n"] == 5
    # Exponential backoff schedule between attempts (1, 2, 4, 8).
    assert delays == [1.0, 2.0, 4.0, 8.0]
    # Fell back to the last known snapshot rather than raising.
    assert result.value == {"price": "₹62,200"}
    assert result.is_snapshot is True


def test_backoff_delay_capped_at_max_delay() -> None:
    clock = _Clock()
    delays: list[float] = []
    client = InMemoryRedis(clock=clock.epoch)
    cache = PriceCache(
        client,
        max_retries=10,
        base_delay=1.0,
        max_delay=5.0,
        jitter=False,
        sleep=delays.append,
        now=clock.dt,
    )

    def always_rate_limited() -> dict:
        raise RateLimitError()

    with pytest.raises(CacheUnavailable):
        # No prior snapshot -> exhausting retries raises CacheUnavailable.
        cache.get_or_fetch("missing", always_rate_limited)

    assert delays, "expected backoff sleeps to have occurred"
    assert all(d <= 5.0 for d in delays), "delays must be capped at max_delay"


def test_jitter_keeps_delays_within_bound() -> None:
    clock = _Clock()
    delays: list[float] = []
    client = InMemoryRedis(clock=clock.epoch)
    cache = PriceCache(
        client,
        max_retries=6,
        base_delay=2.0,
        max_delay=20.0,
        jitter=True,
        sleep=delays.append,
        rng=random.Random(123),
        now=clock.dt,
    )

    def always_rate_limited() -> dict:
        raise RateLimitError()

    with pytest.raises(CacheUnavailable):
        cache.get_or_fetch("missing", always_rate_limited)

    assert all(0.0 <= d <= 20.0 for d in delays)


def test_retry_after_hint_is_respected() -> None:
    clock = _Clock()
    delays: list[float] = []
    client = InMemoryRedis(clock=clock.epoch)
    cache = PriceCache(
        client,
        max_retries=1,
        base_delay=1.0,
        max_delay=100.0,
        jitter=False,
        sleep=delays.append,
        now=clock.dt,
    )

    calls = {"n": 0}

    def rate_limited_once() -> dict:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError(retry_after=7.5)
        return {"price": "ok"}

    snap = cache.get_or_fetch("k", rate_limited_once)
    assert snap.value == {"price": "ok"}
    assert delays == [7.5]  # honoured the provider's retry-after hint
