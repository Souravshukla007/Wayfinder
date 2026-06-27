"""Property test for Property 16: time decay follows a 180-day half-life.

Feature: wayfinder-travel-planner, Property 16

*For any* stored signal with original weight ``w`` and elapsed time ``t`` days,
the effective contribution SHALL equal ``w * 0.5 ** (t / 180)``, reducing to 50%
of ``w`` at ``t = 180`` and decaying proportionally thereafter.

Validates: Requirements 10.5

The test exercises the real Memory Layer decay arithmetic on two surfaces:

1. :func:`decay_factor` directly — the pure 180-day half-life multiplier, with
   an injectable reference time so the half-life is exactly testable.
2. :func:`load_context` end-to-end — a preference persisted to an in-memory
   SQLite database with a known ``updated_at`` is loaded ``t`` days later and its
   returned (decayed) weight is asserted against the closed-form formula. This
   mirrors the in-memory-DB approach of the sibling memory-layer property tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.memory.memory_layer import HALF_LIFE_DAYS, decay_factor, load_context
from app.models.db import Base, User, UserPreference

# A fixed, timezone-aware reference instant. Using a constant "now" keeps the
# generated ages deterministic and avoids coupling the property to wall-clock
# time.
_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

# Original stored weights span the full valid [0, 1] preference range, including
# the exact boundaries.
_WEIGHTS = st.one_of(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.sampled_from([0.0, 1.0, 0.5]),
)

# Elapsed time since the signal was stamped, in days. Covers fresh signals,
# fractional ages, exactly one half-life, and several multiples of the
# half-life so the proportional ("...and thereafter") decay is exercised.
_ELAPSED_DAYS = st.one_of(
    st.floats(min_value=0.0, max_value=5.0 * HALF_LIFE_DAYS, allow_nan=False, allow_infinity=False),
    st.sampled_from([0.0, 1.0, HALF_LIFE_DAYS, 2.0 * HALF_LIFE_DAYS, 3.0 * HALF_LIFE_DAYS]),
)

# Absolute and relative tolerances for the floating-point comparison.
_ABS_TOL = 1e-9
_REL_TOL = 1e-9


def _new_session() -> Session:
    """Create a fresh in-memory SQLite session with the full ORM schema."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return factory()


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(weight=_WEIGHTS, elapsed_days=_ELAPSED_DAYS)
def test_decay_factor_follows_180_day_half_life(weight: float, elapsed_days: float) -> None:
    """Property 16: the decay multiplier is exactly ``0.5 ** (t / 180)``.

    Feature: wayfinder-travel-planner, Property 16
    Validates: Requirements 10.5
    """
    updated_at = _NOW - timedelta(days=elapsed_days)

    factor = decay_factor(updated_at, _NOW)
    expected_factor = 0.5 ** (elapsed_days / HALF_LIFE_DAYS)

    assert factor == expected_factor or abs(factor - expected_factor) <= _ABS_TOL, (
        f"decay factor {factor!r} != expected {expected_factor!r} for "
        f"elapsed_days={elapsed_days!r}"
    )

    # The effective contribution is the original weight scaled by the factor.
    effective = weight * factor
    expected_effective = weight * expected_factor
    assert abs(effective - expected_effective) <= _ABS_TOL + _REL_TOL * abs(expected_effective)

    # Half-life anchor: at exactly t = 180 days the contribution is 50% of w.
    if elapsed_days == HALF_LIFE_DAYS:
        assert abs(effective - 0.5 * weight) <= _ABS_TOL


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(weight=_WEIGHTS, elapsed_days=_ELAPSED_DAYS)
def test_load_context_applies_half_life_decay(weight: float, elapsed_days: float) -> None:
    """Property 16 end-to-end: ``load_context`` returns ``w * 0.5 ** (t / 180)``.

    Feature: wayfinder-travel-planner, Property 16
    Validates: Requirements 10.5
    """
    session = _new_session()
    try:
        user = User(email="prop16@example.com", prefs={})
        session.add(user)
        session.commit()

        updated_at = _NOW - timedelta(days=elapsed_days)
        # Persist a single known preference so the cold-start seeding path is
        # bypassed and the decayed weight reflects exactly this stored signal.
        session.add(
            UserPreference(
                user_id=user.id,
                topic="weather",
                weight=weight,
                source="explicit",
                updated_at=updated_at,
            )
        )
        session.commit()

        context = load_context(session, user.id, now=_NOW)
        loaded = context.topic_weights()
        assert "weather" in loaded

        expected = weight * (0.5 ** (elapsed_days / HALF_LIFE_DAYS))
        # load_context clamps the effective weight into [0, 1]; since the factor
        # is <= 1 for non-negative ages and weight <= 1, expected already lies in
        # range, so a direct comparison is valid.
        assert abs(loaded["weather"] - expected) <= _ABS_TOL + _REL_TOL * abs(expected), (
            f"load_context decayed weight {loaded['weather']!r} != expected "
            f"{expected!r} for weight={weight!r}, elapsed_days={elapsed_days!r}"
        )
    finally:
        session.close()
