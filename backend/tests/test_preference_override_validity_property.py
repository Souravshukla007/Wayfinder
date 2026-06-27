"""Property test for Property 18: preference override validity.

Feature: wayfinder-travel-planner, Property 18

*For any* override value within 0.0 to 1.0 inclusive, the memory layer SHALL
replace the stored weight with that value and update its updated-at timestamp;
*for any* override value outside that range, it SHALL reject the override,
retain the previously stored weight unchanged, and return an error identifying
the invalid value.

Validates: Requirements 10.8, 10.9

The test drives the real ``override_preference`` write path against an in-memory
SQLite database (mirroring the existing memory-layer tests). For each generated
case it seeds a known stored preference (weight + ``updated_at``) and then:

- In-range branch (Requirement 10.8): asserts the stored weight is replaced with
  the supplied value, the ``updated_at`` timestamp is re-stamped to the new time,
  and the returned :class:`PreferenceEntry` reflects the override.
- Out-of-range branch (Requirement 10.9): asserts the override is rejected with a
  :class:`PreferenceOverrideError` whose payload *identifies the invalid value*,
  and that the previously stored weight and timestamp are retained unchanged.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.memory.memory_layer import (
    WEIGHT_MAX,
    WEIGHT_MIN,
    PreferenceOverrideError,
    _as_utc,
    override_preference,
)
from app.models.db import Base, User, UserPreference
from app.models.schemas import FEATURES

# A fixed reference instant; the seeded preference is stamped relative to this
# and overrides are applied at a strictly later instant so the re-stamp is
# observable.
_NOW = datetime(2024, 10, 1, 12, 0, tzinfo=timezone.utc)

_TOPICS = st.sampled_from(list(FEATURES) + ["custom_topic_a", "custom_topic_b"])

# A valid stored starting weight in [0, 1] for the seeded preference.
_STORED_WEIGHT = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

# In-range override values: the inclusive [0, 1] interval, plus the exact
# boundaries which must be accepted.
_IN_RANGE = st.one_of(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.sampled_from([WEIGHT_MIN, WEIGHT_MAX, 0.0, 1.0, -0.0]),
)

# Out-of-range override values: strictly below 0, strictly above 1, and the
# non-finite values (nan, +/-inf) that validation must also reject.
_OUT_OF_RANGE = st.one_of(
    st.floats(min_value=-1e6, max_value=-1e-6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1.0 + 1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.sampled_from([math.nan, math.inf, -math.inf, -0.01, 1.01, -5.0, 5.0]),
)

# Elapsed days between the seeded stamp and the override instant (> 0 so the
# re-stamp moves the timestamp forward).
_ELAPSED_DAYS = st.integers(min_value=1, max_value=3650)


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


def _seed_user_with_preference(
    session: Session, topic: str, weight: float, stamped_at: datetime
) -> User:
    user = User(email="prop18@example.com", prefs={})
    session.add(user)
    session.commit()
    session.add(
        UserPreference(
            user_id=user.id,
            topic=topic,
            weight=weight,
            source="implicit",
            updated_at=stamped_at,
        )
    )
    session.commit()
    return user


# ---------------------------------------------------------------------------
# Property 18 — in-range branch (Requirement 10.8)
# ---------------------------------------------------------------------------
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    topic=_TOPICS,
    stored=_STORED_WEIGHT,
    new_value=_IN_RANGE,
    elapsed_days=_ELAPSED_DAYS,
)
def test_in_range_override_replaces_weight_and_restamps(
    topic: str, stored: float, new_value: float, elapsed_days: int
) -> None:
    """Property 18 (in-range): replace stored weight and update timestamp.

    Feature: wayfinder-travel-planner, Property 18
    Validates: Requirements 10.8
    """
    session = _new_session()
    try:
        stamped_at = _NOW - timedelta(days=elapsed_days)
        user = _seed_user_with_preference(session, topic, stored, stamped_at)

        entry = override_preference(session, user.id, topic, new_value, now=_NOW)

        # Returned entry reflects the override exactly.
        assert entry.topic == topic
        assert entry.weight == new_value
        assert _as_utc(entry.updated_at) == _NOW

        # Persisted state: weight replaced and timestamp re-stamped to `now`,
        # with no duplicate rows created.
        session.expire_all()
        rows = list(
            session.scalars(
                select(UserPreference).where(
                    UserPreference.user_id == user.id,
                    UserPreference.topic == topic,
                )
            )
        )
        assert len(rows) == 1
        assert rows[0].weight == new_value
        assert _as_utc(rows[0].updated_at) == _NOW
        # The stored weight always lands back in the valid [0, 1] range.
        assert WEIGHT_MIN <= rows[0].weight <= WEIGHT_MAX
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Property 18 — out-of-range branch (Requirement 10.9)
# ---------------------------------------------------------------------------
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    topic=_TOPICS,
    stored=_STORED_WEIGHT,
    bad_value=_OUT_OF_RANGE,
    elapsed_days=_ELAPSED_DAYS,
)
def test_out_of_range_override_is_rejected_and_retains_prior_weight(
    topic: str, stored: float, bad_value: float, elapsed_days: int
) -> None:
    """Property 18 (out-of-range): reject, retain prior weight, identify value.

    Feature: wayfinder-travel-planner, Property 18
    Validates: Requirements 10.9
    """
    session = _new_session()
    try:
        stamped_at = _NOW - timedelta(days=elapsed_days)
        user = _seed_user_with_preference(session, topic, stored, stamped_at)

        try:
            override_preference(session, user.id, topic, bad_value, now=_NOW)
            raised = None
        except PreferenceOverrideError as exc:  # noqa: PERF203 - test clarity
            raised = exc

        # The override MUST be rejected with an error that identifies the
        # invalid value (Requirement 10.9).
        assert raised is not None, f"out-of-range value {bad_value!r} was not rejected"
        assert raised.value == bad_value or (
            math.isnan(bad_value) and isinstance(raised.value, float) and math.isnan(raised.value)
        )
        assert raised.topic == topic
        assert repr(bad_value) in str(raised)

        # The previously stored weight and timestamp are retained unchanged.
        session.expire_all()
        row = session.scalars(
            select(UserPreference).where(
                UserPreference.user_id == user.id,
                UserPreference.topic == topic,
            )
        ).one()
        assert row.weight == stored
        assert _as_utc(row.updated_at) == stamped_at
    finally:
        session.close()
