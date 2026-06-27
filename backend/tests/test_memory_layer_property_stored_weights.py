"""Property test for Property 15: stored preference weights stay in [0, 1].

Feature: wayfinder-travel-planner, Property 15

*For any* sequence of explicit and implicit signals written to the memory layer
(plus user overrides), every stored preference weight SHALL remain within the
inclusive range 0.0 to 1.0.

Validates: Requirements 10.3

The test drives the real Memory Layer write paths — ``write_signals`` (explicit
+ implicit signal merges) and ``override_preference`` (user-supplied overrides,
including out-of-range values that MUST be rejected without corrupting stored
state) — against an in-memory SQLite database, mirroring the existing
memory-layer unit tests. After each operation in a randomly generated sequence
it asserts the [0, 1] invariant holds for every persisted weight.
"""

from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.memory.memory_layer import (
    PreferenceOverrideError,
    UnknownPreferenceTopicError,
    override_preference,
    write_signals,
)
from app.models.db import Base, User, UserPreference
from app.models.schemas import FEATURES

# Topics drawn from the required feature set plus a couple of extra names so the
# generated signals exercise both known and novel topics.
_TOPICS = list(FEATURES) + ["custom_topic_a", "custom_topic_b"]

# Raw signal/override values: deliberately span well below 0, well above 1, the
# exact boundaries, in-range values, and the non-finite values (nan, +/-inf)
# that the clamping/validation logic must also keep in range.
_VALUES = st.one_of(
    st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.sampled_from([0.0, 1.0, -0.0, math.nan, math.inf, -math.inf, -5.0, 5.0]),
)

_signal_dict = st.dictionaries(st.sampled_from(_TOPICS), _VALUES, max_size=len(_TOPICS))

# An operation is either a signal write (explicit + implicit maps) or a user
# preference override (topic + value).
_write_op = st.tuples(st.just("write"), _signal_dict, _signal_dict)
_override_op = st.tuples(st.just("override"), st.sampled_from(_TOPICS), _VALUES)
_operations = st.lists(st.one_of(_write_op, _override_op), max_size=25)


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


def _assert_all_weights_in_unit_range(session: Session, user_id) -> None:
    weights = list(
        session.scalars(
            select(UserPreference.weight).where(UserPreference.user_id == user_id)
        )
    )
    for weight in weights:
        assert math.isfinite(weight), f"stored weight {weight!r} is not finite"
        assert 0.0 <= weight <= 1.0, f"stored weight {weight!r} escaped [0, 1]"


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(operations=_operations)
def test_stored_preference_weights_stay_in_unit_range(operations) -> None:
    """Property 15: every persisted preference weight stays within [0, 1].

    Feature: wayfinder-travel-planner, Property 15
    Validates: Requirements 10.3
    """
    session = _new_session()
    try:
        user = User(email="prop15@example.com", prefs={})
        session.add(user)
        session.commit()

        for op in operations:
            if op[0] == "write":
                _, explicit, implicit = op
                write_signals(
                    session,
                    "trip-prop15",
                    user.id,
                    explicit=explicit or None,
                    implicit=implicit or None,
                )
            else:
                _, topic, value = op
                try:
                    override_preference(session, user.id, topic, value)
                except (PreferenceOverrideError, UnknownPreferenceTopicError):
                    # Out-of-range overrides are rejected and unknown topics have
                    # nothing to replace; in both cases stored state is left
                    # unchanged, so the invariant must still hold below.
                    session.rollback()

            # Invariant after every operation in the sequence.
            _assert_all_weights_in_unit_range(session, user.id)
    finally:
        session.close()
