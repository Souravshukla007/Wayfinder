"""Property test for Property 17: explicit signals weighted at least twice implicit.

Feature: wayfinder-travel-planner, Property 17

*For any* topic with both an explicit stated signal and an implicit inferred
signal, the combination SHALL apply to the explicit signal a weight at least
twice that applied to the implicit signal.

Validates: Requirements 10.6

The Memory Layer combines a topic's explicit and implicit raw signals through
``combine_signal`` (a weighted average with explicit weight
``EXPLICIT_SIGNAL_WEIGHT`` and implicit weight ``IMPLICIT_SIGNAL_WEIGHT``) and
persists the merged weight via ``write_signals``. The "explicit weighted at
least twice implicit" rule is observable as the combination's *marginal
sensitivity*: changing the explicit input by some delta moves the combined
weight at least twice as much as changing the implicit input by the same delta.
This test asserts that sensitivity relationship across generated in-range
signal pairs, and grounds it in the real persistence path by writing the
signals to an in-memory database and reading the merged weight back.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.memory.memory_layer import (
    EXPLICIT_SIGNAL_WEIGHT,
    IMPLICIT_SIGNAL_WEIGHT,
    combine_signal,
    write_signals,
)
from app.models.db import Base, User, UserPreference

# Absolute tolerance for floating-point comparisons of the weighted average.
_TOL = 1e-9


@st.composite
def _signal_pair_with_perturbation(draw):
    """Draw an (explicit, implicit, delta) triple kept inside the unit range.

    ``explicit`` and ``implicit`` are in-range raw signals; ``delta`` is a
    non-negative perturbation small enough that ``explicit + delta`` and
    ``implicit + delta`` both remain within [0, 1]. Staying in range keeps
    ``combine_signal`` in its linear (un-clamped) regime so the marginal
    sensitivity to each input is exactly its configured weight ratio.
    """
    explicit = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    implicit = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    headroom = min(1.0 - explicit, 1.0 - implicit)
    delta = draw(st.floats(min_value=0.0, max_value=headroom, allow_nan=False, allow_infinity=False))
    return explicit, implicit, delta


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


@settings(max_examples=200, deadline=None)
@given(triple=_signal_pair_with_perturbation())
def test_explicit_signal_weighted_at_least_twice_implicit(triple) -> None:
    """Property 17: explicit signals weighted at least twice implicit.

    Feature: wayfinder-travel-planner, Property 17
    Validates: Requirements 10.6
    """
    explicit, implicit, delta = triple

    base = combine_signal(explicit, implicit)

    # Marginal effect on the combined weight of moving each input by the SAME
    # delta. Both moves increase the weighted average, so both deltas are >= 0.
    d_explicit = combine_signal(explicit + delta, implicit) - base
    d_implicit = combine_signal(explicit, implicit + delta) - base

    # Core invariant: the explicit input pulls the combined weight at least
    # twice as hard as the implicit input does (Requirement 10.6 / Property 17).
    assert d_explicit >= 2.0 * d_implicit - _TOL, (
        f"explicit sensitivity {d_explicit!r} is not >= 2x implicit "
        f"sensitivity {d_implicit!r} for explicit={explicit!r}, "
        f"implicit={implicit!r}, delta={delta!r}"
    )

    # The configured weights themselves encode the >= 2x rule.
    assert EXPLICIT_SIGNAL_WEIGHT >= 2.0 * IMPLICIT_SIGNAL_WEIGHT

    # When both signals are present, the merged weight equals the
    # explicit-dominant weighted average, never below an equal-weight blend:
    # the explicit signal is favored whenever it exceeds the implicit one.
    equal_weight_blend = (explicit + implicit) / 2.0
    if explicit >= implicit:
        assert base >= equal_weight_blend - _TOL
    else:
        assert base <= equal_weight_blend + _TOL

    # Ground the property in the real persistence path: write_signals must
    # persist exactly the explicit-weighted-twice combination for the topic.
    session = _new_session()
    try:
        user = User(email="prop17@example.com", prefs={})
        session.add(user)
        session.commit()

        write_signals(
            session,
            "trip-prop17",
            user.id,
            explicit={"food_score": explicit},
            implicit={"food_score": implicit},
        )

        stored = session.scalars(
            select(UserPreference.weight).where(
                UserPreference.user_id == user.id,
                UserPreference.topic == "food_score",
            )
        ).one()

        assert math.isfinite(stored)
        assert abs(stored - base) <= _TOL, (
            f"persisted weight {stored!r} != combined weight {base!r} within {_TOL}"
        )
    finally:
        session.close()
