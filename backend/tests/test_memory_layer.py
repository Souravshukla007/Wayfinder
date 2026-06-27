"""Unit tests for the Travel Memory Layer ``load_context`` (task 10.1).

Covers:
- Requirement 10.4: cold-start initializes the predefined default preference set
  so every required topic has a starting weight.
- Requirement 10.5: a 180-day half-life time decay is applied to each preference
  weight using its ``updated_at`` timestamp.
- Requirement 10.1: stored preferences and liked/disliked lists are loaded
  (within a 2s budget) so they need not be re-requested.

These tests exercise the deterministic core with an injectable ``now`` so the
half-life arithmetic is exact, against an in-memory SQLite database.
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.memory.memory_layer import (
    EXPLICIT_SIGNAL_WEIGHT,
    EXPLICIT_TO_IMPLICIT_RATIO,
    HALF_LIFE_DAYS,
    IMPLICIT_SIGNAL_WEIGHT,
    LOAD_CONTEXT_BUDGET_SECONDS,
    WRITE_SIGNALS_BUDGET_SECONDS,
    MemoryContext,
    PreferenceOverrideError,
    UnknownPreferenceTopicError,
    combine_signal,
    decay_factor,
    default_preferences,
    get_preferences,
    load_context,
    override_preference,
    write_signals,
)
from app.memory.memory_layer import _as_utc  # internal helper for tz-aware compare
from app.models.db import (
    Base,
    DislikedDestination,
    LikedDestination,
    User,
    UserPreference,
)
from app.models.schemas import FEATURES

NOW = datetime(2024, 10, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def session() -> Session:
    """A SQLite in-memory session with the full ORM schema created."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    db = factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _make_user(db: Session) -> User:
    user = User(email="traveler@example.com", prefs={})
    db.add(user)
    db.commit()
    return user


# --- Cold start (Requirement 10.4) ------------------------------------------


def test_cold_start_initializes_every_required_topic(session: Session) -> None:
    user = _make_user(session)
    ctx = load_context(session, user.id, now=NOW)

    topics = {p.topic for p in ctx.preferences}
    assert topics == set(FEATURES)
    # Every required topic has a defined, in-range starting weight.
    assert all(0.0 <= p.weight <= 1.0 for p in ctx.preferences)


def test_cold_start_weights_match_configured_base(session: Session) -> None:
    user = _make_user(session)
    base = get_settings().base_feature_weights()
    ctx = load_context(session, user.id, now=NOW)

    weights = ctx.topic_weights()
    for feature in FEATURES:
        assert weights[feature] == pytest.approx(base[feature])


def test_cold_start_persists_defaults_for_durable_reload(session: Session) -> None:
    user = _make_user(session)
    load_context(session, user.id, now=NOW)

    rows = session.query(UserPreference).filter_by(user_id=user.id).all()
    assert {r.topic for r in rows} == set(FEATURES)

    # A second load returns the same vector and does not duplicate rows.
    load_context(session, user.id, now=NOW)
    rows_again = session.query(UserPreference).filter_by(user_id=user.id).all()
    assert len(rows_again) == len(FEATURES)


def test_default_preferences_are_stamped_at_now() -> None:
    defaults = default_preferences(now=NOW)
    assert {d.topic for d in defaults} == set(FEATURES)
    assert all(d.updated_at == NOW for d in defaults)


# --- Time decay (Requirement 10.5) ------------------------------------------


def test_decay_factor_is_one_half_at_one_half_life() -> None:
    updated = NOW - timedelta(days=HALF_LIFE_DAYS)
    assert decay_factor(updated, NOW) == pytest.approx(0.5)


def test_decay_factor_is_one_quarter_at_two_half_lives() -> None:
    updated = NOW - timedelta(days=2 * HALF_LIFE_DAYS)
    assert decay_factor(updated, NOW) == pytest.approx(0.25)


def test_decay_factor_is_one_for_fresh_and_future_signals() -> None:
    assert decay_factor(NOW, NOW) == pytest.approx(1.0)
    assert decay_factor(NOW + timedelta(days=10), NOW) == pytest.approx(1.0)


def test_load_context_applies_half_life_decay_to_stored_weight(session: Session) -> None:
    user = _make_user(session)
    session.add(
        UserPreference(
            user_id=user.id,
            topic="food_score",
            weight=0.8,
            source="explicit",
            updated_at=NOW - timedelta(days=HALF_LIFE_DAYS),
        )
    )
    session.commit()

    ctx = load_context(session, user.id, now=NOW)
    weights = ctx.topic_weights()
    # 0.8 decayed by exactly one half-life -> 0.4.
    assert weights["food_score"] == pytest.approx(0.4)


def test_decayed_weights_stay_in_range(session: Session) -> None:
    user = _make_user(session)
    session.add_all(
        [
            UserPreference(
                user_id=user.id,
                topic="budget_fit",
                weight=1.0,
                source="explicit",
                updated_at=NOW - timedelta(days=3 * HALF_LIFE_DAYS),
            ),
            UserPreference(
                user_id=user.id,
                topic="weather_fit",
                weight=0.0,
                source="implicit",
                updated_at=NOW - timedelta(days=10),
            ),
        ]
    )
    session.commit()

    ctx = load_context(session, user.id, now=NOW)
    assert all(0.0 <= p.weight <= 1.0 for p in ctx.preferences)


# --- Loading existing context (Requirement 10.1) ----------------------------


def test_loads_existing_preferences_without_cold_start(session: Session) -> None:
    user = _make_user(session)
    session.add(
        UserPreference(
            user_id=user.id,
            topic="photography_score",
            weight=0.9,
            source="explicit",
            updated_at=NOW,
        )
    )
    session.commit()

    ctx = load_context(session, user.id, now=NOW)
    # Only the stored preference is present — defaults are NOT seeded when any
    # preference already exists.
    assert [p.topic for p in ctx.preferences] == ["photography_score"]
    assert ctx.topic_weights()["photography_score"] == pytest.approx(0.9)


def test_loads_liked_and_disliked_destinations(session: Session) -> None:
    user = _make_user(session)
    session.add_all(
        [
            UserPreference(user_id=user.id, topic="food_score", weight=0.5,
                           source="explicit", updated_at=NOW),
            LikedDestination(user_id=user.id, place="Kyoto", updated_at=NOW),
            DislikedDestination(user_id=user.id, place="Tourist Trap", updated_at=NOW),
        ]
    )
    session.commit()

    ctx = load_context(session, user.id, now=NOW)
    assert ctx.liked == ["Kyoto"]
    assert ctx.disliked == ["Tourist Trap"]


def test_returns_memory_context_within_budget(session: Session) -> None:
    user = _make_user(session)
    start = _time.perf_counter()
    ctx = load_context(session, user.id, now=NOW)
    elapsed = _time.perf_counter() - start

    assert isinstance(ctx, MemoryContext)
    assert elapsed < LOAD_CONTEXT_BUDGET_SECONDS


# --- Signal writing: explicit >= 2x implicit (Requirements 10.2, 10.6) ------


def test_explicit_weight_is_at_least_twice_implicit_constant() -> None:
    # The configured weighting ratio satisfies the >= 2x rule (Requirement 10.6).
    assert EXPLICIT_SIGNAL_WEIGHT >= 2 * IMPLICIT_SIGNAL_WEIGHT
    assert EXPLICIT_TO_IMPLICIT_RATIO >= 2.0


def test_combine_signal_weights_explicit_twice_implicit_for_same_topic() -> None:
    # When both signals are present the explicit value pulls the combined weight
    # exactly twice as hard as the implicit one: (2*e + i) / 3.
    assert combine_signal(1.0, 0.0) == pytest.approx(2.0 / 3.0)
    assert combine_signal(0.0, 1.0) == pytest.approx(1.0 / 3.0)
    assert combine_signal(0.9, 0.3) == pytest.approx((2 * 0.9 + 0.3) / 3.0)

    # Sensitivity to the explicit input is exactly twice that of the implicit
    # input, holding the other fixed.
    base = combine_signal(0.4, 0.4)
    d_explicit = combine_signal(0.5, 0.4) - base
    d_implicit = combine_signal(0.4, 0.5) - base
    assert d_explicit == pytest.approx(2 * d_implicit)


def test_combine_signal_single_source_uses_value_directly() -> None:
    assert combine_signal(explicit_value=0.7) == pytest.approx(0.7)
    assert combine_signal(implicit_value=0.7) == pytest.approx(0.7)


def test_combine_signal_clamps_to_unit_range() -> None:
    assert combine_signal(5.0, -1.0) == pytest.approx(2.0 / 3.0)
    assert 0.0 <= combine_signal(2.0, 2.0) <= 1.0


def test_write_signals_persists_combined_weight_for_same_topic(session: Session) -> None:
    user = _make_user(session)
    write_signals(
        session,
        "trip-1",
        user.id,
        explicit={"food_score": 0.9},
        implicit={"food_score": 0.3},
        now=NOW,
    )

    row = (
        session.query(UserPreference)
        .filter_by(user_id=user.id, topic="food_score")
        .one()
    )
    # (2 * 0.9 + 0.3) / 3 -> explicit dominates at the >= 2x ratio.
    assert row.weight == pytest.approx((2 * 0.9 + 0.3) / 3.0)
    # Explicit contributed, so the recorded source is 'explicit'.
    assert row.source == "explicit"
    assert _as_utc(row.updated_at) == NOW


def test_write_signals_records_correct_source_per_topic(session: Session) -> None:
    user = _make_user(session)
    write_signals(
        session,
        "trip-1",
        user.id,
        explicit={"photography_score": 0.8},
        implicit={"crowd_score": 0.6},
        now=NOW,
    )

    rows = {
        r.topic: r
        for r in session.query(UserPreference).filter_by(user_id=user.id).all()
    }
    assert rows["photography_score"].source == "explicit"
    assert rows["crowd_score"].source == "implicit"
    assert _as_utc(rows["photography_score"].updated_at) == NOW
    assert _as_utc(rows["crowd_score"].updated_at) == NOW


def test_write_signals_updates_existing_row_in_place(session: Session) -> None:
    user = _make_user(session)
    session.add(
        UserPreference(
            user_id=user.id,
            topic="food_score",
            weight=0.2,
            source="implicit",
            updated_at=NOW - timedelta(days=30),
        )
    )
    session.commit()

    later = NOW + timedelta(days=1)
    write_signals(
        session,
        "trip-2",
        user.id,
        explicit={"food_score": 1.0},
        implicit=None,
        now=later,
    )

    rows = session.query(UserPreference).filter_by(user_id=user.id, topic="food_score").all()
    assert len(rows) == 1  # updated in place, not duplicated
    assert rows[0].weight == pytest.approx(1.0)
    assert rows[0].source == "explicit"
    assert _as_utc(rows[0].updated_at) == later


def test_write_signals_keeps_weights_in_unit_range(session: Session) -> None:
    user = _make_user(session)
    write_signals(
        session,
        "trip-3",
        user.id,
        explicit={"budget_fit": 5.0},
        implicit={"budget_fit": -2.0, "weather_fit": 9.0},
        now=NOW,
    )
    rows = session.query(UserPreference).filter_by(user_id=user.id).all()
    assert all(0.0 <= r.weight <= 1.0 for r in rows)


def test_write_signals_no_topics_is_noop(session: Session) -> None:
    user = _make_user(session)
    write_signals(session, "trip-4", user.id, explicit={}, implicit={}, now=NOW)
    assert session.query(UserPreference).filter_by(user_id=user.id).count() == 0


def test_write_signals_completes_within_budget(session: Session) -> None:
    user = _make_user(session)
    explicit = {f"topic_{n}": 0.5 for n in range(50)}
    implicit = {f"topic_{n}": 0.4 for n in range(50)}

    start = _time.perf_counter()
    write_signals(session, "trip-5", user.id, explicit=explicit, implicit=implicit, now=NOW)
    elapsed = _time.perf_counter() - start

    assert elapsed < WRITE_SIGNALS_BUDGET_SECONDS


# --- Preference read & override (Requirements 10.3, 10.7, 10.8, 10.9) -------


def test_get_preferences_returns_topic_weight_and_timestamp(session: Session) -> None:
    user = _make_user(session)
    session.add_all(
        [
            UserPreference(user_id=user.id, topic="food_score", weight=0.7,
                           source="explicit", updated_at=NOW),
            UserPreference(user_id=user.id, topic="budget_fit", weight=0.2,
                           source="implicit", updated_at=NOW - timedelta(days=5)),
        ]
    )
    session.commit()

    prefs = get_preferences(session, user.id)
    # Stored vector returned as (topic, weight, updated_at), topic-sorted.
    assert [p.topic for p in prefs] == ["budget_fit", "food_score"]
    by_topic = {p.topic: p for p in prefs}
    # Weights are returned as stored (NOT time-decayed).
    assert by_topic["food_score"].weight == pytest.approx(0.7)
    assert by_topic["budget_fit"].weight == pytest.approx(0.2)
    assert _as_utc(by_topic["food_score"].updated_at) == NOW
    assert _as_utc(by_topic["budget_fit"].updated_at) == NOW - timedelta(days=5)


def test_get_preferences_empty_for_user_without_vector(session: Session) -> None:
    user = _make_user(session)
    assert get_preferences(session, user.id) == []


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_override_preference_replaces_and_restamps_in_range(
    session: Session, value: float
) -> None:
    user = _make_user(session)
    session.add(
        UserPreference(user_id=user.id, topic="food_score", weight=0.2,
                       source="implicit", updated_at=NOW - timedelta(days=30))
    )
    session.commit()

    later = NOW + timedelta(days=1)
    entry = override_preference(session, user.id, "food_score", value, now=later)

    # Returned entry reflects the override.
    assert entry.topic == "food_score"
    assert entry.weight == pytest.approx(value)
    assert entry.source == "explicit"
    assert _as_utc(entry.updated_at) == later

    # Persisted, in place (no duplicate rows), and re-stamped.
    rows = session.query(UserPreference).filter_by(user_id=user.id, topic="food_score").all()
    assert len(rows) == 1
    assert rows[0].weight == pytest.approx(value)
    assert rows[0].source == "explicit"
    assert _as_utc(rows[0].updated_at) == later


@pytest.mark.parametrize("bad", [-0.01, 1.01, 5.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_override_preference_rejects_out_of_range_and_retains_prior_weight(
    session: Session, bad: float
) -> None:
    user = _make_user(session)
    session.add(
        UserPreference(user_id=user.id, topic="food_score", weight=0.42,
                       source="explicit", updated_at=NOW)
    )
    session.commit()

    with pytest.raises(PreferenceOverrideError) as exc_info:
        override_preference(session, user.id, "food_score", bad, now=NOW + timedelta(days=1))

    # The error identifies the invalid value (Requirement 10.9).
    assert exc_info.value.value == pytest.approx(bad, nan_ok=True)
    assert exc_info.value.topic == "food_score"
    assert repr(bad) in str(exc_info.value)

    # Prior stored weight and timestamp are retained unchanged.
    session.expire_all()
    row = session.query(UserPreference).filter_by(user_id=user.id, topic="food_score").one()
    assert row.weight == pytest.approx(0.42)
    assert _as_utc(row.updated_at) == NOW


def test_override_preference_unknown_topic_raises(session: Session) -> None:
    user = _make_user(session)
    with pytest.raises(UnknownPreferenceTopicError):
        override_preference(session, user.id, "no_such_topic", 0.5, now=NOW)
