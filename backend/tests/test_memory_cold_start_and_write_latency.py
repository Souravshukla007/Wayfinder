"""Unit tests for Memory Layer cold-start defaults and write latency (task 10.9).

Focused coverage for two Memory Layer guarantees:

- Requirement 10.4 — *cold-start initializes every required topic*: a user with
  no stored preferences gets the predefined default preference set, and that set
  covers **every** required Decision Engine feature topic with an in-range,
  durable starting weight.
- Requirement 10.2 — *signals are written within 60s of trip completion*: a
  ``write_signals`` call derived from a completed trip persists within the
  ``WRITE_SIGNALS_BUDGET_SECONDS`` (60s) budget, even for a realistically large
  signal set.

These complement ``test_memory_layer.py`` by asserting the two task-10.9
guarantees directly and independently, against an in-memory SQLite database with
an injectable ``now`` so timing is deterministic where it matters.
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.memory.memory_layer import (
    WRITE_SIGNALS_BUDGET_SECONDS,
    default_preferences,
    load_context,
    write_signals,
)
from app.memory.memory_layer import _as_utc  # internal helper for tz-aware compare
from app.models.db import Base, User, UserPreference
from app.models.schemas import FEATURES

NOW = datetime(2024, 10, 1, 12, 0, tzinfo=timezone.utc)

# The full set of required preference topics is the six Decision Engine
# features. Cold-start must cover every one of them (Requirement 10.4).
REQUIRED_TOPICS = set(FEATURES)


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


# --- Cold-start defaults: every required topic (Requirement 10.4) -----------


def test_default_preferences_cover_every_required_topic() -> None:
    """The predefined default set names exactly the required topics, no gaps."""
    defaults = default_preferences(now=NOW)
    topics = [d.topic for d in defaults]

    # Every required topic is present...
    assert set(topics) == REQUIRED_TOPICS
    # ...exactly once (no duplicate or missing topic).
    assert len(topics) == len(REQUIRED_TOPICS)


def test_default_preferences_have_in_range_starting_weights() -> None:
    """Each defaulted topic has a defined, in-range [0, 1] starting weight."""
    defaults = default_preferences(now=NOW)
    assert all(0.0 <= d.weight <= 1.0 for d in defaults)
    # Defaults are seeded weights, recorded as implicit (system-seeded) signals.
    assert all(d.source == "implicit" for d in defaults)


def test_cold_start_initializes_every_required_topic(session: Session) -> None:
    """A user with no stored vector gets every required topic seeded on load."""
    user = _make_user(session)

    ctx = load_context(session, user.id, now=NOW)

    loaded_topics = {p.topic for p in ctx.preferences}
    assert loaded_topics == REQUIRED_TOPICS
    # No required topic is left without a starting weight.
    assert REQUIRED_TOPICS - loaded_topics == set()
    assert all(0.0 <= p.weight <= 1.0 for p in ctx.preferences)


def test_cold_start_seeds_match_configured_base_weights(session: Session) -> None:
    """The seeded starting weights come from the configured base feature set."""
    user = _make_user(session)
    base = get_settings().base_feature_weights()

    ctx = load_context(session, user.id, now=NOW)

    weights = ctx.topic_weights()
    for topic in REQUIRED_TOPICS:
        assert weights[topic] == pytest.approx(base[topic])


def test_cold_start_persists_every_required_topic_durably(session: Session) -> None:
    """Cold-start writes one row per required topic, durable across reloads."""
    user = _make_user(session)

    load_context(session, user.id, now=NOW)

    rows = session.query(UserPreference).filter_by(user_id=user.id).all()
    assert {r.topic for r in rows} == REQUIRED_TOPICS
    assert len(rows) == len(REQUIRED_TOPICS)

    # A second load is stable: same topics, no duplicate rows seeded.
    load_context(session, user.id, now=NOW)
    rows_again = session.query(UserPreference).filter_by(user_id=user.id).all()
    assert {r.topic for r in rows_again} == REQUIRED_TOPICS
    assert len(rows_again) == len(REQUIRED_TOPICS)


# --- Write latency on trip completion (Requirement 10.2) --------------------


def test_write_signals_persists_within_60s_budget(session: Session) -> None:
    """Signals derived on trip completion persist within the 60s budget."""
    user = _make_user(session)

    start = _time.perf_counter()
    write_signals(
        session,
        "trip-complete-1",
        user.id,
        explicit={"food_score": 0.9, "budget_fit": 0.6},
        implicit={"food_score": 0.4, "weather_fit": 0.5},
        now=NOW,
    )
    elapsed = _time.perf_counter() - start

    # The 60s budget is the contract (Requirement 10.2).
    assert WRITE_SIGNALS_BUDGET_SECONDS == pytest.approx(60.0)
    assert elapsed < WRITE_SIGNALS_BUDGET_SECONDS

    # And the signals are actually durably written.
    written = {
        r.topic for r in session.query(UserPreference).filter_by(user_id=user.id).all()
    }
    assert written == {"food_score", "budget_fit", "weather_fit"}


def test_write_signals_large_batch_within_budget(session: Session) -> None:
    """Even a realistically large signal set writes within the 60s budget."""
    user = _make_user(session)
    explicit = {f"topic_{n}": 0.5 for n in range(200)}
    implicit = {f"topic_{n}": 0.4 for n in range(200)}

    start = _time.perf_counter()
    write_signals(
        session,
        "trip-complete-2",
        user.id,
        explicit=explicit,
        implicit=implicit,
        now=NOW,
    )
    elapsed = _time.perf_counter() - start

    assert elapsed < WRITE_SIGNALS_BUDGET_SECONDS
    assert session.query(UserPreference).filter_by(user_id=user.id).count() == 200


def test_write_signals_stamps_completion_time(session: Session) -> None:
    """Written signals are stamped at the trip-completion time (within budget)."""
    user = _make_user(session)

    write_signals(
        session,
        "trip-complete-3",
        user.id,
        explicit={"photography_score": 0.8},
        implicit=None,
        now=NOW,
    )

    row = (
        session.query(UserPreference)
        .filter_by(user_id=user.id, topic="photography_score")
        .one()
    )
    # The signal carries the completion timestamp, anchoring the "within 60s of
    # completion" guarantee to the recorded write time.
    assert _as_utc(row.updated_at) == NOW
