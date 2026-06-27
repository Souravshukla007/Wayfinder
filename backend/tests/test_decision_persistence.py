"""Unit tests for Decision Engine persistence (task 7.4).

Covers Requirements 5.8, 8.4, 8.5, 15.2:
- Persist the chosen destination's final score + per-feature breakdown to the
  ``recommendations.scores`` column, and read it back unchanged.
- Persist one decision trace per candidate to ``decision_traces.factors_json``.
- A failure during persistence rolls back, leaving **no partial** recommendation
  or trace, and raises ``TracePersistenceError``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.decision.engine import RawCandidate, ScoredCandidate, score_candidates
from app.decision.persistence import (
    PersistedDecision,
    TracePersistenceError,
    persist_decision,
)
from app.models.db import Base, DecisionTrace, Recommendation, Trip, User
from app.models.schemas import FEATURES

CREATED_AT = datetime(2024, 10, 1, 12, 30, tzinfo=timezone.utc)

# Equal weights summing to 1.0 (within tolerance) for deterministic scoring.
_EQUAL_WEIGHTS = {feature: 1.0 / len(FEATURES) for feature in FEATURES}


@event.listens_for(Engine, "connect")
def _enable_sqlite_fk(dbapi_connection, _record) -> None:  # pragma: no cover - glue
    """Enforce SQLite foreign keys so FK violations raise like Postgres."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@pytest.fixture()
def session() -> Session:
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


def _make_trip(db: Session) -> Trip:
    user = User(email="traveler@example.com")
    db.add(user)
    db.commit()
    trip = Trip(user_id=user.id, origin="DEL", status="processing")
    db.add(trip)
    db.commit()
    return trip


def _scored() -> list[ScoredCandidate]:
    """Two valid candidates scored by the real engine, ranked descending."""
    candidates = [
        RawCandidate(
            destination="Kyoto",
            features={f: 80.0 for f in FEATURES},
            metadata={"country": "Japan"},
        ),
        RawCandidate(
            destination="Osaka",
            features={f: 50.0 for f in FEATURES},
            metadata={"country": "Japan"},
        ),
    ]
    return score_candidates(candidates, _EQUAL_WEIGHTS).ranked


def test_persist_round_trip_recommendation_and_traces(session: Session) -> None:
    trip = _make_trip(session)
    scored = _scored()
    chosen = scored[0]

    result = persist_decision(session, trip.id, scored, created_at=CREATED_AT)

    assert isinstance(result, PersistedDecision)
    assert len(result.trace_ids) == 2

    # --- recommendation: final score + per-feature breakdown round-trips ---
    rec = session.scalars(
        select(Recommendation).where(Recommendation.trip_id == trip.id)
    ).one()
    assert rec.decision == chosen.destination
    assert rec.scores["final"] == pytest.approx(chosen.score)
    breakdown = rec.scores["breakdown"]
    assert set(breakdown) == set(FEATURES)
    assert sum(breakdown.values()) == pytest.approx(chosen.score, abs=1e-6)
    assert {alt["name"] for alt in rec.alternatives} == {"Osaka"}

    # --- decision traces: one per candidate, factors_json round-trips ---
    traces = session.scalars(
        select(DecisionTrace).where(DecisionTrace.trip_id == trip.id)
    ).all()
    assert {t.destination for t in traces} == {"Kyoto", "Osaka"}
    for trace in traces:
        assert {f["feature"] for f in trace.factors_json} == set(FEATURES)
        assert sum(f["contribution"] for f in trace.factors_json) == pytest.approx(
            trace.score, abs=1e-6
        )
        assert trace.created_at.replace(tzinfo=timezone.utc) == CREATED_AT


def test_persistence_failure_rolls_back_with_no_partial_trace(session: Session) -> None:
    """A bad trip_id triggers an FK violation; nothing must be written."""
    scored = _scored()

    with pytest.raises(TracePersistenceError) as exc_info:
        persist_decision(session, "nonexistent-trip-id", scored, created_at=CREATED_AT)

    assert exc_info.value.trip_id == "nonexistent-trip-id"

    # No partial write: neither table has any rows after the rollback.
    assert session.scalars(select(DecisionTrace)).all() == []
    assert session.scalars(select(Recommendation)).all() == []


def test_persist_empty_scored_raises_value_error(session: Session) -> None:
    trip = _make_trip(session)
    with pytest.raises(ValueError):
        persist_decision(session, trip.id, [], created_at=CREATED_AT)
