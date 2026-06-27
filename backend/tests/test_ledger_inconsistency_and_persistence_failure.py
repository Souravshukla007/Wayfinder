"""Unit tests for ledger inconsistency and trace-persistence failure (task 7.15).

Focused, example-based unit tests for the two *error* contracts of the
explainability ledger and the persistence boundary:

- **Requirement 8.3** -- IF the sum of per-feature contributions deviates from
  the destination's final score by more than 0.001, THEN the Decision Engine
  SHALL reject the Decision_Trace and produce an error identifying the affected
  destination. Asserted via :class:`LedgerInconsistencyError`.
- **Requirement 8.5** -- IF persisting a Decision_Trace fails, THEN the system
  SHALL produce an error identifying the affected trip/destination and SHALL
  NOT record a partial Decision_Trace (the write is transactional/all-or-nothing).
  Asserted via :class:`TracePersistenceError`.

Validates Requirements 8.3, 8.5.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.decision.engine import RawCandidate, ScoredCandidate, score_candidates
from app.decision.ledger import (
    LEDGER_TOLERANCE,
    LedgerInconsistencyError,
    validate_ledger,
)
from app.decision.persistence import (
    TracePersistenceError,
    persist_decision,
)
from app.models.db import Base, DecisionTrace, Recommendation, Trip, User
from app.models.schemas import FEATURES, FeatureContribution

CREATED_AT = datetime(2024, 10, 1, 12, 30, tzinfo=timezone.utc)
_EQUAL_WEIGHTS = {feature: 1.0 / len(FEATURES) for feature in FEATURES}


# ---------------------------------------------------------------------------
# Requirement 8.3 -- ledger inconsistency raises, identifying the destination
# ---------------------------------------------------------------------------

def _factors(contribution: float) -> list[FeatureContribution]:
    """One contribution per feature; sum == contribution * len(FEATURES)."""
    return [FeatureContribution(feature=f, contribution=contribution) for f in FEATURES]


def test_deviation_just_over_tolerance_raises_identifying_destination() -> None:
    """Sum deviating by more than 0.001 rejects the trace and names the dest."""
    factors = _factors(10.0)  # sum == 60.0
    contribution_sum = 10.0 * len(FEATURES)
    # Deviate by just over the 0.001 tolerance.
    score = contribution_sum + (LEDGER_TOLERANCE + 1e-6)

    with pytest.raises(LedgerInconsistencyError) as exc:
        validate_ledger("Reykjavik", score, factors)

    err = exc.value
    # The error identifies the affected destination (Req 8.3) ...
    assert err.destination == "Reykjavik"
    assert "Reykjavik" in str(err)
    # ... and carries the offending numbers for diagnosis.
    assert err.expected_score == score
    assert err.contribution_sum == pytest.approx(contribution_sum)
    assert abs(err.contribution_sum - err.expected_score) > LEDGER_TOLERANCE


def test_large_deviation_raises_identifying_destination() -> None:
    factors = _factors(5.0)  # sum == 30.0
    with pytest.raises(LedgerInconsistencyError) as exc:
        validate_ledger("Lisbon", 95.0, factors)
    assert exc.value.destination == "Lisbon"
    assert "Lisbon" in str(exc.value)
    assert exc.value.contribution_sum == pytest.approx(30.0)


def test_deviation_exactly_at_tolerance_is_accepted() -> None:
    """Boundary: a deviation of exactly 0.001 is within tolerance (not > 0.001)."""
    factors = _factors(10.0)  # sum == 60.0
    score = 60.0 + LEDGER_TOLERANCE  # deviation == tolerance, not greater
    validate_ledger("Oslo", score, factors)  # must not raise


def test_consistent_ledger_does_not_raise() -> None:
    factors = _factors(10.0)  # sum == 60.0
    validate_ledger("Bergen", 60.0, factors)  # must not raise


# ---------------------------------------------------------------------------
# Requirement 8.5 -- persistence failure: error + no partial write
# ---------------------------------------------------------------------------

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
    candidates = [
        RawCandidate(destination="Kyoto", features={f: 80.0 for f in FEATURES}),
        RawCandidate(destination="Osaka", features={f: 50.0 for f in FEATURES}),
    ]
    return score_candidates(candidates, _EQUAL_WEIGHTS).ranked


def test_persistence_failure_raises_and_writes_no_partial_trace(session: Session) -> None:
    """An FK violation (bad trip_id) fails the commit; nothing is persisted."""
    scored = _scored()

    with pytest.raises(TracePersistenceError) as exc:
        persist_decision(session, "nonexistent-trip", scored, created_at=CREATED_AT)

    # Error identifies the affected trip (Req 8.5).
    assert exc.value.trip_id == "nonexistent-trip"
    assert "nonexistent-trip" in str(exc.value)

    # No partial write: neither traces nor the recommendation survive.
    assert session.scalars(select(DecisionTrace)).all() == []
    assert session.scalars(select(Recommendation)).all() == []


def test_commit_failure_rolls_back_leaving_no_partial_trace(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-transaction commit failure is transactional: rollback, no partial rows.

    Injects a failure at ``commit`` (after the recommendation and traces are
    added to the session) to prove the write is all-or-nothing even when the
    recommendation row would otherwise have been valid.
    """
    trip = _make_trip(session)
    scored = _scored()

    rollback_calls: list[bool] = []
    real_rollback = session.rollback

    def _tracking_rollback() -> None:
        rollback_calls.append(True)
        real_rollback()

    def _boom() -> None:
        raise RuntimeError("simulated decision_traces insert failure")

    monkeypatch.setattr(session, "commit", _boom)
    monkeypatch.setattr(session, "rollback", _tracking_rollback)

    with pytest.raises(TracePersistenceError) as exc:
        persist_decision(session, trip.id, scored, created_at=CREATED_AT)

    assert exc.value.trip_id == trip.id
    # The originating error is chained for diagnosis.
    assert isinstance(exc.value.__cause__, RuntimeError)
    # Rollback was invoked before raising (transactional contract).
    assert rollback_calls == [True]

    # Restore real commit and confirm absolutely nothing was persisted.
    monkeypatch.undo()
    assert session.scalars(select(DecisionTrace)).all() == []
    assert session.scalars(select(Recommendation)).all() == []
