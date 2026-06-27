"""Property-based test for decision-data persistence round-trip (task 7.14).

Covers:
- **Property 10: Decision-data persistence round-trip** (Task 7.14) - for any
  completed scoring, the final score and per-feature component breakdown read
  back from the ``recommendations`` store, and the decision trace read back
  from the ``decision_traces`` store (destination, score, contributions,
  timestamp), equal the computed values, with no data loss.
  Validates: Requirements 5.8, 8.4, 15.2.
  Tagged: Feature: wayfinder-travel-planner, Property 10.

This is a property test only. It persists real scoring output through the real
``persist_decision`` path into an in-memory SQLite database, reads the rows
back, and compares them against the values computed by the engine/ledger. A
counterexample here indicates genuine data loss or corruption on the
persistence round-trip to report rather than mask.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.decision.engine import RawCandidate, score_candidates
from app.decision.ledger import build_decision_trace
from app.decision.persistence import persist_decision
from app.models.db import Base, DecisionTrace, Recommendation, Trip, User
from app.models.schemas import FEATURES, FeatureWeights

# ---------------------------------------------------------------------------
# Property 10: Decision-data persistence round-trip
# Feature: wayfinder-travel-planner, Property 10
# Validates: Requirements 5.8, 8.4, 15.2
# ---------------------------------------------------------------------------

# Absolute tolerance for float round-trips through the JSON columns. The stored
# values are the engine's own floats serialized to JSON, so equality should be
# exact up to JSON float formatting; a small epsilon guards rounding only.
_ROUNDTRIP_TOLERANCE = 1e-9


@event.listens_for(Engine, "connect")
def _enable_sqlite_fk(dbapi_connection, _record) -> None:  # pragma: no cover - glue
    """Enforce SQLite foreign keys so the schema behaves like Postgres."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _fresh_session() -> tuple[Session, Engine]:
    """Build an isolated in-memory database + session for one example.

    Hypothesis drives many examples through a single test function, so each
    example needs its own database to stay independent (and to avoid trip-id
    collisions). The caller disposes of the engine when done.
    """
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return factory(), engine


def _make_trip(db: Session) -> Trip:
    """Persist a user + trip so the recommendation/trace FKs resolve."""
    user = User(email="traveler@example.com")
    db.add(user)
    db.commit()
    trip = Trip(user_id=user.id, origin="DEL", status="processing")
    db.add(trip)
    db.commit()
    return trip


# Raw feature signals span a wide band -- including values outside the engine's
# [0, 100] input bounds -- so the round-trip is exercised across clamping,
# normalization, and weighting paths.
_raw_value = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _valid_candidate(draw: st.DrawFn) -> RawCandidate:
    """A fully-valid candidate (every feature numeric) that will be scored."""
    features = {feature: draw(_raw_value) for feature in FEATURES}
    destination = draw(st.text(min_size=1, max_size=12))
    return RawCandidate(destination=destination, features=features)


@st.composite
def _weights(draw: st.DrawFn) -> FeatureWeights:
    """Generate a valid weight config: non-negative, renormalized to sum 1.0."""
    raw = [
        draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        for _ in FEATURES
    ]
    total = sum(raw)
    if total <= 0.0:
        raw = [1.0 for _ in FEATURES]
        total = float(len(FEATURES))
    normalized = [value / total for value in raw]
    # Absorb residual float drift into the last weight so the sum lands well
    # within the engine's 0.001 tolerance regardless of rounding. Clamp at zero
    # so float drift can never synthesize a tiny negative weight (which is
    # outside the valid weight-config input space this generator targets).
    normalized[-1] = max(0.0, 1.0 - sum(normalized[:-1]))
    return FeatureWeights(**dict(zip(FEATURES, normalized)))


# Distinct destination names so each candidate maps to its own trace row and we
# can look traces up unambiguously by destination on read-back.
@st.composite
def _unique_candidates(draw: st.DrawFn) -> list[RawCandidate]:
    candidates = draw(st.lists(_valid_candidate(), min_size=1, max_size=8))
    seen: dict[str, RawCandidate] = {}
    for index, candidate in enumerate(candidates):
        key = f"{candidate.destination}#{index}"
        seen[key] = RawCandidate(
            destination=key,
            features=candidate.features,
            metadata=candidate.metadata,
        )
    return list(seen.values())


_created_at = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2100, 1, 1),
).map(lambda dt: dt.replace(tzinfo=timezone.utc))


@settings(max_examples=150, deadline=None)
@given(candidates=_unique_candidates(), weights=_weights(), created_at=_created_at)
def test_decision_data_persistence_round_trip(
    candidates: list[RawCandidate],
    weights: FeatureWeights,
    created_at: datetime,
) -> None:
    """Persisted decision data reads back identically to the computed values.

    For a completed scoring run we:
    1. Compute the scored candidates and the chosen recommendation in memory.
    2. Persist them via ``persist_decision``.
    3. Read the recommendation and decision traces back from the database.
    4. Assert the final score + per-feature breakdown (recommendations store)
       and each trace's destination, score, contributions, and timestamp
       (decision_traces store) equal the computed values -- no data loss
       (Requirements 5.8, 8.4, 15.2).
    """
    result = score_candidates(candidates, weights)
    scored = result.ranked
    assert scored, "all generated candidates are valid, so at least one is scored"

    chosen = scored[0]
    # Expected traces (destination, score, contributions, timestamp) computed by
    # the same ledger the persistence layer uses, stamped with the real time.
    expected_traces = {
        candidate.destination: build_decision_trace(candidate, created_at)
        for candidate in scored
    }

    db, engine = _fresh_session()
    try:
        trip = _make_trip(db)
        persisted = persist_decision(db, trip.id, scored, created_at=created_at)
        # Drop any identity-map cached state so the assertions read true
        # database round-trip values rather than the in-memory instances.
        db.expire_all()

        # --- recommendations store: final score + per-feature breakdown ------
        rec = db.scalars(
            select(Recommendation).where(Recommendation.trip_id == trip.id)
        ).one()
        assert rec.id == persisted.recommendation_id
        assert rec.decision == chosen.destination
        assert rec.scores["final"] == chosen.score

        breakdown = rec.scores["breakdown"]
        assert set(breakdown) == set(FEATURES)
        for feature in FEATURES:
            assert breakdown[feature] == chosen.contributions[feature]

        normalized = rec.scores["normalized"]
        assert set(normalized) == set(FEATURES)
        for feature in FEATURES:
            assert normalized[feature] == chosen.normalized[feature]

        # --- decision_traces store: one per candidate, read back identically -
        traces = db.scalars(
            select(DecisionTrace).where(DecisionTrace.trip_id == trip.id)
        ).all()
        assert len(traces) == len(scored)
        assert {t.destination for t in traces} == set(expected_traces)

        for trace_row in traces:
            expected = expected_traces[trace_row.destination]

            # destination + score round-trip exactly.
            assert trace_row.destination == expected.destination
            assert trace_row.score == expected.score

            # contributions round-trip: exactly one per feature, same values.
            stored = {f["feature"]: f["contribution"] for f in trace_row.factors_json}
            assert set(stored) == set(FEATURES)
            expected_factors = {f.feature: f.contribution for f in expected.factors}
            for feature in FEATURES:
                assert stored[feature] == expected_factors[feature]

            # contributions still sum to the stored score (no data loss).
            assert abs(sum(stored.values()) - trace_row.score) <= _ROUNDTRIP_TOLERANCE

            # timestamp round-trips (SQLite drops tzinfo; re-stamp UTC to compare).
            persisted_ts = trace_row.created_at
            if persisted_ts.tzinfo is None:
                persisted_ts = persisted_ts.replace(tzinfo=timezone.utc)
            assert persisted_ts == created_at
    finally:
        db.close()
        engine.dispose()
