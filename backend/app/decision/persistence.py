"""Persist Decision Engine scores and decision traces (task 7.4).

This module is the bridge between the *pure* Decision Engine (scoring in
``engine.py``, explainability in ``ledger.py``) and durable storage. Where the
engine and ledger are deterministic and side-effect free, persistence is the
one place that touches the database and supplies the real wall-clock time.

Responsibilities (Requirements 5.8, 8.4, 8.5, 15.2):
- Persist the chosen destination's final score **and** its per-feature
  breakdown to the ``recommendations`` table (the ``scores`` JSON column).
- Insert one decision trace per scored candidate into the ``decision_traces``
  table (the ``factors_json`` column) **transactionally** -- the recommendation
  and every trace are written all-or-nothing in a single transaction.
- On any failure, roll back so **no partial trace** (and no partial
  recommendation) is left behind, and raise :class:`TracePersistenceError`
  (Requirement 8.5).

Timestamps: the ledger defaults trace ``created_at`` to a fixed epoch to stay
pure. Persistence is the impure boundary, so the caller supplies the real
creation time which is stamped onto every trace and persisted row.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.decision.engine import ScoredCandidate
from app.decision.ledger import build_decision_trace
from app.models.db import DecisionTrace, Recommendation
from app.models.schemas import DecisionTraceModel


class TracePersistenceError(Exception):
    """Raised when persisting the recommendation/decision traces fails.

    The originating database error is chained as ``__cause__``. By contract the
    persisting transaction has been rolled back before this is raised, so no
    partial recommendation or trace remains in the database (Requirement 8.5).
    """

    def __init__(self, trip_id: str, message: str) -> None:
        self.trip_id = trip_id
        super().__init__(f"Failed to persist decision data for trip '{trip_id}': {message}")


@dataclass(frozen=True)
class PersistedDecision:
    """Identifiers of the rows written by :func:`persist_decision`."""

    recommendation_id: str
    trace_ids: list[str]


def _factors_payload(trace: DecisionTraceModel) -> list[dict[str, float | str]]:
    """Serialize a trace's additive factors for the ``factors_json`` column."""
    return [
        {"feature": factor.feature, "contribution": factor.contribution}
        for factor in trace.factors
    ]


def _build_scores_payload(chosen: ScoredCandidate) -> dict[str, object]:
    """Build the ``recommendations.scores`` payload for the chosen candidate.

    Contains the final score plus the per-feature breakdown (both the
    normalized [0, 1] values and the weighted contributions), so the stored
    recommendation fully explains the final number (Requirements 5.8, 8.4).
    """
    return {
        "final": chosen.score,
        "normalized": dict(chosen.normalized),
        "breakdown": dict(chosen.contributions),
    }


def persist_decision(
    session: Session,
    trip_id: str,
    scored: Sequence[ScoredCandidate],
    *,
    created_at: datetime,
    chosen: ScoredCandidate | None = None,
) -> PersistedDecision:
    """Persist the chosen recommendation and all decision traces atomically.

    Builds a validated decision trace for every candidate (reusing the ledger
    so the additivity invariant from task 7.3 is enforced), then writes the
    chosen destination's recommendation and all traces in a single
    transaction. If anything fails the transaction is rolled back, leaving no
    partial write, and :class:`TracePersistenceError` is raised
    (Requirements 8.4, 8.5, 15.2).

    Args:
        session: an active SQLAlchemy session.
        trip_id: the trip these results belong to.
        scored: scored candidates (typically ``ScoringResult.ranked``); the
            first is treated as the chosen destination unless ``chosen`` is
            given.
        created_at: real creation timestamp stamped on every trace (the ledger
            stays pure and defaults to epoch; persistence supplies real time).
        chosen: optional explicit chosen candidate; defaults to ``scored[0]``.

    Returns:
        A :class:`PersistedDecision` with the new recommendation id and the
        ids of every persisted decision trace, ordered to match ``scored``.

    Raises:
        ValueError: if ``scored`` is empty (nothing to persist).
        TracePersistenceError: if the database write fails; the transaction is
            rolled back first so no partial trace remains.
    """
    if not scored:
        raise ValueError("cannot persist a decision with no scored candidates")

    chosen_candidate = chosen if chosen is not None else scored[0]

    # Build validated traces up front (ledger enforces one-per-feature
    # additivity, Requirement 8.2) using the caller-supplied real time.
    traces: list[DecisionTraceModel] = [
        build_decision_trace(candidate, created_at) for candidate in scored
    ]

    recommendation = Recommendation(
        trip_id=trip_id,
        decision=chosen_candidate.destination,
        rationale=None,
        alternatives=[
            {"name": candidate.destination, "score": candidate.score}
            for candidate in scored
            if candidate.destination != chosen_candidate.destination
        ],
        scores=_build_scores_payload(chosen_candidate),
    )

    trace_rows: list[DecisionTrace] = [
        DecisionTrace(
            trip_id=trip_id,
            destination=trace.destination,
            score=trace.score,
            factors_json=_factors_payload(trace),
            created_at=trace.created_at,
        )
        for trace in traces
    ]

    # All-or-nothing: a single commit either persists the recommendation and
    # every trace together, or rolls everything back (Requirements 8.4, 8.5).
    try:
        session.add(recommendation)
        session.add_all(trace_rows)
        session.commit()
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error below
        session.rollback()
        raise TracePersistenceError(trip_id, str(exc)) from exc

    return PersistedDecision(
        recommendation_id=recommendation.id,
        trace_ids=[row.id for row in trace_rows],
    )


__all__ = [
    "TracePersistenceError",
    "PersistedDecision",
    "persist_decision",
]
