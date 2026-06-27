"""The explainability ledger: decision-trace construction and validation (task 7.3).

The ledger turns a scored destination into an auditable :class:`DecisionTraceModel`
whose per-feature contributions explain exactly how the final score was reached.
Like the rest of the Decision Engine, every function here is **pure and
deterministic**: no LLM, no randomness, no system clock, no network. Timestamps
are supplied by the caller (defaulting to a fixed epoch) so identical inputs
always yield identical traces.

Responsibilities (Requirements 8.1, 8.2, 8.3):
- Build exactly one additive :class:`FeatureContribution` per feature for a
  scored candidate -- no feature is missing and none is duplicated
  (Requirement 8.1 / Property 9).
- Validate that the sum of the per-feature contributions equals the candidate's
  final score within an absolute tolerance of 0.001 (Requirement 8.2 /
  Property 8).
- Raise :class:`LedgerInconsistencyError`, identifying the affected destination,
  when the contribution sum deviates from the final score by more than 0.001
  (Requirement 8.3). A destination with an inconsistent ledger is surfaced as an
  error rather than persisted with an invalid trace.

The engine (task 7.2) already exposes each :class:`ScoredCandidate.contributions`
as a per-feature weighted contribution that sums to ``score``. The ledger uses
those contributions directly as the additive entries rather than recomputing,
keeping the trace faithful to the scoring math.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from app.decision.engine import ScoredCandidate
from app.models.schemas import (
    FEATURES,
    WEIGHT_SUM_TOLERANCE,
    DecisionTraceModel,
    FeatureContribution,
)

# Absolute tolerance for the additivity check. Shared with the weight-sum
# tolerance so the whole engine uses one documented epsilon (Requirement 8.2).
LEDGER_TOLERANCE = WEIGHT_SUM_TOLERANCE

# Deterministic default timestamp. The ledger never reads the system clock so it
# stays pure; the persistence layer (task 7.4) supplies the real creation time.
_DEFAULT_CREATED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)


class LedgerInconsistencyError(Exception):
    """Raised when a decision trace's contributions do not sum to its score.

    Carries the affected ``destination`` plus the expected score and the actual
    contribution sum so callers can report the specific failing destination
    (Requirement 8.3).
    """

    def __init__(
        self,
        destination: str,
        expected_score: float,
        contribution_sum: float,
        tolerance: float = LEDGER_TOLERANCE,
    ) -> None:
        self.destination = destination
        self.expected_score = expected_score
        self.contribution_sum = contribution_sum
        self.tolerance = tolerance
        deviation = abs(contribution_sum - expected_score)
        super().__init__(
            f"Decision trace for destination '{destination}' is inconsistent: "
            f"contributions sum to {contribution_sum} but the final score is "
            f"{expected_score} (deviation {deviation} exceeds tolerance {tolerance})."
        )


def build_contributions(
    contributions: Mapping[str, float],
    destination: str,
) -> list[FeatureContribution]:
    """Build exactly one :class:`FeatureContribution` per feature (Req 8.1).

    Iterates over the canonical ``FEATURES`` tuple so the result contains one
    entry for every scored feature, in a stable order, with no duplicates and
    none missing (Property 9). A feature absent from ``contributions`` means the
    scoring math is incomplete, so the destination's trace is treated as
    inconsistent rather than silently zero-filled.
    """
    entries: list[FeatureContribution] = []
    for feature in FEATURES:
        if feature not in contributions:
            raise LedgerInconsistencyError(
                destination,
                expected_score=float("nan"),
                contribution_sum=float("nan"),
            )
        entries.append(
            FeatureContribution(feature=feature, contribution=float(contributions[feature]))
        )
    return entries


def validate_ledger(
    destination: str,
    score: float,
    factors: list[FeatureContribution],
    tolerance: float = LEDGER_TOLERANCE,
) -> None:
    """Validate one-per-feature coverage and additivity (Req 8.1, 8.2, 8.3).

    Raises :class:`LedgerInconsistencyError` (identifying ``destination``) when
    the factors do not cover each feature exactly once, or when their sum
    deviates from ``score`` by more than ``tolerance`` (Requirement 8.3).
    """
    seen = [f.feature for f in factors]
    # Exactly one contribution per feature -- no more, no fewer (Requirement 8.1).
    if sorted(seen) != sorted(FEATURES):
        raise LedgerInconsistencyError(
            destination,
            expected_score=score,
            contribution_sum=float("nan"),
        )

    contribution_sum = sum(f.contribution for f in factors)
    if abs(contribution_sum - score) > tolerance:
        raise LedgerInconsistencyError(
            destination,
            expected_score=score,
            contribution_sum=contribution_sum,
            tolerance=tolerance,
        )


def build_decision_trace(
    scored: ScoredCandidate,
    created_at: datetime | None = None,
    *,
    tolerance: float = LEDGER_TOLERANCE,
) -> DecisionTraceModel:
    """Build a validated :class:`DecisionTraceModel` for a scored candidate.

    Uses the engine's per-feature ``contributions`` directly as additive ledger
    entries (one per feature, Requirement 8.1), validates that they sum to the
    candidate's final score within ``tolerance`` (Requirement 8.2), and raises
    :class:`LedgerInconsistencyError` identifying the destination otherwise
    (Requirement 8.3).

    Args:
        scored: the scored candidate produced by the Decision Engine.
        created_at: creation timestamp; defaults to a fixed epoch to keep the
            function pure/deterministic. The persistence layer supplies the
            real time.
        tolerance: absolute additivity tolerance (default 0.001).

    Returns:
        A :class:`DecisionTraceModel` with destination, final score, exactly one
        contribution per feature, and the creation timestamp.
    """
    factors = build_contributions(scored.contributions, scored.destination)
    validate_ledger(scored.destination, scored.score, factors, tolerance)

    return DecisionTraceModel(
        destination=scored.destination,
        score=scored.score,
        factors=factors,
        created_at=created_at or _DEFAULT_CREATED_AT,
    )


__all__ = [
    "LEDGER_TOLERANCE",
    "LedgerInconsistencyError",
    "build_contributions",
    "validate_ledger",
    "build_decision_trace",
]
