"""Unit tests for the explainability ledger (task 7.3).

Validates Requirements:
- 8.1: a decision trace records exactly one contribution per scored feature.
- 8.2: the per-feature contributions sum to the destination's final score
  within an absolute tolerance of 0.001.
- 8.3: a contribution sum deviating by more than 0.001 raises
  ``LedgerInconsistencyError`` identifying the affected destination.

These are example-based unit tests; the universally-quantified property tests
for ledger additivity and one-contribution-per-feature are scheduled separately
(tasks 7.11, 7.12, 7.15).
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import pytest

from app.decision.engine import RawCandidate, ScoredCandidate, score_candidates
from app.decision.ledger import (
    LEDGER_TOLERANCE,
    LedgerInconsistencyError,
    build_contributions,
    build_decision_trace,
    validate_ledger,
)
from app.models.schemas import FEATURES, FeatureContribution


def _full_features(value: float) -> dict[str, float]:
    return {feature: value for feature in FEATURES}


def _base_weights_map() -> dict[str, float]:
    return {
        "budget_fit": 0.25,
        "weather_fit": 0.20,
        "crowd_score": 0.15,
        "food_score": 0.15,
        "photography_score": 0.15,
        "travel_efficiency": 0.10,
    }


def _score_one(destination: str, value: float) -> ScoredCandidate:
    result = score_candidates([RawCandidate(destination, _full_features(value))], _base_weights_map())
    return result.ranked[0]


# --- One contribution per feature (Req 8.1 / Property 9) ---------------------

def test_one_contribution_per_feature() -> None:
    scored = _score_one("Tokyo", 80.0)
    trace = build_decision_trace(scored)

    seen = [f.feature for f in trace.factors]
    assert len(trace.factors) == len(FEATURES)
    assert sorted(seen) == sorted(FEATURES)
    # No feature appears more than once.
    assert all(count == 1 for count in Counter(seen).values())


def test_build_contributions_covers_every_feature_in_order() -> None:
    scored = _score_one("Kyoto", 55.0)
    factors = build_contributions(scored.contributions, scored.destination)
    assert [f.feature for f in factors] == list(FEATURES)


def test_missing_feature_in_contributions_is_inconsistent() -> None:
    partial = {f: 1.0 for f in FEATURES}
    del partial["food_score"]
    with pytest.raises(LedgerInconsistencyError) as exc:
        build_contributions(partial, "Osaka")
    assert exc.value.destination == "Osaka"


# --- Ledger additivity (Req 8.2 / Property 8) -------------------------------

def test_contributions_sum_to_final_score() -> None:
    scored = _score_one("Sapporo", 72.0)
    trace = build_decision_trace(scored)
    total = sum(f.contribution for f in trace.factors)
    assert total == pytest.approx(trace.score, abs=LEDGER_TOLERANCE)
    assert trace.score == pytest.approx(scored.score)


def test_validate_ledger_accepts_within_tolerance() -> None:
    factors = [FeatureContribution(feature=f, contribution=10.0) for f in FEATURES]
    # Sum is 60.0; a score within 0.001 is accepted.
    validate_ledger("Nara", 60.0005, factors)  # should not raise


# --- Ledger inconsistency raises identifying the destination (Req 8.3) ------

def test_validate_ledger_rejects_beyond_tolerance() -> None:
    factors = [FeatureContribution(feature=f, contribution=10.0) for f in FEATURES]
    with pytest.raises(LedgerInconsistencyError) as exc:
        validate_ledger("Hiroshima", 70.0, factors)  # sum 60 vs score 70
    assert exc.value.destination == "Hiroshima"
    assert exc.value.expected_score == 70.0
    assert exc.value.contribution_sum == pytest.approx(60.0)
    assert "Hiroshima" in str(exc.value)


def test_validate_ledger_rejects_duplicate_feature() -> None:
    factors = [FeatureContribution(feature="budget_fit", contribution=10.0) for _ in FEATURES]
    with pytest.raises(LedgerInconsistencyError) as exc:
        validate_ledger("Fukuoka", 60.0, factors)
    assert exc.value.destination == "Fukuoka"


def test_just_over_tolerance_is_rejected() -> None:
    factors = [FeatureContribution(feature=f, contribution=10.0) for f in FEATURES]
    # Sum 60.0; deviate by just over 0.001.
    with pytest.raises(LedgerInconsistencyError):
        validate_ledger("Kobe", 60.0 + LEDGER_TOLERANCE * 2, factors)


# --- build_decision_trace integration & determinism -------------------------

def test_build_decision_trace_is_deterministic() -> None:
    scored = _score_one("Tokyo", 64.0)
    first = build_decision_trace(scored)
    second = build_decision_trace(scored)
    assert [(f.feature, f.contribution) for f in first.factors] == [
        (f.feature, f.contribution) for f in second.factors
    ]
    assert first.score == second.score


def test_build_decision_trace_uses_supplied_timestamp() -> None:
    scored = _score_one("Tokyo", 64.0)
    ts = datetime(2024, 10, 1, 12, 0, tzinfo=timezone.utc)
    trace = build_decision_trace(scored, created_at=ts)
    assert trace.created_at == ts


def test_extreme_scores_build_consistent_traces() -> None:
    for value in (0.0, 100.0):
        scored = _score_one("Edge", value)
        trace = build_decision_trace(scored)
        total = sum(f.contribution for f in trace.factors)
        assert total == pytest.approx(trace.score, abs=LEDGER_TOLERANCE)
