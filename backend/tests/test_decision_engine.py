"""Unit tests for the pure Decision Engine scoring function (task 7.2).

Validates Requirements:
- 5.1, 5.7: pure & deterministic scoring (same inputs => identical outputs).
- 5.2: normalized features lie in [0, 1].
- 5.3: out-of-bounds raw values are clamped to the nearest bound.
- 5.4: invalid candidates are rejected while valid ones are preserved.
- 5.5: final score equals the weighted sum on a 0-100 scale.
- 5.6: invalid weight configurations are rejected before any scoring.
- 7.1: candidates are returned ranked by score in descending order.

These are example-based unit tests; the universally-quantified property tests
for the same behaviors are scheduled separately (tasks 7.5-7.10).
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.decision.engine import (
    FEATURE_INPUT_MAX,
    FINAL_SCORE_SCALE,
    RawCandidate,
    score_candidates,
)
from app.models.schemas import FEATURES, FeatureWeights


def _full_features(value: float) -> dict[str, float]:
    return {feature: value for feature in FEATURES}


def _equal_weights() -> FeatureWeights:
    # Six features summing to 1.0 (1/6 each, last absorbs rounding).
    share = round(1.0 / 6.0, 6)
    values = {feature: share for feature in FEATURES}
    values["travel_efficiency"] = 1.0 - share * 5
    return FeatureWeights(**values)


def _base_weights() -> FeatureWeights:
    return FeatureWeights(
        budget_fit=0.25,
        weather_fit=0.20,
        crowd_score=0.15,
        food_score=0.15,
        photography_score=0.15,
        travel_efficiency=0.10,
    )


# --- Final score / weighted sum (Req 5.5) -----------------------------------

def test_all_max_features_scores_100() -> None:
    candidate = RawCandidate("Tokyo", _full_features(FEATURE_INPUT_MAX))
    result = score_candidates([candidate], _base_weights())
    assert result.ranked[0].score == pytest.approx(100.0)


def test_all_min_features_scores_0() -> None:
    candidate = RawCandidate("Osaka", _full_features(0.0))
    result = score_candidates([candidate], _base_weights())
    assert result.ranked[0].score == pytest.approx(0.0)


def test_uniform_midpoint_scores_half() -> None:
    candidate = RawCandidate("Kyoto", _full_features(50.0))
    result = score_candidates([candidate], _base_weights())
    assert result.ranked[0].score == pytest.approx(50.0)


def test_score_equals_weighted_sum_of_contributions() -> None:
    candidate = RawCandidate(
        "Mixed",
        {
            "budget_fit": 80.0,
            "weather_fit": 60.0,
            "crowd_score": 40.0,
            "food_score": 100.0,
            "photography_score": 20.0,
            "travel_efficiency": 50.0,
        },
    )
    weights = _base_weights()
    result = score_candidates([candidate], weights)
    scored = result.ranked[0]

    expected = FINAL_SCORE_SCALE * math.fsum(
        getattr(weights, f) * (scored.normalized[f]) for f in FEATURES
    )
    assert scored.score == pytest.approx(expected)
    # Contributions are additive and sum to the final score (feeds the ledger).
    assert math.fsum(scored.contributions.values()) == pytest.approx(scored.score)


# --- Normalization in [0, 1] (Req 5.2) --------------------------------------

def test_normalized_features_within_unit_range() -> None:
    candidate = RawCandidate("Edge", _full_features(73.0))
    scored = score_candidates([candidate], _base_weights()).ranked[0]
    for feature in FEATURES:
        assert 0.0 <= scored.normalized[feature] <= 1.0


def test_midpoint_normalizes_to_half() -> None:
    candidate = RawCandidate("Half", _full_features(50.0))
    scored = score_candidates([candidate], _base_weights()).ranked[0]
    assert scored.normalized["budget_fit"] == pytest.approx(0.5)


# --- Clamping out-of-bounds (Req 5.3) ---------------------------------------

def test_above_max_clamps_to_one() -> None:
    candidate = RawCandidate("High", _full_features(1000.0))
    scored = score_candidates([candidate], _base_weights()).ranked[0]
    assert scored.score == pytest.approx(100.0)
    assert all(scored.normalized[f] == pytest.approx(1.0) for f in FEATURES)


def test_below_min_clamps_to_zero() -> None:
    candidate = RawCandidate("Low", _full_features(-500.0))
    scored = score_candidates([candidate], _base_weights()).ranked[0]
    assert scored.score == pytest.approx(0.0)
    assert all(scored.normalized[f] == pytest.approx(0.0) for f in FEATURES)


def test_out_of_bounds_equivalent_to_clamped_value() -> None:
    over = RawCandidate("Over", _full_features(150.0))
    at_max = RawCandidate("AtMax", _full_features(FEATURE_INPUT_MAX))
    over_score = score_candidates([over], _base_weights()).ranked[0].score
    max_score = score_candidates([at_max], _base_weights()).ranked[0].score
    assert over_score == pytest.approx(max_score)


# --- Invalid-candidate rejection preserves valid ones (Req 5.4) -------------

def test_missing_feature_is_rejected_others_preserved() -> None:
    valid = RawCandidate("Valid", _full_features(70.0))
    missing = _full_features(70.0)
    del missing["food_score"]
    invalid = RawCandidate("MissingFood", missing)

    result = score_candidates([valid, invalid], _base_weights())

    assert [c.destination for c in result.ranked] == ["Valid"]
    assert len(result.rejected) == 1
    assert result.rejected[0].destination == "MissingFood"
    assert "food_score" in result.rejected[0].reason


def test_non_numeric_feature_is_rejected() -> None:
    bad = _full_features(70.0)
    bad["weather_fit"] = "warm"  # type: ignore[assignment]
    result = score_candidates([RawCandidate("Bad", bad)], _base_weights())
    assert result.ranked == []
    assert result.rejected[0].destination == "Bad"
    assert "weather_fit" in result.rejected[0].reason


def test_nan_and_inf_and_none_and_bool_are_rejected() -> None:
    for label, bad_value in (
        ("nan", math.nan),
        ("inf", math.inf),
        ("none", None),
        ("bool", True),
    ):
        bad = _full_features(70.0)
        bad["crowd_score"] = bad_value  # type: ignore[assignment]
        result = score_candidates([RawCandidate(label, bad)], _base_weights())
        assert result.ranked == [], f"{label} should have been rejected"
        assert result.rejected[0].destination == label


def test_all_invalid_returns_empty_ranking() -> None:
    invalid = [
        RawCandidate("a", {}),
        RawCandidate("b", {"budget_fit": 10.0}),
    ]
    result = score_candidates(invalid, _base_weights())
    assert result.ranked == []
    assert {c.destination for c in result.rejected} == {"a", "b"}


# --- Ranking descending (Req 7.1) -------------------------------------------

def test_candidates_ranked_descending_by_score() -> None:
    low = RawCandidate("Low", _full_features(10.0))
    high = RawCandidate("High", _full_features(90.0))
    mid = RawCandidate("Mid", _full_features(50.0))

    result = score_candidates([low, high, mid], _base_weights())

    scores = [c.score for c in result.ranked]
    assert scores == sorted(scores, reverse=True)
    assert [c.destination for c in result.ranked] == ["High", "Mid", "Low"]


def test_equal_scores_preserve_input_order() -> None:
    first = RawCandidate("First", _full_features(50.0))
    second = RawCandidate("Second", _full_features(50.0))
    result = score_candidates([first, second], _equal_weights())
    assert [c.destination for c in result.ranked] == ["First", "Second"]


# --- Weight-config validation (Req 5.6) -------------------------------------

def test_negative_weight_rejected_before_scoring() -> None:
    bad = {
        "budget_fit": -0.1,
        "weather_fit": 0.3,
        "crowd_score": 0.2,
        "food_score": 0.2,
        "photography_score": 0.2,
        "travel_efficiency": 0.2,
    }
    with pytest.raises(ValidationError):
        score_candidates([RawCandidate("x", _full_features(50.0))], bad)


def test_non_unit_sum_weight_rejected_before_scoring() -> None:
    bad = {f: 0.5 for f in FEATURES}  # sums to 3.0
    with pytest.raises(ValidationError):
        score_candidates([RawCandidate("x", _full_features(50.0))], bad)


def test_mapping_weights_are_accepted() -> None:
    weights = {
        "budget_fit": 0.25,
        "weather_fit": 0.20,
        "crowd_score": 0.15,
        "food_score": 0.15,
        "photography_score": 0.15,
        "travel_efficiency": 0.10,
    }
    result = score_candidates([RawCandidate("x", _full_features(100.0))], weights)
    assert result.ranked[0].score == pytest.approx(100.0)


# --- Determinism (Req 5.1, 5.7) ---------------------------------------------

def test_repeated_runs_are_bit_for_bit_identical() -> None:
    candidates = [
        RawCandidate("A", {**_full_features(33.0), "food_score": 88.0}),
        RawCandidate("B", _full_features(61.0)),
        RawCandidate("C", {**_full_features(72.0), "budget_fit": 12.0}),
    ]
    weights = _base_weights()

    first = score_candidates(candidates, weights)
    second = score_candidates(candidates, weights)

    assert [(c.destination, c.score) for c in first.ranked] == [
        (c.destination, c.score) for c in second.ranked
    ]
    for a, b in zip(first.ranked, second.ranked):
        assert a.normalized == b.normalized
        assert a.contributions == b.contributions


def test_metadata_passed_through_untouched() -> None:
    meta = {"country": "Japan", "flag": "🇯🇵"}
    candidate = RawCandidate("Tokyo", _full_features(80.0), metadata=meta)
    scored = score_candidates([candidate], _base_weights()).ranked[0]
    assert scored.metadata == meta
