"""Property-based test for Decision Engine out-of-bounds clamping (task 7.7).

Covers:
- **Property 3: Out-of-bounds clamping** - for any raw feature value outside its
  defined input bounds, the value used for normalization equals the nearest
  bound (minimum or maximum).
  Validates: Requirements 5.3.
  Tagged: Feature: wayfinder-travel-planner, Property 3.

The engine (``app/decision/engine.py``) does not expose the post-clamp raw
value directly, but normalization is a strictly monotonic linear map from
``[low, high]`` onto ``[0, 1]`` and is therefore invertible:
``value_used = normalized * (high - low) + low``. We invert it to recover the
exact value the engine fed into normalization and assert it equals the nearest
bound for every out-of-bounds input. A second test cross-checks against a
pre-clamped "twin" candidate scored independently.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.decision.engine import (
    FEATURE_BOUNDS,
    FEATURE_INPUT_MAX,
    FEATURE_INPUT_MIN,
    RawCandidate,
    score_candidates,
)
from app.models.schemas import FEATURES, FeatureWeights


def _base_weights() -> FeatureWeights:
    """A valid weight configuration (non-negative, sums to 1.0)."""
    return FeatureWeights(
        budget_fit=0.25,
        weather_fit=0.20,
        crowd_score=0.15,
        food_score=0.15,
        photography_score=0.15,
        travel_efficiency=0.10,
    )


# Strictly out-of-bounds raw values, with a comfortable margin past each bound
# so float rounding near the boundary never makes a value land back in range.
_below_min = st.floats(
    min_value=-1.0e6,
    max_value=FEATURE_INPUT_MIN - 1.0e-3,
    allow_nan=False,
    allow_infinity=False,
)
_above_max = st.floats(
    min_value=FEATURE_INPUT_MAX + 1.0e-3,
    max_value=1.0e6,
    allow_nan=False,
    allow_infinity=False,
)
_out_of_bounds = st.one_of(_below_min, _above_max)
_oob_feature_map = st.fixed_dictionaries(
    {feature: _out_of_bounds for feature in FEATURES}
)


def _nearest_bound(value: float, low: float, high: float) -> float:
    return low if value < low else high


# ---------------------------------------------------------------------------
# Property 3: Out-of-bounds clamping
# Feature: wayfinder-travel-planner, Property 3
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(features=_oob_feature_map)
def test_property_3_out_of_bounds_clamped_to_nearest_bound(
    features: dict[str, float],
) -> None:
    """Every out-of-bounds feature is normalized as if clamped to its bound.

    **Validates: Requirements 5.3**
    """
    scored = score_candidates(
        [RawCandidate("oob", features)], _base_weights()
    ).ranked[0]

    for feature in FEATURES:
        low, high = FEATURE_BOUNDS[feature]
        span = high - low
        expected_bound = _nearest_bound(features[feature], low, high)

        # Below the minimum must normalize to the low bound's position (0.0);
        # above the maximum to the high bound's (1.0) -- the observable clamp.
        expected_normalized = (expected_bound - low) / span
        assert scored.normalized[feature] == pytest.approx(expected_normalized)

        # Invert normalization to recover the value the engine actually used and
        # assert it equals the nearest bound (Requirement 5.3).
        value_used = scored.normalized[feature] * span + low
        assert value_used == pytest.approx(expected_bound)


@settings(max_examples=200)
@given(features=_oob_feature_map)
def test_property_3_out_of_bounds_matches_clamped_twin(
    features: dict[str, float],
) -> None:
    """An out-of-bounds candidate scores identically to its pre-clamped twin.

    **Validates: Requirements 5.3**

    Building a second candidate whose features are pre-clamped to the nearest
    bound and showing both produce the same normalized values and final score
    independently confirms the engine clamps before normalizing.
    """
    clamped = {
        feature: _nearest_bound(features[feature], *FEATURE_BOUNDS[feature])
        for feature in FEATURES
    }

    oob = score_candidates([RawCandidate("oob", features)], _base_weights()).ranked[0]
    twin = score_candidates(
        [RawCandidate("twin", clamped)], _base_weights()
    ).ranked[0]

    for feature in FEATURES:
        assert oob.normalized[feature] == pytest.approx(twin.normalized[feature])
    assert oob.score == pytest.approx(twin.score)
