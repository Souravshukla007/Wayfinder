"""Property-based test for Decision Engine final-score computation.

Feature: wayfinder-travel-planner, Property 5: Final score equals the weighted
sum on a 0-100 scale.

*For any* valid candidate and valid weight configuration, the final score SHALL
equal the weighted sum of the six normalized features scaled to the inclusive
range 0 to 100 (within floating tolerance).

Validates: Requirements 5.5

This is the universally-quantified counterpart to the example-based weighted-sum
tests in ``test_decision_engine.py``. It is intentionally self-contained (its
own strategies and helpers) so it does not depend on, or collide with, the
other decision-engine property tests.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.decision.engine import (
    FEATURE_INPUT_MAX,
    FEATURE_INPUT_MIN,
    FINAL_SCORE_SCALE,
    RawCandidate,
    score_candidates,
)
from app.models.schemas import FEATURES, FeatureWeights

# Raw feature signals deliberately span well past the defined [0, 100] input
# bounds so the property holds for in-bounds, boundary, and out-of-bounds values
# alike. NaN/inf are excluded: those make a candidate invalid (Requirement 5.4)
# and never produce a final score to check.
_feature_value = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)

_feature_map = st.fixed_dictionaries({feature: _feature_value for feature in FEATURES})


@st.composite
def _valid_weights(draw: st.DrawFn) -> FeatureWeights:
    """Draw a valid weight configuration: non-negative, summing to 1.0.

    Six non-negative reals are drawn and renormalized by their total, which
    yields a sum within float epsilon of 1.0 -- comfortably inside the engine's
    0.001 acceptance tolerance (Requirement 5.6) and never negative.
    """
    raw = draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=len(FEATURES),
            max_size=len(FEATURES),
        ).filter(lambda xs: sum(xs) > 0.0)
    )
    total = sum(raw)
    normalized = [x / total for x in raw]
    return FeatureWeights(**dict(zip(FEATURES, normalized)))


def _clamp_to_bounds(value: float) -> float:
    """Independent clamp into the engine's defined input band [0, 100]."""
    return max(FEATURE_INPUT_MIN, min(FEATURE_INPUT_MAX, value))


# ---------------------------------------------------------------------------
# Property 5: Final score equals the weighted sum on a 0-100 scale
# Feature: wayfinder-travel-planner, Property 5
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(features=_feature_map, weights=_valid_weights())
def test_final_score_equals_weighted_sum_on_0_100_scale(
    features: dict[str, float], weights: FeatureWeights
) -> None:
    """The final score equals the weighted sum of the six normalized features
    scaled to the inclusive 0-100 range (within floating tolerance).

    **Validates: Requirements 5.5**

    Two independent checks: (1) against a from-scratch oracle that clamps,
    normalizes, weights, and scales the raw inputs; and (2) the defining
    Property-5 relationship stated directly against the engine's own normalized
    values, score == 100 * sum(weight_f * normalized_f).
    """
    scored = score_candidates([RawCandidate("X", features)], weights).ranked[0]

    # (1) From-scratch oracle: clamp -> normalize -> weighted sum -> scale.
    expected = FINAL_SCORE_SCALE * math.fsum(
        getattr(weights, f) * (_clamp_to_bounds(features[f]) / FEATURE_INPUT_MAX)
        for f in FEATURES
    )
    assert scored.score == pytest.approx(expected, abs=1e-6)

    # (2) Property-5 relationship against the engine's own normalized values.
    weighted_sum_of_normalized = FINAL_SCORE_SCALE * math.fsum(
        getattr(weights, f) * scored.normalized[f] for f in FEATURES
    )
    assert scored.score == pytest.approx(weighted_sum_of_normalized, abs=1e-9)

    # The weighted sum lands on the inclusive 0-100 scale.
    assert -1e-9 <= scored.score <= FINAL_SCORE_SCALE + 1e-9
