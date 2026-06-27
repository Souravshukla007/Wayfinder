"""Property-based test for Decision Engine feature normalization (task 7.6).

Feature: wayfinder-travel-planner, Property 2: Normalized features lie in [0, 1].

*For any* candidate (including extreme and out-of-bounds raw feature values),
every one of the six normalized feature values used in scoring SHALL fall
within the inclusive range 0.0 to 1.0 before weighting.

Validates: Requirements 5.2

This is the universally-quantified counterpart to the example-based
normalization tests in ``test_decision_engine.py``. It is kept in its own
module so the single Property-2 test lives next to the Decision Engine code it
validates without colliding with the other Decision Engine property tests.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from app.decision.engine import RawCandidate, score_candidates
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


# Raw feature signals are nominally on a 0-100 scale, but Property 2 must hold
# for *arbitrary* numeric inputs - far below the minimum bound, far above the
# maximum bound, and everywhere in between - so the generator spans a wide
# finite range (and mixes ints with floats) to stress the clamp + normalize
# pipeline. NaN/inf are intentionally excluded: those are rejected as invalid
# candidates (Requirement 5.4) and never produce a normalized value to check.
_feature_value = st.one_of(
    st.floats(
        min_value=-1e9,
        max_value=1e9,
        allow_nan=False,
        allow_infinity=False,
    ),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
)

# A complete feature mapping: every one of the six features present and numeric
# so the candidate survives validation and is actually scored.
_feature_map = st.fixed_dictionaries({feature: _feature_value for feature in FEATURES})


# ---------------------------------------------------------------------------
# Property 2: Normalized features lie in [0, 1]
# Feature: wayfinder-travel-planner, Property 2
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    features=_feature_map,
    extra=st.lists(_feature_map, min_size=0, max_size=4),
)
def test_normalized_features_lie_in_unit_interval(
    features: dict[str, float],
    extra: list[dict[str, float]],
) -> None:
    """Every normalized feature of every scored candidate lies within [0, 1].

    **Validates: Requirements 5.2**

    No matter how extreme or out-of-bounds the raw signals are, the value used
    for weighting must be normalized into the inclusive unit interval, for each
    candidate in a single- or multi-candidate run.
    """
    candidates = [
        RawCandidate(f"cand-{index}", feature_map)
        for index, feature_map in enumerate([features, *extra])
    ]

    result = score_candidates(candidates, _base_weights())

    # Fully-numeric candidates are valid and must all be scored, not rejected.
    assert result.rejected == []
    assert len(result.ranked) == len(candidates)

    for scored in result.ranked:
        # Exactly the six features are normalized...
        assert set(scored.normalized) == set(FEATURES)
        # ...and every normalized value lies in the inclusive [0, 1] range.
        for feature in FEATURES:
            value = scored.normalized[feature]
            assert 0.0 <= value <= 1.0, (
                f"{scored.destination}/{feature} normalized to {value}, "
                "outside [0, 1]"
            )
