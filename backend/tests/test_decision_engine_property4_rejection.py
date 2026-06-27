"""Property-based test for Decision Engine invalid-candidate rejection.

Feature: wayfinder-travel-planner, Property 4: Invalid-candidate rejection
preserves valid candidates.

*For any* set of candidates containing some candidates with a missing or
non-numeric feature, the engine SHALL exclude every invalid candidate from the
results while computing and preserving scores for all remaining valid
candidates.

Validates: Requirements 5.4

This is the universally-quantified counterpart to the example-based rejection
tests in ``test_decision_engine.py``. It generates mixed batches of valid and
invalid candidates (missing features and non-numeric features, including
``None``, ``bool``, ``NaN``, ``inf``, strings, and lists) and asserts the
engine partitions them correctly and scores every surviving valid candidate
exactly as it would be scored on its own.

This Property 4 test lives in its own module (rather than a shared
``*_properties.py``) so it is self-contained and isolated.
"""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from app.decision.engine import RawCandidate, score_candidates
from app.models.schemas import FEATURES, FeatureWeights


def _base_weights() -> FeatureWeights:
    """A fixed valid weight configuration (sums to 1.0, all non-negative)."""
    return FeatureWeights(
        budget_fit=0.25,
        weather_fit=0.20,
        crowd_score=0.15,
        food_score=0.15,
        photography_score=0.15,
        travel_efficiency=0.10,
    )


# Finite numeric feature values, intentionally spanning beyond the [0, 100]
# input bounds so clamping/normalization are exercised on the valid path.
_valid_value = st.floats(
    min_value=-1.0e6,
    max_value=1.0e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)

# A complete, all-numeric feature mapping -> a valid candidate.
_valid_features = st.fixed_dictionaries({feature: _valid_value for feature in FEATURES})

# Values the engine must treat as non-numeric (Requirement 5.4): booleans,
# None, NaN, infinities, strings, and lists.
_non_numeric_value = st.one_of(
    st.text(max_size=8),
    st.none(),
    st.booleans(),
    st.just(math.nan),
    st.just(math.inf),
    st.just(-math.inf),
    st.lists(st.integers(), max_size=3),
)


@st.composite
def _invalid_features(draw: st.DrawFn) -> dict:
    """Build a feature mapping guaranteed to be rejected by the engine.

    Either drops one or more required features (missing) or replaces one or
    more with a non-numeric value. At least one defect is always introduced.
    """
    base = dict(draw(_valid_features))
    mode = draw(st.sampled_from(("missing", "non_numeric")))
    targets = draw(
        st.lists(st.sampled_from(FEATURES), min_size=1, max_size=len(FEATURES), unique=True)
    )
    if mode == "missing":
        for feature in targets:
            base.pop(feature, None)
    else:
        for feature in targets:
            base[feature] = draw(_non_numeric_value)
    return base


# Each element is tagged with whether it is expected to be valid.
_candidate_spec = st.one_of(
    _valid_features.map(lambda feats: (True, feats)),
    _invalid_features().map(lambda feats: (False, feats)),
)


def _expected_score(features: dict, weights: FeatureWeights) -> float:
    """Independently recompute the engine's score (clamp -> normalize -> weight).

    Mirrors the engine's documented pipeline (Requirements 5.2, 5.3, 5.5) via a
    separate derivation so the test validates correctness, not just internal
    self-consistency.
    """
    contributions = []
    for feature in FEATURES:
        value = float(features[feature])
        clamped = min(max(value, 0.0), 100.0)
        normalized = clamped / 100.0
        contributions.append(getattr(weights, feature) * normalized * 100.0)
    return math.fsum(contributions)


@settings(max_examples=200, deadline=None)
@given(specs=st.lists(_candidate_spec, min_size=1, max_size=10))
def test_invalid_candidate_rejection_preserves_valid_candidates(specs: list) -> None:
    """Feature: wayfinder-travel-planner, Property 4.

    Validates: Requirements 5.4
    """
    weights = _base_weights()

    candidates: list[RawCandidate] = []
    valid_features_by_dest: dict[str, dict] = {}
    invalid_dests: set[str] = set()

    for index, (is_valid, features) in enumerate(specs):
        destination = f"dest_{index}"  # index-based => destinations are unique
        candidates.append(RawCandidate(destination, features))
        if is_valid:
            valid_features_by_dest[destination] = features
        else:
            invalid_dests.add(destination)

    result = score_candidates(candidates, weights)

    ranked_dests = {c.destination for c in result.ranked}
    rejected_dests = {c.destination for c in result.rejected}

    # The result is a clean partition of the input: valid -> ranked, invalid ->
    # rejected, with no overlap, nothing invented, and nothing dropped.
    assert ranked_dests == set(valid_features_by_dest)
    assert rejected_dests == invalid_dests
    assert ranked_dests.isdisjoint(rejected_dests)
    assert len(result.ranked) == len(valid_features_by_dest)
    assert len(result.rejected) == len(invalid_dests)

    scored_by_dest = {c.destination: c for c in result.ranked}
    for destination, features in valid_features_by_dest.items():
        scored = scored_by_dest[destination]

        # Correctly scored: matches an independent re-derivation.
        assert scored.score == _expected_score(features, weights)
        for feature in FEATURES:
            assert 0.0 <= scored.normalized[feature] <= 1.0

        # Preserved: presence of invalid candidates in the batch does not alter
        # a valid candidate's score versus scoring it on its own.
        alone = score_candidates([RawCandidate(destination, features)], weights).ranked[0]
        assert scored.score == alone.score
        assert scored.normalized == alone.normalized
        assert scored.contributions == alone.contributions
