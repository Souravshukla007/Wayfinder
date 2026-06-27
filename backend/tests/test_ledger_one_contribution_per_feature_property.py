"""Property-based test for one contribution per feature (task 7.12).

Covers:
- **Property 9: One contribution per feature** (Task 7.12) - for any scored
  destination, its decision trace contains exactly one contribution value for
  each feature used in the scoring: no feature is missing and none is
  duplicated.
  Validates: Requirements 8.1.
  Tagged: Feature: wayfinder-travel-planner, Property 9.

This is a property test only. It does not modify the ledger or engine
implementation; a counterexample here would indicate a genuine
one-per-feature coverage bug to report rather than mask.
"""

from __future__ import annotations

from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from app.decision.engine import RawCandidate, score_candidates
from app.decision.ledger import build_decision_trace
from app.models.schemas import FEATURES, FeatureWeights

# ---------------------------------------------------------------------------
# Property 9: One contribution per feature
# Feature: wayfinder-travel-planner, Property 9
# Validates: Requirements 8.1
# ---------------------------------------------------------------------------

# Raw feature signals are drawn from a wide band -- including values well
# outside the engine's [0, 100] input bounds -- so the trace is exercised over
# clamping, normalization, and weighting paths alike.
_raw_value = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _valid_features(draw: st.DrawFn) -> dict[str, float]:
    """Build a fully-valid feature mapping (every feature numeric).

    Only valid candidates produce a ScoredCandidate that can be turned into a
    decision trace, so the one-per-feature invariant is what we exercise here.
    """
    return {feature: draw(_raw_value) for feature in FEATURES}


@st.composite
def _valid_candidate(draw: st.DrawFn) -> RawCandidate:
    destination = draw(st.text(min_size=1, max_size=12))
    return RawCandidate(destination=destination, features=draw(_valid_features()))


@st.composite
def _weights(draw: st.DrawFn) -> FeatureWeights:
    """Generate a valid weight config: non-negative, renormalized to sum 1.0."""
    raw = [
        draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
        for _ in FEATURES
    ]
    total = sum(raw)
    if total <= 0.0:
        # Degenerate all-zero draw -> fall back to uniform weights.
        raw = [1.0 for _ in FEATURES]
        total = float(len(FEATURES))
    normalized = [value / total for value in raw]
    # Absorb residual float drift into the last weight so the sum is 1.0 well
    # within the engine's 0.001 tolerance regardless of rounding. Clamp at zero
    # so float drift can never synthesize a tiny negative weight (which is
    # outside the valid weight-config input space this generator targets).
    normalized[-1] = max(0.0, 1.0 - sum(normalized[:-1]))
    return FeatureWeights(**dict(zip(FEATURES, normalized)))


@settings(max_examples=200)
@given(candidates=st.lists(_valid_candidate(), min_size=1, max_size=8), weights=_weights())
def test_decision_trace_has_exactly_one_contribution_per_feature(
    candidates: list[RawCandidate], weights: FeatureWeights
) -> None:
    """For every scored destination, the decision trace contains exactly one
    contribution per scored feature -- no feature missing and none duplicated.

    Each surviving candidate is scored and its decision trace built. The set of
    features named in the trace must equal the canonical feature set, each must
    appear exactly once, and the count of contributions must equal the number of
    scored features (Requirement 8.1).
    """
    result = score_candidates(candidates, weights)

    expected_features = set(FEATURES)

    for scored in result.ranked:
        trace = build_decision_trace(scored)

        feature_names = [f.feature for f in trace.factors]
        counts = Counter(feature_names)

        # Exactly one contribution per feature: no duplicates.
        duplicates = [name for name, count in counts.items() if count > 1]
        assert not duplicates, (
            f"duplicate feature contributions for '{scored.destination}': {duplicates}"
        )

        # Coverage is exact: the named features equal the canonical feature set,
        # so none is missing and none is extraneous.
        assert set(feature_names) == expected_features, (
            f"feature coverage mismatch for '{scored.destination}': "
            f"got {sorted(set(feature_names))}, expected {sorted(expected_features)}"
        )

        # One entry per feature -> the contribution count equals the feature count.
        assert len(trace.factors) == len(FEATURES), (
            f"expected {len(FEATURES)} contributions for '{scored.destination}', "
            f"got {len(trace.factors)}"
        )
