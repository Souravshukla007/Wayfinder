"""Property-based test for explainability-ledger additivity (task 7.11).

Covers:
- **Property 8: Ledger additivity** (Task 7.11) - for any scored destination,
  the sum of the per-feature contributions in its decision trace equals that
  destination's final score within an absolute tolerance of 0.001.
  Validates: Requirements 8.2.
  Tagged: Feature: wayfinder-travel-planner, Property 8.

This is a property test only. It does not modify the ledger or engine
implementation; a counterexample here would indicate a genuine additivity bug
to report rather than mask.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from app.decision.engine import RawCandidate, score_candidates
from app.decision.ledger import LEDGER_TOLERANCE, build_decision_trace
from app.models.schemas import FEATURES, FeatureWeights

# ---------------------------------------------------------------------------
# Property 8: Ledger additivity
# Feature: wayfinder-travel-planner, Property 8
# Validates: Requirements 8.2
# ---------------------------------------------------------------------------

# Raw feature signals are drawn from a wide band -- including values well
# outside the engine's [0, 100] input bounds -- so additivity is exercised over
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
    decision trace, so the additivity invariant is what we exercise here.
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
    # within the engine's 0.001 tolerance regardless of rounding. Clamp away
    # negative drift (e.g. -2.2e-16) so the non-negativity validator is not
    # tripped; the residual is far inside the 0.001 sum tolerance.
    normalized[-1] = max(0.0, 1.0 - sum(normalized[:-1]))
    return FeatureWeights(**dict(zip(FEATURES, normalized)))


@settings(max_examples=200)
@given(candidates=st.lists(_valid_candidate(), min_size=1, max_size=8), weights=_weights())
def test_ledger_contributions_sum_to_final_score(
    candidates: list[RawCandidate], weights: FeatureWeights
) -> None:
    """For every scored destination, the per-feature contributions in its
    decision trace sum to the destination's final score within 0.001.

    Each surviving candidate is scored, its decision trace is built by the
    ledger, and the sum of the trace's per-feature contributions is checked
    against the trace's recorded score (which equals the engine's final score).
    A deviation beyond 0.001 would be a real additivity violation
    (Requirement 8.2).
    """
    result = score_candidates(candidates, weights)

    for scored in result.ranked:
        trace = build_decision_trace(scored)

        # The trace's score is the candidate's final score.
        assert trace.score == scored.score

        contribution_sum = sum(f.contribution for f in trace.factors)
        assert abs(contribution_sum - trace.score) <= LEDGER_TOLERANCE, (
            f"ledger additivity violated for '{scored.destination}': "
            f"contributions sum to {contribution_sum} but score is {trace.score}"
        )
