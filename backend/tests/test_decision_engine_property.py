"""Property-based test for Decision Engine scoring determinism (task 7.5).

Covers:
- **Property 1: Scoring determinism** (Task 7.5) - for any fixed set of
  candidates, feature weights, and constraints, computing scores two or more
  times produces bit-for-bit identical final scores and per-feature component
  breakdowns, with no dependence on randomness, system time, or network.
  Validates: Requirements 5.1, 5.7.
  Tagged: Feature: wayfinder-travel-planner, Property 1.

This is a property test only. It does not modify the engine implementation; a
counterexample here would indicate a genuine bug to report rather than mask.
"""

from __future__ import annotations

import random
import time

from hypothesis import given, settings
from hypothesis import strategies as st

from app.decision.engine import (
    RawCandidate,
    ScoringResult,
    score_candidates,
)
from app.models.schemas import FEATURES, FeatureWeights

# ---------------------------------------------------------------------------
# Property 1: Scoring determinism
# Feature: wayfinder-travel-planner, Property 1
# Validates: Requirements 5.1, 5.7
# ---------------------------------------------------------------------------

# Raw feature signals are deliberately drawn from a wide band -- including
# values well outside the engine's [0, 100] input bounds -- so determinism is
# exercised over clamping, normalization, and weighting paths alike.
_raw_value = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)

# A single feature cell: usually a valid numeric value, but occasionally a
# non-numeric/absent marker so the rejection path's determinism is covered too.
_NON_NUMERIC = st.sampled_from([None, "warm", True, [1, 2]])


@st.composite
def _features(draw: st.DrawFn) -> dict[str, object]:
    """Build a feature mapping that may be valid or invalid (per feature)."""
    features: dict[str, object] = {}
    for feature in FEATURES:
        roll = draw(st.integers(min_value=0, max_value=11))
        if roll == 0:
            # Drop the feature entirely (missing -> candidate rejected).
            continue
        if roll == 1:
            features[feature] = draw(_NON_NUMERIC)
        else:
            features[feature] = draw(_raw_value)
    return features


@st.composite
def _candidate(draw: st.DrawFn) -> RawCandidate:
    destination = draw(st.text(min_size=1, max_size=12))
    return RawCandidate(destination=destination, features=draw(_features()))


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
    # Absorb any residual float drift into the last weight so the sum is 1.0
    # well within the engine's 0.001 tolerance regardless of rounding. Clamp at
    # zero so float drift can never synthesize a tiny negative weight (which is
    # outside the valid weight-config input space this generator targets).
    normalized[-1] = max(0.0, 1.0 - sum(normalized[:-1]))
    return FeatureWeights(**dict(zip(FEATURES, normalized)))


def _fingerprint(result: ScoringResult) -> tuple:
    """A fully-ordered, hashable snapshot of a scoring run for equality checks."""
    ranked = tuple(
        (
            c.destination,
            c.score,
            tuple((f, c.normalized[f]) for f in FEATURES),
            tuple((f, c.contributions[f]) for f in FEATURES),
        )
        for c in result.ranked
    )
    rejected = tuple((r.destination, r.reason) for r in result.rejected)
    return (ranked, rejected)


@settings(max_examples=200)
@given(candidates=st.lists(_candidate(), max_size=8), weights=_weights())
def test_scoring_is_deterministic_across_repeated_runs(
    candidates: list[RawCandidate], weights: FeatureWeights
) -> None:
    """Same (candidates, weights) scored 3x => bit-for-bit identical output.

    Between runs the process-global random state and the perceived wall clock
    are perturbed; an engine that secretly depended on randomness, system time,
    or any nondeterministic source would diverge. Bit-for-bit equality of the
    final scores, the full ranking order, and every per-feature normalized
    value and weighted contribution confirms purity (Requirements 5.1, 5.7).
    """
    first = _fingerprint(score_candidates(candidates, weights))

    # Perturb global nondeterminism sources between runs.
    random.seed(12345)
    _ = random.random()
    _ = time.time()

    second = _fingerprint(score_candidates(candidates, weights))

    random.seed(98765)
    _ = random.random()

    third = _fingerprint(score_candidates(candidates, weights))

    assert first == second
    assert second == third


@settings(max_examples=100)
@given(candidates=st.lists(_candidate(), max_size=8), weights=_weights())
def test_repeated_scores_are_bitwise_equal(
    candidates: list[RawCandidate], weights: FeatureWeights
) -> None:
    """Final scores match by exact float bit pattern, not just approximately."""
    first = score_candidates(candidates, weights)
    second = score_candidates(candidates, weights)

    assert [c.destination for c in first.ranked] == [
        c.destination for c in second.ranked
    ]
    for a, b in zip(first.ranked, second.ranked):
        # Exact equality (==) on floats is intentional: determinism demands
        # identical bit patterns, not closeness.
        assert a.score == b.score
        assert a.normalized == b.normalized
        assert a.contributions == b.contributions
