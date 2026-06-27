"""Property-based test for ranked descending output with top-three presentation.

Feature: wayfinder-travel-planner, Property 11: Ranked descending with
top-three presentation.

*For any* set of valid candidates, the returned list SHALL be ordered by final
score in non-increasing (descending) order, and when three or more valid
candidates exist, at least three options SHALL be presented.

Validates: Requirements 7.1, 7.2

The property is exercised against the two layers that together realize it:

* the deterministic Decision Engine (:func:`score_candidates`), which owns the
  ranking and produces a list ordered by final score descending (Req 7.1); and
* the Coordinator's merge step (:func:`merge_final_plan`), the presentation
  layer that surfaces the ranked options to the user. The engine's ranked
  output is mapped onto the shared-state schema exactly as the graph does
  (mirroring ``graph._to_scored_state``) before being presented, so the test
  follows the real pipeline rather than a hand-rolled shortcut.

The presentation must preserve the engine's descending order and, when three or
more valid candidates exist, surface at least three options (Req 7.2); with
fewer than three it surfaces all of them.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from app.decision.engine import RawCandidate, score_candidates
from app.models.schemas import FEATURES, FeatureWeights
from app.orchestration.nodes.coordinator import merge_final_plan
from app.orchestration.state import ScoredCandidate as StateScoredCandidate

# Raw feature signals span past the engine's [0, 100] input band so the property
# holds for in-bounds, boundary, and out-of-bounds values alike. NaN/inf are
# excluded since those make a candidate invalid and never reach the ranking.
_feature_value = st.floats(
    min_value=-500.0,
    max_value=500.0,
    allow_nan=False,
    allow_infinity=False,
)

_feature_map = st.fixed_dictionaries({feature: _feature_value for feature in FEATURES})


@st.composite
def _valid_weights(draw: st.DrawFn) -> FeatureWeights:
    """Draw a valid weight configuration: non-negative, summing to 1.0.

    Six non-negative reals are drawn and renormalized by their total, landing
    well within the engine's 0.001 sum tolerance (Requirement 5.6).
    """
    raw = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=len(FEATURES),
            max_size=len(FEATURES),
        ).filter(lambda xs: sum(xs) > 0.0)
    )
    total = sum(raw)
    return FeatureWeights(**dict(zip(FEATURES, (x / total for x in raw))))


@st.composite
def _candidates(draw: st.DrawFn) -> list[RawCandidate]:
    """Draw 0-8 valid candidates with distinct names and full feature maps.

    The size range deliberately straddles the boundary at three so the test
    covers fewer-than-three, exactly-three, and more-than-three cases.
    """
    count = draw(st.integers(min_value=0, max_value=8))
    return [
        RawCandidate(
            destination=f"C{i}",
            features=draw(_feature_map),
            metadata={"country": "Testland"},
        )
        for i in range(count)
    ]


def _present(ranked: list) -> list:
    """Map engine ranked output -> presented destinations, like the real graph.

    Mirrors ``graph._to_scored_state`` (engine candidate -> shared-state
    ``ScoredCandidate``) and then runs the Coordinator's merge presentation
    layer, returning ``FinalPlan.destinations`` (the surfaced options).
    """
    scored_state = [
        StateScoredCandidate(
            name=candidate.destination,
            country=str(candidate.metadata.get("country", "")),
            score=candidate.score,
            components=dict(candidate.contributions),
            rank=position,
        )
        for position, candidate in enumerate(ranked, start=1)
    ]
    plan, _withheld = merge_final_plan(scored_state, [], None, None)
    return plan.destinations


# ---------------------------------------------------------------------------
# Property 11: Ranked descending with top-three presentation
# Feature: wayfinder-travel-planner, Property 11
# Validates: Requirements 7.1, 7.2
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(candidates=_candidates(), weights=_valid_weights())
def test_ranked_descending_with_top_three_presentation(
    candidates: list[RawCandidate], weights: FeatureWeights
) -> None:
    """Results rank by score descending and the presentation surfaces the top
    three (or fewer when fewer exist).

    **Validates: Requirements 7.1, 7.2**
    """
    ranked = score_candidates(candidates, weights).ranked

    # --- Requirement 7.1: engine output is non-increasing by final score. ---
    scores = [c.score for c in ranked]
    assert all(earlier >= later for earlier, later in zip(scores, scores[1:])), (
        f"engine ranking is not non-increasing: {scores}"
    )
    # All valid candidates survive (these generated candidates are all valid).
    assert len(ranked) == len(candidates)

    # --- Presentation layer surfaces the ranked options. ---
    presented = _present(ranked)

    # The presentation preserves the engine's descending order verbatim — names
    # and scores match position-for-position; the presenter never reorders.
    assert [(d.name, d.score) for d in presented] == [
        (c.destination, c.score) for c in ranked
    ]

    presented_scores = [d.score for d in presented]
    assert all(
        earlier >= later
        for earlier, later in zip(presented_scores, presented_scores[1:])
    ), f"presented options are not non-increasing: {presented_scores}"

    # --- Requirement 7.2: at least three options when three or more exist. ---
    if len(ranked) >= 3:
        assert len(presented) >= 3
        # The surfaced top-three are exactly the engine's three highest-scored.
        assert [(d.name, d.score) for d in presented[:3]] == [
            (c.destination, c.score) for c in ranked[:3]
        ]
    else:
        # Fewer than three valid candidates: surface all of them, none dropped.
        assert len(presented) == len(ranked)
