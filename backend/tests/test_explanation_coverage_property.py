"""Property-based test for explanation coverage (task 13.8).

Feature: wayfinder-travel-planner, Property 12: Explanation coverage.

*For any* produced plan, every destination recommendation SHALL carry a
rationale, and any ranked output SHALL include each option's final score
together with an explanation of why higher-ranked options outrank lower-ranked
ones. Any major recommendation that cannot be accompanied by a rationale SHALL
be withheld and the missing-rationale condition recorded — no unexplained
recommendation is ever presented to the user.

Validates: Requirements 1.1, 1.2, 7.3

The property is exercised against the Coordinator's merge step
(:func:`merge_final_plan`), which is the presentation gate every destination
recommendation passes through before reaching the user. A destination's
rationale is derived deterministically from its additive per-feature
contributions (``components``): a candidate carrying contributions gets a
rationale and is surfaced; a candidate with no contributions has no basis for
an explanation, so it is withheld and recorded as a
:class:`~app.orchestration.state.WithheldRecommendation` (Requirement 1.3 — the
mechanism that guarantees the coverage invariant holds).

Bundle and constraints are passed as ``None`` so this test isolates the
destination explanation-coverage invariant; flight/hotel rationale presence is
covered separately by Property 13 (task 13.9).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from app.orchestration.nodes.coordinator import merge_final_plan
from app.orchestration.state import ScoredCandidate

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# The six Decision-Engine features that may appear as additive contributions.
_FEATURE_NAMES = ("budget", "weather", "crowd", "interest", "safety", "novelty")

_contribution_value = st.floats(
    min_value=0.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def _scored_candidates(draw: st.DrawFn) -> list[ScoredCandidate]:
    """Draw a ranked list mixing explainable and unexplainable candidates.

    Each candidate is given a distinct name so withheld items can be matched
    back unambiguously. Roughly half are drawn with a non-empty ``components``
    map (an explainable recommendation that must be surfaced) and the rest with
    an empty map (no basis for a rationale -> must be withheld and recorded).
    The mix deliberately straddles the boundary so every run exercises both the
    "presented with explanation" and "withheld and recorded" branches.
    """
    count = draw(st.integers(min_value=0, max_value=8))

    raw: list[tuple[float, dict[str, float]]] = []
    for _ in range(count):
        score = draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False))
        has_components = draw(st.booleans())
        if has_components:
            components = draw(
                st.dictionaries(
                    keys=st.sampled_from(_FEATURE_NAMES),
                    values=_contribution_value,
                    min_size=1,
                    max_size=len(_FEATURE_NAMES),
                )
            )
        else:
            components = {}
        raw.append((score, components))

    # Emulate the engine: order by score descending and assign rank positions.
    raw.sort(key=lambda item: item[0], reverse=True)
    return [
        ScoredCandidate(
            name=f"C{index}",
            country="Testland",
            score=score,
            components=components,
            rank=position,
        )
        for position, (index, (score, components)) in enumerate(
            enumerate(raw), start=1
        )
    ]


# ---------------------------------------------------------------------------
# Property 12: Explanation coverage
# Feature: wayfinder-travel-planner, Property 12
# Validates: Requirements 1.1, 1.2, 7.3
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(scored=_scored_candidates())
def test_explanation_coverage(scored: list[ScoredCandidate]) -> None:
    """Every surfaced destination carries an explanation; the rest are withheld.

    Three guarantees are checked for each produced plan:

    1. **No unexplained recommendation is presented (Req 1.1).** Every
       destination in ``FinalPlan.destinations`` carries a non-empty rationale.
    2. **Withhold-and-record (Req 1.3, the coverage mechanism).** Every
       candidate lacking a basis for an explanation (no contributions) is absent
       from the presented destinations and recorded as a withheld destination;
       every candidate with contributions is surfaced. No candidate is silently
       dropped — presented + withheld accounts for all of them.
    3. **Ranked output carries scores and an ordering explanation
       (Req 1.2, 7.3).** Each surfaced option includes its final score, the
       presented order is non-increasing by score, and each rationale states the
       option's rank and score so a reader can see why higher options outrank
       lower ones.
    """
    plan, withheld = merge_final_plan(scored, [], None, None)

    explainable = [c for c in scored if c.components]
    unexplainable = [c for c in scored if not c.components]

    presented_names = {d.name for d in plan.destinations}
    withheld_destinations = [w for w in withheld if w.kind == "destination"]
    withheld_names = {w.identifier for w in withheld_destinations}

    # (1) No unexplained recommendation reaches the user.
    for destination in plan.destinations:
        assert destination.rationale, (
            f"presented destination {destination.name!r} has no rationale"
        )
        assert destination.rationale.strip() != ""

    # (2) Withhold-and-record: explainable surfaced, unexplainable recorded.
    assert presented_names == {c.name for c in explainable}
    assert withheld_names == {c.name for c in unexplainable}
    for w in withheld_destinations:
        assert w.reason, "withheld recommendation must record why it was withheld"
    # Nothing is silently dropped: every candidate is either presented or withheld.
    assert len(plan.destinations) + len(withheld_destinations) == len(scored)

    # (3) Ranked output: scores present, order preserved, ordering explained.
    presented_scores = [d.score for d in plan.destinations]
    assert all(
        earlier >= later
        for earlier, later in zip(presented_scores, presented_scores[1:])
    ), f"presented options are not non-increasing by score: {presented_scores}"

    for destination in plan.destinations:
        # The option's final score is surfaced verbatim alongside it.
        score_text = f"{destination.score:.1f}"
        assert score_text in destination.rationale, (
            f"rationale for {destination.name!r} omits its final score {score_text}"
        )
        # The ordering is explained: the rationale states the option's rank,
        # which is what lets a reader see why higher options outrank lower ones.
        assert f"#{destination.rank}" in destination.rationale, (
            f"rationale for {destination.name!r} omits its rank #{destination.rank}"
        )
        # The auditable per-feature factors back the explanation.
        assert destination.factors, (
            f"surfaced destination {destination.name!r} carries no factors"
        )
