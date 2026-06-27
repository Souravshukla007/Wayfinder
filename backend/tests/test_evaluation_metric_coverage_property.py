"""Property-based test for evaluation metric coverage (task 22.4).

Feature: wayfinder-travel-planner, Property 29: Evaluation metric coverage.

*For any* produced plan scored by the evaluation harness, the harness SHALL emit
a score for each of budget adherence, constraint satisfaction, feasibility,
explanation coverage, and personalization.

Validates: Requirements 14.1

The property is exercised directly against :func:`app.eval.harness.score_plan`,
which is the harness's scoring gate: given a golden scenario and the final graph
state a planning run produced, it must emit a :class:`MetricScores` covering
every metric in :data:`EVAL_METRICS`. Rather than running the (expensive,
deterministic) planner, this test generates a *wide space of plan/graph states*
— feasible skeletons, solver rejections, empty/partial plans, varied
constraints and interests — and asserts that for every such produced state the
harness still scores all five metrics, each normalized to ``[0, 1]``.

The generators deliberately straddle the scorer's branches (skeleton present vs
rejection vs neither; final plan present vs absent; matching vs non-matching
interests) so each run pushes the coverage invariant through the full range of
states a real run could land in.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from app.eval.harness import score_plan
from app.eval.scenarios.base import EVAL_METRICS, GoldenScenario, MetricThresholds
from app.models.schemas import FeatureContribution, FlightDTO, HotelDTO
from app.orchestration.nodes.destination import DESTINATION_CATALOG
from app.orchestration.state import (
    FinalPlan,
    GraphState,
    ItineraryLeg,
    ItinerarySkeleton,
    RecommendedDestination,
    ScoredCandidate,
    SolverRejection,
    WithheldRecommendation,
)

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# Real catalog city names so the personalization scorer (which looks up a city's
# interest tags in the catalog) exercises both matching and non-matching paths.
_CATALOG_NAMES = tuple(entry.name for entry in DESTINATION_CATALOG)

# A pool of interest tags mixing catalog tags (which can match) with free-form
# tags (which generally won't), so personalization spans its full output range.
_INTEREST_POOL = (
    "food",
    "photography",
    "history",
    "nature",
    "less crowded",
    "nightlife",
    "snorkelling",
    "opera",
    "desert safari",
)

_DIETARY_POOL = ("vegetarian", "vegan", "halal", "kosher", "gluten-free")

_money = st.integers(min_value=1, max_value=50_000)
_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_rationale = st.text(min_size=1, max_size=24)


@st.composite
def _constraints(draw: st.DrawFn):
    """Draw a valid :class:`TripConstraints`-shaped mapping for a scenario."""
    return dict(
        budget=Decimal(draw(_money)),
        duration_days=draw(st.integers(min_value=1, max_value=30)),
        max_cities=draw(st.integers(min_value=1, max_value=8)),
        max_hotel_changes=draw(st.integers(min_value=0, max_value=8)),
        max_same_day_travel=timedelta(hours=draw(st.integers(min_value=1, max_value=12))),
        dietary=draw(st.lists(st.sampled_from(_DIETARY_POOL), max_size=3, unique=True)),
        interests=draw(st.lists(st.sampled_from(_INTEREST_POOL), max_size=4, unique=True)),
    )


@st.composite
def _scenario(draw: st.DrawFn) -> GoldenScenario:
    """Draw a golden scenario with arbitrary (valid) thresholds and budget cap."""
    from app.models.schemas import TripConstraints

    constraints = TripConstraints(**draw(_constraints()))
    thresholds = MetricThresholds(
        budget_adherence=draw(_unit),
        constraint_satisfaction=draw(_unit),
        feasibility=draw(_unit),
        explanation_coverage=draw(_unit),
        personalization=draw(_unit),
    )
    return GoldenScenario(
        id=f"gen-{draw(st.integers(min_value=0, max_value=10_000))}",
        name="generated scenario",
        prompt="plan me a trip",
        interests=draw(st.lists(st.sampled_from(_INTEREST_POOL), max_size=4, unique=True)),
        constraints=constraints,
        thresholds=thresholds,
        budget_cap=constraints.budget,
    )


@st.composite
def _skeleton(draw: st.DrawFn) -> ItinerarySkeleton:
    """Draw a non-empty feasible itinerary skeleton over catalog cities."""
    n = draw(st.integers(min_value=1, max_value=5))
    legs: list[ItineraryLeg] = []
    for _ in range(n):
        arrival = draw(st.integers(min_value=0, max_value=20))
        stay = draw(st.integers(min_value=0, max_value=5))
        legs.append(
            ItineraryLeg(
                city=draw(st.sampled_from(_CATALOG_NAMES)),
                arrival_day=arrival,
                departure_day=arrival + stay,
                hotel_change=draw(st.booleans()),
            )
        )
    return ItinerarySkeleton(
        legs=legs,
        total_cost=Decimal(draw(st.integers(min_value=0, max_value=80_000))),
        hotel_changes=draw(st.integers(min_value=0, max_value=8)),
    )


def _factor() -> st.SearchStrategy[FeatureContribution]:
    return st.builds(
        FeatureContribution,
        feature=st.sampled_from(
            ("budget", "weather", "crowd", "interest", "safety", "novelty")
        ),
        contribution=st.floats(
            min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
    )


def _flight() -> st.SearchStrategy[FlightDTO]:
    return st.builds(
        FlightDTO,
        airline=st.text(min_size=0, max_size=10),
        price=st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
        duration=st.text(min_size=0, max_size=8),
        # rationale may be empty or present so explanation_coverage spans its range.
        rationale=st.text(min_size=0, max_size=20),
    )


def _hotel() -> st.SearchStrategy[HotelDTO]:
    return st.builds(
        HotelDTO,
        name=st.text(min_size=0, max_size=12),
        rating=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        pricePerNight=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        distanceKm=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        rationale=st.text(min_size=0, max_size=20),
    )


@st.composite
def _recommended_destination(draw: st.DrawFn) -> RecommendedDestination:
    return RecommendedDestination(
        name=draw(st.sampled_from(_CATALOG_NAMES)),
        country="Japan",
        score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        rank=draw(st.integers(min_value=1, max_value=8)),
        rationale=draw(_rationale),
        factors=draw(st.lists(_factor(), max_size=4)),
    )


@st.composite
def _final_plan(draw: st.DrawFn) -> FinalPlan:
    return FinalPlan(
        destinations=draw(st.lists(_recommended_destination(), max_size=5)),
        flights=draw(st.lists(_flight(), max_size=3)),
        hotels=draw(st.lists(_hotel(), max_size=3)),
        estimated_total_cost=Decimal(draw(st.integers(min_value=0, max_value=80_000))),
        within_budget=draw(st.booleans()),
    )


@st.composite
def _scored_candidate(draw: st.DrawFn) -> ScoredCandidate:
    return ScoredCandidate(
        name=draw(st.sampled_from(_CATALOG_NAMES)),
        country="Japan",
        score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        rank=draw(st.integers(min_value=1, max_value=8)),
    )


@st.composite
def _withheld(draw: st.DrawFn) -> WithheldRecommendation:
    return WithheldRecommendation(
        kind=draw(st.sampled_from(("destination", "flight", "hotel"))),
        identifier=draw(st.text(min_size=0, max_size=10)),
    )


@st.composite
def _graph_state(draw: st.DrawFn, scenario: GoldenScenario) -> GraphState:
    """Draw a final graph state spanning the scorer's branches.

    Independently decides whether a feasible skeleton, a solver rejection, a
    final plan, scored candidates, and withheld recommendations are present, so
    the generated states cover feasible runs, rejections, and partial/empty
    outputs alike.
    """
    has_skeleton = draw(st.booleans())
    has_rejection = draw(st.booleans())
    has_plan = draw(st.booleans())

    state = GraphState(
        trip_id="prop-eval",
        user_id="prop",
        prompt=scenario.prompt,
        constraints=scenario.constraints,
    )
    if has_skeleton:
        state.skeleton = draw(_skeleton())
    if has_rejection:
        state.rejection = SolverRejection(
            unsatisfied_constraints=draw(
                st.lists(st.text(min_size=0, max_size=8), max_size=3)
            ),
            timed_out=draw(st.booleans()),
        )
    if has_plan:
        state.final_plan = draw(_final_plan())
    state.scored = draw(st.lists(_scored_candidate(), max_size=5))
    state.withheld_recommendations = draw(st.lists(_withheld(), max_size=3))
    return state


@st.composite
def _scenario_and_state(draw: st.DrawFn) -> tuple[GoldenScenario, GraphState]:
    scenario = draw(_scenario())
    state = draw(_graph_state(scenario))
    return scenario, state


# ---------------------------------------------------------------------------
# Property 29: Evaluation metric coverage
# Feature: wayfinder-travel-planner, Property 29
# Validates: Requirements 14.1
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(scenario_and_state=_scenario_and_state())
def test_evaluation_metric_coverage(
    scenario_and_state: tuple[GoldenScenario, GraphState],
) -> None:
    """Every produced plan is scored on all five metrics, each in ``[0, 1]``.

    For any scenario + final graph state a run could produce, ``score_plan``:

    1. **Covers every metric (Req 14.1 / Property 29).** The emitted scores
       expose exactly the canonical metric set :data:`EVAL_METRICS` — budget
       adherence, constraint satisfaction, feasibility, explanation coverage,
       and personalization — with none missing or extra.
    2. **Normalizes every score to ``[0, 1]``.** Each metric's score is a finite
       value within the normalized range, so it is comparable against a
       threshold on the same scale (Requirement 14.2's precondition).
    """
    scenario, state = scenario_and_state

    scores = score_plan(scenario, state)
    score_map = scores.as_dict()

    # (1) Exactly the five canonical metrics are scored — full coverage.
    assert set(score_map) == set(EVAL_METRICS)
    assert len(score_map) == len(EVAL_METRICS)

    # (2) Every metric carries a finite, normalized score in [0, 1].
    for metric in EVAL_METRICS:
        value = score_map[metric]
        assert isinstance(value, float), f"{metric} score is not a float: {value!r}"
        assert value == value, f"{metric} score is NaN"  # noqa: PLR0124
        assert 0.0 <= value <= 1.0, f"{metric} score {value} outside [0, 1]"
