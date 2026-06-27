"""Unit tests for the Coordinator agent node (Task 13.2, Requirements 3.2, 1.3).

Covers the three Coordinator responsibilities exercised by the task:
- decomposing a free-text prompt into structured ``TripConstraints``;
- loading the Memory Layer preference vector (cold-start) into shared state;
- merging the final explained plan while *withholding* any major recommendation
  that lacks a rationale and *recording* the missing-rationale condition.
"""

from __future__ import annotations

from decimal import Decimal

from app.config import Settings
from app.models.schemas import FEATURES
from app.orchestration.nodes import coordinator as coord
from app.orchestration.state import GraphState, ScoredCandidate


def _settings() -> Settings:
    # Explicit no-credential settings => mock providers, deterministic results.
    return Settings()


# --------------------------------------------------------------------------- #
# decompose_prompt
# --------------------------------------------------------------------------- #
def test_decompose_japan_example_extracts_budget_duration_interests() -> None:
    prompt = (
        "Plan a 7-day Japan trip in October under \u20b92 lakh. "
        "I like anime, photography, local food, and less crowded places."
    )
    constraints = coord.decompose_prompt(prompt, settings=_settings())

    assert constraints.budget == Decimal("200000")
    assert constraints.duration_days == 7
    # ceil(7/3) == 3 cities, one fewer hotel change.
    assert constraints.max_cities == 3
    assert constraints.max_hotel_changes == 2
    assert constraints.interests == ["anime", "photography", "food", "less_crowded"]


def test_decompose_parses_grouped_rupee_figure_and_nights() -> None:
    constraints = coord.decompose_prompt(
        "5 nights in Goa, budget \u20b91,20,000, vegetarian food", settings=_settings()
    )
    assert constraints.budget == Decimal("120000")
    assert constraints.duration_days == 6  # 5 nights -> 6 days
    assert "vegetarian" in constraints.dietary


def test_decompose_uses_defaults_when_unspecified() -> None:
    constraints = coord.decompose_prompt("Take me somewhere fun", settings=_settings())
    assert constraints.budget == coord._DEFAULT_BUDGET
    assert constraints.duration_days == coord._DEFAULT_DURATION_DAYS
    assert constraints.max_cities >= 1


def test_explicit_interests_merge_ahead_of_inferred() -> None:
    constraints = coord.decompose_prompt(
        "A trip with great photography", interests=["food"], settings=_settings()
    )
    # Explicitly supplied interests come first, inferred ones follow, deduped.
    assert constraints.interests == ["food", "photography"]


# --------------------------------------------------------------------------- #
# intake node: decompose + load memory
# --------------------------------------------------------------------------- #
def test_intake_node_sets_constraints_and_cold_start_preferences() -> None:
    state = GraphState(
        trip_id="t1",
        user_id="u1",
        prompt="Plan a 7-day Japan trip under \u20b92 lakh, I like anime",
    )
    out = coord.coordinator_intake_node(state, session=None, settings=_settings())

    assert out.constraints is not None
    assert out.constraints.duration_days == 7
    # Cold-start defaults: every required feature topic has a starting weight.
    topics = {p.topic for p in out.preference_vector}
    assert set(FEATURES).issubset(topics)


# --------------------------------------------------------------------------- #
# merge: rationale gating (Requirement 1.3 / 1.4)
# --------------------------------------------------------------------------- #
def _scored(name: str, rank: int, components: dict[str, float]) -> ScoredCandidate:
    score = sum(components.values())
    return ScoredCandidate(
        name=name, country="Japan", score=score, components=components, rank=rank
    )


def test_merge_withholds_destination_lacking_rationale_and_records_it() -> None:
    good = _scored("Kyoto", 1, {"budget_fit": 20.0, "food_score": 15.0})
    # No components => no basis for a rationale => must be withheld (Req 1.3).
    bad = _scored("Nowhere", 2, {})

    plan, withheld = coord.merge_final_plan(
        [good, bad], itinerary=[], bundle=None, constraints=None
    )

    names = [d.name for d in plan.destinations]
    assert "Kyoto" in names
    assert "Nowhere" not in names
    assert all(d.rationale for d in plan.destinations)

    withheld_dest = [w for w in withheld if w.kind == "destination"]
    assert len(withheld_dest) == 1
    assert withheld_dest[0].identifier == "Nowhere"


def test_merge_node_sets_missing_rationale_flag() -> None:
    state = GraphState(trip_id="t", user_id="u", prompt="p")
    state.constraints = coord.decompose_prompt("3-day trip", settings=_settings())
    state.scored = [_scored("Kyoto", 1, {"budget_fit": 30.0}), _scored("X", 2, {})]

    out = coord.coordinator_merge_node(state, settings=_settings())

    assert out.final_plan is not None
    assert out.missing_rationale is True
    assert any(w.identifier == "X" for w in out.withheld_recommendations)


def test_merge_attaches_rationales_to_flights_and_hotels() -> None:
    """Flight/hotel options each carry a rationale (Req 1.4); mock data has tags."""
    state = GraphState(trip_id="t", user_id="u", prompt="p")
    state.constraints = coord.decompose_prompt(
        "7-day Kyoto trip under \u20b92 lakh", settings=_settings()
    )
    state.scored = [_scored("Kyoto", 1, {"budget_fit": 25.0, "weather_fit": 20.0})]

    out = coord.coordinator_merge_node(state, settings=_settings())

    assert out.final_plan is not None
    assert out.final_plan.flights, "expected mock flights for Kyoto"
    assert out.final_plan.hotels, "expected mock hotels for Kyoto"
    assert all(f.rationale for f in out.final_plan.flights)
    assert all(h.rationale for h in out.final_plan.hotels)
    # Budget tool ran and produced a positive estimate.
    assert out.final_plan.estimated_total_cost > 0
