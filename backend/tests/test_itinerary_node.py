"""Unit tests for the Itinerary agent node (task 13.4).

Validates Requirements:
- 3.4: WHEN the Constraint Solver emits a feasible skeleton, the Itinerary
  agent enriches the skeleton into a day-by-day plan; and the critical gate —
  the agent runs ONLY when a feasible skeleton exists (skeleton present, no
  rejection). A solver rejection/timeout must prevent the agent from running.

These are example-based unit tests covering the gate, the day-by-day
enrichment, determinism, and that the LLM narration never alters the schedule.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from app.llm.base import LLMResponse
from app.models.schemas import TripConstraints
from app.orchestration.nodes.itinerary import (
    enrich_skeleton,
    itinerary_agent_should_run,
    itinerary_node,
    narrate_day,
)
from app.orchestration.state import (
    DayPlan,
    GraphState,
    ItineraryLeg,
    ItinerarySkeleton,
    SolverRejection,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _constraints(interests: list[str] | None = None) -> TripConstraints:
    return TripConstraints(
        budget=Decimal("200000"),
        duration_days=7,
        max_cities=3,
        max_hotel_changes=2,
        max_same_day_travel=timedelta(hours=4),
        dietary=[],
        interests=interests or [],
    )


def _skeleton() -> ItinerarySkeleton:
    return ItinerarySkeleton(
        legs=[
            ItineraryLeg(city="Tokyo", arrival_day=1, departure_day=3, hotel_change=False),
            ItineraryLeg(city="Kyoto", arrival_day=4, departure_day=5, hotel_change=True),
        ],
        total_cost=Decimal("150000"),
        hotel_changes=1,
    )


def _state(
    *,
    skeleton: ItinerarySkeleton | None = None,
    rejection: SolverRejection | None = None,
    interests: list[str] | None = None,
) -> GraphState:
    return GraphState(
        trip_id="trip-1",
        user_id="user-1",
        prompt="Plan a trip",
        constraints=_constraints(interests),
        skeleton=skeleton,
        rejection=rejection,
    )


class _RewritingLLM:
    """A hostile LLM stub that tries to bolt on an unplanned activity."""

    def complete(self, prompt, *, system=None, temperature=0.0) -> LLMResponse:
        return LLMResponse(
            text="Actually, add a day trip to Osaka and skip Kyoto.",
            model="rewriting-stub",
        )


# --------------------------------------------------------------------------- #
# The critical gate (Requirement 3.4)
# --------------------------------------------------------------------------- #
def test_runs_only_when_feasible_skeleton_present() -> None:
    assert itinerary_agent_should_run(_state(skeleton=_skeleton())) is True


def test_does_not_run_on_rejection() -> None:
    rejection = SolverRejection(unsatisfied_constraints=["budget_cap"], message="over budget")
    state = _state(skeleton=None, rejection=rejection)
    assert itinerary_agent_should_run(state) is False
    # The node returns an empty update and leaves the itinerary empty.
    assert itinerary_node(state) == {}


def test_does_not_run_on_timeout_rejection() -> None:
    # A timeout is a rejection with no skeleton; the agent must not run.
    timeout = SolverRejection(unsatisfied_constraints=[], timed_out=True, message="timed out")
    state = _state(skeleton=None, rejection=timeout)
    assert itinerary_agent_should_run(state) is False
    assert itinerary_node(state) == {}


def test_does_not_run_when_no_skeleton_and_no_rejection() -> None:
    # Defensive: nothing to enrich means nothing to do.
    state = _state(skeleton=None, rejection=None)
    assert itinerary_agent_should_run(state) is False
    assert itinerary_node(state) == {}


def test_rejection_takes_precedence_even_if_skeleton_set() -> None:
    # If both are somehow present, a rejection forbids running (no skeleton is
    # the contract, but the gate is conservative).
    rejection = SolverRejection(unsatisfied_constraints=["city_count_limit"])
    state = _state(skeleton=_skeleton(), rejection=rejection)
    assert itinerary_agent_should_run(state) is False
    assert itinerary_node(state) == {}


# --------------------------------------------------------------------------- #
# Day-by-day enrichment (Requirement 3.4)
# --------------------------------------------------------------------------- #
def test_enriches_skeleton_into_day_by_day_plan() -> None:
    state = _state(skeleton=_skeleton(), interests=["photography", "food"])
    update = itinerary_node(state)

    assert "itinerary" in update
    itinerary: list[DayPlan] = update["itinerary"]
    # Tokyo days 1-3 + Kyoto days 4-5 == 5 days.
    assert [d.day for d in itinerary] == [1, 2, 3, 4, 5]
    assert [d.city for d in itinerary] == ["Tokyo", "Tokyo", "Tokyo", "Kyoto", "Kyoto"]
    # Every day carries at least one activity.
    assert all(d.activities for d in itinerary)


def test_arrival_and_hotel_change_markers() -> None:
    itinerary = enrich_skeleton(_skeleton(), _constraints(["food"]), include_events=False)
    by_day = {d.day: d for d in itinerary}
    # Day 1 (Tokyo, no hotel change): plain arrival.
    assert any("Arrive in Tokyo" in a for a in by_day[1].activities)
    # Day 4 (Kyoto, hotel change flagged): arrival mentions hotel check-in.
    assert any("check in" in a.lower() for a in by_day[4].activities)


def test_interests_drive_activities() -> None:
    itinerary = enrich_skeleton(
        _skeleton(), _constraints(["anime"]), include_events=False
    )
    flat = [a for day in itinerary for a in day.activities]
    assert any("anime" in a.lower() for a in flat)


def test_no_interests_uses_default_activity() -> None:
    itinerary = enrich_skeleton(_skeleton(), _constraints([]), include_events=False)
    flat = [a for day in itinerary for a in day.activities]
    assert any("highlights" in a.lower() for a in flat)


def test_start_date_fills_in_dates() -> None:
    itinerary = enrich_skeleton(
        _skeleton(),
        _constraints(["food"]),
        include_events=False,
        start_date=date(2025, 10, 1),
    )
    assert itinerary[0].date == "2025-10-01"
    # Day 4 is three days after day 1.
    day4 = next(d for d in itinerary if d.day == 4)
    assert day4.date == "2025-10-04"


def test_dates_empty_without_start_date() -> None:
    itinerary = enrich_skeleton(_skeleton(), _constraints(["food"]), include_events=False)
    assert all(d.date == "" for d in itinerary)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_enrichment_is_deterministic() -> None:
    first = enrich_skeleton(_skeleton(), _constraints(["photography", "food"]))
    second = enrich_skeleton(_skeleton(), _constraints(["photography", "food"]))
    assert [(d.day, d.city, tuple(d.activities)) for d in first] == [
        (d.day, d.city, tuple(d.activities)) for d in second
    ]


# --------------------------------------------------------------------------- #
# Narration never alters the schedule (LLM narrates only)
# --------------------------------------------------------------------------- #
def test_hostile_llm_does_not_change_itinerary() -> None:
    state = _state(skeleton=_skeleton(), interests=["food"])
    update = itinerary_node(state, llm=_RewritingLLM())
    # The structured plan is built deterministically before any LLM call, so a
    # hostile narration cannot add/remove cities or days.
    itinerary = update["itinerary"]
    assert [d.city for d in itinerary] == ["Tokyo", "Tokyo", "Tokyo", "Kyoto", "Kyoto"]


def test_narrate_day_returns_text_without_changing_plan() -> None:
    day = DayPlan(day=1, city="Tokyo", activities=["Arrive in Tokyo", "Local food tour"])
    text = narrate_day(day)
    assert isinstance(text, str) and text
    # The plan object is untouched by narration.
    assert day.activities == ["Arrive in Tokyo", "Local food tour"]
