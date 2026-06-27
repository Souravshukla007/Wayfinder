"""The Itinerary agent node (task 13.4, Requirement 3.4).

The Itinerary agent is the third reasoning agent of the LangGraph pipeline. Its
single responsibility is to **enrich the Constraint Solver's feasible skeleton
into a day-by-day plan**:

    ... -> CP-SAT Solver -> Itinerary Agent -> Coordinator (merge)

**Hard gate (Requirement 3.4 / 9.8 / 9.9).** The agent runs *only* when the
solver emitted a feasible skeleton. Concretely, it runs iff
``state.skeleton is not None and state.rejection is None``. If the solver
produced a :class:`~app.orchestration.state.SolverRejection` (infeasible) or a
:class:`~app.solver.cp_sat.SolverTimeout`, **no skeleton is present** and the
Itinerary agent does nothing — it returns an empty state update so the
``state.itinerary`` stays empty and the pipeline moves on without an enriched
plan. :func:`itinerary_agent_should_run` exposes that gate as a pure predicate
so the graph assembly (Task 13.5) and tests can reason about it explicitly.

**What enrichment does.** A skeleton is a sequence of city *legs*
(:class:`~app.orchestration.state.ItineraryLeg`) with arrival/departure trip-day
indices and hotel-change flags. Enrichment expands every leg into one
:class:`~app.orchestration.state.DayPlan` per occupied day, assigning concrete,
deterministic activities driven by the trip's stated interests (and any events
discovered for the city). The structured day-by-day plan is built
deterministically — no LLM, no randomness, no clock — so a feasible skeleton
always enriches to the same itinerary. The pluggable LLM provider
(:mod:`app.llm`) is used only to add optional friendly *narration*; it never
invents the schedule, mirroring the guardrail the Destination agent applies to
scores.

Requirements: 3.4.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from typing import Any, Optional

from app.config import Settings, get_settings
from app.llm.base import LLMProvider, get_llm_provider
from app.models.schemas import TripConstraints
from app.orchestration.state import (
    DayPlan,
    GraphState,
    ItineraryLeg,
    ItinerarySkeleton,
)
from app.providers.base import EventOption, EventQuery
from app.tools.events import search_events

# System prompt that confines the LLM to narration. The day-by-day schedule is
# fixed before the LLM is ever called; the model only phrases it nicely.
NARRATION_SYSTEM_PROMPT = (
    "You are a travel consultant writing a short, friendly summary of an "
    "itinerary day that has ALREADY been planned. Describe the day's city and "
    "activities in plain language. You MUST NOT add, remove, or reschedule any "
    "activity, city, or day; treat the provided plan as fixed."
)

# Interest tag -> a deterministic activity phrasing. Tags are the normalized
# interest tags produced by the Coordinator's prompt decomposition.
_INTEREST_ACTIVITIES: tuple[tuple[str, str], ...] = (
    ("anime", "Explore the anime & manga district"),
    ("photography", "Golden-hour photography walk"),
    ("food", "Local food tour and street-eats tasting"),
    ("less_crowded", "Wander quieter, off-the-beaten-path neighborhoods"),
    ("nature", "Half-day nature excursion"),
    ("history", "Visit historical landmarks"),
    ("culture", "Temples, shrines, and cultural sites"),
    ("beach", "Relax at the waterfront"),
    ("nightlife", "Evening out exploring the local nightlife"),
    ("shopping", "Browse local markets and shops"),
    ("adventure", "Outdoor adventure activity"),
)

# A neutral fallback so every day has at least one activity even when the trip
# states no interests.
_DEFAULT_ACTIVITY = "Explore the city's main highlights"


def itinerary_agent_should_run(state: GraphState) -> bool:
    """Return whether the Itinerary agent may run for this state (Req 3.4).

    The agent enriches a plan only when the Constraint Solver produced a
    feasible skeleton. A solver rejection or timeout leaves ``state.skeleton``
    unset (and ``state.rejection`` set), which forbids enrichment. This is a
    pure predicate over the shared state with no side effects.
    """
    return state.skeleton is not None and state.rejection is None


def _interest_activities(interests: Sequence[str]) -> list[str]:
    """Map the trip's interests to ordered, de-duplicated activity phrasings."""
    wanted = {tag.strip().lower() for tag in interests if tag and tag.strip()}
    activities = [phrase for tag, phrase in _INTEREST_ACTIVITIES if tag in wanted]
    return activities


def _events_for_city(
    city: str,
    *,
    settings: Settings,
) -> list[EventOption]:
    """Fetch events for a city through the deterministic events tool (mock-first).

    Best-effort enrichment: any provider error degrades to no events rather than
    failing the whole plan, since events are supplementary to the skeleton.
    """
    try:
        return search_events(EventQuery(city=city), settings=settings)
    except Exception:  # pragma: no cover - defensive; events are supplementary.
        return []


def _day_date(start_date: date | None, day_index: int) -> str:
    """Render the ISO date for a 1-indexed trip day, or ``""`` when unknown."""
    if start_date is None:
        return ""
    return (start_date + timedelta(days=day_index - 1)).isoformat()


def _activities_for_day(
    *,
    city: str,
    position_in_stay: int,
    is_arrival: bool,
    is_departure: bool,
    hotel_change: bool,
    interest_activities: Sequence[str],
    city_events: Sequence[EventOption],
) -> list[str]:
    """Build the deterministic activity list for a single day of a city stay.

    The arrival day leads with a check-in marker (noting a hotel change when the
    leg flags one); interior days rotate through the trip's interest activities
    so consecutive days differ; the departure day notes onward travel. Events
    discovered for the city are woven in by stay position so the assignment is
    stable for a given skeleton.
    """
    activities: list[str] = []

    if is_arrival:
        if hotel_change:
            activities.append(f"Arrive in {city} and check in to your hotel")
        else:
            activities.append(f"Arrive in {city}")

    # Pick one interest-driven activity, rotating by the day's position so a
    # multi-day stay shows variety while staying fully deterministic.
    if interest_activities:
        chosen = interest_activities[position_in_stay % len(interest_activities)]
        activities.append(chosen)
    else:
        activities.append(_DEFAULT_ACTIVITY)

    # Weave in a city event deterministically by stay position when available.
    if city_events:
        event = city_events[position_in_stay % len(city_events)]
        activities.append(f"Attend {event.name} ({event.type})")

    if is_departure and not is_arrival:
        activities.append(f"Depart {city}")

    return activities


def _enrich_leg(
    leg: ItineraryLeg,
    *,
    interest_activities: Sequence[str],
    city_events: Sequence[EventOption],
    start_date: date | None,
) -> list[DayPlan]:
    """Expand one skeleton leg into one :class:`DayPlan` per occupied day."""
    days: list[DayPlan] = []
    # Guard against malformed spans: ensure at least the arrival day is emitted.
    last_day = max(leg.arrival_day, leg.departure_day)
    for position, day in enumerate(range(leg.arrival_day, last_day + 1)):
        activities = _activities_for_day(
            city=leg.city,
            position_in_stay=position,
            is_arrival=(day == leg.arrival_day),
            is_departure=(day == last_day),
            hotel_change=leg.hotel_change,
            interest_activities=interest_activities,
            city_events=city_events,
        )
        days.append(
            DayPlan(
                day=day,
                date=_day_date(start_date, day),
                city=leg.city,
                activities=activities,
            )
        )
    return days


def enrich_skeleton(
    skeleton: ItinerarySkeleton,
    constraints: TripConstraints | None,
    *,
    settings: Settings | None = None,
    start_date: date | None = None,
    include_events: bool = True,
) -> list[DayPlan]:
    """Enrich a feasible solver skeleton into a deterministic day-by-day plan.

    Expands every leg of the skeleton into one :class:`DayPlan` per occupied
    trip day, attaching interest-driven activities (and, when ``include_events``
    is set, any events discovered for the leg's city through the events tool).
    The result is a pure function of ``(skeleton, constraints, start_date)`` for
    a fixed provider configuration — no LLM, randomness, or clock — so a given
    feasible skeleton always enriches to the same itinerary.

    Args:
        skeleton: the feasible skeleton emitted by the Constraint Solver.
        constraints: the trip's constraints; ``interests`` steer the activities.
        settings: optional settings override (provider selection for events).
        start_date: optional first-day date; when given, each ``DayPlan.date`` is
            filled in, otherwise dates are left empty.
        include_events: when ``True`` (default) fold city events into the plan.

    Returns:
        The day-by-day itinerary ordered by trip day.
    """
    settings = settings or get_settings()
    interests = list(constraints.interests) if constraints is not None else []
    interest_activities = _interest_activities(interests)

    plan: list[DayPlan] = []
    for leg in skeleton.legs:
        city_events = (
            _events_for_city(leg.city, settings=settings) if include_events else []
        )
        plan.extend(
            _enrich_leg(
                leg,
                interest_activities=interest_activities,
                city_events=city_events,
                start_date=start_date,
            )
        )
    return plan


def narrate_day(day_plan: DayPlan, *, llm: LLMProvider | None = None) -> str:
    """Produce optional friendly narration for one planned day (LLM, narration only).

    The schedule in ``day_plan`` is already fixed; the LLM is handed it as
    immutable context and only phrases it. With the default mock provider the
    output is deterministic, keeping whole runs reproducible.
    """
    provider = llm if llm is not None else get_llm_provider()
    activities = "; ".join(day_plan.activities) or "free day"
    prompt = (
        f"Day {day_plan.day} in {day_plan.city}. Planned activities: {activities}. "
        "Write a one-sentence friendly summary. Do not change the plan."
    )
    return provider.complete(
        prompt, system=NARRATION_SYSTEM_PROMPT, temperature=0.0
    ).text


def itinerary_node(
    state: GraphState,
    *,
    settings: Settings | None = None,
    start_date: date | None = None,
    llm: LLMProvider | None = None,
) -> dict[str, Any]:
    """LangGraph node for the Itinerary agent.

    Enriches the solver's feasible skeleton into a day-by-day plan **only when a
    feasible skeleton exists** (Requirement 3.4). When the solver rejected the
    plan or timed out — i.e. ``state.skeleton`` is unset or ``state.rejection``
    is set — the agent does not run and returns an empty update, leaving
    ``state.itinerary`` empty.

    Args:
        state: the shared graph state.
        settings: optional settings override (event-provider selection).
        start_date: optional first-day date used to fill in each day's date.
        llm: optional narration provider (defaults to the config-selected one).

    Returns:
        A partial-state update dict (the LangGraph node contract):
        ``{"itinerary": [...]}`` when a feasible skeleton was enriched, or an
        empty dict when the agent did not run.
    """
    if not itinerary_agent_should_run(state):
        # Solver produced a rejection/timeout (or no skeleton): do not run.
        return {}

    settings = settings or get_settings()
    assert state.skeleton is not None  # guaranteed by the gate above.
    itinerary = enrich_skeleton(
        state.skeleton,
        state.constraints,
        settings=settings,
        start_date=start_date,
    )
    return {"itinerary": itinerary}


__all__ = [
    "NARRATION_SYSTEM_PROMPT",
    "itinerary_agent_should_run",
    "enrich_skeleton",
    "narrate_day",
    "itinerary_node",
]
