"""Coordinator agent node (Task 13.2, Requirements 3.2, 1.3).

The Coordinator is the orchestrating reasoning agent of the LangGraph pipeline.
Per Requirement 3.2 it: decomposes the planning goal, manages shared state,
delegates to tools, and merges results into a final explained plan. It appears
at both ends of the graph:

    Coordinator(intake) -> tools -> Destination -> Decision Engine
        -> CP-SAT Solver -> Itinerary -> Coordinator(merge)

This module implements both phases as plain functions over the shared
``GraphState`` so the graph assembly (Task 13.5) can wire them as nodes:

* :func:`coordinator_intake_node` — decompose the user prompt into
  :class:`TripConstraints` and load the Memory Layer preference vector into the
  shared state. Deterministic, no LLM: constraint extraction is rule-based so a
  given prompt always yields the same constraints.
* :func:`coordinator_merge_node` — invoke the deterministic tools (flights,
  hotels, events, weather, budget) for the chosen destination and merge the
  ranked scored candidates + solver-fed itinerary into a single explained
  :class:`FinalPlan`.

**Decision transparency (Requirement 1.3).** Every *major* recommendation that
reaches the user must carry a rationale. The merge step gates each destination
recommendation and each flight/hotel option on a non-empty rationale: anything
lacking one is *withheld* (excluded from the ``FinalPlan``) and the
missing-rationale condition is *recorded* as a
:class:`~app.orchestration.state.WithheldRecommendation`, with
``GraphState.missing_rationale`` flipped to ``True``. Destination rationales are
derived deterministically from the Decision Engine's additive per-feature
contributions — the LLM only narrates scores, it never generates or reorders
them (Requirement 5.10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.config import Settings, get_settings
from app.memory import memory_layer
from app.models.schemas import (
    EventDTO,
    FeatureContribution,
    FlightDTO,
    HotelDTO,
    PreferenceEntry,
    TripConstraints,
)
from app.orchestration.state import (
    FinalPlan,
    GraphState,
    RecommendedDestination,
    ScoredCandidate,
    WithheldRecommendation,
)
from app.providers.base import (
    DateRange,
    EventOption,
    EventQuery,
    FlightOption,
    FlightQuery,
    GeoPoint,
    HotelOption,
    HotelQuery,
    WeatherForecast,
)
from app.tools import budget as budget_tool
from app.tools.events import search_events
from app.tools.flights import search_flights
from app.tools.hotels import search_hotels
from app.tools.weather import get_forecast

# --------------------------------------------------------------------------- #
# Prompt decomposition (deterministic, rule-based)
# --------------------------------------------------------------------------- #

# Fallback budget when a prompt states no figure (a sensible mid-range trip
# budget in INR). Kept explicit so decomposition is total and deterministic.
_DEFAULT_BUDGET = Decimal("150000")
_DEFAULT_DURATION_DAYS = 5
_DEFAULT_SAME_DAY_TRAVEL = timedelta(hours=6)

# Interest keyword -> normalized interest tag. Order is preserved on output so
# the same prompt always yields the same interest ordering.
_INTEREST_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("anime", "anime"),
    ("manga", "anime"),
    ("photograph", "photography"),
    ("photo", "photography"),
    ("local food", "food"),
    ("street food", "food"),
    ("cuisine", "food"),
    ("food", "food"),
    ("less crowd", "less_crowded"),
    ("uncrowded", "less_crowded"),
    ("offbeat", "less_crowded"),
    ("quiet", "less_crowded"),
    ("nature", "nature"),
    ("hiking", "nature"),
    ("beach", "beach"),
    ("history", "history"),
    ("historic", "history"),
    ("culture", "culture"),
    ("temple", "culture"),
    ("nightlife", "nightlife"),
    ("shopping", "shopping"),
    ("adventure", "adventure"),
)

# Dietary keyword -> normalized dietary tag.
_DIETARY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("vegetarian", "vegetarian"),
    ("vegan", "vegan"),
    ("halal", "halal"),
    ("jain", "jain"),
    ("gluten-free", "gluten_free"),
    ("gluten free", "gluten_free"),
    ("kosher", "kosher"),
)


def _parse_budget(prompt: str) -> Decimal | None:
    """Extract a budget figure from free text, or ``None`` if none is stated.

    Understands Indian shorthand (``lakh``/``L``, ``crore``/``cr``) and plain
    grouped figures (``₹2,00,000`` / ``₹200000``). Deterministic and total.
    """
    text = prompt.lower()
    crore = re.search(r"(\d+(?:\.\d+)?)\s*(?:crore|cr)\b", text)
    if crore:
        return Decimal(crore.group(1)) * Decimal(10_000_000)
    lakh = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?|l)\b", text)
    if lakh:
        return Decimal(lakh.group(1)) * Decimal(100_000)
    # Plain figure, optionally prefixed by a rupee sign and grouped with commas.
    figure = re.search(r"(?:₹|rs\.?|inr)?\s*(\d[\d,]{3,})", text)
    if figure:
        digits = figure.group(1).replace(",", "")
        if digits:
            return Decimal(digits)
    return None


def _parse_duration_days(prompt: str) -> int:
    """Extract a trip length in days, defaulting when none is stated."""
    text = prompt.lower()
    day = re.search(r"(\d+)\s*-?\s*days?\b", text)
    if day:
        return max(1, int(day.group(1)))
    night = re.search(r"(\d+)\s*nights?\b", text)
    if night:
        # N nights spans N+1 days of presence.
        return max(1, int(night.group(1)) + 1)
    week = re.search(r"(\d+)\s*-?\s*weeks?\b", text)
    if week:
        return max(1, int(week.group(1)) * 7)
    return _DEFAULT_DURATION_DAYS


def _ordered_unique(values: list[str]) -> list[str]:
    """Return ``values`` de-duplicated while preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _extract_tags(prompt: str, mapping: tuple[tuple[str, str], ...]) -> list[str]:
    """Map keyword hits in ``prompt`` to their normalized tags (ordered, unique)."""
    text = prompt.lower()
    hits = [tag for keyword, tag in mapping if keyword in text]
    return _ordered_unique(hits)


def decompose_prompt(
    prompt: str,
    interests: list[str] | None = None,
    *,
    settings: Settings | None = None,
) -> TripConstraints:
    """Decompose a free-text travel goal into structured :class:`TripConstraints`.

    Rule-based and deterministic (Requirement 3.2): the same prompt always
    produces the same constraints, with no LLM, randomness, or clock access.
    Explicitly supplied ``interests`` are merged ahead of any interests inferred
    from the prompt text.
    """
    settings = settings or get_settings()

    budget = _parse_budget(prompt) or _DEFAULT_BUDGET
    duration_days = _parse_duration_days(prompt)

    inferred_interests = _extract_tags(prompt, _INTEREST_KEYWORDS)
    interests = _ordered_unique([*(interests or []), *inferred_interests])

    dietary = _extract_tags(prompt, _DIETARY_KEYWORDS)

    # Derive trip-shape limits from the duration: roughly one city per ~3 days,
    # at least one, and one fewer hotel change than cities.
    max_cities = max(1, -(-duration_days // 3))  # ceil(duration / 3)
    max_hotel_changes = max(0, max_cities - 1)

    return TripConstraints(
        budget=budget,
        duration_days=duration_days,
        max_cities=max_cities,
        max_hotel_changes=max_hotel_changes,
        max_same_day_travel=_DEFAULT_SAME_DAY_TRAVEL,
        dietary=dietary,
        interests=interests,
        solver_timeout=timedelta(seconds=settings.solver_timeout_seconds),
    )


# --------------------------------------------------------------------------- #
# Memory load
# --------------------------------------------------------------------------- #
def load_preferences(
    state: GraphState,
    *,
    session=None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> list[PreferenceEntry]:
    """Load the user's preference vector for the planning session (Req 10.1/3.2).

    When a database ``session`` is supplied, the full Memory Layer
    ``load_context`` is used (decay + cold-start persistence). Without a session
    — e.g. lightweight graph runs in tests — the predefined cold-start defaults
    are returned so every required topic still has a starting weight.
    """
    settings = settings or get_settings()
    now = now or datetime.now(timezone.utc)
    if session is None:
        return memory_layer.default_preferences(settings, now=now)
    context = memory_layer.load_context(
        session, state.user_id, now=now, settings=settings
    )
    return context.preferences


# --------------------------------------------------------------------------- #
# Tool invocation
# --------------------------------------------------------------------------- #
@dataclass
class ToolBundle:
    """Deterministic-tool results gathered by the Coordinator for one destination."""

    destination: str
    flights: list[FlightOption] = field(default_factory=list)
    hotels: list[HotelOption] = field(default_factory=list)
    events: list[EventOption] = field(default_factory=list)
    weather: WeatherForecast | None = None


def gather_tool_bundle(
    destination: str,
    constraints: TripConstraints,
    *,
    origin: str = "Delhi",
    settings: Settings | None = None,
) -> ToolBundle:
    """Invoke the deterministic tools for ``destination`` (Requirement 3.5).

    Calls flights, hotels, events, and weather tools through the provider layer
    (mock by default). No LLM reasoning happens here — the Coordinator merely
    delegates to the tools and collects their results.
    """
    settings = settings or get_settings()

    flights = search_flights(
        FlightQuery(origin=origin or "Delhi", destination=destination),
        settings=settings,
    )
    hotels = search_hotels(HotelQuery(city=destination), settings=settings)
    events = search_events(EventQuery(city=destination), settings=settings)
    # Weather uses the GeoPoint name for mock lookups; coordinates are unused by
    # the mock provider, so a labelled origin point is sufficient.
    weather = get_forecast(
        GeoPoint(lat=0.0, lon=0.0, name=destination),
        DateRange(
            start=datetime.now(timezone.utc).date(),
            end=(datetime.now(timezone.utc) + constraints.max_same_day_travel).date(),
        ),
        settings=settings,
    )
    return ToolBundle(
        destination=destination,
        flights=flights,
        hotels=hotels,
        events=events,
        weather=weather,
    )


# --------------------------------------------------------------------------- #
# Money / number parsing helpers (mock fixtures carry display strings)
# --------------------------------------------------------------------------- #
def _parse_money(text: str) -> float:
    """Parse a display price like ``"₹14,500/n"`` into a float (``14500.0``)."""
    head = text.split("/")[0]
    digits = re.sub(r"[^\d.]", "", head)
    return float(digits) if digits else 0.0


def _parse_leading_number(text: str) -> float:
    """Parse the first number in a free-text distance like ``"12 min"``."""
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else 0.0


# --------------------------------------------------------------------------- #
# Rationale derivation + merge (Requirements 1.1, 1.3, 1.4, 17.4)
# --------------------------------------------------------------------------- #
def _destination_rationale(candidate: ScoredCandidate) -> str | None:
    """Build a destination rationale from its additive score contributions.

    Returns ``None`` when there is no basis for an explanation (no per-feature
    components), which signals the recommendation must be withheld (Req 1.3).
    The rationale is derived purely from Decision Engine output — the engine
    owns the numbers; this only narrates them (Requirement 5.10).
    """
    components = candidate.components or {}
    if not components:
        return None
    top = sorted(components.items(), key=lambda kv: kv[1], reverse=True)[:3]
    drivers = ", ".join(f"{name} (+{value:.1f})" for name, value in top)
    return (
        f"Ranked #{candidate.rank} with a score of {candidate.score:.1f}/100, "
        f"led by {drivers}."
    )


def _flight_rationale(option: FlightOption) -> str | None:
    """Derive a flight option's rationale from its provider tag (Req 1.4/17.4)."""
    tag = (option.tag or "").strip()
    if not tag:
        return None
    return f"{tag}: {option.airline} at {option.price} ({option.duration})."


def _hotel_rationale(option: HotelOption) -> str | None:
    """Derive a hotel option's rationale from its provider tag (Req 1.4/17.4)."""
    tag = (option.tag or "").strip()
    if not tag:
        return None
    return (
        f"{tag}: {option.name} ({option.rating}\u2605), {option.price}, "
        f"{option.distance}."
    )


def _factors(candidate: ScoredCandidate) -> list[FeatureContribution]:
    """Convert a scored candidate's component map into ledger contributions."""
    return [
        FeatureContribution(feature=name, contribution=value)
        for name, value in (candidate.components or {}).items()
    ]


def merge_final_plan(
    scored: list[ScoredCandidate],
    itinerary,
    bundle: ToolBundle | None,
    constraints: TripConstraints | None,
) -> tuple[FinalPlan, list[WithheldRecommendation]]:
    """Merge scored candidates + tools + itinerary into one explained plan.

    Returns the :class:`FinalPlan` (only rationale-backed recommendations) and
    the list of withheld recommendations (the recorded missing-rationale
    conditions, Requirement 1.3). Flight and hotel options each carry their own
    rationale (Requirement 1.4 / 17.4); any option lacking one is withheld.
    """
    withheld: list[WithheldRecommendation] = []

    # --- Destinations: gate each on a derivable rationale (Req 1.1, 1.3) ---
    destinations: list[RecommendedDestination] = []
    for candidate in scored:
        rationale = _destination_rationale(candidate)
        if rationale is None:
            withheld.append(
                WithheldRecommendation(
                    kind="destination",
                    identifier=candidate.name,
                    reason="no per-feature contributions to justify the ranking",
                )
            )
            continue
        destinations.append(
            RecommendedDestination(
                name=candidate.name,
                country=candidate.country,
                score=candidate.score,
                rank=candidate.rank,
                rationale=rationale,
                factors=_factors(candidate),
            )
        )

    # --- Flights / hotels / events for the chosen destination ---
    flights: list[FlightDTO] = []
    hotels: list[HotelDTO] = []
    events: list[EventDTO] = []
    if bundle is not None:
        for option in bundle.flights:
            rationale = _flight_rationale(option)
            if rationale is None:
                withheld.append(
                    WithheldRecommendation(
                        kind="flight",
                        identifier=option.airline,
                        reason="no rationale available for this flight option",
                    )
                )
                continue
            flights.append(
                FlightDTO(
                    airline=option.airline,
                    price=_parse_money(option.price),
                    duration=option.duration,
                    rationale=rationale,
                )
            )
        for option in bundle.hotels:
            rationale = _hotel_rationale(option)
            if rationale is None:
                withheld.append(
                    WithheldRecommendation(
                        kind="hotel",
                        identifier=option.name,
                        reason="no rationale available for this hotel option",
                    )
                )
                continue
            hotels.append(
                HotelDTO(
                    name=option.name,
                    rating=float(option.rating),
                    pricePerNight=_parse_money(option.price),
                    distanceKm=_parse_leading_number(option.distance),
                    rationale=rationale,
                )
            )
        events = [
            EventDTO(name=e.name, date=e.date, type=e.type) for e in bundle.events
        ]

    # --- Budget tool: estimate the chosen trip's total and budget fit ---
    estimated_total = Decimal(0)
    within = True
    if constraints is not None and (flights or hotels):
        cheapest_flight = min((f.price for f in flights), default=0.0)
        cheapest_hotel = min((h.pricePerNight for h in hotels), default=0.0)
        nights = max(1, constraints.duration_days)
        estimated_total = budget_tool.total_cost(
            [Decimal(str(cheapest_flight)), Decimal(str(cheapest_hotel)) * nights]
        )
        within = budget_tool.within_budget(constraints.budget, estimated_total)

    plan = FinalPlan(
        destinations=destinations,
        flights=flights,
        hotels=hotels,
        events=events,
        itinerary=list(itinerary or []),
        estimated_total_cost=estimated_total,
        within_budget=within,
    )
    return plan, withheld


# --------------------------------------------------------------------------- #
# Graph node entry points
# --------------------------------------------------------------------------- #
def coordinator_intake_node(
    state: GraphState,
    *,
    session=None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> GraphState:
    """Intake phase: decompose the goal and load memory into shared state.

    Sets ``state.constraints`` (from the prompt) and ``state.preference_vector``
    (from the Memory Layer). Runs first in the graph, before the deterministic
    tools and the Destination Agent (Requirement 3.2).
    """
    settings = settings or get_settings()
    state.constraints = decompose_prompt(state.prompt, settings=settings)
    state.preference_vector = load_preferences(
        state, session=session, settings=settings, now=now
    )
    return state


def coordinator_merge_node(
    state: GraphState,
    *,
    settings: Settings | None = None,
) -> GraphState:
    """Merge phase: invoke tools for the chosen destination and assemble the plan.

    Picks the top-ranked scored candidate as the chosen destination, gathers its
    tool bundle, and merges everything into ``state.final_plan``. Any major
    recommendation lacking a rationale is withheld and recorded in
    ``state.withheld_recommendations`` with ``state.missing_rationale`` set
    (Requirement 1.3).
    """
    settings = settings or get_settings()

    chosen = state.scored[0].name if state.scored else ""
    bundle = (
        gather_tool_bundle(chosen, state.constraints, settings=settings)
        if chosen and state.constraints is not None
        else None
    )

    plan, withheld = merge_final_plan(
        state.scored, state.itinerary, bundle, state.constraints
    )
    state.final_plan = plan
    state.withheld_recommendations = withheld
    state.missing_rationale = bool(withheld)
    return state


# The Coordinator's primary graph entry point is its intake phase; the merge
# phase is wired as the graph's terminal node (Task 13.5).
coordinator_node = coordinator_intake_node


__all__ = [
    "decompose_prompt",
    "load_preferences",
    "ToolBundle",
    "gather_tool_bundle",
    "merge_final_plan",
    "coordinator_intake_node",
    "coordinator_merge_node",
    "coordinator_node",
]
