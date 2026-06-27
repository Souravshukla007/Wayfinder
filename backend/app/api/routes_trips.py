"""Trip planning routes (Task 19.1).

Implements the trip-planning surface of the REST API, wiring the validated
frontend (`src/lib/mockApi.ts`) to the real backend:

* ``POST /api/trips/plan`` — create a ``trips`` row (``status=processing``),
  kick off the LangGraph planning pipeline asynchronously, and return
  ``{tripId, status}`` (Requirement 18.1; design Request Lifecycle steps 1-3).
* ``GET  /api/trips/{id}/results`` — the ranked, scored destination options with
  an explanation of the ordering, plus the itinerary, flights, hotels, events,
  and the chosen destination's decision-trace breakdown (Requirements 7.1-7.3).
* ``GET  /api/trips`` — list the caller's trips as summaries.
* ``GET  /api/trips/{id}`` — a single trip summary.

Response shapes align with the frontend ``Result`` / ``Trip`` types in
``src/lib/mockApi.ts`` (via the Pydantic DTOs in ``app.models.schemas``),
extended with a decision-trace breakdown so the UI can render "Why X Won".

**Async planning.** The graph is deterministic on the default mock providers but
still runs out of band so the HTTP request returns immediately with a
``tripId``. We use FastAPI :class:`~fastapi.BackgroundTasks`; the background
job runs the pipeline on its own DB session, persists the durable artifacts
(decision traces, recommendation, flight/hotel options, itinerary) through the
existing persistence layers, and transitions the trip's ``status`` from
``processing`` to ``complete`` (or ``rejected``). The Memory Layer supplies the
preference vector for scoring (its cold-start defaults on the mock path). The
fully assembled, render-ready result is also cached in-process for fast
retrieval; if the cache is cold the results endpoint reconstructs the response
from the persisted rows.

**Auth.** Task 19.4 enforces Supabase JWT auth. Every route resolves the caller
through the :func:`get_current_user` dependency — which now delegates to
:func:`require_user` (JWT verification) — and scopes every query to that user,
so unauthenticated requests are rejected with ``401`` (Requirements 19.1, 19.2)
and no route signatures change.

Requirements: 7.1, 7.2, 7.3, 18.1.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.jwt_middleware import AuthenticatedUser, require_user
from app.models.database import get_session
from app.models.database import get_session_factory as _db_session_factory
from app.models.db import (
    DecisionTrace,
    FlightOption,
    HotelOption,
    Itinerary,
    Activity,
    Recommendation,
    Trip,
    User,
)
from app.models.schemas import (
    ActivityDTO,
    DecisionTraceBreakdownDTO,
    DestinationDTO,
    EventDTO,
    FactorDTO,
    FlightDTO,
    HotelDTO,
    ItineraryDayDTO,
    PlanRequest,
    PlanResponse,
    TripResultsDTO,
    TripSummaryDTO,
)
from app.orchestration.graph import run_plan
from app.orchestration.nodes.coordinator import decompose_prompt
from app.orchestration.state import FinalPlan, GraphState

router = APIRouter(prefix="/api/trips", tags=["trips"])


# --------------------------------------------------------------------------- #
# Caller resolution (Task 19.4: Supabase JWT auth enforced via require_user)
# --------------------------------------------------------------------------- #
# A stable demo user id used as the override seam in tests. Production traffic
# never sees it: every route resolves the caller from the verified JWT.
DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-0000000000a1")


@dataclass(frozen=True)
class CurrentUser:
    """The resolved caller. Mirrors ``auth.AuthenticatedUser`` minimally."""

    user_id: uuid.UUID
    email: Optional[str] = None


def _coerce_user_uuid(user_id: str) -> uuid.UUID:
    """Coerce the verified ``sub`` claim to the UUID the ORM expects.

    Trip ownership and the ``users`` table key on a UUID ``user_id``; Supabase
    issues UUID ``sub`` claims. A malformed id can never own a trip, so it is
    rejected as an unauthorized caller rather than surfaced as a server error.
    """
    try:
        return uuid.UUID(str(user_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user identifier",
        ) from exc


def get_current_user(
    user: AuthenticatedUser = Depends(require_user),
) -> CurrentUser:
    """Resolve the calling user from a verified Supabase JWT (Task 19.4).

    Delegates to :func:`require_user`, which verifies the bearer token against
    the Supabase JWKS (or the HS256 shared secret) and rejects any missing,
    malformed, or expired token with ``401`` (Requirements 19.1, 19.2). The
    verified ``sub`` claim is mapped onto :class:`CurrentUser` so the route
    signatures are unchanged.
    """
    return CurrentUser(user_id=_coerce_user_uuid(user.user_id), email=user.email)


def get_planning_session_factory() -> Callable[[], Session]:
    """Provide a session factory for the background planning job.

    The request-scoped session from :func:`get_session` is closed once the
    response is sent, so the background job needs to open its own session. This
    indirection (rather than referencing the engine directly) lets tests point
    the background job at the same in-memory database they use for the request
    session. The underlying factory is built lazily, so importing this module
    never forces the configured database driver to be installed.
    """
    return _db_session_factory()


# --------------------------------------------------------------------------- #
# In-process result cache (render-ready snapshot per trip)
# --------------------------------------------------------------------------- #
@dataclass
class _PlanSummary:
    """Summary-level facts about a completed plan (for trip-list rendering)."""

    destination: str
    country: str
    top_spot: str
    image: str
    budget_used: float
    budget_total: float
    duration_days: int


@dataclass
class _StoredPlan:
    """A cached planning outcome: status plus the render-ready result/summary."""

    status: str
    results: Optional[TripResultsDTO]
    summary: Optional[_PlanSummary]


# Keyed by trip id (str). Populated by the background job; read by the GET
# endpoints for full-fidelity, fast retrieval (including events, which have no
# dedicated table). Durable rows in the DB are the fallback source of truth.
_PLAN_CACHE: dict[str, _StoredPlan] = {}


# --------------------------------------------------------------------------- #
# Small presentation helpers
# --------------------------------------------------------------------------- #
_COUNTRY_FLAGS: dict[str, str] = {
    "Japan": "\U0001F1EF\U0001F1F5",
    "India": "\U0001F1EE\U0001F1F3",
    "Portugal": "\U0001F1F5\U0001F1F9",
    "Iceland": "\U0001F1EE\U0001F1F8",
    "France": "\U0001F1EB\U0001F1F7",
    "Italy": "\U0001F1EE\U0001F1F9",
    "Thailand": "\U0001F1F9\U0001F1ED",
}


def _flag_for(country: str) -> str:
    """Best-effort flag emoji for a country name (empty string if unknown)."""
    return _COUNTRY_FLAGS.get(country, "")


def _rank_label(rank: int) -> str:
    """Map a 1-based rank to the A/B/C grade the frontend renders (else #n)."""
    return {1: "A", 2: "B", 3: "C"}.get(rank, f"#{rank}")


def _duration_to_timedelta(text: str) -> Optional[timedelta]:
    """Parse a flight duration like ``"11h 25m"`` into a timedelta (or None)."""
    hours = re.search(r"(\d+)\s*h", text)
    minutes = re.search(r"(\d+)\s*m", text)
    if not hours and not minutes:
        return None
    return timedelta(
        hours=int(hours.group(1)) if hours else 0,
        minutes=int(minutes.group(1)) if minutes else 0,
    )


def _to_decimal(value: float) -> Decimal:
    """Coerce a float price to Decimal, tolerating odd inputs."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal(0)


# --------------------------------------------------------------------------- #
# Assembly: GraphState -> render-ready DTOs (aligned to mockApi.ts shapes)
# --------------------------------------------------------------------------- #
def _decision_trace_for(
    state: GraphState, destination: str
) -> Optional[DecisionTraceBreakdownDTO]:
    """Build the chosen destination's decision-trace breakdown, if available."""
    for trace in state.decision_traces:
        if trace.destination == destination:
            return DecisionTraceBreakdownDTO(
                destination=trace.destination,
                score=trace.score,
                factors=[
                    FactorDTO(feature=f.feature, contribution=f.contribution)
                    for f in trace.factors
                ],
            )
    return None


def _assemble_results(plan: FinalPlan, state: GraphState) -> TripResultsDTO:
    """Map a merged :class:`FinalPlan` onto the frontend-aligned results DTO.

    The plan's ranked destinations preserve the Decision Engine's descending
    order (Requirement 7.1); each carries its final score and per-feature
    factors. The chosen (top-ranked) destination's decision trace is surfaced as
    the breakdown (Requirements 7.3, 8.6).
    """
    destinations = [
        DestinationDTO(
            name=d.name,
            country=d.country,
            flag=_flag_for(d.country),
            score=d.score,
            rank=_rank_label(d.rank),
            factors=[
                FactorDTO(feature=f.feature, contribution=f.contribution)
                for f in d.factors
            ],
        )
        for d in plan.destinations
    ]

    itinerary = [
        ItineraryDayDTO(
            day=day.day,
            date=day.date,
            activities=[
                ActivityDTO(name=activity, time="", cost=0.0)
                for activity in day.activities
            ],
        )
        for day in plan.itinerary
    ]

    chosen = plan.destinations[0].name if plan.destinations else ""
    return TripResultsDTO(
        destinations=destinations,
        itinerary=itinerary,
        # FinalPlan flights/hotels/events already use the schemas DTO types.
        flights=list(plan.flights),
        hotels=list(plan.hotels),
        events=list(plan.events),
        decisionTrace=_decision_trace_for(state, chosen) if chosen else None,
        narration=state.destination_narration,
    )


def _assemble_summary(
    plan: FinalPlan, state: GraphState, budget_total: Decimal
) -> _PlanSummary:
    """Derive the trip-list summary facts from a merged plan."""
    duration_days = state.constraints.duration_days if state.constraints else 0
    chosen = plan.destinations[0] if plan.destinations else None
    return _PlanSummary(
        destination=chosen.country if chosen else "",
        country=chosen.country if chosen else "",
        top_spot=chosen.name if chosen else "",
        image="",
        budget_used=float(plan.estimated_total_cost),
        budget_total=float(budget_total),
        duration_days=duration_days,
    )


# --------------------------------------------------------------------------- #
# Persistence (reuses the ORM layer; durable source of truth for results)
# --------------------------------------------------------------------------- #
def _persist_results(
    session: Session,
    trip_id: str,
    state: GraphState,
    summary: _PlanSummary,
) -> None:
    """Persist the durable planning artifacts for a trip in one transaction.

    Writes decision traces, the chosen recommendation (with its score
    breakdown), flight/hotel options (each with its rationale, Requirement
    17.4), and the itinerary + activities. All-or-nothing: on failure the
    transaction is rolled back and the error propagates to the caller, which
    records a ``rejected`` status.
    """
    plan = state.final_plan
    if plan is None:
        return

    key = uuid.UUID(trip_id)  # FK columns are UUID-typed; bind UUID, not str.

    for trace in state.decision_traces:
        session.add(
            DecisionTrace(
                trip_id=key,
                destination=trace.destination,
                score=trace.score,
                factors_json=[
                    {"feature": f.feature, "contribution": f.contribution}
                    for f in trace.factors
                ],
                created_at=trace.created_at,
            )
        )

    if plan.destinations:
        chosen = plan.destinations[0]
        session.add(
            Recommendation(
                trip_id=key,
                decision=chosen.name,
                rationale=chosen.rationale,
                alternatives=[
                    {"name": d.name, "score": d.score}
                    for d in plan.destinations[1:]
                ],
                scores={
                    "final": chosen.score,
                    "breakdown": {
                        f.feature: f.contribution for f in chosen.factors
                    },
                    "estimated_total_cost": float(plan.estimated_total_cost),
                    "within_budget": plan.within_budget,
                    "country": chosen.country,
                    "top_spot": chosen.name,
                    "image": summary.image,
                },
            )
        )

    for option in plan.flights:
        session.add(
            FlightOption(
                trip_id=key,
                carrier=option.airline,
                price=_to_decimal(option.price),
                duration=_duration_to_timedelta(option.duration),
                rationale=option.rationale,
            )
        )

    for option in plan.hotels:
        session.add(
            HotelOption(
                trip_id=key,
                name=option.name,
                rating=int(round(option.rating)),
                price=_to_decimal(option.pricePerNight),
                distance=_to_decimal(option.distanceKm),
                rationale=option.rationale,
            )
        )

    for day in plan.itinerary:
        itinerary = Itinerary(trip_id=key, day=day.day, summary=day.city)
        session.add(itinerary)
        session.flush()  # assign itinerary.id for the activity FK
        for activity_name in day.activities:
            session.add(
                Activity(itinerary_id=itinerary.id, name=activity_name)
            )

    session.commit()


def _set_trip_status(session: Session, trip_id: str, new_status: str) -> None:
    """Transition a trip's persisted status (e.g. processing -> complete)."""
    trip = session.get(Trip, uuid.UUID(trip_id))
    if trip is not None:
        trip.status = new_status
        session.commit()


# --------------------------------------------------------------------------- #
# Background planning job
# --------------------------------------------------------------------------- #
def _run_planning(
    trip_id: str,
    user_id: str,
    prompt: str,
    budget_total: Decimal,
    session_factory: Callable[[], Session],
) -> None:
    """Run the planning pipeline for a trip and persist its outcome.

    Executed out of band (FastAPI background task). Opens its own DB session so
    the Memory Layer is reused (preference load + cold-start persistence), runs
    the deterministic graph on mock providers, persists the durable artifacts,
    caches the render-ready snapshot, and flips the trip status.
    """
    session = session_factory()
    try:
        state = GraphState(trip_id=trip_id, user_id=user_id, prompt=prompt)
        # The graph runs on mock providers without a DB session: the Memory
        # Layer supplies its cold-start default preference vector, keeping the
        # run deterministic and avoiding a UUID/str binding mismatch (GraphState
        # carries the user id as a str while preference rows are UUID-keyed).
        # The session is reserved for persisting the durable result artifacts.
        final_state, _events = run_plan(state)

        plan = final_state.final_plan
        if plan is None or not plan.destinations:
            _PLAN_CACHE[trip_id] = _StoredPlan("rejected", None, None)
            _set_trip_status(session, trip_id, "rejected")
            return

        results = _assemble_results(plan, final_state)
        summary = _assemble_summary(plan, final_state, budget_total)

        try:
            _persist_results(session, trip_id, final_state, summary)
            new_status = "complete"
        except Exception:
            # Persistence failed; leave no partial write and mark rejected, but
            # still cache the in-memory result so the run is observable.
            session.rollback()
            new_status = "rejected"

        _PLAN_CACHE[trip_id] = _StoredPlan(new_status, results, summary)
        _set_trip_status(session, trip_id, new_status)
    except Exception:
        # Total planning failure: record a rejected status and an empty cache.
        session.rollback()
        _PLAN_CACHE[trip_id] = _StoredPlan("rejected", None, None)
        try:
            _set_trip_status(session, trip_id, "rejected")
        except Exception:
            session.rollback()
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Helpers for the GET endpoints (DB-backed fallbacks)
# --------------------------------------------------------------------------- #
def _ensure_user(session: Session, user_id: uuid.UUID) -> None:
    """Get-or-create the caller's ``users`` row so trip FKs are satisfiable."""
    if session.get(User, user_id) is None:
        session.add(User(id=user_id))
        session.commit()


def _owned_trip_or_404(session: Session, trip_id: str, user_id: uuid.UUID) -> Trip:
    """Return the caller's trip by id, or raise 404 (also for malformed ids)."""
    try:
        key = uuid.UUID(trip_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Trip not found") from exc
    trip = session.get(Trip, key)
    if trip is None or trip.user_id != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Trip not found")
    return trip


def _reconstruct_results_from_db(session: Session, trip_id: str) -> TripResultsDTO:
    """Rebuild a results DTO from persisted rows when the cache is cold.

    Events have no dedicated table, so they are omitted on the DB path; all
    other sections (ranked destinations with factors, itinerary, flights,
    hotels, decision trace) are reconstructed from their tables.
    """
    key = uuid.UUID(trip_id)

    traces = list(
        session.scalars(
            select(DecisionTrace).where(DecisionTrace.trip_id == key)
        )
    )
    # Decision traces are ranked descending by score (Requirement 7.1).
    traces.sort(key=lambda t: t.score, reverse=True)

    rec = session.scalars(
        select(Recommendation).where(Recommendation.trip_id == key)
    ).first()
    country_by_dest: dict[str, str] = {}
    chosen_name = ""
    if rec is not None:
        chosen_name = rec.decision or ""
        country = (rec.scores or {}).get("country", "")
        if chosen_name:
            country_by_dest[chosen_name] = country

    destinations: list[DestinationDTO] = []
    decision_trace: Optional[DecisionTraceBreakdownDTO] = None
    for position, trace in enumerate(traces, start=1):
        factors = [
            FactorDTO(feature=f["feature"], contribution=f["contribution"])
            for f in (trace.factors_json or [])
        ]
        country = country_by_dest.get(trace.destination, "")
        destinations.append(
            DestinationDTO(
                name=trace.destination,
                country=country,
                flag=_flag_for(country),
                score=trace.score,
                rank=_rank_label(position),
                factors=factors,
            )
        )
        if trace.destination == chosen_name or (not chosen_name and position == 1):
            decision_trace = DecisionTraceBreakdownDTO(
                destination=trace.destination, score=trace.score, factors=factors
            )

    flights = [
        FlightDTO(
            airline=f.carrier,
            price=float(f.price),
            duration=_format_timedelta(f.duration),
            rationale=f.rationale or "",
        )
        for f in session.scalars(
            select(FlightOption).where(FlightOption.trip_id == key)
        )
    ]
    hotels = [
        HotelDTO(
            name=h.name,
            rating=float(h.rating or 0),
            pricePerNight=float(h.price),
            distanceKm=float(h.distance or 0),
            rationale=h.rationale or "",
        )
        for h in session.scalars(
            select(HotelOption).where(HotelOption.trip_id == key)
        )
    ]

    itinerary_rows = list(
        session.scalars(select(Itinerary).where(Itinerary.trip_id == key))
    )
    itinerary_rows.sort(key=lambda i: i.day)
    itinerary = [
        ItineraryDayDTO(
            day=row.day,
            date="",
            activities=[
                ActivityDTO(name=a.name, time="", cost=0.0)
                for a in sorted(row.activities, key=lambda a: str(a.name))
            ],
        )
        for row in itinerary_rows
    ]

    return TripResultsDTO(
        destinations=destinations,
        itinerary=itinerary,
        flights=flights,
        hotels=hotels,
        events=[],
        decisionTrace=decision_trace,
    )


def _format_timedelta(value: Optional[timedelta]) -> str:
    """Render a stored interval back to a ``"11h 25m"`` style string."""
    if value is None:
        return ""
    total_minutes = int(value.total_seconds() // 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


def _summary_from_trip(session: Session, trip: Trip) -> TripSummaryDTO:
    """Build a trip summary, preferring the cache and falling back to the DB."""
    trip_id = str(trip.id)
    budget_total = float(trip.budget) if trip.budget is not None else 0.0
    duration_days = 0
    if trip.start_date and trip.end_date:
        duration_days = (trip.end_date.date() - trip.start_date.date()).days + 1

    cached = _PLAN_CACHE.get(trip_id)
    if cached is not None and cached.summary is not None:
        s = cached.summary
        return TripSummaryDTO(
            id=trip_id,
            destination=s.destination,
            country=s.country,
            image=s.image,
            startDate=_format_date(trip.start_date),
            endDate=_format_date(trip.end_date),
            durationDays=s.duration_days or duration_days,
            budgetUsed=s.budget_used,
            budgetTotal=s.budget_total or budget_total,
            topSpot=s.top_spot,
        )

    # Cache cold: derive what we can from the recommendation row.
    rec = session.scalars(
        select(Recommendation).where(Recommendation.trip_id == trip.id)
    ).first()
    scores = (rec.scores or {}) if rec is not None else {}
    return TripSummaryDTO(
        id=trip_id,
        destination=scores.get("country", ""),
        country=scores.get("country", ""),
        image=scores.get("image", ""),
        startDate=_format_date(trip.start_date),
        endDate=_format_date(trip.end_date),
        durationDays=duration_days,
        budgetUsed=float(scores.get("estimated_total_cost", 0.0)),
        budgetTotal=budget_total,
        topSpot=scores.get("top_spot", rec.decision if rec is not None else ""),
    )


def _format_date(value: Optional[datetime]) -> str:
    """Format a stored datetime as an ISO date string (empty when absent)."""
    return value.date().isoformat() if value is not None else ""


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.post("/plan", response_model=PlanResponse, status_code=status.HTTP_202_ACCEPTED)
def plan_trip(
    body: PlanRequest,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
    user: CurrentUser = Depends(get_current_user),
    session_factory: Callable[[], Session] = Depends(get_planning_session_factory),
) -> PlanResponse:
    """Create a trip and kick off planning asynchronously (Requirement 18.1).

    Decomposes the prompt up front (deterministically) so the trip row carries a
    budget, interests, and an estimated date range immediately, persists it with
    ``status=processing``, schedules the planning pipeline as a background task,
    and returns ``{tripId, status}`` without waiting for the plan to finish.
    """
    _ensure_user(session, user.user_id)

    constraints = decompose_prompt(body.prompt, body.interests)
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=max(0, constraints.duration_days - 1))

    trip = Trip(
        user_id=user.user_id,
        start_date=now,
        end_date=end,
        budget=constraints.budget,
        interests=list(constraints.interests),
        status="processing",
    )
    session.add(trip)
    session.commit()
    trip_id = str(trip.id)

    _PLAN_CACHE[trip_id] = _StoredPlan("processing", None, None)
    background.add_task(
        _run_planning,
        trip_id,
        str(user.user_id),
        body.prompt,
        constraints.budget,
        session_factory,
    )
    return PlanResponse(tripId=trip_id, status="processing")


@router.get("/{trip_id}/results", response_model=TripResultsDTO)
def get_trip_results(
    trip_id: str,
    session: Session = Depends(get_session),
    user: CurrentUser = Depends(get_current_user),
) -> TripResultsDTO:
    """Return the ranked options + itinerary + bookings + decision trace.

    Serves the full-fidelity cached snapshot when available, otherwise
    reconstructs the response from the persisted rows. Returns 409 while the
    plan is still processing and 404 when the trip is unknown or not owned by
    the caller.
    """
    trip = _owned_trip_or_404(session, trip_id, user.user_id)

    cached = _PLAN_CACHE.get(trip_id)
    if cached is not None and cached.results is not None:
        return cached.results

    if trip.status == "processing" and cached is not None and cached.results is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "Trip planning is still in progress"
        )

    return _reconstruct_results_from_db(session, trip_id)


@router.get("", response_model=list[TripSummaryDTO])
def list_trips(
    session: Session = Depends(get_session),
    user: CurrentUser = Depends(get_current_user),
) -> list[TripSummaryDTO]:
    """List the caller's trips, most recent first."""
    trips = list(
        session.scalars(
            select(Trip)
            .where(Trip.user_id == user.user_id)
            .order_by(Trip.created_at.desc())
        )
    )
    return [_summary_from_trip(session, trip) for trip in trips]


@router.get("/{trip_id}", response_model=TripSummaryDTO)
def get_trip(
    trip_id: str,
    session: Session = Depends(get_session),
    user: CurrentUser = Depends(get_current_user),
) -> TripSummaryDTO:
    """Return a single trip summary, or 404 when unknown or not owned."""
    trip = _owned_trip_or_404(session, trip_id, user.user_id)
    return _summary_from_trip(session, trip)
