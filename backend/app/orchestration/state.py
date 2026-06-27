"""Shared LangGraph graph state for the planning pipeline.

This module defines the single shared-state contract that every agent node
in the LangGraph pipeline reads from and writes to:

    Coordinator -> Destination Agent -> Decision Engine -> Constraint Solver
    -> Itinerary Agent -> Coordinator (merge)

`GraphState` is intentionally a *state definition only*. The agent nodes
(tasks 13.2-13.4) and graph assembly (task 13.5) are implemented later; this
file just establishes the typed fields those stages will populate.

Established domain types are reused from ``app.models.schemas``
(``TripConstraints``, ``PreferenceEntry``, and ``DecisionTrace``). Stage
outputs that do not yet have a dedicated type elsewhere in the codebase
(candidates, scored candidates, the solver skeleton/rejection, and the final
itinerary) are defined here as precise structures so downstream stages share a
single contract. Fields a stage has not produced yet are ``Optional`` or
default to an empty collection.

Requirements: 3.1.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.schemas import (
    DecisionTraceModel as DecisionTrace,
    EventDTO,
    FeatureContribution,
    FlightDTO,
    HotelDTO,
    PreferenceEntry,
    TripConstraints,
)

__all__ = [
    "Candidate",
    "ScoredCandidate",
    "ItineraryLeg",
    "ItinerarySkeleton",
    "SolverRejection",
    "DayPlan",
    "RecommendedDestination",
    "WithheldRecommendation",
    "FinalPlan",
    "GraphState",
    "DecisionTrace",
]


# --------------------------------------------------------------------------- #
# Stage output structures (not yet typed elsewhere)
# --------------------------------------------------------------------------- #
class Candidate(BaseModel):
    """A candidate destination proposed by the Destination Agent.

    Carries the raw per-feature signals consumed by the deterministic Decision
    Engine. Feature values are kept in a mapping (rather than fixed fields) so
    the engine's invalid-candidate handling — missing or non-numeric features —
    operates on the raw input as designed.
    """

    name: str
    country: str = ""
    features: dict[str, float] = Field(default_factory=dict)


class ScoredCandidate(BaseModel):
    """A candidate after the Decision Engine has scored and ranked it.

    The deterministic engine produces the final 0-100 score, the additive
    per-feature component breakdown, and a descending rank position.
    """

    name: str
    country: str = ""
    score: float = Field(ge=0.0, le=100.0)
    components: dict[str, float] = Field(default_factory=dict)
    rank: int = Field(ge=0)


class ItineraryLeg(BaseModel):
    """A single city stay within a feasible solver skeleton."""

    city: str
    arrival_day: int = Field(ge=0)
    departure_day: int = Field(ge=0)
    hotel_change: bool = False


class ItinerarySkeleton(BaseModel):
    """A provably feasible itinerary skeleton emitted by the CP-SAT solver.

    Present only when the solver finds a plan satisfying every hard constraint.
    Enriched into a day-by-day plan by the Itinerary Agent.
    """

    legs: list[ItineraryLeg] = Field(default_factory=list)
    total_cost: Decimal = Decimal(0)
    hotel_changes: int = Field(default=0, ge=0)


class SolverRejection(BaseModel):
    """A solver outcome indicating no feasible plan was produced.

    Set when hard constraints are infeasible or the solver times out. When
    present, the Itinerary Agent must not run.
    """

    unsatisfied_constraints: list[str] = Field(default_factory=list)
    timed_out: bool = False
    message: str = ""


class DayPlan(BaseModel):
    """One enriched day in the final itinerary produced by the Itinerary Agent."""

    day: int = Field(ge=1)
    date: str = ""
    city: str = ""
    activities: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Coordinator merge outputs
# --------------------------------------------------------------------------- #
class RecommendedDestination(BaseModel):
    """A ranked destination recommendation carrying its required rationale.

    Produced by the Coordinator's merge step from the Decision Engine's scored
    candidates. Every recommendation that reaches the user MUST carry a
    non-empty ``rationale`` (Requirements 1.1, 1.3); candidates that cannot be
    given one are withheld (recorded as a :class:`WithheldRecommendation`)
    instead of appearing here. ``factors`` echoes the additive per-feature
    contributions behind the score so the explanation is auditable.
    """

    name: str
    country: str = ""
    score: float = Field(ge=0.0, le=100.0)
    rank: int = Field(ge=0)
    rationale: str = Field(min_length=1)
    factors: list[FeatureContribution] = Field(default_factory=list)


class WithheldRecommendation(BaseModel):
    """A major recommendation withheld because it lacked a rationale (Req 1.3).

    Recording the withheld item — rather than silently dropping it — is the
    "record the missing-rationale condition" half of Requirement 1.3, so the
    condition is observable to the API, the evaluation harness, and tracing.
    """

    kind: Literal["destination", "flight", "hotel"]
    identifier: str
    reason: str = "missing rationale"


class FinalPlan(BaseModel):
    """The merged, fully explained plan the Coordinator hands back.

    Only recommendations that carry a rationale are present here; anything
    withheld lives in ``GraphState.withheld_recommendations``. Flight and hotel
    options each carry a ``rationale`` (Requirement 1.4 / 17.4).
    """

    destinations: list[RecommendedDestination] = Field(default_factory=list)
    flights: list[FlightDTO] = Field(default_factory=list)
    hotels: list[HotelDTO] = Field(default_factory=list)
    events: list[EventDTO] = Field(default_factory=list)
    itinerary: list[DayPlan] = Field(default_factory=list)
    estimated_total_cost: Decimal = Decimal(0)
    within_budget: bool = True


# --------------------------------------------------------------------------- #
# Shared graph state
# --------------------------------------------------------------------------- #
class GraphState(BaseModel):
    """Shared state passed between all nodes of the LangGraph pipeline.

    Fields are populated progressively as the pipeline advances. Inputs
    (``trip_id``, ``user_id``, ``prompt``) are set at kickoff; ``constraints``
    and ``preference_vector`` are filled by the Coordinator; ``candidates`` by
    the Destination Agent; ``scored`` and ``decision_traces`` by the Decision
    Engine; ``skeleton`` *or* ``rejection`` by the Constraint Solver; and
    ``itinerary`` by the Itinerary Agent. Stages that have not run yet leave
    their outputs at the ``None`` / empty-collection defaults.
    """

    # Inputs (set at kickoff)
    trip_id: str
    user_id: str
    prompt: str

    # Coordinator outputs
    constraints: Optional[TripConstraints] = None
    preference_vector: list[PreferenceEntry] = Field(default_factory=list)

    # Destination Agent output
    candidates: list[Candidate] = Field(default_factory=list)

    # Decision Engine outputs
    scored: list[ScoredCandidate] = Field(default_factory=list)
    decision_traces: list[DecisionTrace] = Field(default_factory=list)
    # Human-readable narration of the ranking, built from the engine's scores
    # only (LLM prose appended when a vendor is configured). Never a source of
    # scores or order — purely explanatory (Requirements 3.3, 5.10, 19.4).
    destination_narration: str = ""

    # Constraint Solver outputs (mutually exclusive)
    skeleton: Optional[ItinerarySkeleton] = None
    rejection: Optional[SolverRejection] = None

    # Itinerary Agent output
    itinerary: list[DayPlan] = Field(default_factory=list)

    # Coordinator merge outputs
    final_plan: Optional[FinalPlan] = None
    withheld_recommendations: list[WithheldRecommendation] = Field(
        default_factory=list
    )
    missing_rationale: bool = False
