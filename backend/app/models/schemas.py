"""Pydantic v2 schemas: domain models, scoring types, and API DTOs.

API DTOs are aligned with the frontend's `src/lib/mockApi.ts` types (Result,
Trip, Preference) plus a decision-trace breakdown, so the frontend can switch
from mock to real with minimal change.

Requirements: 20.2.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FEATURES = (
    "budget_fit",
    "weather_fit",
    "crowd_score",
    "food_score",
    "photography_score",
    "travel_efficiency",
)

WEIGHT_SUM_TOLERANCE = 0.001


# --------------------------------------------------------------------------- #
# Planning / scoring domain types
# --------------------------------------------------------------------------- #
class TripConstraints(BaseModel):
    budget: Decimal
    duration_days: int = Field(gt=0)
    max_cities: int = Field(gt=0)
    max_hotel_changes: int = Field(ge=0)
    max_same_day_travel: timedelta
    dietary: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    solver_timeout: timedelta = timedelta(seconds=10)


class FeatureWeights(BaseModel):
    budget_fit: float
    weather_fit: float
    crowd_score: float
    food_score: float
    photography_score: float
    travel_efficiency: float

    @field_validator(*FEATURES)
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("feature weight must be non-negative")
        return v

    @model_validator(mode="after")
    def _sums_to_one(self) -> "FeatureWeights":
        total = sum(getattr(self, f) for f in FEATURES)
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"weights must sum to 1.0 (got {total})")
        return self


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class DecisionTraceModel(BaseModel):
    destination: str
    score: float = Field(ge=0.0, le=100.0)
    factors: list[FeatureContribution]
    created_at: datetime

    @model_validator(mode="after")
    def _additive(self) -> "DecisionTraceModel":
        total = sum(f.contribution for f in self.factors)
        if abs(total - self.score) > WEIGHT_SUM_TOLERANCE:
            raise ValueError("contributions must sum to score within tolerance")
        return self


class PreferenceEntry(BaseModel):
    topic: str
    weight: float = Field(ge=0.0, le=1.0)
    source: Literal["explicit", "implicit"] = "explicit"
    updated_at: datetime | None = None


# --------------------------------------------------------------------------- #
# API DTOs (aligned with frontend mockApi.ts)
# --------------------------------------------------------------------------- #
class PlanRequest(BaseModel):
    prompt: str
    interests: list[str] = Field(default_factory=list)


class PlanResponse(BaseModel):
    tripId: str
    status: Literal["processing", "complete", "rejected"]


class FactorDTO(BaseModel):
    feature: str
    contribution: float


class DestinationDTO(BaseModel):
    name: str
    country: str
    flag: str = ""
    score: float
    rank: str
    factors: list[FactorDTO]


class FlightDTO(BaseModel):
    airline: str
    price: float
    duration: str
    rationale: str


class HotelDTO(BaseModel):
    name: str
    rating: float
    pricePerNight: float
    distanceKm: float
    rationale: str


class EventDTO(BaseModel):
    name: str
    date: str
    type: str


class ActivityDTO(BaseModel):
    name: str
    time: str
    cost: float


class ItineraryDayDTO(BaseModel):
    day: int
    date: str
    activities: list[ActivityDTO]


class DecisionTraceBreakdownDTO(BaseModel):
    """The chosen destination's explainability ledger ("Why X Won").

    ``factors`` are the additive per-feature contributions that sum to ``score``
    within tolerance (Requirements 8.2, 8.6). Surfaced alongside the ranked
    results so the frontend can render the decision-trace breakdown.
    """

    destination: str
    score: float
    factors: list[FactorDTO]


class TripResultsDTO(BaseModel):
    destinations: list[DestinationDTO]
    itinerary: list[ItineraryDayDTO]
    flights: list[FlightDTO]
    hotels: list[HotelDTO]
    events: list[EventDTO]
    # Decision-trace breakdown for the chosen (top-ranked) destination; null
    # when no valid trace is available (frontend shows a fallback message).
    decisionTrace: DecisionTraceBreakdownDTO | None = None
    # Human-readable ranking narration (deterministic facts plus LLM prose when
    # a vendor is configured). Explanatory only — never a source of scores or
    # order. Empty when reconstructed from the DB (it is not persisted).
    narration: str = ""


class TripSummaryDTO(BaseModel):
    id: str
    destination: str
    country: str
    image: str = ""
    startDate: str
    endDate: str
    durationDays: int
    budgetUsed: float
    budgetTotal: float
    topSpot: str


class PreferencesDTO(BaseModel):
    preferences: list[PreferenceEntry]
