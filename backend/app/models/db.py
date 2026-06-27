"""SQLAlchemy ORM models for the Wayfinder data model (blueprint schema).

Covers: users, trips, itineraries, flight_options, hotel_options, activities,
agent_runs, recommendations, decision_traces, user_preferences, trip_feedback,
liked_destinations, disliked_destinations.

Per the design data model (blueprint §8): primary keys are UUIDs, flight
``duration`` is an interval, hotel ``distance`` is numeric, and
``decision_traces.factors_json`` stores the additive per-feature contributions
that sum to ``score``.

All models are declared against the single shared :class:`Base` from
:mod:`app.models.base` so ``Base.metadata`` reaches every table exactly once
(no duplicate/redefined tables). ``Base`` is re-exported here for the
session/engine layer and tests that import it from this module.

Requirements: 17.1, 17.2, 17.3, 17.4.
"""

from __future__ import annotations

import uuid
from datetime import datetime, time, timedelta
from decimal import Decimal

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Interval,
    Numeric,
    String,
    Text,
    Time,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

__all__ = [
    "Base",
    "User",
    "Trip",
    "Itinerary",
    "Activity",
    "FlightOption",
    "HotelOption",
    "AgentRun",
    "Recommendation",
    "DecisionTrace",
    "DecisionTraceRecord",
    "UserPreference",
    "TripFeedback",
    "LikedDestination",
    "DislikedDestination",
]


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str | None] = mapped_column(String(255), index=True)
    prefs: Mapped[dict] = mapped_column(JSON, default=dict)

    trips: Mapped[list["Trip"]] = relationship(back_populates="user")


class Trip(Base):
    __tablename__ = "trips"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    origin: Mapped[str | None] = mapped_column(String(120))
    start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    budget: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    interests: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="processing")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="trips")
    itineraries: Mapped[list["Itinerary"]] = relationship(back_populates="trip")
    flight_options: Mapped[list["FlightOption"]] = relationship(back_populates="trip")
    hotel_options: Mapped[list["HotelOption"]] = relationship(back_populates="trip")


class Itinerary(Base):
    __tablename__ = "itineraries"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    trip_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trips.id"), index=True)
    day: Mapped[int] = mapped_column(Integer)
    summary: Mapped[str | None] = mapped_column(Text)

    trip: Mapped[Trip] = relationship(back_populates="itineraries")
    activities: Mapped[list["Activity"]] = relationship(back_populates="itinerary")


class Activity(Base):
    __tablename__ = "activities"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    itinerary_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("itineraries.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    time: Mapped[time | None] = mapped_column(Time)
    cost: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    category: Mapped[str | None] = mapped_column(String(60))

    itinerary: Mapped[Itinerary] = relationship(back_populates="activities")


class FlightOption(Base):
    __tablename__ = "flight_options"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    trip_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trips.id"), index=True)
    carrier: Mapped[str] = mapped_column(String(120))
    price: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    duration: Mapped[timedelta | None] = mapped_column(Interval)  # interval (design §8)
    rationale: Mapped[str | None] = mapped_column(Text)  # Requirement 17.4

    trip: Mapped[Trip] = relationship(back_populates="flight_options")


class HotelOption(Base):
    __tablename__ = "hotel_options"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    trip_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trips.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    rating: Mapped[int | None] = mapped_column(Integer)
    price: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    distance: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))  # numeric (design §8)
    rationale: Mapped[str | None] = mapped_column(Text)  # Requirement 17.4

    trip: Mapped[Trip] = relationship(back_populates="hotel_options")


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    trip_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trips.id"), index=True)
    agent: Mapped[str] = mapped_column(String(80))
    input: Mapped[dict] = mapped_column(JSON, default=dict)
    output: Mapped[dict] = mapped_column(JSON, default=dict)
    tokens: Mapped[int | None] = mapped_column(Integer)
    latency: Mapped[float | None] = mapped_column(Float)
    trace_id: Mapped[str | None] = mapped_column(String(80), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    trip_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trips.id"), index=True)
    decision: Mapped[str | None] = mapped_column(String(255))
    rationale: Mapped[str | None] = mapped_column(Text)
    alternatives: Mapped[list] = mapped_column(JSON, default=list)
    scores: Mapped[dict] = mapped_column(JSON, default=dict)  # final score + breakdown


class DecisionTrace(Base):
    __tablename__ = "decision_traces"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    trip_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trips.id"), index=True)
    destination: Mapped[str] = mapped_column(String(120))
    score: Mapped[float] = mapped_column(Float)
    factors_json: Mapped[list] = mapped_column(JSON, default=list)  # additive contributions
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# Backwards/forwards-compatible alias. The design and persistence layer refer to
# this model as ``DecisionTrace``; the model-persistence tests (task 2.4) import
# it as ``DecisionTraceRecord``. Both names point at the single ORM class so the
# ``decision_traces`` table is defined exactly once.
DecisionTraceRecord = DecisionTrace


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    topic: Mapped[str] = mapped_column(String(80))
    weight: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(20), default="explicit")  # explicit | implicit
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class TripFeedback(Base):
    __tablename__ = "trip_feedback"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    trip_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("trips.id"), index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    rating: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class LikedDestination(Base):
    __tablename__ = "liked_destinations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    place: Mapped[str] = mapped_column(String(120))
    signal_strength: Mapped[float] = mapped_column(Float, default=1.0)
    source: Mapped[str] = mapped_column(String(20), default="explicit")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DislikedDestination(Base):
    __tablename__ = "disliked_destinations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    place: Mapped[str] = mapped_column(String(120))
    signal_strength: Mapped[float] = mapped_column(Float, default=1.0)
    source: Mapped[str] = mapped_column(String(20), default="explicit")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
