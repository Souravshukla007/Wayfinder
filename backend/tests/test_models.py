"""Unit tests for ORM model persistence (task 2.4, Requirement 17).

Round-trips each blueprint record type through an in-memory SQLite engine and
asserts the values read back match what was written. In particular it asserts
that the ``rationale`` field is stored on both flight_options and hotel_options
(Requirement 17.4).
"""

from __future__ import annotations

import uuid
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.db import (
    Activity,
    AgentRun,
    DecisionTraceRecord,
    DislikedDestination,
    FlightOption,
    HotelOption,
    Itinerary,
    LikedDestination,
    Recommendation,
    Trip,
    TripFeedback,
    User,
    UserPreference,
)


@pytest.fixture()
def session() -> Session:
    """A SQLite in-memory session with the full schema created."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    db = factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


def _make_user(db: Session) -> User:
    user = User(email="traveler@example.com", prefs={"theme": "dark"})
    db.add(user)
    db.commit()
    return user


def _make_trip(db: Session, user: User) -> Trip:
    trip = Trip(
        user_id=user.id,
        origin="DEL",
        start_date=datetime(2024, 10, 18, tzinfo=timezone.utc),
        end_date=datetime(2024, 10, 25, tzinfo=timezone.utc),
        budget=Decimal("200000.00"),
        interests=["anime", "photography", "food"],
        status="processing",
    )
    db.add(trip)
    db.commit()
    return trip


def test_user_round_trip(session: Session) -> None:
    user = _make_user(session)
    fetched = session.get(User, user.id)
    assert fetched is not None
    assert fetched.email == "traveler@example.com"
    assert fetched.prefs == {"theme": "dark"}


def test_trip_round_trip(session: Session) -> None:
    user = _make_user(session)
    trip = _make_trip(session, user)
    fetched = session.get(Trip, trip.id)
    assert fetched is not None
    assert fetched.user_id == user.id
    assert fetched.budget == Decimal("200000.00")
    assert fetched.interests == ["anime", "photography", "food"]
    assert fetched.status == "processing"


def test_itinerary_and_activity_round_trip(session: Session) -> None:
    user = _make_user(session)
    trip = _make_trip(session, user)
    itinerary = Itinerary(trip_id=trip.id, day=1, summary="Arrive Kyoto")
    session.add(itinerary)
    session.commit()
    activity = Activity(
        itinerary_id=itinerary.id,
        name="Gion stroll",
        time=time(18, 30),
        cost=Decimal("0.00"),
        category="sightseeing",
    )
    session.add(activity)
    session.commit()

    fetched_itin = session.get(Itinerary, itinerary.id)
    fetched_act = session.get(Activity, activity.id)
    assert fetched_itin is not None and fetched_itin.day == 1
    assert fetched_act is not None
    assert fetched_act.itinerary_id == itinerary.id
    assert fetched_act.time == time(18, 30)
    assert fetched_act.category == "sightseeing"


def test_flight_option_stores_rationale(session: Session) -> None:
    """Requirement 17.4: flight options carry a rationale."""
    user = _make_user(session)
    trip = _make_trip(session, user)
    flight = FlightOption(
        trip_id=trip.id,
        carrier="ANA NH 829",
        price=Decimal("68400.00"),
        duration=timedelta(hours=11, minutes=25),
        rationale="Best balance of price and duration for the dates.",
    )
    session.add(flight)
    session.commit()

    fetched = session.get(FlightOption, flight.id)
    assert fetched is not None
    assert fetched.carrier == "ANA NH 829"
    assert fetched.price == Decimal("68400.00")
    assert fetched.duration == timedelta(hours=11, minutes=25)
    assert fetched.rationale == "Best balance of price and duration for the dates."


def test_hotel_option_stores_rationale(session: Session) -> None:
    """Requirement 17.4: hotel options carry a rationale."""
    user = _make_user(session)
    trip = _make_trip(session, user)
    hotel = HotelOption(
        trip_id=trip.id,
        name="Nazuna Kyoto Gosho",
        rating=4,
        price=Decimal("14500.00"),
        distance=Decimal("8.00"),
        rationale="Best value within walking distance of the center.",
    )
    session.add(hotel)
    session.commit()

    fetched = session.get(HotelOption, hotel.id)
    assert fetched is not None
    assert fetched.rating == 4
    assert fetched.rationale == "Best value within walking distance of the center."


def test_agent_run_round_trip(session: Session) -> None:
    user = _make_user(session)
    trip = _make_trip(session, user)
    run = AgentRun(
        trip_id=trip.id,
        agent="coordinator",
        input={"prompt": "Plan a 7-day Japan trip"},
        output={"candidates": 3},
        tokens=1234,
        latency=Decimal("2.5000"),
        trace_id="trace-abc",
    )
    session.add(run)
    session.commit()

    fetched = session.get(AgentRun, run.id)
    assert fetched is not None
    assert fetched.agent == "coordinator"
    assert fetched.input == {"prompt": "Plan a 7-day Japan trip"}
    assert fetched.output == {"candidates": 3}
    assert fetched.tokens == 1234
    assert fetched.trace_id == "trace-abc"


def test_recommendation_stores_scores_jsonb(session: Session) -> None:
    user = _make_user(session)
    trip = _make_trip(session, user)
    scores = {
        "final": 92.0,
        "components": {"budget_fit": 8.0, "photography_score": 18.0},
    }
    rec = Recommendation(
        trip_id=trip.id,
        decision="Kyoto + Hakone loop",
        rationale="Highest weighted score.",
        alternatives=[{"name": "Osaka", "score": 88.0}],
        scores=scores,
    )
    session.add(rec)
    session.commit()

    fetched = session.get(Recommendation, rec.id)
    assert fetched is not None
    assert fetched.scores == scores
    assert fetched.alternatives == [{"name": "Osaka", "score": 88.0}]


def test_decision_trace_stores_factors_json(session: Session) -> None:
    user = _make_user(session)
    trip = _make_trip(session, user)
    factors = [
        {"feature": "photography_score", "contribution": 18.0},
        {"feature": "budget_fit", "contribution": 8.0},
    ]
    trace = DecisionTraceRecord(
        trip_id=trip.id,
        destination="Kyoto + Hakone loop",
        score=Decimal("92.000"),
        factors_json={"factors": factors},
        created_at=datetime(2024, 10, 1, 12, 0, tzinfo=timezone.utc),
    )
    session.add(trace)
    session.commit()

    fetched = session.get(DecisionTraceRecord, trace.id)
    assert fetched is not None
    assert fetched.destination == "Kyoto + Hakone loop"
    assert fetched.score == Decimal("92.000")
    assert fetched.factors_json == {"factors": factors}


def test_user_preference_round_trip(session: Session) -> None:
    user = _make_user(session)
    pref = UserPreference(
        user_id=user.id,
        topic="photography",
        weight=Decimal("0.8200"),
        source="explicit",
        updated_at=datetime(2024, 9, 1, tzinfo=timezone.utc),
    )
    session.add(pref)
    session.commit()

    fetched = session.scalars(
        select(UserPreference).where(UserPreference.user_id == user.id)
    ).one()
    assert fetched.topic == "photography"
    assert fetched.source == "explicit"
    assert Decimal(str(fetched.weight)) == Decimal("0.8200")


def test_trip_feedback_round_trip(session: Session) -> None:
    user = _make_user(session)
    trip = _make_trip(session, user)
    fb = TripFeedback(
        trip_id=trip.id,
        user_id=user.id,
        rating=5,
        notes="Loved the quiet temples.",
        created_at=datetime(2024, 10, 26, tzinfo=timezone.utc),
    )
    session.add(fb)
    session.commit()

    fetched = session.get(TripFeedback, fb.id)
    assert fetched is not None
    assert fetched.rating == 5
    assert fetched.notes == "Loved the quiet temples."


def test_liked_and_disliked_destinations_round_trip(session: Session) -> None:
    user = _make_user(session)
    liked = LikedDestination(
        user_id=user.id,
        place="Kyoto",
        signal_strength=Decimal("0.9000"),
        source="explicit",
        updated_at=datetime(2024, 9, 1, tzinfo=timezone.utc),
    )
    disliked = DislikedDestination(
        user_id=user.id,
        place="Crowded Tourist Trap",
        signal_strength=Decimal("0.7000"),
        source="implicit",
        updated_at=datetime(2024, 9, 1, tzinfo=timezone.utc),
    )
    session.add_all([liked, disliked])
    session.commit()

    fetched_liked = session.get(LikedDestination, liked.id)
    fetched_disliked = session.get(DislikedDestination, disliked.id)
    assert fetched_liked is not None and fetched_liked.place == "Kyoto"
    assert fetched_disliked is not None
    assert fetched_disliked.place == "Crowded Tourist Trap"
    assert fetched_disliked.source == "implicit"


def test_all_tables_created() -> None:
    """The migration scaffold registers every blueprint table."""
    expected = {
        "users",
        "trips",
        "itineraries",
        "flight_options",
        "hotel_options",
        "activities",
        "agent_runs",
        "recommendations",
        "decision_traces",
        "user_preferences",
        "trip_feedback",
        "liked_destinations",
        "disliked_destinations",
    }
    assert expected.issubset(set(Base.metadata.tables.keys()))


def test_uuid_primary_keys_are_uuid(session: Session) -> None:
    user = _make_user(session)
    assert isinstance(user.id, uuid.UUID)
