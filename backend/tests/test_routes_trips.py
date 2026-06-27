"""Integration tests for the trip planning routes (Task 19.1).

Exercises the REST surface end-to-end through a FastAPI ``TestClient`` against an
in-memory SQLite database, on the default mock providers (no paid keys):

* ``POST /api/trips/plan`` creates a trip and returns ``{tripId, status}``; the
  background planning job then runs to completion (TestClient drives background
  tasks synchronously).
* ``GET /api/trips/{id}/results`` returns ranked, scored destinations (descending
  order, top-three present), each carrying its factor breakdown, plus the
  itinerary, flights/hotels (with rationales), events, and the chosen
  destination's decision-trace breakdown whose contributions sum to its score.
* ``GET /api/trips`` and ``GET /api/trips/{id}`` return the caller's trips.

Covers the Japan driving example (Requirement 18.1-18.3) and the ranked-output
contract (Requirements 7.1-7.3). Also verifies the DB-reconstruction fallback
when the in-process result cache is cold, and that response shapes match the
``mockApi.ts`` DTOs.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes_trips
from app.api.routes_trips import CurrentUser, DEMO_USER_ID, get_current_user, get_planning_session_factory
from app.main import create_app
from app.models.base import Base
from app.models.database import get_session

JAPAN_PROMPT = (
    "Plan a 7-day Japan trip in October under \u20b92 lakh. "
    "I like anime, photography, local food, and less crowded places."
)
JAPAN_INTERESTS = ["anime", "photography", "food", "less_crowded"]
JAPAN_BUDGET = 200_000.0


@pytest.fixture()
def db_factory() -> Iterator[sessionmaker]:
    """A shared in-memory SQLite engine with the full schema created.

    A ``StaticPool`` keeps a single connection so the request session and the
    background job's session see the same in-memory database.
    """
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    yield factory
    engine.dispose()


@pytest.fixture()
def client(db_factory: sessionmaker) -> Iterator[TestClient]:
    """A TestClient with the DB session + session factory pointed at SQLite."""
    app = create_app()

    def _override_session() -> Iterator[Session]:
        db = db_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_planning_session_factory] = lambda: db_factory
    # Task 19.4 enforces Supabase JWT auth on every route; inject a verified
    # caller here so the route behaviour is exercised without minting a JWT
    # (real JWT verification is covered in test_auth.py and the 401 test below).
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(user_id=DEMO_USER_ID)

    # Isolate the in-process result cache per test.
    routes_trips._PLAN_CACHE.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    routes_trips._PLAN_CACHE.clear()


def _plan_japan(client: TestClient) -> str:
    resp = client.post(
        "/api/trips/plan",
        json={"prompt": JAPAN_PROMPT, "interests": JAPAN_INTERESTS},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "processing"
    assert body["tripId"]
    return body["tripId"]


# --------------------------------------------------------------------------- #
# POST /api/trips/plan
# --------------------------------------------------------------------------- #
def test_plan_creates_trip_and_returns_tripid(client: TestClient) -> None:
    trip_id = _plan_japan(client)
    # The trip is immediately listable while planning runs/finishes.
    listing = client.get("/api/trips")
    assert listing.status_code == 200
    assert any(t["id"] == trip_id for t in listing.json())


# --------------------------------------------------------------------------- #
# GET /api/trips/{id}/results  (Requirements 7.1, 7.2, 7.3, 18.1-18.3)
# --------------------------------------------------------------------------- #
def test_results_are_ranked_scored_and_explained(client: TestClient) -> None:
    trip_id = _plan_japan(client)

    resp = client.get(f"/api/trips/{trip_id}/results")
    assert resp.status_code == 200, resp.text
    results = resp.json()

    destinations = results["destinations"]
    assert destinations, "expected ranked destinations"
    # Requirement 7.2: at least the top three options are present.
    assert len(destinations) >= 3
    # Requirement 7.1: ordered by final score descending.
    scores = [d["score"] for d in destinations]
    assert scores == sorted(scores, reverse=True)
    # Requirement 7.3 / 8.1: each option carries its per-feature factors.
    assert all(d["factors"] for d in destinations)
    # A/B/C grade labels for the top three.
    assert [d["rank"] for d in destinations[:3]] == ["A", "B", "C"]


def test_results_include_decision_trace_summing_to_score(client: TestClient) -> None:
    trip_id = _plan_japan(client)
    results = client.get(f"/api/trips/{trip_id}/results").json()

    trace = results["decisionTrace"]
    assert trace is not None
    # Requirement 18.3: contributions sum to the chosen destination's score.
    total = sum(f["contribution"] for f in trace["factors"])
    assert total == pytest.approx(trace["score"], abs=1e-2)
    # The trace belongs to the top-ranked (chosen) destination.
    assert trace["destination"] == results["destinations"][0]["name"]


def test_results_include_itinerary_flights_hotels_with_rationales(
    client: TestClient,
) -> None:
    trip_id = _plan_japan(client)
    results = client.get(f"/api/trips/{trip_id}/results").json()

    assert results["itinerary"], "expected a day-by-day itinerary"
    assert results["flights"] and all(f["rationale"] for f in results["flights"])
    assert results["hotels"] and all(h["rationale"] for h in results["hotels"])


def test_results_shape_matches_mockapi_dtos(client: TestClient) -> None:
    """Response keys align with mockApi.ts Result-derived DTOs."""
    trip_id = _plan_japan(client)
    results = client.get(f"/api/trips/{trip_id}/results").json()

    assert set(results) == {
        "destinations",
        "itinerary",
        "flights",
        "hotels",
        "events",
        "decisionTrace",
        "narration",
    }
    assert set(results["destinations"][0]) == {
        "name",
        "country",
        "flag",
        "score",
        "rank",
        "factors",
    }
    assert set(results["flights"][0]) == {"airline", "price", "duration", "rationale"}
    assert set(results["hotels"][0]) == {
        "name",
        "rating",
        "pricePerNight",
        "distanceKm",
        "rationale",
    }


def test_results_404_for_unknown_trip(client: TestClient) -> None:
    assert client.get("/api/trips/not-a-real-id/results").status_code == 404
    import uuid

    missing = uuid.uuid4()
    assert client.get(f"/api/trips/{missing}/results").status_code == 404


# --------------------------------------------------------------------------- #
# GET /api/trips and GET /api/trips/{id}  (budget / summary)
# --------------------------------------------------------------------------- #
def test_trip_summary_reports_budget_within_cap(client: TestClient) -> None:
    trip_id = _plan_japan(client)
    summary = client.get(f"/api/trips/{trip_id}").json()

    assert summary["id"] == trip_id
    assert summary["budgetTotal"] == pytest.approx(JAPAN_BUDGET)
    # Requirement 18.2: the plan's estimated cost stays within the ₹2L cap.
    assert summary["budgetUsed"] <= summary["budgetTotal"]
    assert summary["country"] == "Japan"
    assert summary["topSpot"]
    assert summary["durationDays"] == 7


def test_get_single_trip_404_for_unknown(client: TestClient) -> None:
    import uuid

    assert client.get(f"/api/trips/{uuid.uuid4()}").status_code == 404


# --------------------------------------------------------------------------- #
# DB-reconstruction fallback when the in-process cache is cold
# --------------------------------------------------------------------------- #
def test_results_reconstructed_from_db_when_cache_is_cold(client: TestClient) -> None:
    trip_id = _plan_japan(client)
    # Drop the cached snapshot so the endpoint must rebuild from persisted rows.
    routes_trips._PLAN_CACHE.clear()

    resp = client.get(f"/api/trips/{trip_id}/results")
    assert resp.status_code == 200, resp.text
    results = resp.json()

    # Destinations + factors + decision trace survive the DB round-trip.
    assert results["destinations"]
    scores = [d["score"] for d in results["destinations"]]
    assert scores == sorted(scores, reverse=True)
    assert results["decisionTrace"] is not None
    assert results["flights"] and results["hotels"]


def test_planning_persists_durable_rows(client: TestClient, db_factory: sessionmaker) -> None:
    """The background job persists decision traces and bookings to the DB."""
    import uuid

    from app.models.db import DecisionTrace, FlightOption, HotelOption

    trip_id = _plan_japan(client)
    key = uuid.UUID(trip_id)

    db = db_factory()
    try:
        traces = db.query(DecisionTrace).filter_by(trip_id=key).all()
        flights = db.query(FlightOption).filter_by(trip_id=key).all()
        hotels = db.query(HotelOption).filter_by(trip_id=key).all()
    finally:
        db.close()

    assert traces, "decision traces should be persisted"
    assert flights and all(f.rationale for f in flights)
    assert hotels and all(h.rationale for h in hotels)


# --------------------------------------------------------------------------- #
# Authentication enforcement (Task 19.4, Requirements 19.1, 19.2)
# --------------------------------------------------------------------------- #
def test_routes_require_authentication(db_factory: sessionmaker) -> None:
    """Without a verified caller, every trip route rejects the request (401).

    Builds the app with only the DB session overridden (no auth override), so
    ``require_user`` runs and rejects the unauthenticated request before the
    handler executes (Requirement 19.2).
    """
    import uuid

    app = create_app()

    def _override_session() -> Iterator[Session]:
        db = db_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_planning_session_factory] = lambda: db_factory

    with TestClient(app) as test_client:
        assert test_client.get("/api/trips").status_code == 401
        assert test_client.get(f"/api/trips/{uuid.uuid4()}").status_code == 401
        assert test_client.get(f"/api/trips/{uuid.uuid4()}/results").status_code == 401
        assert (
            test_client.post(
                "/api/trips/plan",
                json={"prompt": JAPAN_PROMPT, "interests": JAPAN_INTERESTS},
            ).status_code
            == 401
        )
    app.dependency_overrides.clear()
