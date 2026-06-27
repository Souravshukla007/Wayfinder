"""Integration test for endpoint/response shape alignment (Task 19.7).

Asserts that the REST API's JSON responses line up with the frontend's
``src/lib/mockApi.ts`` ``Result`` / ``Trip`` / ``Preference`` types (plus the
decision-trace breakdown that extends ``Result``), so the prototype can swap its
mock function bodies for real ``fetch()`` calls without changing the component
layer (Requirement 18.1; design "Frontend swap" and the API DTO alignment in
``app.models.schemas``).

Rather than hard-coding the expected frontend field names, this test *parses the
actual TypeScript type declarations out of ``mockApi.ts``* and checks the live
REST responses against them through an explicit, documented field-mapping table.
That makes the test a genuine two-sided drift detector:

* If the **frontend** type gains/renames a data field that the backend feeds,
  the mapping table no longer covers the parsed TS fields and the test fails.
* If the **backend** response drops/renames a mapped field, the response no
  longer contains the mapped key and the test fails.

The whole suite runs end-to-end through a FastAPI ``TestClient`` against an
in-memory SQLite database on the default mock providers (no paid keys), driving
the Japan example from Requirement 18.

Requirements: 18.1.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes_trips
from app.api.routes_preferences import current_user_id
from app.api.routes_trips import (
    CurrentUser,
    DEMO_USER_ID,
    get_current_user,
    get_planning_session_factory,
)
from app.main import create_app
from app.models.base import Base
from app.models.database import get_session
from app.models.db import User, UserPreference

JAPAN_PROMPT = (
    "Plan a 7-day Japan trip in October under \u20b92 lakh. "
    "I like anime, photography, local food, and less crowded places."
)
JAPAN_INTERESTS = ["anime", "photography", "food", "less_crowded"]

# mockApi.ts lives in the frontend source tree, a sibling of backend/.
MOCK_API_TS = Path(__file__).resolve().parents[2] / "src" / "lib" / "mockApi.ts"


# --------------------------------------------------------------------------- #
# Parse the real frontend TypeScript types out of mockApi.ts
# --------------------------------------------------------------------------- #
def _parse_ts_type_fields(source: str, type_name: str) -> set[str]:
    """Extract the field names declared on ``export type <type_name> = { ... }``.

    Handles optional fields (``badge?:``) and both single-line
    (``FlightOpt = { ... };``) and multi-line (``Result = { ... };``)
    declarations. The frontend types are flat (they reference other named types
    rather than nesting object literals), so the first closing brace delimits
    the body.
    """
    match = re.search(
        rf"export type {re.escape(type_name)}\s*=\s*\{{(.*?)\}}",
        source,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"could not locate `export type {type_name}` in mockApi.ts")
    body = match.group(1)
    # Field declarations are `;`-separated (works for single- and multi-line
    # bodies). Take the identifier before the first `:` of each segment, e.g.
    # `name: Type` -> name, `badge?: string` -> badge.
    fields: set[str] = set()
    for segment in body.split(";"):
        decl = re.match(r"\s*(\w+)\??\s*:", segment)
        if decl is not None:
            fields.add(decl.group(1))
    return fields


@pytest.fixture(scope="module")
def ts_types() -> dict[str, set[str]]:
    """The parsed frontend field sets for the shapes under test."""
    assert MOCK_API_TS.exists(), f"mockApi.ts not found at {MOCK_API_TS}"
    source = MOCK_API_TS.read_text(encoding="utf-8")
    types = {
        name: _parse_ts_type_fields(source, name)
        for name in (
            "Result",
            "Trip",
            "Preference",
            "FlightOpt",
            "HotelOpt",
            "EventOpt",
            "DayPlan",
            "BreakdownRow",
        )
    }
    # Guard: parsing actually found fields (catches a future refactor of the file).
    for name, fields in types.items():
        assert fields, f"parsed no fields for frontend type {name}"
    return types


# --------------------------------------------------------------------------- #
# Field-mapping tables: frontend field -> backend response field (or None when
# the frontend field is UI-only / presentation-derived and not backend-fed).
# Documenting the UI-only fields explicitly keeps the test honest: every parsed
# frontend field must be accounted for, either mapped to a backend field or
# deliberately marked UI-only.
# --------------------------------------------------------------------------- #

# Result is split across the backend results DTO: per-destination ranking fields
# come from `destinations[]`, while the booking/itinerary collections are shared
# at the top level.
RESULT_DEST_MAP = {
    "name": "name",
    "country": "country",
    "flag": "flag",
    "score": "score",
    "grade": "rank",  # A/B/C grade <- backend rank label
    "breakdown": "factors",  # BreakdownRow[] <- additive FactorDTO[]
}
RESULT_TOPLEVEL_MAP = {
    "itinerary": "itinerary",
    "flights": "flights",
    "hotels": "hotels",
    "events": "events",
    "decisionTrace": "decisionTrace",  # "Why X Won" breakdown <- decision trace
}
# Presentation-only fields the backend does not supply (the frontend derives or
# styles these locally); not part of the data contract.
RESULT_UI_ONLY = {"image", "micro", "teaser", "badge"}

FLIGHT_MAP = {
    "airline": "airline",
    "price": "price",
    "duration": "duration",
    "tag": "rationale",  # short "why" label <- rationale
}
HOTEL_MAP = {
    "name": "name",
    "rating": "rating",
    "price": "pricePerNight",
    "distance": "distanceKm",
    "tag": "rationale",
}
EVENT_MAP = {"name": "name", "date": "date", "type": "type"}
DAYPLAN_MAP = {
    "day": "day",
    "items": "activities",  # string[] <- ActivityDTO[]
    "title": None,  # UI-only heading; backend itinerary day has no title field
}
# Trip aligns 1:1 with TripSummaryDTO.
TRIP_MAP = {
    "id": "id",
    "destination": "destination",
    "country": "country",
    "image": "image",
    "startDate": "startDate",
    "endDate": "endDate",
    "durationDays": "durationDays",
    "budgetUsed": "budgetUsed",
    "budgetTotal": "budgetTotal",
    "topSpot": "topSpot",
}


def _assert_mapping_covers(frontend_fields: set[str], mapping: dict, ui_only: set[str]) -> None:
    """Every parsed frontend field is either mapped or explicitly UI-only."""
    accounted = set(mapping) | ui_only
    missing = frontend_fields - accounted
    assert not missing, (
        f"frontend fields not accounted for by the mapping (frontend drift): {missing}"
    )


def _assert_backend_has(obj: dict, mapping: dict) -> None:
    """Every mapped (non-None) backend field is present on the response object."""
    for fe_field, be_field in mapping.items():
        if be_field is None:
            continue
        assert be_field in obj, (
            f"backend response missing field {be_field!r} "
            f"(maps from frontend {fe_field!r}); response keys={sorted(obj)}"
        )


# --------------------------------------------------------------------------- #
# App / DB fixtures (full app: trips + preferences routers, mock providers)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def db_factory() -> Iterator[sessionmaker]:
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
    """TestClient with DB + planning factory + auth overridden to the demo user.

    Seeds a stored preference for the demo user so ``GET /api/preferences``
    returns a non-empty vector whose entries can be shape-checked.
    """
    app = create_app()

    def _override_session() -> Iterator[Session]:
        db = db_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_planning_session_factory] = lambda: db_factory
    app.dependency_overrides[get_current_user] = lambda: CurrentUser(user_id=DEMO_USER_ID)
    # The preferences router resolves the caller via its own dependency; point it
    # at the same demo user so the preference seed is visible to GET.
    app.dependency_overrides[current_user_id] = lambda: str(DEMO_USER_ID)

    # Seed the demo user + one stored preference for the preferences endpoint.
    seed = db_factory()
    try:
        seed.add(User(id=DEMO_USER_ID, email="demo@example.com", prefs={}))
        seed.add(
            UserPreference(
                user_id=DEMO_USER_ID,
                topic="photography_score",
                weight=0.82,
                source="explicit",
                updated_at=datetime(2024, 10, 1, tzinfo=timezone.utc),
            )
        )
        seed.commit()
    finally:
        seed.close()

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
    return resp.json()["tripId"]


# --------------------------------------------------------------------------- #
# Result alignment: GET /api/trips/{id}/results
# --------------------------------------------------------------------------- #
def test_results_align_with_frontend_result_type(
    client: TestClient, ts_types: dict[str, set[str]]
) -> None:
    """The results payload covers every data field of the frontend `Result`."""
    result_fields = ts_types["Result"]
    # Every parsed Result field is either mapped (to destinations[] or top-level)
    # or explicitly presentation-only.
    combined_map = {**RESULT_DEST_MAP, **RESULT_TOPLEVEL_MAP}
    _assert_mapping_covers(result_fields, combined_map, RESULT_UI_ONLY)

    trip_id = _plan_japan(client)
    resp = client.get(f"/api/trips/{trip_id}/results")
    assert resp.status_code == 200, resp.text
    results = resp.json()

    # Top-level collections (itinerary/flights/hotels/events) are present.
    _assert_backend_has(results, RESULT_TOPLEVEL_MAP)

    # Per-destination ranking fields are present on each destination entry.
    assert results["destinations"], "expected ranked destinations"
    for dest in results["destinations"]:
        _assert_backend_has(dest, RESULT_DEST_MAP)


def test_flight_hotel_event_entries_align_with_frontend_opts(
    client: TestClient, ts_types: dict[str, set[str]]
) -> None:
    """Flight/hotel/event entries map onto FlightOpt/HotelOpt/EventOpt."""
    _assert_mapping_covers(ts_types["FlightOpt"], FLIGHT_MAP, set())
    _assert_mapping_covers(ts_types["HotelOpt"], HOTEL_MAP, set())
    _assert_mapping_covers(ts_types["EventOpt"], EVENT_MAP, set())

    trip_id = _plan_japan(client)
    results = client.get(f"/api/trips/{trip_id}/results").json()

    assert results["flights"], "expected flight options"
    for flight in results["flights"]:
        _assert_backend_has(flight, FLIGHT_MAP)

    assert results["hotels"], "expected hotel options"
    for hotel in results["hotels"]:
        _assert_backend_has(hotel, HOTEL_MAP)

    # Events may be empty on some paths, but when present must match the shape.
    for event in results["events"]:
        _assert_backend_has(event, EVENT_MAP)


def test_itinerary_days_align_with_frontend_dayplan(
    client: TestClient, ts_types: dict[str, set[str]]
) -> None:
    """Each itinerary day maps onto the frontend DayPlan type."""
    _assert_mapping_covers(ts_types["DayPlan"], DAYPLAN_MAP, set())

    trip_id = _plan_japan(client)
    results = client.get(f"/api/trips/{trip_id}/results").json()

    assert results["itinerary"], "expected a day-by-day itinerary"
    for day in results["itinerary"]:
        _assert_backend_has(day, DAYPLAN_MAP)


def test_results_carry_decision_trace_breakdown(client: TestClient) -> None:
    """The Result is extended with a decision-trace breakdown ("Why X Won").

    The breakdown carries the chosen destination, its score, and the additive
    per-feature factors that sum to that score (Requirements 8.6, 18.3) — the
    field that the frontend ``Result`` type does not have but the design adds.
    """
    trip_id = _plan_japan(client)
    results = client.get(f"/api/trips/{trip_id}/results").json()

    assert "decisionTrace" in results, "results must include the decision-trace field"
    trace = results["decisionTrace"]
    assert trace is not None
    assert set(trace) == {"destination", "score", "factors"}
    assert trace["destination"] == results["destinations"][0]["name"]
    assert trace["factors"], "breakdown must list per-feature contributions"
    for factor in trace["factors"]:
        assert set(factor) == {"feature", "contribution"}
    total = sum(f["contribution"] for f in trace["factors"])
    assert total == pytest.approx(trace["score"], abs=1e-2)


# --------------------------------------------------------------------------- #
# Trip alignment: GET /api/trips and GET /api/trips/{id}
# --------------------------------------------------------------------------- #
def test_trip_summary_aligns_exactly_with_frontend_trip_type(
    client: TestClient, ts_types: dict[str, set[str]]
) -> None:
    """The trip summary's keys match the frontend ``Trip`` type one-to-one."""
    trip_fields = ts_types["Trip"]
    _assert_mapping_covers(trip_fields, TRIP_MAP, set())
    # Trip is a 1:1 alignment: the mapped backend names equal the frontend names.
    assert set(TRIP_MAP.values()) == trip_fields

    trip_id = _plan_japan(client)

    # GET /api/trips/{id}
    summary = client.get(f"/api/trips/{trip_id}").json()
    assert set(summary) == trip_fields

    # GET /api/trips (list)
    listing = client.get("/api/trips")
    assert listing.status_code == 200
    rows = listing.json()
    assert rows, "expected at least one trip in the listing"
    for row in rows:
        assert set(row) == trip_fields


# --------------------------------------------------------------------------- #
# Preference alignment: GET /api/preferences
# --------------------------------------------------------------------------- #
def test_preferences_align_with_frontend_preference_type(
    client: TestClient, ts_types: dict[str, set[str]]
) -> None:
    """Each returned preference is a superset of the frontend ``Preference`` type.

    The frontend ``Preference`` is ``{ topic, weight }``; the backend additionally
    carries ``source`` and ``updated_at``, so the frontend shape is a subset that
    the prototype can consume unchanged.
    """
    pref_fields = ts_types["Preference"]
    assert pref_fields == {"topic", "weight"}

    resp = client.get("/api/preferences")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "preferences" in body
    assert body["preferences"], "expected a seeded preference vector"
    for entry in body["preferences"]:
        # Frontend Preference fields are present on each entry.
        assert pref_fields <= set(entry), (
            f"preference entry missing frontend fields; entry keys={sorted(entry)}"
        )
        assert isinstance(entry["topic"], str)
        assert isinstance(entry["weight"], (int, float))
