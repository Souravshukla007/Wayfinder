"""API tests for the preference routes (task 19.2).

Covers ``GET /api/preferences`` and ``PATCH /api/preferences`` end to end through
a FastAPI ``TestClient``, with the DB session backed by an in-memory SQLite
database and the authenticated user injected via a dependency override (real
JWT enforcement is exercised separately in ``test_auth.py``; blanket route
enforcement lands in task 19.4).

Validates:
- Requirement 10.7: GET returns the stored vector with (topic, weight, updated_at).
- Requirement 10.8: PATCH replaces an in-range weight and returns the new vector.
- Requirement 10.9: PATCH rejects an out-of-range override, the error identifies
  the invalid value, and the prior stored weight is retained unchanged.
- Response shape aligns with the frontend ``Preference`` type (``{ topic, weight }``
  is a subset of each returned entry).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import routes_preferences
from app.api.routes_preferences import current_user_id
from app.models.database import get_session
from app.models.db import Base, User, UserPreference

NOW = datetime(2024, 10, 1, 12, 0, tzinfo=timezone.utc)


def _build_app() -> FastAPI:
    """Build a minimal app exposing only the preferences router.

    Avoids importing ``app.main`` (which eagerly constructs the full app and the
    configured database engine via other routers); these tests exercise the
    preference routes in isolation against an in-memory database.
    """
    app = FastAPI()
    app.include_router(routes_preferences.router)
    return app


@pytest.fixture()
def engine():
    eng = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture()
def session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@pytest.fixture()
def user_id(session_factory) -> str:
    """Seed a user and return its id as a string (mirrors a Supabase ``sub``)."""
    db: Session = session_factory()
    try:
        user = User(id=uuid.uuid4(), email="traveler@example.com", prefs={})
        db.add(user)
        db.commit()
        return str(user.id)
    finally:
        db.close()


@pytest.fixture()
def client(session_factory, user_id):
    """A TestClient with DB session + authenticated user dependencies overridden."""
    app = _build_app()

    def _override_session():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[current_user_id] = lambda: user_id

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _seed_pref(session_factory, user_id: str, topic: str, weight: float, *, source="explicit") -> None:
    db: Session = session_factory()
    try:
        db.add(
            UserPreference(
                user_id=uuid.UUID(user_id),
                topic=topic,
                weight=weight,
                source=source,
                updated_at=NOW,
            )
        )
        db.commit()
    finally:
        db.close()


# --- GET /api/preferences (Requirement 10.7) --------------------------------


def test_get_preferences_returns_stored_vector(client, session_factory, user_id) -> None:
    _seed_pref(session_factory, user_id, "food_score", 0.7)
    _seed_pref(session_factory, user_id, "budget_fit", 0.2, source="implicit")

    resp = client.get("/api/preferences")
    assert resp.status_code == 200

    body = resp.json()
    assert "preferences" in body
    # Topic-sorted, each entry carries topic + weight + updated_at (Req 10.7).
    topics = [p["topic"] for p in body["preferences"]]
    assert topics == ["budget_fit", "food_score"]
    by_topic = {p["topic"]: p for p in body["preferences"]}
    assert by_topic["food_score"]["weight"] == pytest.approx(0.7)
    assert by_topic["budget_fit"]["weight"] == pytest.approx(0.2)
    assert by_topic["food_score"]["updated_at"] is not None
    # Frontend Preference shape ({topic, weight}) is a subset of each entry.
    assert {"topic", "weight"} <= set(by_topic["food_score"].keys())


def test_get_preferences_empty_when_no_vector(client) -> None:
    resp = client.get("/api/preferences")
    assert resp.status_code == 200
    assert resp.json() == {"preferences": []}


# --- PATCH /api/preferences (Requirements 10.8, 10.9) -----------------------


def test_patch_preferences_overrides_in_range_weight(client, session_factory, user_id) -> None:
    _seed_pref(session_factory, user_id, "food_score", 0.2, source="implicit")

    resp = client.patch(
        "/api/preferences",
        json={"preferences": [{"topic": "food_score", "weight": 0.9}]},
    )
    assert resp.status_code == 200

    by_topic = {p["topic"]: p for p in resp.json()["preferences"]}
    assert by_topic["food_score"]["weight"] == pytest.approx(0.9)
    # Override is an explicit user-stated signal.
    assert by_topic["food_score"]["source"] == "explicit"


def test_patch_preferences_applies_multiple_overrides(client, session_factory, user_id) -> None:
    _seed_pref(session_factory, user_id, "food_score", 0.2)
    _seed_pref(session_factory, user_id, "budget_fit", 0.5)

    resp = client.patch(
        "/api/preferences",
        json={
            "preferences": [
                {"topic": "food_score", "weight": 0.1},
                {"topic": "budget_fit", "weight": 0.8},
            ]
        },
    )
    assert resp.status_code == 200
    by_topic = {p["topic"]: p["weight"] for p in resp.json()["preferences"]}
    assert by_topic["food_score"] == pytest.approx(0.1)
    assert by_topic["budget_fit"] == pytest.approx(0.8)


@pytest.mark.parametrize("bad", [-0.01, 1.01, 5.0, -1.0])
def test_patch_preferences_rejects_out_of_range_and_identifies_value(
    client, session_factory, user_id, bad
) -> None:
    _seed_pref(session_factory, user_id, "food_score", 0.42)

    resp = client.patch(
        "/api/preferences",
        json={"preferences": [{"topic": "food_score", "weight": bad}]},
    )
    assert resp.status_code == 422
    # The error identifies the invalid value (Requirement 10.9).
    assert repr(bad) in resp.json()["detail"]

    # Prior stored weight is retained unchanged.
    db: Session = session_factory()
    try:
        row = (
            db.query(UserPreference)
            .filter_by(user_id=uuid.UUID(user_id), topic="food_score")
            .one()
        )
        assert row.weight == pytest.approx(0.42)
    finally:
        db.close()


def test_patch_out_of_range_in_batch_does_not_apply_any(client, session_factory, user_id) -> None:
    _seed_pref(session_factory, user_id, "food_score", 0.42)
    _seed_pref(session_factory, user_id, "budget_fit", 0.5)

    resp = client.patch(
        "/api/preferences",
        json={
            "preferences": [
                {"topic": "food_score", "weight": 0.1},  # valid
                {"topic": "budget_fit", "weight": 9.0},  # invalid -> rejects batch
            ]
        },
    )
    assert resp.status_code == 422

    # Neither weight changed (all-or-nothing).
    db: Session = session_factory()
    try:
        rows = {
            r.topic: r.weight
            for r in db.query(UserPreference).filter_by(user_id=uuid.UUID(user_id)).all()
        }
        assert rows["food_score"] == pytest.approx(0.42)
        assert rows["budget_fit"] == pytest.approx(0.5)
    finally:
        db.close()


def test_patch_unknown_topic_returns_404(client, session_factory, user_id) -> None:
    resp = client.patch(
        "/api/preferences",
        json={"preferences": [{"topic": "no_such_topic", "weight": 0.5}]},
    )
    assert resp.status_code == 404


# --- Authentication wiring --------------------------------------------------


def test_preferences_require_authentication(session_factory) -> None:
    """Without the auth override, the unauthenticated request is rejected (401)."""
    app = _build_app()

    def _override_session():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_session] = _override_session
    with TestClient(app) as test_client:
        assert test_client.get("/api/preferences").status_code == 401
    app.dependency_overrides.clear()
