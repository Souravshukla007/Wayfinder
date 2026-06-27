"""Tests for the WebSocket progress stream (Task 19.3).

Covers ``WS /api/trips/{tripId}/progress``:

* Requirement 13.1 — progress events are streamed over the WebSocket channel
  while a planning run is processed.
* Requirement 13.2 — a stage event is emitted on each agent/tool start and
  completion.
* Requirement 13.3 — only progress events are streamed; never live/real-time
  prices.

The hub buffers events per trip, so a client that connects after the run has
started (or finished) still receives every stage event. Tests publish events to
the shared hub, then connect a :class:`fastapi.testclient.TestClient`
WebSocket and read them. One integration test drives the real planning graph
through ``hub.reporter_for`` to confirm the end-to-end stream.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from app.api.progress_hub import get_progress_hub
from app.api.routes_ws import authorize_ws_handshake
from app.config import Settings
from app.orchestration.graph import ProgressEvent, run_plan
from app.main import create_app
from app.orchestration.state import GraphState

# A live/real-time price would show a currency marker, a "price" word, or a
# multi-digit amount. No progress message may match this (Requirement 13.3).
_PRICE_PATTERN = re.compile(r"[\u20b9$\u20ac\u00a3]|\bprice\b|\d{3,}", re.IGNORECASE)


def _client() -> TestClient:
    """A TestClient whose WebSocket handshake auth is satisfied (Task 19.4).

    The handshake authorization dependency is overridden to admit the
    connection so these tests focus on the streaming behaviour; rejection of
    unauthenticated handshakes is covered by
    :func:`test_unauthenticated_handshake_is_rejected_before_stream`.
    """
    app = create_app()
    app.dependency_overrides[authorize_ws_handshake] = lambda: True
    return TestClient(app)


def test_stream_relays_buffered_stage_events_then_closes() -> None:
    hub = get_progress_hub()
    trip_id = "ws-trip-buffered"
    hub.reset(trip_id)

    # A producer (the plan route in 19.1) publishes start/complete per stage.
    emitted = [
        ProgressEvent(stage="coordinator", phase="start", message="Understanding"),
        ProgressEvent(stage="coordinator", phase="complete", message="Understanding — done"),
        ProgressEvent(stage="tools", phase="start", message="Searching"),
        ProgressEvent(stage="tools", phase="complete", message="Searching — done"),
    ]
    for event in emitted:
        hub.publish(trip_id, event)
    hub.complete(trip_id)

    received: list[dict] = []
    with _client() as client:
        with client.websocket_connect(f"/api/trips/{trip_id}/progress") as ws:
            for _ in emitted:
                received.append(ws.receive_json())

    assert [(e["stage"], e["phase"]) for e in received] == [
        (e.stage, e.phase) for e in emitted
    ]


def test_stream_emits_start_and_complete_for_each_stage() -> None:
    hub = get_progress_hub()
    trip_id = "ws-trip-phases"
    hub.reset(trip_id)

    stages = ("coordinator", "destination", "tools")
    for stage in stages:
        hub.publish(trip_id, ProgressEvent(stage=stage, phase="start", message=stage))
        hub.publish(
            trip_id, ProgressEvent(stage=stage, phase="complete", message=f"{stage} — done")
        )
    hub.complete(trip_id)

    received: list[dict] = []
    with _client() as client:
        with client.websocket_connect(f"/api/trips/{trip_id}/progress") as ws:
            for _ in range(len(stages) * 2):
                received.append(ws.receive_json())

    # Every stage emitted exactly a start followed by a complete (Req 13.2).
    for stage in stages:
        phases = [e["phase"] for e in received if e["stage"] == stage]
        assert phases == ["start", "complete"], f"{stage}: {phases}"


def test_full_graph_run_streams_stage_events_and_no_prices() -> None:
    """End-to-end: drive the real graph through the hub and stream it."""
    hub = get_progress_hub()
    trip_id = "ws-trip-graph"
    hub.reset(trip_id)

    prompt = (
        "Plan a 7-day Japan trip in October under \u20b92 lakh. "
        "I like anime, photography, local food, and less crowded places."
    )
    state = GraphState(trip_id=trip_id, user_id="user-1", prompt=prompt)

    # Run the pipeline, publishing each node's start/complete to the hub, then
    # close the channel — mirroring how the plan route (19.1) will wire it.
    run_plan(state, settings=Settings(), progress=hub.reporter_for(trip_id))
    hub.complete(trip_id)

    received: list[dict] = []
    with _client() as client:
        with client.websocket_connect(f"/api/trips/{trip_id}/progress") as ws:
            try:
                while True:
                    received.append(ws.receive_json())
            except Exception:
                # Socket closed by the server once the run's stream ended.
                pass

    assert received, "expected progress events to be streamed (Req 13.1)"

    # Each visited stage has a start and a complete event (Req 13.2).
    for stage in ("coordinator", "destination", "decision_engine", "tools", "solver", "merge"):
        phases = [e["phase"] for e in received if e["stage"] == stage]
        assert phases == ["start", "complete"], f"{stage}: {phases}"

    # Only stage progress is streamed — never a live/real-time price (Req 13.3).
    for event in received:
        assert not _PRICE_PATTERN.search(event["message"]), event["message"]
        assert set(event.keys()) == {"stage", "phase", "message"}


# --------------------------------------------------------------------------- #
# Handshake authentication (Task 19.4, Requirements 19.1, 19.2)
# --------------------------------------------------------------------------- #
def test_unauthenticated_handshake_is_rejected_before_stream() -> None:
    """An unauthenticated handshake is closed before any event is streamed.

    The app is built WITHOUT overriding the handshake auth dependency and the
    client connects with no token, so the server must close the socket before
    ``accept()`` — the connection never yields a progress event.
    """
    import pytest
    from starlette.websockets import WebSocketDisconnect

    hub = get_progress_hub()
    trip_id = "ws-trip-unauth"
    hub.reset(trip_id)
    # Buffer an event so a (wrongly) accepted socket would have something to send.
    hub.publish(trip_id, ProgressEvent(stage="coordinator", phase="start", message="x"))
    hub.complete(trip_id)

    received: list[dict] = []
    with TestClient(create_app()) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(f"/api/trips/{trip_id}/progress") as ws:
                received.append(ws.receive_json())

    assert received == [], "no progress may be streamed to an unauthenticated client"


def test_valid_token_admits_handshake_via_query_param() -> None:
    """A valid Supabase HS256 token in the query string admits the handshake.

    Exercises the real verification path (no auth override): the token is
    extracted from the handshake, verified against the configured shared
    secret, and the buffered stage events are then streamed (Requirement 19.1).
    """
    import time

    import jwt

    from app.auth.jwks import JWKSCache
    from app.auth.jwt_middleware import get_jwks_cache
    from app.config import get_settings

    secret = "super-secret-shared-key-0123456789abcdef"
    token = jwt.encode(
        {
            "sub": "user-uuid-123",
            "aud": "authenticated",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        },
        secret,
        algorithm="HS256",
    )

    hub = get_progress_hub()
    trip_id = "ws-trip-auth-ok"
    hub.reset(trip_id)
    hub.publish(trip_id, ProgressEvent(stage="coordinator", phase="start", message="go"))
    hub.complete(trip_id)

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        supabase_jwks_url=None,
        supabase_jwt_secret=secret,
        supabase_jwt_audience="authenticated",
    )
    app.dependency_overrides[get_jwks_cache] = lambda: JWKSCache(None)

    received: list[dict] = []
    with TestClient(app) as client:
        with client.websocket_connect(
            f"/api/trips/{trip_id}/progress?token={token}"
        ) as ws:
            received.append(ws.receive_json())

    assert received and received[0]["stage"] == "coordinator"
    app.dependency_overrides.clear()
