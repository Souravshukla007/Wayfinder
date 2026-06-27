"""WebSocket progress-stream route (Task 19.3).

Exposes ``WS /api/trips/{tripId}/progress``: a per-trip channel that relays the
planning graph's stage progress events to the frontend as each agent/tool node
starts and completes (e.g. "Searching flights, hotels, events, and weather",
then "... — done"). See :mod:`app.api.progress_hub` for the producer/consumer
bridge.

**Only progress events are streamed — never live or real-time prices**
(Requirement 13.3 / Property 26). The events relayed here come from
:class:`app.orchestration.graph.ProgressEvent`, which by construction carries
only a stage name, phase, and human-readable message; this handler forwards
them verbatim and never reads pricing data.

**Auth (Task 19.4).** The handshake is verified *before* the socket is accepted
and *before* any event is streamed: :func:`_authorize_handshake` extracts the
Supabase JWT from the handshake and verifies it; on failure the connection is
closed with a policy-violation code *before* ``accept()`` so no progress is ever
streamed to an unauthenticated client (Requirements 19.1, 19.2). The
:func:`authorize_ws_handshake` dependency wraps it so the verification keys are
injected (and overridable in tests).

Requirements: 13.1, 13.2, 13.3, 19.1, 19.2.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.api.progress_hub import ProgressHub, get_progress_hub
from app.auth.jwks import JWKSCache
from app.auth.jwt_middleware import JwtError, get_jwks_cache, verify_supabase_jwt
from app.config import Settings, get_settings

router = APIRouter(tags=["progress"])


def _extract_bearer_token(websocket: WebSocket) -> Optional[str]:
    """Pull the Supabase JWT out of the WebSocket handshake.

    Browser ``WebSocket`` clients cannot set arbitrary headers, so the token is
    accepted from any of the conventional handshake carriers, in order:

    * the ``Authorization: Bearer <jwt>`` header (non-browser / server clients);
    * a ``token`` or ``access_token`` query parameter;
    * the ``Sec-WebSocket-Protocol`` subprotocol list, where the first entry is
      ``bearer``/``authorization`` and the token is the trailing entry.

    Returns ``None`` when no token is present.
    """
    auth_header = websocket.headers.get("authorization")
    if auth_header:
        scheme, _, credentials = auth_header.partition(" ")
        if scheme.lower() == "bearer" and credentials.strip():
            return credentials.strip()

    for param in ("token", "access_token"):
        value = websocket.query_params.get(param)
        if value:
            return value

    protocols = websocket.headers.get("sec-websocket-protocol")
    if protocols:
        parts = [part.strip() for part in protocols.split(",") if part.strip()]
        if len(parts) >= 2 and parts[0].lower() in {"bearer", "authorization"}:
            return parts[-1]

    return None


async def _authorize_handshake(
    websocket: WebSocket,
    settings: Settings,
    jwks_cache: JWKSCache,
) -> bool:
    """Verify the WebSocket handshake before any stream begins.

    Returns ``True`` only when the handshake carries a Supabase JWT that
    verifies (signature, ``aud``, ``exp``) and has a ``sub`` claim. Any
    missing/malformed/expired/invalid token returns ``False`` so the caller
    closes the connection *before* ``accept()`` and before streaming,
    guaranteeing unauthenticated clients receive no progress events
    (Requirements 19.1, 19.2).
    """
    token = _extract_bearer_token(websocket)
    if not token:
        return False
    try:
        claims = verify_supabase_jwt(token, settings, jwks_cache)
    except JwtError:
        return False
    return bool(claims.get("sub"))


async def authorize_ws_handshake(
    websocket: WebSocket,
    settings: Settings = Depends(get_settings),
    jwks_cache: JWKSCache = Depends(get_jwks_cache),
) -> bool:
    """Dependency wrapper around :func:`_authorize_handshake`.

    Injects the configured settings and JWKS cache so verification uses the same
    key material as the REST routes. Exposed as a dependency so the test suite
    can override it with an authenticated stand-in.
    """
    return await _authorize_handshake(websocket, settings, jwks_cache)


@router.websocket("/api/trips/{trip_id}/progress")
async def trip_progress(
    websocket: WebSocket,
    trip_id: str,
    hub: ProgressHub = Depends(get_progress_hub),
    authorized: bool = Depends(authorize_ws_handshake),
) -> None:
    """Relay stage progress events for ``trip_id`` to the connected client.

    Streams one JSON message per progress event (stage start/completion) until
    the run finishes, then closes the socket. Never streams pricing data.
    """
    # Enforce auth before accepting the socket so no events are streamed to an
    # unauthorized client (Requirements 19.1, 19.2).
    if not authorized:
        await websocket.close(code=1008)  # 1008 = policy violation
        return

    await websocket.accept()
    try:
        async for event in hub.stream(trip_id):
            # ProgressEvent only holds stage/phase/message — no prices.
            await websocket.send_json(event.model_dump())
    except WebSocketDisconnect:
        # Client went away; nothing to clean up beyond the stream's own finally.
        return
    else:
        # Run completed: close the stream cleanly.
        await websocket.close()
