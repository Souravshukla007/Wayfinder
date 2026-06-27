"""Property-based test for protected endpoint authentication (task 19.6).

Feature: wayfinder-travel-planner, Property 30: Protected endpoints require
valid authentication.

*For any* network-accessible protected endpoint, a request lacking a valid
Supabase-issued token SHALL be rejected with an authentication error (HTTP
401), and only requests bearing a valid token SHALL be admitted (the auth guard
lets them through to the handler — i.e. the response is never 401).

Validates: Requirements 19.1, 19.2.

The test drives the **real** ``require_user`` JWT verification dependency
through a FastAPI ``TestClient`` (no auth override). The app's settings are
overridden to the HS256 shared-secret verification path so tokens can be minted
locally with no network/JWKS round-trip, mirroring ``test_auth.py``. Only the
database session is overridden (to an in-memory SQLite schema) so the protected
handlers can run when a valid token is admitted.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator

import jwt
import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings, get_settings
from app.main import create_app
from app.models.base import Base
from app.models.database import get_session

# --------------------------------------------------------------------------- #
# Local key material (no network): HS256 shared-secret verification path.
# --------------------------------------------------------------------------- #
AUDIENCE = "authenticated"
# >= 32 bytes so PyJWT does not warn about a short HMAC key.
HS256_SECRET = "super-secret-shared-key-0123456789abcdef"


def _hs256_settings() -> Settings:
    """Settings that verify HS256 tokens against the shared secret only."""
    return Settings(
        _env_file=None,
        supabase_jwks_url=None,
        supabase_jwt_secret=HS256_SECRET,
        supabase_jwt_audience=AUDIENCE,
    )


def _encode(claims: dict, *, secret: str = HS256_SECRET, alg: str = "HS256") -> str:
    """Mint a JWT with the given claims/secret/algorithm."""
    return jwt.encode(claims, secret, algorithm=alg)


def _valid_claims(**overrides: object) -> dict:
    """A well-formed claim set: UUID ``sub``, correct ``aud``, future ``exp``."""
    now = int(time.time())
    base = {
        "sub": str(uuid.uuid4()),
        "email": "traveler@example.com",
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + 3600,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# The protected REST surface (Requirement 19.1: every endpoint is protected).
# --------------------------------------------------------------------------- #
# A placeholder ``{tid}`` is filled with a fresh random trip id per request.
# ``json`` is the request body for write methods (irrelevant on the rejection
# path, since auth fails before the body is read).
_PROTECTED_ENDPOINTS: tuple[tuple[str, str, object], ...] = (
    ("GET", "/api/trips", None),
    ("GET", "/api/trips/{tid}", None),
    ("GET", "/api/trips/{tid}/results", None),
    ("POST", "/api/trips/plan", {"prompt": "Plan a trip", "interests": []}),
    ("GET", "/api/preferences", None),
    ("PATCH", "/api/preferences", {"preferences": []}),
)

# Endpoints safe to call with a *valid* token inside a tight property loop:
# they never kick off the background planning pipeline. They must never be
# rejected as unauthenticated (the admission side of the property).
_SAFE_ENDPOINTS: tuple[tuple[str, str, object], ...] = (
    ("GET", "/api/trips", None),
    ("GET", "/api/trips/{tid}", None),
    ("GET", "/api/trips/{tid}/results", None),
    ("GET", "/api/preferences", None),
    ("PATCH", "/api/preferences", {"preferences": []}),
)


def _request(client: TestClient, endpoint: tuple[str, str, object], headers: dict) -> int:
    """Issue ``endpoint`` through ``client`` with ``headers``; return the status."""
    method, template, body = endpoint
    path = template.format(tid=uuid.uuid4())
    response = client.request(method, path, json=body, headers=headers)
    return response.status_code


# --------------------------------------------------------------------------- #
# App / client fixture (real auth, SQLite-backed handlers).
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client() -> Iterator[TestClient]:
    """A TestClient whose auth runs for real over the HS256 settings path."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    app = create_app()

    def _override_session() -> Iterator[Session]:
        db = factory()
        try:
            yield db
        finally:
            db.close()

    # Override only the DB session and the settings (HS256 path). Critically,
    # the auth dependency (require_user) is NOT overridden, so the real JWT
    # verification guards every route.
    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_settings] = _hs256_settings

    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


# --------------------------------------------------------------------------- #
# Strategies: malformed / missing / invalid credentials.
# --------------------------------------------------------------------------- #
# A safe alphabet so generated junk never produces an illegal HTTP header value.
_HEADER_SAFE = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-",
    min_size=0,
    max_size=48,
)


@st.composite
def invalid_credentials(draw: st.DrawFn) -> dict:
    """Draw an Authorization-headers dict that does NOT carry a valid token.

    Covers the full space of "lacking a valid Supabase-issued token": missing
    header, empty/garbage bearer values, wrong-signature tokens, expired
    tokens, wrong-audience tokens, missing-``sub`` tokens, and unsigned
    (``alg=none``) tokens.
    """
    kind = draw(
        st.sampled_from(
            [
                "missing",
                "empty",
                "garbage",
                "wrong_secret",
                "expired",
                "wrong_audience",
                "missing_sub",
                "alg_none",
            ]
        )
    )

    if kind == "missing":
        return {}
    if kind == "empty":
        return {"Authorization": "Bearer "}
    if kind == "garbage":
        return {"Authorization": f"Bearer {draw(_HEADER_SAFE)}"}
    if kind == "wrong_secret":
        # Correctly-formed HS256 token signed with a secret that is not ours.
        bad_secret = draw(_HEADER_SAFE.filter(lambda s: s != HS256_SECRET)) + "x"
        token = _encode(_valid_claims(), secret=bad_secret)
        return {"Authorization": f"Bearer {token}"}
    if kind == "expired":
        now = int(time.time())
        token = _encode(_valid_claims(iat=now - 7200, exp=now - 3600))
        return {"Authorization": f"Bearer {token}"}
    if kind == "wrong_audience":
        aud = draw(_HEADER_SAFE.filter(lambda s: s != AUDIENCE)) + "!"
        token = _encode(_valid_claims(aud=aud))
        return {"Authorization": f"Bearer {token}"}
    if kind == "missing_sub":
        claims = _valid_claims()
        claims.pop("sub")
        token = _encode(claims)
        return {"Authorization": f"Bearer {token}"}
    # alg_none: unsigned token; verification must refuse the algorithm.
    token = jwt.encode(_valid_claims(), key="", algorithm="none")
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# Property 30: Protected endpoints require valid authentication
# Feature: wayfinder-travel-planner, Property 30
# Validates: Requirements 19.1, 19.2
# --------------------------------------------------------------------------- #
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    endpoint_index=st.integers(min_value=0, max_value=len(_PROTECTED_ENDPOINTS) - 1),
    headers=invalid_credentials(),
)
def test_invalid_token_rejected_on_every_protected_endpoint(
    client: TestClient, endpoint_index: int, headers: dict
) -> None:
    """Any protected endpoint rejects a request lacking a valid token (401).

    Feature: wayfinder-travel-planner, Property 30.
    Validates: Requirements 19.1, 19.2.
    """
    endpoint = _PROTECTED_ENDPOINTS[endpoint_index]
    status_code = _request(client, endpoint, headers)
    assert status_code == 401, (
        f"{endpoint[0]} {endpoint[1]} admitted an invalid credential "
        f"(status {status_code}); expected 401"
    )


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    endpoint_index=st.integers(min_value=0, max_value=len(_SAFE_ENDPOINTS) - 1),
    sub=st.uuids().map(str),
    email=st.emails(),
)
def test_valid_token_is_admitted(
    client: TestClient, endpoint_index: int, sub: str, email: str
) -> None:
    """A request bearing a valid token is admitted (never rejected as 401).

    Feature: wayfinder-travel-planner, Property 30.
    Validates: Requirements 19.1, 19.2.
    """
    endpoint = _SAFE_ENDPOINTS[endpoint_index]
    token = _encode(_valid_claims(sub=sub, email=email))
    headers = {"Authorization": f"Bearer {token}"}
    status_code = _request(client, endpoint, headers)
    assert status_code != 401, (
        f"{endpoint[0]} {endpoint[1]} rejected a valid token (status "
        f"{status_code}); a valid token must be admitted"
    )
