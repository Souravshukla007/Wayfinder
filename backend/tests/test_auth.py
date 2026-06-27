"""Unit tests for Supabase JWT verification (task 3.3, Requirement 19.2).

Generates RSA key material and an HS256 secret locally (no network), then
asserts:

* a valid RS256 token (verified via JWKS) is admitted;
* a valid HS256 token (verified via shared secret) is admitted;
* missing, malformed, expired, and wrong-signature tokens are rejected with 401.

The :func:`require_user` dependency is exercised directly as an async callable,
constructing the bearer credentials by hand, which avoids needing an HTTP test
client.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.auth.jwks import JWKSCache, JWKSError
from app.auth.jwt_middleware import (
    AuthenticatedUser,
    require_user,
    verify_supabase_jwt,
)
from app.config import Settings

AUDIENCE = "authenticated"
TEST_KID = "test-key-1"
# >= 32 bytes so PyJWT does not warn about a short HMAC key.
HS256_SECRET = "super-secret-shared-key-0123456789abcdef"


# --- key material / fixtures ------------------------------------------------


@pytest.fixture(scope="module")
def rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwks_document(rsa_key: rsa.RSAPrivateKey) -> dict:
    """A JWKS document containing the RSA public key under TEST_KID."""
    public_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(rsa_key.public_key(), as_dict=True)
    public_jwk.update({"kid": TEST_KID, "use": "sig", "alg": "RS256"})
    return {"keys": [public_jwk]}


@pytest.fixture()
def jwks_cache(jwks_document: dict) -> JWKSCache:
    """A cache that serves the test JWKS via an injected fetcher (no network)."""
    return JWKSCache("https://example.test/jwks", fetcher=lambda: jwks_document)


@pytest.fixture()
def rs256_settings() -> Settings:
    return Settings(
        _env_file=None,
        supabase_jwks_url="https://example.test/jwks",
        supabase_jwt_secret=None,
        supabase_jwt_audience=AUDIENCE,
    )


@pytest.fixture()
def hs256_settings() -> Settings:
    return Settings(
        _env_file=None,
        supabase_jwks_url=None,
        supabase_jwt_secret="super-secret-shared-key-0123456789abcdef",
        supabase_jwt_audience=AUDIENCE,
    )


# --- token builders ---------------------------------------------------------


def _claims(**overrides) -> dict:
    now = int(time.time())
    base = {
        "sub": "user-uuid-123",
        "email": "traveler@example.com",
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + 3600,
    }
    base.update(overrides)
    return base


def _rs256_token(rsa_key: rsa.RSAPrivateKey, **overrides) -> str:
    return jwt.encode(
        _claims(**overrides),
        rsa_key,
        algorithm="RS256",
        headers={"kid": TEST_KID},
    )


def _hs256_token(secret: str, **overrides) -> str:
    return jwt.encode(_claims(**overrides), secret, algorithm="HS256")


async def _call_require_user(token, settings, jwks_cache) -> AuthenticatedUser:
    creds = (
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        if token is not None
        else None
    )
    return await require_user(creds=creds, settings=settings, jwks_cache=jwks_cache)


# --- valid token admitted ---------------------------------------------------


def test_valid_rs256_token_verifies(rsa_key, rs256_settings, jwks_cache) -> None:
    claims = verify_supabase_jwt(_rs256_token(rsa_key), rs256_settings, jwks_cache)
    assert claims["sub"] == "user-uuid-123"
    assert claims["email"] == "traveler@example.com"


async def test_valid_rs256_token_admits_user(rsa_key, rs256_settings, jwks_cache) -> None:
    user = await _call_require_user(_rs256_token(rsa_key), rs256_settings, jwks_cache)
    assert isinstance(user, AuthenticatedUser)
    assert user.user_id == "user-uuid-123"
    assert user.email == "traveler@example.com"


async def test_valid_hs256_token_admits_user(hs256_settings) -> None:
    cache = JWKSCache(None)  # no JWKS needed for the symmetric path
    token = _hs256_token("super-secret-shared-key-0123456789abcdef")
    user = await _call_require_user(token, hs256_settings, cache)
    assert user.user_id == "user-uuid-123"


# --- rejections (401) -------------------------------------------------------


async def test_missing_token_rejected(rs256_settings, jwks_cache) -> None:
    with pytest.raises(HTTPException) as exc:
        await _call_require_user(None, rs256_settings, jwks_cache)
    assert exc.value.status_code == 401


async def test_malformed_token_rejected(rs256_settings, jwks_cache) -> None:
    with pytest.raises(HTTPException) as exc:
        await _call_require_user("not-a-jwt", rs256_settings, jwks_cache)
    assert exc.value.status_code == 401


async def test_expired_token_rejected(rsa_key, rs256_settings, jwks_cache) -> None:
    expired = _rs256_token(
        rsa_key,
        iat=int(time.time()) - 7200,
        exp=int(time.time()) - 3600,
    )
    with pytest.raises(HTTPException) as exc:
        await _call_require_user(expired, rs256_settings, jwks_cache)
    assert exc.value.status_code == 401


async def test_wrong_signature_rejected(rs256_settings, jwks_cache) -> None:
    """A token signed by a different RSA key fails JWKS verification."""
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged = jwt.encode(_claims(), other_key, algorithm="RS256", headers={"kid": TEST_KID})
    with pytest.raises(HTTPException) as exc:
        await _call_require_user(forged, rs256_settings, jwks_cache)
    assert exc.value.status_code == 401


async def test_wrong_audience_rejected(rsa_key, rs256_settings, jwks_cache) -> None:
    bad_aud = _rs256_token(rsa_key, aud="some-other-audience")
    with pytest.raises(HTTPException) as exc:
        await _call_require_user(bad_aud, rs256_settings, jwks_cache)
    assert exc.value.status_code == 401


async def test_hs256_without_configured_secret_rejected(rs256_settings, jwks_cache) -> None:
    """An HS256 token is rejected when no shared secret is configured."""
    token = _hs256_token("attacker-chosen-secret-0123456789abcdef")
    with pytest.raises(HTTPException) as exc:
        await _call_require_user(token, rs256_settings, jwks_cache)
    assert exc.value.status_code == 401


# --- JWKS fail-closed behaviour (Requirement 19.1) --------------------------


def test_jwks_fails_closed_without_url() -> None:
    cache = JWKSCache(None)
    with pytest.raises(JWKSError):
        cache.get_signing_key("any-kid")


def test_jwks_unknown_kid_fails_closed(jwks_document) -> None:
    cache = JWKSCache("https://example.test/jwks", fetcher=lambda: jwks_document)
    with pytest.raises(JWKSError):
        cache.get_signing_key("nonexistent-kid")
