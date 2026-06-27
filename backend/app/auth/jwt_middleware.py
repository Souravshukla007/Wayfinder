"""Supabase JWT verification dependency (task 3.2, Requirements 19.1, 19.2).

The FastAPI backend does not implement its own auth; it *verifies* JWTs issued
by Supabase. The :func:`require_user` dependency guards every protected route
and the WebSocket handshake.

Verification strategy:

* **Primary (asymmetric):** ``RS256`` / ``ES256`` tokens are verified against
  the cached Supabase JWKS (see :mod:`app.auth.jwks`), checking signature,
  ``aud`` and ``exp``.
* **Fallback (symmetric):** ``HS256`` tokens are verified against the configured
  Supabase shared JWT secret. This path is only available when
  ``settings.supabase_jwt_secret`` is set.

Any failure — missing/malformed token, bad signature, expired, wrong audience,
unconfigured key material — results in a ``401`` and the request is rejected.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.auth.jwks import JWKSCache, JWKSError
from app.config import Settings, get_settings

# Algorithms verified via the JWKS (asymmetric) path.
ASYMMETRIC_ALGORITHMS = ("RS256", "ES256")
# Algorithm verified via the shared-secret (symmetric) fallback path.
SYMMETRIC_ALGORITHM = "HS256"

_bearer_scheme = HTTPBearer(auto_error=False)


class AuthenticatedUser(BaseModel):
    """The verified caller, derived from the JWT claims."""

    user_id: str  # Supabase "sub" claim
    email: Optional[str] = None


class JwtError(Exception):
    """Raised internally when a token cannot be verified."""


@lru_cache(maxsize=8)
def _cache_for(jwks_url: Optional[str], ttl_seconds: int) -> JWKSCache:
    """Return a process-wide :class:`JWKSCache` for a given URL (memoized)."""
    return JWKSCache(jwks_url, ttl_seconds=ttl_seconds)


def get_jwks_cache(settings: Settings = Depends(get_settings)) -> JWKSCache:
    """FastAPI dependency providing the JWKS cache for the configured URL."""
    return _cache_for(settings.supabase_jwks_url, 3600)


def verify_supabase_jwt(
    token: str,
    settings: Settings,
    jwks_cache: JWKSCache,
) -> dict[str, Any]:
    """Verify a Supabase JWT and return its claims, or raise :class:`JwtError`.

    Selects the verification path from the token's ``alg`` header: asymmetric
    algorithms go through the JWKS, ``HS256`` through the shared secret.
    """
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise JwtError(f"malformed token header: {exc}") from exc

    alg = header.get("alg")
    decode_kwargs: dict[str, Any] = {
        "audience": settings.supabase_jwt_audience,
        "options": {"require": ["exp", "sub"]},
    }

    try:
        if alg in ASYMMETRIC_ALGORITHMS:
            signing_key = jwks_cache.get_signing_key(header.get("kid"))
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=list(ASYMMETRIC_ALGORITHMS),
                **decode_kwargs,
            )
        elif alg == SYMMETRIC_ALGORITHM:
            secret = settings.supabase_jwt_secret
            if not secret:
                raise JwtError(
                    "HS256 token received but no shared secret is configured"
                )
            claims = jwt.decode(
                token,
                secret,
                algorithms=[SYMMETRIC_ALGORITHM],
                **decode_kwargs,
            )
        else:
            raise JwtError(f"unsupported token algorithm: {alg!r}")
    except JWKSError as exc:
        raise JwtError(f"no usable verification key: {exc}") from exc
    except jwt.PyJWTError as exc:
        raise JwtError(f"token verification failed: {exc}") from exc

    return claims


async def require_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    settings: Settings = Depends(get_settings),
    jwks_cache: JWKSCache = Depends(get_jwks_cache),
) -> AuthenticatedUser:
    """FastAPI dependency: verify the bearer token and return the caller.

    Raises ``401`` for any missing/invalid/expired token (Requirement 19.2).
    """
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None or not creds.credentials:
        raise unauthorized
    try:
        claims = verify_supabase_jwt(creds.credentials, settings, jwks_cache)
    except JwtError as exc:
        raise unauthorized from exc

    sub = claims.get("sub")
    if not sub:
        raise unauthorized
    return AuthenticatedUser(user_id=sub, email=claims.get("email"))
