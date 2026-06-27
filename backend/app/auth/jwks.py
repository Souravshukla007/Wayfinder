"""Supabase JWKS fetch + cache (task 3.1, Requirement 19.1).

When Supabase issues asymmetric (RS256/ES256) access tokens, their public
verification keys are published at a JWKS endpoint. This module fetches that
key set, caches it with a TTL, and resolves a signing key by ``kid``.

**Fail closed.** If no JWKS URL is configured, or the endpoint cannot be
reached, or the requested ``kid`` is absent from the key set, key resolution
raises :class:`JWKSError`. The caller (the JWT middleware) treats any such
failure as an authentication failure (HTTP 401) rather than admitting the
request — i.e. absence of a usable key never results in a token being trusted.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Any, Callable, Optional

import jwt
from jwt import PyJWK


class JWKSError(Exception):
    """Raised when a usable signing key cannot be obtained (fail closed)."""


# A fetcher returns the raw JWKS document as a dict: ``{"keys": [ {...}, ... ]}``.
Fetcher = Callable[[], dict[str, Any]]


class JWKSCache:
    """Caches a Supabase JWKS document and resolves signing keys by ``kid``.

    Parameters
    ----------
    jwks_url:
        The Supabase JWKS endpoint. ``None`` means "no asymmetric verification
        configured" and any key resolution fails closed.
    ttl_seconds:
        How long a fetched key set is considered fresh before a refresh is
        attempted.
    fetcher:
        Optional injectable fetch function (used in tests to avoid network).
        Defaults to an HTTP GET of ``jwks_url``.
    """

    def __init__(
        self,
        jwks_url: Optional[str],
        *,
        ttl_seconds: int = 3600,
        fetcher: Optional[Fetcher] = None,
    ) -> None:
        self._url = jwks_url
        self._ttl = ttl_seconds
        self._fetcher = fetcher or self._default_fetch
        self._lock = threading.Lock()
        self._keys: dict[str, PyJWK] = {}
        self._fetched_at: float = 0.0

    # -- fetching -----------------------------------------------------------
    def _default_fetch(self) -> dict[str, Any]:
        if not self._url:
            raise JWKSError("no JWKS URL configured")
        try:
            with urllib.request.urlopen(self._url, timeout=5) as resp:  # noqa: S310
                payload = resp.read().decode("utf-8")
        except Exception as exc:  # network/HTTP error -> fail closed
            raise JWKSError(f"failed to fetch JWKS: {exc}") from exc
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise JWKSError("JWKS endpoint returned invalid JSON") from exc

    def _is_stale(self) -> bool:
        return (time.monotonic() - self._fetched_at) >= self._ttl

    def _refresh_locked(self) -> None:
        document = self._fetcher()
        keys = document.get("keys") if isinstance(document, dict) else None
        if not keys:
            raise JWKSError("JWKS document contained no keys")
        parsed: dict[str, PyJWK] = {}
        for entry in keys:
            try:
                key = PyJWK.from_dict(entry)
            except Exception:  # skip malformed individual keys
                continue
            kid = entry.get("kid")
            if kid:
                parsed[kid] = key
        if not parsed:
            raise JWKSError("JWKS document contained no usable keys")
        self._keys = parsed
        self._fetched_at = time.monotonic()

    # -- public API ---------------------------------------------------------
    def get_signing_key(self, kid: Optional[str], *, force_refresh: bool = False) -> PyJWK:
        """Return the :class:`PyJWK` for ``kid``, fetching/refreshing as needed.

        Fails closed (raises :class:`JWKSError`) when no URL is configured, the
        fetch fails, or the ``kid`` is not present after a refresh attempt.
        """
        if not self._url:
            raise JWKSError("no JWKS URL configured; cannot verify asymmetric token")
        with self._lock:
            if force_refresh or not self._keys or self._is_stale():
                self._refresh_locked()
            key = self._keys.get(kid) if kid else None
            if key is None:
                # Unknown kid: the key set may have rotated. Try one refresh.
                self._refresh_locked()
                key = self._keys.get(kid) if kid else None
            if key is None:
                raise JWKSError(f"no signing key found for kid={kid!r}")
            return key
