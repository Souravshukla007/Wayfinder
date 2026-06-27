"""Preference routes over the Travel Memory Layer (task 19.2).

Exposes the two preference endpoints from the API contract, mapped 1:1 to the
prototype's ``getPreferences`` / ``patchPreferences`` functions in
``src/lib/mockApi.ts``:

- ``GET /api/preferences``   — return the user's stored Preference_Vector as
  ``(topic, weight, updated_at)`` entries (Requirement 10.7).
- ``PATCH /api/preferences`` — override stored preference weights; each override
  is validated to lie within [0.0, 1.0] by the Memory Layer, and an
  out-of-range value is rejected with an error that identifies the invalid value
  (Requirement 10.8 / 10.9).

Response bodies use :class:`PreferencesDTO` (``{ "preferences": [...] }``) so the
frontend's ``Preference`` shape (``{ topic, weight }``) is a subset of what is
returned (each entry additionally carries ``source`` and ``updated_at``).

The authenticated user id is resolved through the :func:`current_user_id`
dependency, isolating the auth seam so task 19.4's blanket authentication
enforcement (and test overrides) has a single, well-defined place to act.

Requirements: 10.7, 10.8, 19.1.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.auth.jwt_middleware import AuthenticatedUser, require_user
from app.memory import memory_layer
from app.memory.memory_layer import (
    PreferenceOverrideError,
    UnknownPreferenceTopicError,
    get_preferences,
    override_preference,
)
from app.models.database import get_session
from app.models.schemas import PreferencesDTO

router = APIRouter(prefix="/api/preferences", tags=["preferences"])


def current_user_id(user: AuthenticatedUser = Depends(require_user)) -> str:
    """Resolve the authenticated caller's id for preference operations.

    Kept as its own dependency (rather than reading ``require_user`` inline in
    each route) so that the routes only ever see a user id, and so task 19.4's
    blanket auth enforcement and the test suite have a single override seam.
    """
    return user.user_id


def _resolve_user_uuid(user_id: str) -> uuid.UUID:
    """Coerce the authenticated user id (Supabase ``sub``) to a ``uuid.UUID``.

    The ``user_preferences`` table keys on a UUID ``user_id``; Supabase issues
    UUID ``sub`` claims, so the string id is parsed here. A malformed id can
    never match a stored preference, so it is rejected as an unauthorized
    caller rather than surfaced as a server error.
    """
    try:
        return uuid.UUID(str(user_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user identifier",
        ) from exc


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class PreferenceOverrideDTO(BaseModel):
    """A single ``(topic, weight)`` override.

    ``weight`` is intentionally left unconstrained at the schema layer so that
    range validation is performed by the Memory Layer's
    :func:`override_preference`, which rejects out-of-range values with an error
    that identifies the offending value (Requirement 10.9). Constraining it here
    would surface a generic 422 that does not carry the Memory Layer's
    value-identifying message.
    """

    topic: str
    weight: float


class PreferencesPatchRequest(BaseModel):
    """Body for ``PATCH /api/preferences`` — a batch of weight overrides.

    Mirrors the prototype's ``patchPreferences({ preferences: Preference[] })``.
    Each entry is applied via the Memory Layer's single-topic override.
    """

    preferences: list[PreferenceOverrideDTO] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.get("", response_model=PreferencesDTO)
def read_preferences(
    user_id: Annotated[str, Depends(current_user_id)],
    session: Annotated[Session, Depends(get_session)],
) -> PreferencesDTO:
    """Return the caller's current stored Preference_Vector (Requirement 10.7).

    Each entry carries its ``(topic, weight)`` pair and the ``updated_at``
    timestamp of the stored signal; weights are returned as stored (not
    time-decayed).
    """
    return PreferencesDTO(preferences=get_preferences(session, _resolve_user_uuid(user_id)))


@router.patch("", response_model=PreferencesDTO)
def patch_preferences(
    body: PreferencesPatchRequest,
    user_id: Annotated[str, Depends(current_user_id)],
    session: Annotated[Session, Depends(get_session)],
) -> PreferencesDTO:
    """Override one or more stored preference weights (Requirements 10.8, 10.9).

    Every supplied weight is range-validated *before* any mutation, so a single
    out-of-range value rejects the whole request without altering any stored
    weight. An out-of-range value yields ``422`` with a detail that identifies
    the invalid value; an override targeting a topic the user has no stored
    weight for yields ``404``. On success the full, updated Preference_Vector is
    returned.
    """
    uid = _resolve_user_uuid(user_id)
    # Pre-validate all weights up front so the batch is all-or-nothing: no
    # partial writes occur if any value is out of range. The Memory Layer's
    # bounds are reused so the validation semantics match exactly (non-finite
    # values such as nan/inf are treated as out of range).
    for entry in body.preferences:
        if not (memory_layer.WEIGHT_MIN <= entry.weight <= memory_layer.WEIGHT_MAX):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"preference override weight {entry.weight!r} for topic "
                    f"{entry.topic!r} is out of range; it must be within "
                    f"[{memory_layer.WEIGHT_MIN}, {memory_layer.WEIGHT_MAX}] inclusive"
                ),
            )

    for entry in body.preferences:
        try:
            override_preference(session, uid, entry.topic, entry.weight)
        except PreferenceOverrideError as exc:  # defensive: bounds re-checked
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except UnknownPreferenceTopicError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"no stored preference for topic {entry.topic!r}",
            ) from exc

    return PreferencesDTO(preferences=get_preferences(session, uid))
