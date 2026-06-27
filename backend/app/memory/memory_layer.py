"""Travel Memory Layer: per-user weighted preference vector with time decay.

This module owns reading and writing the personalization signals that drive the
Decision Engine's personalized weights. It is intentionally structured so the
three Memory Layer operations live side by side:

- ``load_context``    (task 10.1, Requirements 10.1, 10.4, 10.5) — implemented here.
- ``write_signals``   (task 10.2, Requirements 10.2, 10.6) — added alongside.
- ``get_preferences`` / ``override_preference`` (task 10.3) — added alongside.

``load_context`` loads a user's stored preferences (plus liked/disliked
destinations) into the planning context, returning within a 2 second budget so
that no preference already known is re-requested from the user (Requirement
10.1). When the user has no stored preferences it initializes the predefined
cold-start default set so every required topic has a starting weight
(Requirement 10.4). Each stored signal carries an ``updated_at`` timestamp; an
effective weight is produced by applying a 180-day half-life time decay
(Requirement 10.5):

    decayed = weight * 0.5 ** (age_days / 180)

The decay reference time ("now") is injectable so the function is pure and the
half-life is exactly testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.models.db import DislikedDestination, LikedDestination, UserPreference
from app.models.schemas import FEATURES, PreferenceEntry

# Half-life for preference time decay, in days (Requirement 10.5).
HALF_LIFE_DAYS = 180.0

# Performance budget for loading the planning context (Requirement 10.1).
LOAD_CONTEXT_BUDGET_SECONDS = 2.0

# Performance budget for writing signals on trip completion (Requirement 10.2).
WRITE_SIGNALS_BUDGET_SECONDS = 60.0

# Relative weights applied when combining an explicit stated signal and an
# implicit inferred signal for the SAME topic (Requirement 10.6 / Property 17).
# Explicit stated preferences MUST be weighted at least twice (>= 2x) the
# implicit inferred ones. These are named constants so the >= 2x ratio is
# explicit in the code and directly assertable by the Property 17 test (10.6).
EXPLICIT_SIGNAL_WEIGHT = 2.0
IMPLICIT_SIGNAL_WEIGHT = 1.0
# The realized explicit-to-implicit weighting ratio. Kept as a derived constant
# (== 2.0) so callers/tests can assert the invariant EXPLICIT >= 2 * IMPLICIT
# without re-deriving it.
EXPLICIT_TO_IMPLICIT_RATIO = EXPLICIT_SIGNAL_WEIGHT / IMPLICIT_SIGNAL_WEIGHT

# Source label applied to the cold-start defaults. These are system-seeded
# starting weights rather than user-stated preferences, so they are recorded as
# implicit signals.
_DEFAULT_SOURCE = "implicit"

# Inclusive bounds for a stored / overridable preference weight (Requirement
# 10.3, 10.8, 10.9). A weight is valid iff ``WEIGHT_MIN <= weight <= WEIGHT_MAX``.
WEIGHT_MIN = 0.0
WEIGHT_MAX = 1.0


class PreferenceOverrideError(ValueError):
    """Raised when a preference override weight is outside [0.0, 1.0] (Req 10.9).

    The offending value is retained on the exception as ``value`` so a caller
    can surface an error indication that *identifies the invalid value*, and the
    ``topic`` it was submitted for is captured for context. The prior stored
    weight is left unchanged because validation happens before any mutation.
    """

    def __init__(self, value: object, *, topic: str | None = None) -> None:
        self.value = value
        self.topic = topic
        super().__init__(
            f"preference override weight {value!r} for topic {topic!r} is out of "
            f"range; it must be within [{WEIGHT_MIN}, {WEIGHT_MAX}] inclusive"
        )


class UnknownPreferenceTopicError(LookupError):
    """Raised when an override targets a topic the user has no stored weight for.

    Override replaces the weight of an *existing* stored preference (Requirement
    10.8); there is nothing to replace for an unknown topic.
    """

    def __init__(self, topic: str, *, user_id: object | None = None) -> None:
        self.topic = topic
        self.user_id = user_id
        super().__init__(f"no stored preference for topic {topic!r}")


def _is_unit_weight(value: float) -> bool:
    """Return True iff ``value`` is a real number within [0.0, 1.0] inclusive.

    Uses a single chained comparison so non-finite values (``nan``, ``inf``) are
    correctly treated as out of range (``nan`` comparisons are always False).
    """
    return WEIGHT_MIN <= value <= WEIGHT_MAX


@dataclass
class MemoryContext:
    """The personalization context loaded for a planning session.

    ``preferences`` carries the user's preference vector with **time-decayed**
    effective weights (Requirement 10.5). ``liked``/``disliked`` are the place
    names the user has previously favored or rejected.
    """

    user_id: str
    preferences: list[PreferenceEntry] = field(default_factory=list)
    liked: list[str] = field(default_factory=list)
    disliked: list[str] = field(default_factory=list)

    def topic_weights(self) -> dict[str, float]:
        """Return the decayed preference vector as a ``{topic: weight}`` map.

        This is the shape the Decision Engine's personalization step consumes.
        """
        return {p.topic: p.weight for p in self.preferences}


def default_preferences(
    settings: Settings | None = None,
    *,
    now: datetime | None = None,
) -> list[PreferenceEntry]:
    """Build the predefined cold-start default preference set (Requirement 10.4).

    Every required preference topic — the six Decision Engine features — is
    given a starting weight from the configured base feature weights, stamped at
    ``now`` so its decay factor starts at 1.0.
    """
    settings = settings or get_settings()
    now = now or datetime.now(timezone.utc)
    base = settings.base_feature_weights()
    return [
        PreferenceEntry(
            topic=feature,
            weight=base[feature],
            source=_DEFAULT_SOURCE,
            updated_at=now,
        )
        for feature in FEATURES
    ]


def _as_utc(moment: datetime) -> datetime:
    """Coerce a possibly-naive timestamp to timezone-aware UTC.

    Storage backends such as SQLite drop timezone info, returning naive
    datetimes. Treat those as UTC so decay arithmetic against an aware ``now``
    is well defined.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def decay_factor(updated_at: datetime, now: datetime, *, half_life_days: float = HALF_LIFE_DAYS) -> float:
    """Return the 180-day half-life decay multiplier for a signal (Req 10.5).

    ``factor = 0.5 ** (age_days / half_life_days)``. A signal exactly one
    half-life old yields 0.5; a freshly stamped signal yields 1.0. Future-dated
    timestamps (age < 0) are clamped to a factor of 1.0 so decay never amplifies
    a weight beyond its stored value.
    """
    age_seconds = (_as_utc(now) - _as_utc(updated_at)).total_seconds()
    age_days = age_seconds / 86_400.0
    if age_days <= 0.0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def _decayed_entry(pref: UserPreference, now: datetime) -> PreferenceEntry:
    """Convert a stored ``UserPreference`` row into a decayed ``PreferenceEntry``."""
    updated_at = pref.updated_at or now
    factor = decay_factor(updated_at, now)
    effective = pref.weight * factor
    # Decay only shrinks weights, but clamp defensively so the result always
    # satisfies the PreferenceEntry [0, 1] invariant.
    effective = min(1.0, max(0.0, effective))
    source = pref.source if pref.source in ("explicit", "implicit") else "implicit"
    return PreferenceEntry(
        topic=pref.topic,
        weight=effective,
        source=source,  # type: ignore[arg-type]
        updated_at=_as_utc(updated_at),
    )


def load_context(
    session: Session,
    user_id: str,
    *,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> MemoryContext:
    """Load a user's personalization context for a planning session.

    Behavior:
    - Reads the stored preference vector and the liked/disliked destination
      lists (Requirement 10.1) — once loaded these are not re-requested.
    - Cold-start (Requirement 10.4): when the user has no stored preferences,
      the predefined default set (every required topic) is initialized,
      persisted, and returned so subsequent loads are stable.
    - Applies a 180-day half-life time decay to each preference weight using its
      ``updated_at`` timestamp (Requirement 10.5).

    ``now`` is injectable (defaulting to the current UTC time) so the decay and
    half-life are exactly testable.
    """
    settings = settings or get_settings()
    now = now or datetime.now(timezone.utc)

    stored = list(
        session.scalars(
            select(UserPreference).where(UserPreference.user_id == user_id)
        )
    )

    if not stored:
        # Cold-start: seed the predefined default preference set so every
        # required topic has a starting weight, then persist it for durability.
        defaults = default_preferences(settings, now=now)
        session.add_all(
            UserPreference(
                user_id=user_id,
                topic=entry.topic,
                weight=entry.weight,
                source=entry.source,
                updated_at=entry.updated_at,
            )
            for entry in defaults
        )
        session.commit()
        preferences = defaults
    else:
        preferences = [_decayed_entry(pref, now) for pref in stored]

    liked = list(
        session.scalars(
            select(LikedDestination.place).where(LikedDestination.user_id == user_id)
        )
    )
    disliked = list(
        session.scalars(
            select(DislikedDestination.place).where(
                DislikedDestination.user_id == user_id
            )
        )
    )

    return MemoryContext(
        user_id=user_id,
        preferences=preferences,
        liked=liked,
        disliked=disliked,
    )


# --------------------------------------------------------------------------- #
# Signal writing (task 10.2). Preference read/override (task 10.3) is
# implemented alongside ``load_context``/``write_signals`` in this module.
# --------------------------------------------------------------------------- #


def _clamp_unit(value: float) -> float:
    """Clamp a raw signal value into the PreferenceEntry [0, 1] range."""
    return min(1.0, max(0.0, float(value)))


def combine_signal(
    explicit_value: float | None = None,
    implicit_value: float | None = None,
) -> float:
    """Combine a topic's explicit and implicit raw signals into one [0, 1] weight.

    The explicit stated signal is weighted ``EXPLICIT_SIGNAL_WEIGHT`` and the
    implicit inferred signal ``IMPLICIT_SIGNAL_WEIGHT`` — a ratio of
    ``EXPLICIT_TO_IMPLICIT_RATIO`` (== 2.0, i.e. >= 2x), satisfying Requirement
    10.6 / Property 17. When both signals are present for the same topic the
    result is their weight-normalized average ``(2*e + i) / 3``, so the explicit
    signal pulls the combined weight exactly twice as hard as the implicit one.
    When only one signal is present, that signal's (clamped) value is used
    directly. The weighted average of values in [0, 1] is itself in [0, 1].
    """
    has_explicit = explicit_value is not None
    has_implicit = implicit_value is not None
    if not has_explicit and not has_implicit:
        raise ValueError("combine_signal requires an explicit and/or implicit value")

    e = _clamp_unit(explicit_value) if has_explicit else 0.0
    i = _clamp_unit(implicit_value) if has_implicit else 0.0
    w_e = EXPLICIT_SIGNAL_WEIGHT if has_explicit else 0.0
    w_i = IMPLICIT_SIGNAL_WEIGHT if has_implicit else 0.0
    return (w_e * e + w_i * i) / (w_e + w_i)


def write_signals(
    session: Session,
    trip_id: str | None,
    user_id: str,
    explicit: dict[str, float] | None,
    implicit: dict[str, float] | None,
    *,
    now: datetime | None = None,
) -> None:
    """Write updated preference signals on trip completion (Requirement 10.2).

    On trip completion the Memory Layer derives preference signals from two
    sources and persists them to ``user_preferences``:

    - ``explicit`` — stated preferences / trip feedback (ratings, stated
      likes/dislikes), keyed ``topic -> value`` with value in [0, 1].
    - ``implicit`` — signals inferred from observed behavior (selections, views,
      skips during the trip), keyed ``topic -> value`` with value in [0, 1].

    For each topic the explicit and implicit signals are merged via
    :func:`combine_signal`, which weights the explicit signal at least twice the
    implicit one for the same topic (Requirement 10.6). Each written row is
    stamped with ``updated_at = now`` (injectable for testability) and the
    correct ``source``: ``'explicit'`` when an explicit signal contributed to
    the topic (explicit takes precedence), otherwise ``'implicit'``. An existing
    row for the ``(user_id, topic)`` pair is updated in place; otherwise a new
    row is inserted. The whole operation is a small bounded set of upserts and
    completes well within the 60 second budget (``WRITE_SIGNALS_BUDGET_SECONDS``).

    ``trip_id`` identifies the completed trip the signals are derived from; it is
    accepted to match the Memory Layer contract and is not itself persisted on
    the preference vector (which is keyed by user and topic).
    """
    now = now or datetime.now(timezone.utc)

    # Drop missing (``None``) values so presence-by-key is unambiguous and the
    # resolved ``source`` matches the signals that actually contributed.
    explicit_signals = {k: v for k, v in (explicit or {}).items() if v is not None}
    implicit_signals = {k: v for k, v in (implicit or {}).items() if v is not None}

    topics = set(explicit_signals) | set(implicit_signals)
    if not topics:
        return

    existing = {
        pref.topic: pref
        for pref in session.scalars(
            select(UserPreference).where(
                UserPreference.user_id == user_id,
                UserPreference.topic.in_(topics),
            )
        )
    }

    for topic in sorted(topics):
        explicit_value = explicit_signals.get(topic)
        implicit_value = implicit_signals.get(topic)
        weight = combine_signal(explicit_value, implicit_value)
        # Explicit stated preferences take precedence for the recorded source.
        source = "explicit" if topic in explicit_signals else "implicit"

        row = existing.get(topic)
        if row is None:
            session.add(
                UserPreference(
                    user_id=user_id,
                    topic=topic,
                    weight=weight,
                    source=source,
                    updated_at=now,
                )
            )
        else:
            row.weight = weight
            row.source = source
            row.updated_at = now

    session.commit()


# --------------------------------------------------------------------------- #
# Preference read & override (task 10.3, Requirements 10.3, 10.7, 10.8, 10.9).
# --------------------------------------------------------------------------- #


def get_preferences(session: Session, user_id: str) -> list[PreferenceEntry]:
    """Return the user's current stored Preference_Vector (Requirement 10.7).

    Each entry carries its ``(topic, weight)`` pair and the ``updated_at``
    timestamp of the stored signal. The **stored** weight is returned as-is —
    i.e. *not* time-decayed — because this is the raw vector a user views and
    overrides; the ``updated_at`` timestamp is what makes the decay applied by
    :func:`load_context` reproducible. Entries are returned in a stable
    topic-sorted order so the view is deterministic.
    """
    rows = session.scalars(
        select(UserPreference).where(UserPreference.user_id == user_id)
    )
    entries = [
        PreferenceEntry(
            topic=row.topic,
            weight=row.weight,
            source=row.source if row.source in ("explicit", "implicit") else "implicit",  # type: ignore[arg-type]
            updated_at=_as_utc(row.updated_at) if row.updated_at else None,
        )
        for row in rows
    ]
    entries.sort(key=lambda entry: entry.topic)
    return entries


def override_preference(
    session: Session,
    user_id: str,
    topic: str,
    weight: float,
    *,
    now: datetime | None = None,
) -> PreferenceEntry:
    """Override a stored preference weight with a user-supplied value.

    Behavior:
    - In range (Requirement 10.8): when ``weight`` is within [0.0, 1.0]
      inclusive, the stored weight for ``(user_id, topic)`` is replaced with the
      user-supplied value, its ``updated_at`` is re-stamped to ``now`` (an
      explicit user-stated override, so ``source`` becomes ``'explicit'``), and
      the updated :class:`PreferenceEntry` is returned.
    - Out of range (Requirement 10.9): when ``weight`` is outside [0.0, 1.0]
      (including non-finite values such as ``nan``/``inf``), the override is
      rejected by raising :class:`PreferenceOverrideError`. Validation happens
      *before* any mutation, so the previously stored weight is retained
      unchanged, and the raised error identifies the invalid value.

    Raises :class:`UnknownPreferenceTopicError` when the user has no stored
    preference for ``topic`` (there is no stored weight to replace).

    ``now`` is injectable (defaulting to the current UTC time) so the re-stamped
    timestamp is exactly testable.
    """
    now = now or datetime.now(timezone.utc)

    # Reject out-of-range values up front so the prior stored weight is never
    # touched and the error identifies the offending value (Requirement 10.9).
    if not _is_unit_weight(weight):
        raise PreferenceOverrideError(weight, topic=topic)

    row = session.scalars(
        select(UserPreference).where(
            UserPreference.user_id == user_id,
            UserPreference.topic == topic,
        )
    ).first()
    if row is None:
        raise UnknownPreferenceTopicError(topic, user_id=user_id)

    # Replace the stored weight and re-stamp (Requirement 10.8).
    row.weight = float(weight)
    row.source = "explicit"
    row.updated_at = now
    session.commit()

    return PreferenceEntry(
        topic=row.topic,
        weight=row.weight,
        source="explicit",
        updated_at=_as_utc(row.updated_at),
    )


__all__ = [
    "HALF_LIFE_DAYS",
    "LOAD_CONTEXT_BUDGET_SECONDS",
    "WRITE_SIGNALS_BUDGET_SECONDS",
    "EXPLICIT_SIGNAL_WEIGHT",
    "IMPLICIT_SIGNAL_WEIGHT",
    "EXPLICIT_TO_IMPLICIT_RATIO",
    "WEIGHT_MIN",
    "WEIGHT_MAX",
    "PreferenceOverrideError",
    "UnknownPreferenceTopicError",
    "MemoryContext",
    "default_preferences",
    "decay_factor",
    "load_context",
    "combine_signal",
    "write_signals",
    "get_preferences",
    "override_preference",
]
