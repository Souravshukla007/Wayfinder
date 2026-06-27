"""Property test for Property 19: loaded preferences are not re-requested.

Feature: wayfinder-travel-planner, Property 19

*For any* user with a stored preference vector, after the planning context is
loaded, no topic already present in the vector SHALL be re-requested from the
user during that planning session.

Validates: Requirements 10.1

There is no literal "ask the user" mechanism in the deterministic core, so a
topic is treated as "re-requested" exactly when it is *absent* from the loaded
``MemoryContext`` — i.e. the planning session has no value for it and would have
to ask. The property therefore drives the real ``load_context`` against an
in-memory SQLite database (mirroring the memory-layer unit tests) over randomly
generated stored preference vectors plus liked/disliked destinations, and
asserts that every stored topic is present in the loaded context (and every
liked/disliked place round-trips), so nothing already known is re-requested.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.memory.memory_layer import load_context
from app.models.db import (
    Base,
    DislikedDestination,
    LikedDestination,
    User,
    UserPreference,
)
from app.models.schemas import FEATURES

NOW = datetime(2024, 10, 1, 12, 0, tzinfo=timezone.utc)

# Topics drawn from the required feature set plus a couple of extra names so the
# stored vector exercises both required and novel (custom) topics.
_TOPICS = list(FEATURES) + ["custom_topic_a", "custom_topic_b"]

# A stored signal: an in-range weight and an age (days) so decay varies across
# examples. Decay never changes a topic's presence, only its magnitude.
_weight = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_age_days = st.floats(min_value=0.0, max_value=720.0, allow_nan=False, allow_infinity=False)

# A NON-EMPTY stored preference vector keyed by topic -> (weight, age_days).
_stored_vector = st.dictionaries(
    st.sampled_from(_TOPICS),
    st.tuples(_weight, _age_days),
    min_size=1,
    max_size=len(_TOPICS),
)

_places = st.lists(
    st.text(min_size=1, max_size=20).filter(lambda s: s.strip() != ""),
    max_size=5,
    unique=True,
)


def _new_session() -> Session:
    """Create a fresh in-memory SQLite session with the full ORM schema."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return factory()


# ---------------------------------------------------------------------------
# Property 19: Loaded preferences are not re-requested
# Feature: wayfinder-travel-planner, Property 19
# Validates: Requirements 10.1
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(stored=_stored_vector, liked=_places, disliked=_places)
def test_loaded_preferences_are_not_rerequested(stored, liked, disliked) -> None:
    """Property 19: every stored topic is present after ``load_context``.

    Feature: wayfinder-travel-planner, Property 19
    Validates: Requirements 10.1
    """
    session = _new_session()
    try:
        user = User(email="prop19@example.com", prefs={})
        session.add(user)
        session.commit()

        for topic, (weight, age_days) in stored.items():
            session.add(
                UserPreference(
                    user_id=user.id,
                    topic=topic,
                    weight=weight,
                    source="explicit",
                    updated_at=NOW - timedelta(days=age_days),
                )
            )
        for place in liked:
            session.add(LikedDestination(user_id=user.id, place=place, updated_at=NOW))
        for place in disliked:
            session.add(
                DislikedDestination(user_id=user.id, place=place, updated_at=NOW)
            )
        session.commit()

        ctx = load_context(session, user.id, now=NOW)

        available = set(ctx.topic_weights())
        stored_topics = set(stored)

        # The core invariant: no topic already present in the stored vector is
        # missing from the loaded context, so none would be re-requested.
        re_requested = stored_topics - available
        assert not re_requested, (
            f"topics already stored but absent from loaded context "
            f"(would be re-requested): {sorted(re_requested)}"
        )

        # Every loaded weight is finite and in range, so it is a usable value the
        # session can rely on instead of asking the user again.
        for entry in ctx.preferences:
            assert math.isfinite(entry.weight)
            assert 0.0 <= entry.weight <= 1.0

        # Liked/disliked destination knowledge is loaded too, so the session does
        # not re-ask about places the user already favored or rejected.
        assert set(ctx.liked) == set(liked)
        assert set(ctx.disliked) == set(disliked)

        # Reloading within the same session is stable: a stored topic never
        # disappears (and so never becomes re-requestable) on a subsequent load.
        ctx_again = load_context(session, user.id, now=NOW)
        assert stored_topics <= set(ctx_again.topic_weights())
    finally:
        session.close()
