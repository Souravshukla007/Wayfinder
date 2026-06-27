"""Property-based test for complete agent-run records (task 17.2).

Covers:
- **Property 28: Agent-run records are complete** (Task 17.2) - for any agent
  run that executes, the persisted ``agent_runs`` record contains every field
  required by Requirement 15.1: agent name, input, output, token usage,
  latency, and a trace identifier (none missing/null).
  Validates: Requirements 15.1.
  Tagged: Feature: wayfinder-travel-planner, Property 28.

This is a property test only. It drives a wide space of agent runs (varied agent
names, input/output payloads, token-usage shapes, and latencies) through the
real tracing API -- both ``record_agent_run`` and the ``trace_agent_run``
context manager -- into an in-memory SQLite store, reads the rows back, and
asserts no required field is missing or null. A counterexample here indicates a
genuine observability gap (an agent run persisted without complete trace data)
to report rather than mask.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.llm.base import LLMResponse
from app.models.db import AgentRun, Base, Trip, User
from app.observability.tracing import record_agent_run, trace_agent_run

# ---------------------------------------------------------------------------
# Property 28: Agent-run records are complete
# Feature: wayfinder-travel-planner, Property 28
# Validates: Requirements 15.1
# ---------------------------------------------------------------------------


@event.listens_for(Engine, "connect")
def _enable_sqlite_fk(dbapi_connection, _record) -> None:  # pragma: no cover - glue
    """Enforce SQLite foreign keys so the schema behaves like Postgres."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def _fresh_session() -> tuple[Session, Engine]:
    """Build an isolated in-memory database + session for one example.

    Hypothesis drives many examples through a single test function, so each
    example needs its own database to stay independent. The caller disposes of
    the engine when done.
    """
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return factory(), engine


def _make_trip(db: Session) -> Trip:
    """Persist a user + trip so the agent_runs FK resolves."""
    user = User(email="traveler@example.com", prefs={})
    db.add(user)
    db.commit()
    trip = Trip(user_id=user.id, origin="DEL", status="processing")
    db.add(trip)
    db.commit()
    return trip


def _assert_complete(run: AgentRun) -> None:
    """Assert every Requirement-15.1 field is present and non-null.

    Required by Property 28: agent name, input, output, token usage, latency,
    and a trace identifier -- none missing/null.
    """
    # agent name
    assert run.agent is not None
    assert run.agent != ""
    # input
    assert run.input is not None
    # output
    assert run.output is not None
    # token usage
    assert run.tokens is not None
    assert run.tokens >= 0
    # latency
    assert run.latency is not None
    assert run.latency >= 0.0
    # trace identifier
    assert run.trace_id is not None
    assert run.trace_id != ""


# --- input strategies ------------------------------------------------------

# Agent names: non-empty (model column is String(80)); cover the real agent
# names plus arbitrary non-empty labels.
_agent_name = st.one_of(
    st.sampled_from(["coordinator", "destination", "itinerary"]),
    st.text(min_size=1, max_size=80).filter(lambda s: s.strip() != ""),
)

# JSON-serializable payloads for input/output, including the empty payload.
_json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=40),
)
_json_payload = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.one_of(_json_scalar, st.lists(_json_scalar, max_size=4)),
    max_size=6,
)

# Token usage spans every shape ``coerce_token_usage`` accepts, including
# negatives and ``None`` (which must coerce to a valid non-negative total).
_token_usage = st.one_of(
    st.none(),
    st.integers(min_value=-500, max_value=500_000),
    st.builds(
        lambda p, c: LLMResponse(
            text="x", model="mock-llm", prompt_tokens=p, completion_tokens=c
        ),
        st.integers(min_value=0, max_value=100_000),
        st.integers(min_value=0, max_value=100_000),
    ),
    st.fixed_dictionaries(
        {
            "prompt_tokens": st.integers(min_value=0, max_value=100_000),
            "completion_tokens": st.integers(min_value=0, max_value=100_000),
        }
    ),
    st.fixed_dictionaries({"total_tokens": st.integers(min_value=0, max_value=200_000)}),
)

_latency = st.floats(min_value=0.0, max_value=86_400.0, allow_nan=False, allow_infinity=False)

# trace_id: either explicitly supplied (non-empty) or omitted so the tracer
# auto-generates one. Either way the persisted record must carry a trace id.
_trace_id = st.one_of(st.none(), st.text(min_size=1, max_size=80).filter(lambda s: s.strip() != ""))


@settings(max_examples=150, deadline=None)
@given(
    agent=_agent_name,
    input_payload=_json_payload,
    output_payload=_json_payload,
    token_usage=_token_usage,
    latency=_latency,
    trace_id=_trace_id,
)
def test_direct_record_agent_run_is_complete(
    agent: str,
    input_payload: dict,
    output_payload: dict,
    token_usage: object,
    latency: float,
    trace_id: str | None,
) -> None:
    """``record_agent_run`` persists a complete record for any agent run."""
    db, engine = _fresh_session()
    try:
        trip = _make_trip(db)
        run = record_agent_run(
            db,
            trip_id=trip.id,
            agent=agent,
            input=input_payload,
            output=output_payload,
            latency=latency,
            token_usage=token_usage,
            trace_id=trace_id,
        )
        db.expire_all()

        fetched = db.scalars(
            select(AgentRun).where(AgentRun.trip_id == trip.id)
        ).one()
        assert fetched.id == run.id
        _assert_complete(fetched)
    finally:
        db.close()
        engine.dispose()


@settings(max_examples=150, deadline=None)
@given(
    agent=_agent_name,
    input_payload=_json_payload,
    output_payload=_json_payload,
    token_usage=_token_usage,
    trace_id=_trace_id,
    raises=st.booleans(),
)
def test_traced_agent_run_is_complete(
    agent: str,
    input_payload: dict,
    output_payload: dict,
    token_usage: object,
    trace_id: str | None,
    raises: bool,
) -> None:
    """``trace_agent_run`` persists a complete record even when the block fails.

    Failed runs must remain observable, so completeness holds whether the
    wrapped block returns normally or raises.
    """
    db, engine = _fresh_session()
    try:
        trip = _make_trip(db)

        if raises:
            try:
                with trace_agent_run(
                    db, trip_id=trip.id, agent=agent, input=input_payload, trace_id=trace_id
                ) as recorder:
                    recorder.set_token_usage(token_usage)
                    raise ValueError("agent exploded")
            except ValueError:
                pass
        else:
            with trace_agent_run(
                db, trip_id=trip.id, agent=agent, input=input_payload, trace_id=trace_id
            ) as recorder:
                recorder.record(output=output_payload, token_usage=token_usage)

        db.expire_all()
        fetched = db.scalars(
            select(AgentRun).where(AgentRun.trip_id == trip.id)
        ).one()
        _assert_complete(fetched)
    finally:
        db.close()
        engine.dispose()
