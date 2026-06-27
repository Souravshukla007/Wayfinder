"""Unit tests for agent-run tracing (task 17.1, Requirement 15.1).

Persists agent-run records through both the direct ``record_agent_run`` function
and the ``trace_agent_run`` context manager, reads them back from an in-memory
SQLite store, and asserts every Requirement-15.1 field is populated: agent name,
input, output, token usage, latency, and trace id. Completeness here is what
underpins Property 28 (task 17.2).
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.llm.base import LLMResponse
from app.models.db import AgentRun, Base, Trip, User
from app.observability.tracing import (
    coerce_token_usage,
    new_trace_id,
    record_agent_run,
    trace_agent_run,
)


@pytest.fixture()
def session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    db = factory()
    try:
        yield db
    finally:
        db.close()
        engine.dispose()


@pytest.fixture()
def trip_id(session: Session) -> str:
    user = User(email="traveler@example.com", prefs={})
    session.add(user)
    session.commit()
    trip = Trip(
        user_id=user.id,
        origin="DEL",
        start_date=datetime(2024, 10, 18, tzinfo=timezone.utc),
        end_date=datetime(2024, 10, 25, tzinfo=timezone.utc),
        budget=Decimal("200000.00"),
        interests=["anime", "photography"],
        status="processing",
    )
    session.add(trip)
    session.commit()
    return trip.id


def _assert_complete(run: AgentRun) -> None:
    """Every Requirement-15.1 field (plus id/created_at) is populated."""
    assert run.id is not None
    assert run.trip_id is not None
    assert run.agent
    assert run.input is not None
    assert run.output is not None
    assert run.tokens is not None and run.tokens >= 0
    assert run.latency is not None and run.latency >= 0
    assert run.trace_id
    assert run.created_at is not None


def test_record_agent_run_persists_complete_record(session: Session, trip_id: str) -> None:
    run = record_agent_run(
        session,
        trip_id=trip_id,
        agent="coordinator",
        input={"prompt": "Plan a 7-day Japan trip"},
        output={"candidates": 3},
        latency=2.5,
        token_usage=LLMResponse(
            text="ranked", model="mock-llm", prompt_tokens=120, completion_tokens=45
        ),
        trace_id="trace-abc",
    )

    fetched = session.get(AgentRun, run.id)
    assert fetched is not None
    _assert_complete(fetched)
    assert fetched.trip_id == trip_id
    assert fetched.agent == "coordinator"
    assert fetched.input == {"prompt": "Plan a 7-day Japan trip"}
    assert fetched.output == {"candidates": 3}
    assert fetched.tokens == 165  # prompt + completion
    assert fetched.latency == pytest.approx(2.5)
    assert fetched.trace_id == "trace-abc"


def test_record_agent_run_generates_trace_id(session: Session, trip_id: str) -> None:
    run = record_agent_run(
        session,
        trip_id=trip_id,
        agent="destination",
        input={"k": 1},
        output={"ok": True},
        latency=0.1,
    )
    fetched = session.get(AgentRun, run.id)
    assert fetched is not None
    assert fetched.trace_id  # auto-generated, non-empty
    assert fetched.tokens == 0  # default token usage


def test_trace_agent_run_times_and_persists(session: Session, trip_id: str) -> None:
    response = LLMResponse(
        text="narration", model="mock-llm", prompt_tokens=10, completion_tokens=7
    )
    with trace_agent_run(
        session, trip_id=trip_id, agent="itinerary", input={"prompt": "enrich"}
    ) as run:
        run.record(output={"days": 7}, token_usage=response)

    persisted = session.query(AgentRun).filter_by(agent="itinerary").one()
    _assert_complete(persisted)
    assert persisted.input == {"prompt": "enrich"}
    assert persisted.output == {"days": 7}
    assert persisted.tokens == 17
    assert persisted.latency >= 0.0


def test_trace_agent_run_records_on_exception(session: Session, trip_id: str) -> None:
    """A failing block still writes a complete (observable) record."""
    with pytest.raises(ValueError):
        with trace_agent_run(
            session, trip_id=trip_id, agent="coordinator", input={"prompt": "boom"}
        ) as run:
            run.set_token_usage({"prompt_tokens": 5, "completion_tokens": 3})
            raise ValueError("agent exploded")

    persisted = session.query(AgentRun).filter_by(agent="coordinator").one()
    _assert_complete(persisted)
    assert persisted.tokens == 8
    assert persisted.output == {}  # never set before the failure


def test_coerce_token_usage_variants() -> None:
    assert coerce_token_usage(None) == 0
    assert coerce_token_usage(42) == 42
    assert coerce_token_usage(-5) == 0
    assert coerce_token_usage({"prompt_tokens": 3, "completion_tokens": 4}) == 7
    assert coerce_token_usage({"total_tokens": 11}) == 11
    assert (
        coerce_token_usage(
            LLMResponse(text="x", model="m", prompt_tokens=2, completion_tokens=9)
        )
        == 11
    )


def test_new_trace_id_is_unique() -> None:
    assert new_trace_id() != new_trace_id()
