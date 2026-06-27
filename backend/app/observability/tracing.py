"""Agent-run tracing: persist one ``agent_runs`` record per agent run.

Requirement 15.1: WHEN an agent run executes, THE Wayfinder_System SHALL store a
record in the ``agent_runs`` store containing the agent name, input, output,
token usage, latency, and a trace identifier.

This module provides an ergonomic API over the :class:`app.models.db.AgentRun`
ORM model:

* :func:`record_agent_run` — a direct function that writes a single complete
  record. Callers that already know the latency/output use this.
* :func:`trace_agent_run` — a context manager that times the wrapped block,
  collects the output and token usage set on the yielded recorder, and writes
  the record on exit (including when the block raises, so failures are still
  observable).

Token usage accepts the :class:`app.llm.base.LLMResponse` usage fields
(``prompt_tokens`` / ``completion_tokens``); see :func:`coerce_token_usage`.

Schema note (alignment, not silent divergence): the ``agent_runs`` table stores
a single aggregate ``tokens`` integer column — there are no separate
``prompt_tokens`` / ``completion_tokens`` columns. This module therefore records
the *total* token usage (prompt + completion). If a per-direction breakdown is
ever required, that needs a schema change to ``app/models/db.py`` rather than a
workaround here.

The functions take an existing SQLAlchemy :class:`~sqlalchemy.orm.Session`
(wired in task 2.3 via ``app.models.database.get_session``); this module performs
no engine/session construction of its own, keeping it decoupled from settings.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from sqlalchemy.orm import Session

from app.llm.base import LLMResponse
from app.models.db import AgentRun

__all__ = [
    "RunRecorder",
    "coerce_token_usage",
    "new_trace_id",
    "record_agent_run",
    "trace_agent_run",
]


def new_trace_id() -> str:
    """Return a fresh trace identifier (hex UUID, fits ``trace_id`` String(80))."""
    return uuid.uuid4().hex


def coerce_token_usage(token_usage: Any) -> int:
    """Normalize a token-usage value to a non-negative total token count.

    Accepts:

    * ``None`` -> ``0``
    * an :class:`~app.llm.base.LLMResponse` (or anything exposing
      ``prompt_tokens`` / ``completion_tokens`` attributes) -> their sum
    * a mapping with ``prompt_tokens`` / ``completion_tokens`` (and/or
      ``total_tokens``) keys -> the total
    * an ``int`` -> itself (already a total)
    """
    if token_usage is None:
        return 0

    if isinstance(token_usage, bool):  # guard: bool is an int subclass
        return int(token_usage)

    if isinstance(token_usage, int):
        return max(0, token_usage)

    if isinstance(token_usage, LLMResponse):
        return max(0, token_usage.total_tokens)

    if isinstance(token_usage, Mapping):
        if "total_tokens" in token_usage and token_usage["total_tokens"] is not None:
            return max(0, int(token_usage["total_tokens"]))
        prompt = int(token_usage.get("prompt_tokens", 0) or 0)
        completion = int(token_usage.get("completion_tokens", 0) or 0)
        return max(0, prompt + completion)

    # Duck-typed object exposing the LLMResponse usage fields.
    total = getattr(token_usage, "total_tokens", None)
    if total is not None:
        return max(0, int(total))
    prompt = int(getattr(token_usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(token_usage, "completion_tokens", 0) or 0)
    return max(0, prompt + completion)


def record_agent_run(
    session: Session,
    *,
    trip_id: str,
    agent: str,
    input: Mapping[str, Any] | None,
    output: Mapping[str, Any] | None,
    latency: float,
    token_usage: Any = 0,
    trace_id: str | None = None,
    commit: bool = True,
) -> AgentRun:
    """Persist a single complete ``agent_runs`` record and return it.

    Every field required by Requirement 15.1 is populated: ``agent``, ``input``,
    ``output``, ``tokens`` (total token usage), ``latency``, and ``trace_id``
    (generated when not supplied). ``id`` and ``created_at`` are filled by the
    model/DB defaults; when ``commit`` is true the row is refreshed so
    ``created_at`` is loaded onto the returned object.
    """
    run = AgentRun(
        trip_id=trip_id,
        agent=agent,
        input=dict(input) if input is not None else {},
        output=dict(output) if output is not None else {},
        tokens=coerce_token_usage(token_usage),
        latency=float(latency),
        trace_id=trace_id or new_trace_id(),
    )
    session.add(run)
    if commit:
        session.commit()
        session.refresh(run)
    else:
        session.flush()
    return run


@dataclass
class RunRecorder:
    """Mutable handle yielded by :func:`trace_agent_run`.

    The wrapped block sets the ``output`` and ``token_usage`` it produced; the
    context manager reads them when writing the record on exit.
    """

    agent: str
    trip_id: str
    trace_id: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    token_usage: Any = 0

    def set_output(self, output: Mapping[str, Any]) -> None:
        self.output = dict(output)

    def set_token_usage(self, token_usage: Any) -> None:
        self.token_usage = token_usage

    def record(self, *, output: Mapping[str, Any], token_usage: Any = None) -> None:
        """Convenience: set output and (optionally) token usage in one call."""
        self.set_output(output)
        if token_usage is not None:
            self.set_token_usage(token_usage)


@contextmanager
def trace_agent_run(
    session: Session,
    *,
    trip_id: str,
    agent: str,
    input: Mapping[str, Any] | None = None,
    trace_id: str | None = None,
    commit: bool = True,
) -> Iterator[RunRecorder]:
    """Time an agent run and persist its ``agent_runs`` record on exit.

    Usage::

        with trace_agent_run(session, trip_id=trip_id, agent="coordinator",
                             input={"prompt": prompt}) as run:
            response = llm.complete(prompt)
            run.record(output={"text": response.text}, token_usage=response)

    The record is written even if the wrapped block raises, so failed runs
    remain observable; the original exception then propagates.
    """
    recorder = RunRecorder(
        agent=agent,
        trip_id=trip_id,
        trace_id=trace_id or new_trace_id(),
        input=dict(input) if input is not None else {},
    )
    start = perf_counter()
    try:
        yield recorder
    finally:
        latency = perf_counter() - start
        record_agent_run(
            session,
            trip_id=recorder.trip_id,
            agent=recorder.agent,
            input=recorder.input,
            output=recorder.output,
            latency=latency,
            token_usage=recorder.token_usage,
            trace_id=recorder.trace_id,
            commit=commit,
        )
