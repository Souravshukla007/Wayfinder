"""Property test for the progress stream's stage events (Task 19.5).

Feature: wayfinder-travel-planner, Property 26

**Property 26: Progress stream emits stage events and never prices.**
*For any* planning run, each agent or tool stage SHALL emit a progress event
when it starts and when it completes, and no emitted event SHALL be a live or
real-time price update.

**Validates: Requirements 13.2, 13.3**

The progress stream relayed over ``WS /api/trips/{tripId}/progress``
(:mod:`app.api.routes_ws`) forwards :class:`app.orchestration.graph.ProgressEvent`
values verbatim from the planning graph. This test drives the *real* planning
graph (mock providers, mock LLM — no network, no paid keys) across a wide,
Hypothesis-generated space of trip prompts and asserts the two halves of the
property hold for every run:

* **Stage coverage (Req 13.2).** Every stage that executes emits exactly one
  ``start`` event followed by one ``complete`` event — start strictly before
  complete, balanced one-to-one. The always-visited stages (coordinator,
  destination, decision engine, tools, solver, merge) are present in every run;
  the itinerary stage is present whenever the solver gate admits it.
* **No prices (Req 13.3).** No emitted event carries a live/real-time price: no
  message contains a currency marker, the word "price", or a multi-digit amount,
  and each event exposes only the ``stage`` / ``phase`` / ``message`` fields the
  contract allows.

Running the graph (rather than hand-built events) is deliberate: it validates
the actual production producer of the stream, so the no-prices guarantee is
checked against what the system really emits.
"""

from __future__ import annotations

import re

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.config import Settings
from app.orchestration.graph import STAGES, run_plan
from app.orchestration.state import GraphState

# Stages that always execute (the itinerary stage is gated on solver feasibility
# and so is only asserted when present).
_ALWAYS_VISITED = ("coordinator", "destination", "decision_engine", "tools", "solver", "merge")
_STAGE_KEYS = frozenset(key for key, _ in STAGES)

# A live/real-time price would surface a currency marker, the word "price", or a
# multi-digit amount. No progress message may match this (Requirement 13.3).
_PRICE_PATTERN = re.compile(r"[\u20b9$\u20ac\u00a3]|\bprice\b|\d{3,}", re.IGNORECASE)

# Mock-first config: zero paid keys, network-free, deterministic. Built once and
# reused across examples so each iteration only pays the (sub-30ms) graph cost.
_SETTINGS = Settings(_env_file=None)


# --------------------------------------------------------------------------- #
# Input space: a broad, realistic family of trip-planning prompts.
# --------------------------------------------------------------------------- #
_MONTHS = ["January", "March", "April", "June", "August", "October", "December"]
_DESTINATIONS = ["Japan", "Italy", "Thailand", "Iceland", "Peru", "Vietnam", "Spain"]
_INTERESTS = [
    "anime", "photography", "local food", "hiking", "museums", "beaches",
    "nightlife", "history", "less crowded places", "temples",
]


@st.composite
def trip_prompts(draw: st.DrawFn) -> str:
    """Generate a varied natural-language trip prompt.

    Varies duration, month, budget, destination, and interests so the graph is
    exercised across a wide input space rather than a single fixed prompt.
    """
    days = draw(st.integers(min_value=2, max_value=21))
    month = draw(st.sampled_from(_MONTHS))
    destination = draw(st.sampled_from(_DESTINATIONS))
    budget_lakh = draw(st.integers(min_value=1, max_value=10))
    interests = draw(
        st.lists(st.sampled_from(_INTERESTS), min_size=1, max_size=4, unique=True)
    )
    interest_text = ", ".join(interests)
    return (
        f"Plan a {days}-day {destination} trip in {month} under "
        f"\u20b9{budget_lakh} lakh. I like {interest_text}."
    )


@given(
    prompt=trip_prompts(),
    trip_id=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=24
    ),
    user_id=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=24
    ),
)
@settings(
    max_examples=150,
    deadline=None,  # first invocation JIT-warms LangGraph; per-example cost is tiny.
    suppress_health_check=[HealthCheck.too_slow],
)
def test_progress_stream_emits_balanced_stage_events_and_never_prices(
    prompt: str, trip_id: str, user_id: str
) -> None:
    state = GraphState(trip_id=trip_id, user_id=user_id, prompt=prompt)

    _final_state, events = run_plan(state, settings=_SETTINGS)

    # The run must emit progress for the stream to relay (Req 13.1/13.2 premise).
    assert events, "a planning run must emit progress events"

    # --- Req 13.2: each executed stage emits start-then-complete, balanced. --- #
    by_stage: dict[str, list[str]] = {}
    for event in events:
        assert event.stage in _STAGE_KEYS, f"unknown stage {event.stage!r}"
        assert event.phase in ("start", "complete"), event.phase
        by_stage.setdefault(event.stage, []).append(event.phase)

    for stage in _ALWAYS_VISITED:
        assert stage in by_stage, f"stage {stage!r} did not emit any event"

    for stage, phases in by_stage.items():
        # Exactly one start and one complete, in that order, per executed stage.
        assert phases == ["start", "complete"], f"{stage}: {phases}"

    # --- Req 13.3: no event carries a live/real-time price. ------------------ #
    for event in events:
        assert not _PRICE_PATTERN.search(event.message), (
            f"progress message must not contain a price: {event.message!r}"
        )
        # The serialized event exposes only stage/phase/message — nothing else
        # through which a price could leak.
        assert set(event.model_dump().keys()) == {"stage", "phase", "message"}
