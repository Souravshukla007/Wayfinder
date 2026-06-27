"""End-to-end smoke tests for the assembled LangGraph pipeline (Task 13.5).

Validates Requirements:
- 3.1: the planning flow is a single graph wiring Coordinator -> tools ->
  Destination -> Decision Engine -> CP-SAT Solver -> Itinerary -> merge, and it
  runs end-to-end on the default mock providers (no paid keys).
- 1.1 / 1.4 / 17.4: every destination, flight, and hotel option that reaches the
  final plan carries a rationale.
- 3.4 (solver gate): the Itinerary node runs only on a feasible skeleton; on a
  solver rejection the graph routes straight to merge and produces no itinerary.
- 13.2 / 13.3 (Property 26): each node emits a start and a complete progress
  event, and no progress event carries a live/real-time price.

These are example-based integration tests on mock data, exercised through the
public ``run_plan`` / ``build_graph`` entry points.
"""

from __future__ import annotations

import re

from app.config import Settings
from app.orchestration.graph import (
    STAGES,
    ProgressEvent,
    ProgressReporter,
    build_graph,
    run_plan,
)
from app.orchestration.state import GraphState

JAPAN_PROMPT = (
    "Plan a 7-day Japan trip in October under \u20b92 lakh. "
    "I like anime, photography, local food, and less crowded places."
)


def _settings() -> Settings:
    # No credentials => mock providers => deterministic, zero-key run.
    return Settings()


def _state(prompt: str = JAPAN_PROMPT, trip_id: str = "trip-1") -> GraphState:
    return GraphState(trip_id=trip_id, user_id="user-1", prompt=prompt)


# --------------------------------------------------------------------------- #
# End-to-end happy path (Requirement 3.1)
# --------------------------------------------------------------------------- #
def test_runs_end_to_end_on_mock_and_produces_explained_plan() -> None:
    final, events = run_plan(_state(), settings=_settings())

    # The full pipeline ran: constraints decomposed, candidates scored, a
    # feasible skeleton found, and the itinerary enriched.
    assert final.constraints is not None
    assert final.scored, "decision engine should have scored candidates"
    assert final.decision_traces, "decision engine should emit traces"
    assert final.skeleton is not None and final.rejection is None
    assert final.itinerary, "feasible skeleton should enrich into a day plan"
    assert final.final_plan is not None
    assert events  # progress was emitted


def test_graph_composition_includes_the_three_agents() -> None:
    run = build_graph(settings=_settings())
    # Coordinator (intake + merge), Destination, and Itinerary agents are wired.
    assert "coordinator" in run.node_names
    assert "destination" in run.node_names
    assert "itinerary" in run.node_names
    assert "merge" in run.node_names


# --------------------------------------------------------------------------- #
# Rationales on every recommendation (Requirements 1.1, 1.4, 17.4)
# --------------------------------------------------------------------------- #
def test_every_recommendation_carries_a_rationale() -> None:
    final, _ = run_plan(_state(), settings=_settings())
    plan = final.final_plan
    assert plan is not None

    assert plan.destinations, "expected ranked destinations"
    assert all(d.rationale.strip() for d in plan.destinations)

    assert plan.flights, "expected mock flights for the chosen city"
    assert all(f.rationale.strip() for f in plan.flights)

    assert plan.hotels, "expected mock hotels for the chosen city"
    assert all(h.rationale.strip() for h in plan.hotels)


# --------------------------------------------------------------------------- #
# Progress hooks: start/complete per node, never a price (Property 26)
# --------------------------------------------------------------------------- #
def test_every_node_emits_start_and_complete_in_order() -> None:
    _, events = run_plan(_state(), settings=_settings())

    # Each visited stage emits exactly a start then a complete event.
    for stage in ("coordinator", "tools", "destination", "decision_engine", "solver", "merge"):
        phases = [e.phase for e in events if e.stage == stage]
        assert phases == ["start", "complete"], f"{stage}: {phases}"

    # On a feasible run the itinerary node also runs start->complete.
    itin = [e.phase for e in events if e.stage == "itinerary"]
    assert itin == ["start", "complete"]

    # Completion of one stage precedes the start of the next (ordered stream).
    order = [e.stage for e in events]
    assert order.index("coordinator") < order.index("solver") < order.index("merge")


def test_progress_events_never_contain_a_price() -> None:
    _, events = run_plan(_state(), settings=_settings())
    # Only progress events are streamed; never live/real-time prices.
    price_pattern = re.compile(r"[\u20b9$]|\bprice\b|\d{3,}", re.IGNORECASE)
    for event in events:
        assert not price_pattern.search(event.message), event.message


def test_progress_callback_is_invoked_live() -> None:
    seen: list[ProgressEvent] = []
    reporter = ProgressReporter(seen.append)
    run_plan(_state(), settings=_settings(), progress=reporter)
    # The callback fired for every recorded event, in the same order.
    assert seen == reporter.events
    assert [s for s, _ in STAGES][0] == "coordinator"


# --------------------------------------------------------------------------- #
# Solver gate: infeasible => no itinerary, routed straight to merge (Req 3.4)
# --------------------------------------------------------------------------- #
def test_infeasible_plan_skips_itinerary_and_still_merges() -> None:
    # A 1-day trip under an impossibly small budget cannot fit any city.
    final, events = run_plan(
        _state(prompt="A 1-day trip under \u20b91000", trip_id="trip-2"),
        settings=_settings(),
    )

    assert final.rejection is not None
    assert final.skeleton is None
    assert final.itinerary == []
    # The itinerary node never ran; the graph routed solver -> merge.
    assert not any(e.stage == "itinerary" for e in events)
    # Merge still produced a plan (with no itinerary).
    assert final.final_plan is not None
    assert final.final_plan.itinerary == []


# --------------------------------------------------------------------------- #
# Determinism (mock providers, pure engine/solver)
# --------------------------------------------------------------------------- #
def test_runs_are_deterministic() -> None:
    first, first_events = run_plan(_state(), settings=_settings())
    second, second_events = run_plan(_state(), settings=_settings())

    assert [d.name for d in first.final_plan.destinations] == [
        d.name for d in second.final_plan.destinations
    ]
    assert [(d.day, d.city, tuple(d.activities)) for d in first.itinerary] == [
        (d.day, d.city, tuple(d.activities)) for d in second.itinerary
    ]
    assert [(e.stage, e.phase) for e in first_events] == [
        (e.stage, e.phase) for e in second_events
    ]
