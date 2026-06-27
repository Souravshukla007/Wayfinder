"""Integration test for LangGraph graph composition (Task 13.10).

Validates Requirements:
- 3.1: the planning flow is assembled as a single LangGraph graph that wires the
  reasoning agents together and runs end-to-end on the default mock providers
  (no paid API keys).

This focused integration test asserts two things about the assembled graph:

1. **Composition** — the compiled graph contains the three named reasoning
   agents: the Coordinator (intake + ``merge``), the Destination agent, and the
   Itinerary agent.
2. **End-to-end on mock** — running the graph on mock providers actually
   executes those three agents (observed via the progress stream) and yields a
   complete explained plan.

These complement the broader smoke tests in ``test_graph_assembly.py`` by
pinning the exact agent-composition contract Requirement 3.1 describes.
"""

from __future__ import annotations

from app.config import Settings
from app.orchestration.graph import build_graph, run_plan
from app.orchestration.state import GraphState

JAPAN_PROMPT = (
    "Plan a 7-day Japan trip in October under \u20b92 lakh. "
    "I like anime, photography, local food, and less crowded places."
)


def _settings() -> Settings:
    # No credentials => mock providers => deterministic, zero-key run.
    return Settings()


def _state() -> GraphState:
    return GraphState(trip_id="trip-compose", user_id="user-1", prompt=JAPAN_PROMPT)


def test_graph_contains_coordinator_destination_and_itinerary_agents() -> None:
    """The assembled graph wires the three named reasoning agents (Req 3.1)."""
    run = build_graph(settings=_settings())

    # Coordinator is present as both the intake node and the merge node; the
    # Destination and Itinerary agents are each their own node.
    assert "coordinator" in run.node_names, "Coordinator (intake) agent missing"
    assert "merge" in run.node_names, "Coordinator (merge) node missing"
    assert "destination" in run.node_names, "Destination agent missing"
    assert "itinerary" in run.node_names, "Itinerary agent missing"


def test_graph_runs_end_to_end_on_mock_executing_the_three_agents() -> None:
    """Running on mock providers executes the agents and yields a full plan."""
    final, events = run_plan(_state(), settings=_settings())

    # End-to-end success: constraints decomposed, candidates scored, a feasible
    # skeleton enriched into an itinerary, and an explained plan assembled.
    assert final.constraints is not None
    assert final.scored, "Destination/Decision agents should produce ranked candidates"
    assert final.skeleton is not None and final.rejection is None
    assert final.itinerary, "Itinerary agent should enrich the feasible skeleton"
    assert final.final_plan is not None

    # The three named agents actually ran end-to-end (observed via progress).
    executed_stages = {e.stage for e in events}
    assert {"coordinator", "destination", "itinerary", "merge"} <= executed_stages
