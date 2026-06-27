"""LangGraph graph assembly and stage progress (Task 13.5).

This module wires the full planning pipeline into a single LangGraph
``StateGraph`` over the shared :class:`~app.orchestration.state.GraphState`:

    Coordinator(intake) -> tools -> Destination(propose) -> Decision Engine
        -> CP-SAT Solver --(feasible skeleton)--> Itinerary -> Coordinator(merge)
                          \\-(infeasible / timeout)----------------> merge

Each stage is a node that reads from and writes to the shared state. The graph
respects the **solver gate** (Requirement 3.4 / 9.8 / 9.9): the Itinerary node
runs *only* when the solver produced a feasible skeleton; on a rejection or
timeout the graph routes straight to the Coordinator's merge so no enriched
plan is fabricated from an infeasible skeleton.

**Decision transparency (Requirements 1.1, 1.4, 17.4).** The merge node attaches
a rationale to every destination, flight, and hotel option that reaches the
user (delegated to :func:`app.orchestration.nodes.coordinator.merge_final_plan`,
which withholds and records anything lacking one). The graph never lets an LLM
invent or reorder scores — scoring is the deterministic Decision Engine's job
and the Destination narration is built from those scores only.

**Progress hooks (Requirement 13.2 / Property 26).** Every node emits a
``start`` progress event when it begins and a ``complete`` event when it
finishes. Events are plain stage notifications (e.g. "Searching flights,
hotels, events, and weather... done") and never carry live/real-time prices, so
the WebSocket progress stream (Task 19.3) can consume them directly. Supply a
:class:`ProgressReporter` (or a bare callback) to :func:`run_plan` /
:func:`build_graph` to observe them; omit it for a silent run.

The whole pipeline is deterministic on the default mock providers: identical
inputs always produce identical plans and the same ordered sequence of progress
events, so it runs end-to-end with no paid API keys.

Requirements: 3.1, 1.1, 1.4, 17.4.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal, Optional

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.decision.engine import RawCandidate, score_candidates
from app.decision.ledger import LedgerInconsistencyError, build_decision_trace
from app.decision.weights import personalized_weights
from app.llm.base import LLMProvider
from app.orchestration.nodes.coordinator import (
    ToolBundle,
    _parse_money,
    coordinator_intake_node,
    gather_tool_bundle,
    merge_final_plan,
)
from app.orchestration.nodes.destination import narrate_ranking, propose_candidates
from app.orchestration.nodes.itinerary import enrich_skeleton, itinerary_agent_should_run
from app.orchestration.state import GraphState, ScoredCandidate
from app.solver.cp_sat import CandidateCity, solve_skeleton

try:  # LangGraph is a project dependency; import is guarded for import-safety.
    from langgraph.graph import END, START, StateGraph

    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without LangGraph.
    StateGraph = None  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    END = "__end__"  # type: ignore[assignment]
    _LANGGRAPH_AVAILABLE = False


__all__ = [
    "STAGES",
    "ProgressPhase",
    "ProgressEvent",
    "ProgressReporter",
    "PlanningRun",
    "build_graph",
    "run_plan",
]


# --------------------------------------------------------------------------- #
# Stage identifiers and progress contract
# --------------------------------------------------------------------------- #
# Ordered (key, human label) for every pipeline stage. The label is the
# user-facing progress text the WebSocket stream relays; the key is the stable
# node name used in the graph and in progress events.
#
# Note on ordering: the design names the stages "Coordinator -> tools ->
# Destination -> ...", but the deterministic tool layer gathers data *per
# candidate city* (flight/hotel costs feeding the solver's budget cap), so it
# can only run once the Destination agent has proposed candidates. The graph
# therefore runs Destination before tools while still emitting every named
# stage. Consumers of the progress stream key off the stage name, not a fixed
# position, so the ordering is an implementation detail.
STAGES: tuple[tuple[str, str], ...] = (
    ("coordinator", "Understanding your trip request"),
    ("destination", "Proposing candidate destinations"),
    ("decision_engine", "Scoring destinations"),
    ("tools", "Searching flights, hotels, events, and weather"),
    ("solver", "Checking itinerary feasibility"),
    ("itinerary", "Building your day-by-day itinerary"),
    ("merge", "Assembling your explained plan"),
)
_STAGE_LABELS: dict[str, str] = dict(STAGES)

ProgressPhase = Literal["start", "complete"]


class ProgressEvent(BaseModel):
    """A single progress notification emitted as a stage starts or completes.

    Carries only stage metadata and a human-readable message — never a live or
    real-time price (Requirement 13.3 / Property 26). ``message`` is the text a
    UI shows directly (e.g. "Searching flights... done").
    """

    stage: str
    phase: ProgressPhase
    message: str


# A progress callback receives each event as it is emitted.
ProgressCallback = Callable[[ProgressEvent], None]


class ProgressReporter:
    """Collects progress events and forwards them to an optional callback.

    The reporter is the bridge between the graph's node wrappers and any
    consumer (the WebSocket progress stream in Task 19.3, or a test). It records
    every event in :attr:`events` and, when a ``callback`` is supplied, invokes
    it synchronously as each event fires.
    """

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self._callback = callback
        self.events: list[ProgressEvent] = []

    def emit(self, stage: str, phase: ProgressPhase) -> ProgressEvent:
        """Record and forward one progress event for ``stage``/``phase``."""
        label = _STAGE_LABELS.get(stage, stage)
        # "...": started; "... done": completed. Plain stage text, no prices.
        message = label if phase == "start" else f"{label} — done"
        event = ProgressEvent(stage=stage, phase=phase, message=message)
        self.events.append(event)
        if self._callback is not None:
            self._callback(event)
        return event


def _coerce_reporter(
    progress: ProgressReporter | ProgressCallback | None,
) -> ProgressReporter:
    """Accept a reporter, a bare callback, or ``None`` and return a reporter."""
    if progress is None:
        return ProgressReporter()
    if isinstance(progress, ProgressReporter):
        return progress
    return ProgressReporter(progress)


# --------------------------------------------------------------------------- #
# Stage wiring helpers (deterministic, mock-first)
# --------------------------------------------------------------------------- #
def _derive_city_cost(bundle: ToolBundle, nights: int) -> Decimal:
    """Estimate the cost of including a city: cheapest flight + hotel * nights.

    Mirrors the budget math the Coordinator's merge uses for the chosen city, so
    the solver's budget feasibility check and the final plan's estimate agree.
    """
    cheapest_flight = min((_parse_money(f.price) for f in bundle.flights), default=0.0)
    cheapest_hotel = min((_parse_money(h.price) for h in bundle.hotels), default=0.0)
    nights = max(1, nights)
    return Decimal(str(cheapest_flight)) + Decimal(str(cheapest_hotel)) * nights


def _to_scored_state(
    ranked: list[Any],
) -> list[ScoredCandidate]:
    """Map Decision Engine scored candidates onto the shared-state schema.

    The engine owns the numbers; this only re-shapes them. ``components`` carries
    the engine's per-feature contributions (which sum to the score) so the
    Coordinator can derive an auditable rationale, and ``rank`` is the 1-based
    descending position.
    """
    scored: list[ScoredCandidate] = []
    for position, candidate in enumerate(ranked, start=1):
        scored.append(
            ScoredCandidate(
                name=candidate.destination,
                country=str(candidate.metadata.get("country", "")),
                score=candidate.score,
                components=dict(candidate.contributions),
                rank=position,
            )
        )
    return scored


# --------------------------------------------------------------------------- #
# The graph nodes (closures over run dependencies)
# --------------------------------------------------------------------------- #
def _make_nodes(
    *,
    settings: Settings,
    llm: LLMProvider | None,
    session: Any,
    now: datetime | None,
    start_date: date | None,
    reporter: ProgressReporter,
    ctx: dict[str, Any],
) -> dict[str, Callable[[GraphState], dict[str, Any]]]:
    """Build the per-run node callables.

    Each node returns a dict of shared-state field updates (the LangGraph node
    contract). ``ctx`` is a run-scoped scratchpad for data that is computed once
    and reused across nodes (tool bundles, per-city costs) but is not part of
    the documented :class:`GraphState` contract.
    """

    def _with_progress(
        stage: str, fn: Callable[[GraphState], dict[str, Any]]
    ) -> Callable[[GraphState], dict[str, Any]]:
        def node(state: GraphState) -> dict[str, Any]:
            reporter.emit(stage, "start")
            update = fn(state)
            reporter.emit(stage, "complete")
            return update

        return node

    # --- coordinator (intake): decompose prompt + load memory -------------- #
    def _coordinator(state: GraphState) -> dict[str, Any]:
        out = coordinator_intake_node(
            state, session=session, settings=settings, now=now
        )
        return {
            "constraints": out.constraints,
            "preference_vector": out.preference_vector,
        }

    # --- destination (propose candidate cities) ---------------------------- #
    def _destination(state: GraphState) -> dict[str, Any]:
        candidates = state.candidates or propose_candidates(state)
        return {"candidates": candidates}

    # --- tools (deterministic provider layer) ------------------------------ #
    def _tools(state: GraphState) -> dict[str, Any]:
        # Gather each candidate's tool bundle once and derive its trip cost so
        # the solver can enforce the budget cap. Bundles are cached for reuse by
        # the merge node (which only needs the chosen city's bundle).
        nights = state.constraints.duration_days if state.constraints else 1
        bundles: dict[str, ToolBundle] = {}
        costs: dict[str, Decimal] = {}
        for candidate in state.candidates:
            bundle = gather_tool_bundle(
                candidate.name, state.constraints, settings=settings
            )
            bundles[candidate.name] = bundle
            costs[candidate.name] = _derive_city_cost(bundle, nights)
        ctx["bundles"] = bundles
        ctx["costs"] = costs
        return {}

    # --- decision engine (deterministic scoring) --------------------------- #
    def _decision_engine(state: GraphState) -> dict[str, Any]:
        weights = personalized_weights(state.preference_vector, settings=settings)
        raw = [
            RawCandidate(
                destination=candidate.name,
                features=candidate.features,
                metadata={"country": candidate.country},
            )
            for candidate in state.candidates
        ]
        result = score_candidates(raw, weights, state.constraints)
        scored = _to_scored_state(result.ranked)

        traces = []
        for engine_candidate in result.ranked:
            try:
                traces.append(build_decision_trace(engine_candidate))
            except LedgerInconsistencyError:
                # A trace that fails additivity is surfaced as an error rather
                # than persisted; skip it but keep the (valid) score/ranking.
                continue

        # Narrate the ranking using the engine's scores only (never reorders).
        narration = narrate_ranking(scored, llm=llm)
        ctx["destination_narration"] = narration
        return {
            "scored": scored,
            "decision_traces": traces,
            "destination_narration": narration,
        }

    # --- CP-SAT solver (feasible skeleton XOR rejection) ------------------- #
    def _solver(state: GraphState) -> dict[str, Any]:
        if state.constraints is None or not state.scored:
            # Nothing to place; the solver rejects with no skeleton.
            empty = solve_skeleton([], _require_constraints(state))
            return {"skeleton": empty.skeleton, "rejection": empty.rejection}

        costs: dict[str, Decimal] = ctx.get("costs", {})
        cities = [
            CandidateCity(
                name=sc.name,
                cost=costs.get(sc.name, Decimal(0)),
                score=sc.score,
                # Mock catalog cities carry no dietary/interest restrictions, so
                # they are treated as neutral (eligible) by the solver.
                dietary_conflicts=[],
                interest_tags=[],
            )
            for sc in state.scored
        ]
        result = solve_skeleton(cities, state.constraints)
        return {"skeleton": result.skeleton, "rejection": result.rejection}

    # --- itinerary (enrich feasible skeleton) ------------------------------ #
    def _itinerary(state: GraphState) -> dict[str, Any]:
        # The gate is also enforced by graph routing, but re-checked here so the
        # node is correct if invoked directly.
        if not itinerary_agent_should_run(state):
            return {}
        assert state.skeleton is not None
        itinerary = enrich_skeleton(
            state.skeleton,
            state.constraints,
            settings=settings,
            start_date=start_date,
        )
        return {"itinerary": itinerary}

    # --- coordinator (merge): assemble the explained plan ------------------ #
    def _merge(state: GraphState) -> dict[str, Any]:
        chosen = state.scored[0].name if state.scored else ""
        bundles: dict[str, ToolBundle] = ctx.get("bundles", {})
        bundle: Optional[ToolBundle] = bundles.get(chosen)
        if bundle is None and chosen and state.constraints is not None:
            bundle = gather_tool_bundle(chosen, state.constraints, settings=settings)

        plan, withheld = merge_final_plan(
            state.scored, state.itinerary, bundle, state.constraints
        )
        return {
            "final_plan": plan,
            "withheld_recommendations": withheld,
            "missing_rationale": bool(withheld),
        }

    return {
        "coordinator": _with_progress("coordinator", _coordinator),
        "tools": _with_progress("tools", _tools),
        "destination": _with_progress("destination", _destination),
        "decision_engine": _with_progress("decision_engine", _decision_engine),
        "solver": _with_progress("solver", _solver),
        "itinerary": _with_progress("itinerary", _itinerary),
        "merge": _with_progress("merge", _merge),
    }


def _require_constraints(state: GraphState):
    """Return the state's constraints, importing a default lazily if unset.

    The solver always needs a ``TripConstraints``; if the coordinator somehow
    left it unset, decompose a trivial one so the solver can still reject
    cleanly rather than raising.
    """
    if state.constraints is not None:
        return state.constraints
    from app.orchestration.nodes.coordinator import decompose_prompt

    return decompose_prompt(state.prompt or "")


def _solver_route(state: GraphState) -> str:
    """Route after the solver: enrich only on a feasible skeleton (Req 3.4)."""
    return "itinerary" if itinerary_agent_should_run(state) else "merge"


# LangGraph forbids registering a node whose name collides with a state channel.
# :class:`GraphState` has an ``itinerary`` field, so the Itinerary agent must be
# registered under a distinct graph-node id. Its public stage name (used in
# ``node_names`` and progress events) stays "itinerary" so the documented
# pipeline and the progress stream are unchanged.
_GRAPH_NODE_IDS: dict[str, str] = {"itinerary": "itinerary_agent"}


def _graph_node_id(stage: str) -> str:
    """Map a public stage name to its (collision-free) LangGraph node id."""
    return _GRAPH_NODE_IDS.get(stage, stage)


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
class PlanningRun:
    """A compiled planning pipeline bound to one set of run dependencies.

    Wraps the compiled LangGraph app together with the :class:`ProgressReporter`
    observing it, so callers can invoke the run and then read the ordered
    progress events. Construct one via :func:`build_graph`.
    """

    def __init__(self, app: Any, reporter: ProgressReporter, node_names: list[str]):
        self._app = app
        self.reporter = reporter
        self.node_names = node_names

    @property
    def events(self) -> list[ProgressEvent]:
        """The progress events recorded so far for this run."""
        return self.reporter.events

    def invoke(self, state: GraphState) -> GraphState:
        """Execute the pipeline for ``state`` and return the final state."""
        result = self._app.invoke(state)
        if isinstance(result, GraphState):
            return result
        # LangGraph returns the channels as a mapping; rebuild the typed state.
        return GraphState(**dict(result))


def build_graph(
    *,
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
    session: Any = None,
    now: datetime | None = None,
    start_date: date | None = None,
    progress: ProgressReporter | ProgressCallback | None = None,
) -> PlanningRun:
    """Assemble and compile the planning pipeline as a LangGraph graph.

    Wires Coordinator -> tools -> Destination -> Decision Engine -> CP-SAT
    Solver -> Itinerary -> merge, with a conditional edge after the solver that
    enforces the feasible-skeleton gate (Requirement 3.4). Every node emits
    start/complete progress events through the returned run's reporter.

    Args:
        settings: configuration (defaults to :func:`get_settings`); selects mock
            vs. real providers and the LLM vendor.
        llm: optional narration provider; defaults to the config-selected one.
        session: optional DB session for the Memory Layer (cold-start without).
        now: optional clock override for memory decay (kept for determinism).
        start_date: optional first trip day used to fill in itinerary dates.
        progress: a :class:`ProgressReporter`, a bare callback, or ``None``.

    Returns:
        A :class:`PlanningRun` wrapping the compiled graph and its progress
        reporter.

    Raises:
        RuntimeError: if LangGraph is not importable in this environment.
    """
    if not _LANGGRAPH_AVAILABLE:  # pragma: no cover - dependency is pinned.
        raise RuntimeError(
            "LangGraph (langgraph) is required to assemble the planning graph "
            "but is not installed in this environment."
        )

    settings = settings or get_settings()
    reporter = _coerce_reporter(progress)
    ctx: dict[str, Any] = {}

    # Wire the configured LLM provider in by default so narration uses the
    # vendor selected by settings (e.g. live Gemini) — resolved once here and
    # threaded into the nodes, rather than each node implicitly re-resolving it.
    # With no LLM key configured this is the deterministic mock provider.
    if llm is None:
        from app.llm.base import get_llm_provider

        llm = get_llm_provider(settings)

    nodes = _make_nodes(
        settings=settings,
        llm=llm,
        session=session,
        now=now,
        start_date=start_date,
        reporter=reporter,
        ctx=ctx,
    )

    builder = StateGraph(GraphState)
    for name, fn in nodes.items():
        builder.add_node(_graph_node_id(name), fn)

    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "destination")
    builder.add_edge("destination", "decision_engine")
    builder.add_edge("decision_engine", "tools")
    builder.add_edge("tools", "solver")
    builder.add_conditional_edges(
        "solver",
        _solver_route,
        {"itinerary": _graph_node_id("itinerary"), "merge": "merge"},
    )
    builder.add_edge(_graph_node_id("itinerary"), "merge")
    builder.add_edge("merge", END)

    app = builder.compile()
    return PlanningRun(app, reporter, list(nodes.keys()))


def run_plan(
    state: GraphState,
    *,
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
    session: Any = None,
    now: datetime | None = None,
    start_date: date | None = None,
    progress: ProgressReporter | ProgressCallback | None = None,
) -> tuple[GraphState, list[ProgressEvent]]:
    """Run the full planning pipeline for ``state`` end-to-end.

    Convenience entry point used by the API (Task 19.x) and tests: builds the
    graph, executes it, and returns the final :class:`GraphState` together with
    the ordered list of progress events emitted during the run.

    The run is deterministic on the default mock providers — the same input
    state yields the same final plan and the same sequence of progress events.
    """
    reporter = _coerce_reporter(progress)
    run = build_graph(
        settings=settings,
        llm=llm,
        session=session,
        now=now,
        start_date=start_date,
        progress=reporter,
    )
    final_state = run.invoke(state)
    return final_state, reporter.events
