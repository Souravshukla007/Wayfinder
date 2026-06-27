"""Integration test: the Japan driving example end-to-end on mock providers.

Task 22.5 — exercises the full LangGraph planning pipeline against the Japan
golden scenario fixture (``JAPAN_SCENARIO``) using only the default mock
providers, and asserts the four acceptance criteria of Requirement 18:

- **18.1** ranked, *scored* destination options are produced with an
  explanation of the ordering;
- **18.2** the produced itinerary skeleton's total cost does not exceed
  ₹2,00,000;
- **18.3** the explainability ledger's per-feature contributions sum to the
  chosen destination's score (within the 0.001 tolerance);
- **18.4** the run completes with no paid API key configured (mock providers
  selected for every domain).

This is an *end-to-end* test: it drives the real graph via
:func:`app.orchestration.graph.run_plan` rather than re-checking the fixture in
isolation (covered by ``test_japan_golden_scenario_fixture.py``) or the harness
scoring (covered by ``test_eval_harness.py``).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.config import PROVIDER_DOMAINS, Settings
from app.eval.scenarios.japan import (
    JAPAN_BUDGET_CAP,
    JAPAN_PROMPT,
    JAPAN_SCENARIO,
)
from app.orchestration.graph import run_plan
from app.orchestration.state import GraphState

# Additive tolerance for the explainability ledger (Requirement 8.2 / 18.3).
LEDGER_TOLERANCE = 0.001


def _no_key_settings() -> Settings:
    """Settings with no provider credentials => mock providers everywhere."""
    return Settings()


def _japan_state() -> GraphState:
    """Seed the graph from the verbatim Japan prompt + fixture constraints."""
    return GraphState(
        trip_id="japan-e2e",
        user_id="integration-test",
        prompt=JAPAN_PROMPT,
        constraints=JAPAN_SCENARIO.constraints,
    )


@pytest.fixture(scope="module")
def japan_run() -> tuple[GraphState, list]:
    """Run the Japan example once end-to-end on mock providers (no paid key)."""
    settings = _no_key_settings()
    # Requirement 18.4 precondition: every domain must resolve to mock with no
    # credentials configured, so the run needs no paid API key.
    assert all(settings.uses_mock(domain) for domain in PROVIDER_DOMAINS)

    state = _japan_state()
    final_state, events = run_plan(state, settings=settings)
    return final_state, events


def test_japan_run_completes_without_a_paid_api_key(
    japan_run: tuple[GraphState, list],
) -> None:
    """Requirement 18.4: the example completes on mock providers, no paid key."""
    final, events = japan_run

    # The pipeline ran to completion: constraints decomposed and a final plan
    # assembled (the merge node only produces a plan on a completed run).
    assert final.constraints is not None
    assert final.final_plan is not None
    assert events, "progress events should have been emitted during the run"
    # The merge stage ran, i.e. the run reached the end of the pipeline.
    assert any(e.stage == "merge" and e.phase == "complete" for e in events)


def test_japan_produces_ranked_scored_options_with_ordering_explanation(
    japan_run: tuple[GraphState, list],
) -> None:
    """Requirement 18.1: ranked, scored options with an ordering explanation."""
    final, _events = japan_run

    # At least the top three options (A/B/C) are produced (Req 7.2).
    assert len(final.scored) >= 3

    # Each option carries a numeric 0-100 score and a descending rank position.
    scores = [sc.score for sc in final.scored]
    assert all(0.0 <= s <= 100.0 for s in scores)

    # Ranked in non-increasing order of final score (Req 7.1).
    assert scores == sorted(scores, reverse=True)
    assert [sc.rank for sc in final.scored] == list(
        range(1, len(final.scored) + 1)
    )

    # The presented plan explains the ordering: every recommended destination
    # carries a non-empty rationale and its additive factors (Req 7.3 / 1.1).
    presented = final.final_plan.destinations
    assert presented, "the merged plan should present ranked destinations"
    assert all(d.rationale.strip() for d in presented)
    assert all(d.factors for d in presented)
    # Presented destinations preserve the engine's descending ranking.
    presented_scores = [d.score for d in presented]
    assert presented_scores == sorted(presented_scores, reverse=True)


def test_japan_itinerary_total_within_two_lakh(
    japan_run: tuple[GraphState, list],
) -> None:
    """Requirement 18.2: itinerary skeleton total cost does not exceed ₹2 lakh."""
    final, _events = japan_run

    # A feasible skeleton was emitted (not a rejection/timeout).
    assert final.rejection is None
    assert final.skeleton is not None
    assert final.skeleton.legs, "a feasible skeleton should contain legs"

    # The provably-feasible solver total respects the ₹2,00,000 cap (equality
    # accepted, any overshoot rejected — Req 9.1 / 18.2).
    assert Decimal(final.skeleton.total_cost) <= JAPAN_BUDGET_CAP

    # The merged plan's estimate agrees and is flagged within budget.
    assert Decimal(final.final_plan.estimated_total_cost) <= JAPAN_BUDGET_CAP
    assert final.final_plan.within_budget is True


def test_japan_ledger_contributions_sum_to_chosen_score(
    japan_run: tuple[GraphState, list],
) -> None:
    """Requirement 18.3: ledger contributions sum to the chosen destination's score."""
    final, _events = japan_run

    # The chosen destination is the top-ranked scored candidate.
    chosen = final.scored[0]

    # A decision trace exists for the chosen destination.
    traces_by_dest = {t.destination: t for t in final.decision_traces}
    assert chosen.name in traces_by_dest, "chosen destination must have a ledger"
    trace = traces_by_dest[chosen.name]

    # The ledger records the chosen destination's final score.
    assert abs(trace.score - chosen.score) <= LEDGER_TOLERANCE

    # The per-feature contributions sum to that score within tolerance (8.2).
    contributions_sum = sum(f.contribution for f in trace.factors)
    assert abs(contributions_sum - trace.score) <= LEDGER_TOLERANCE


def test_japan_run_is_deterministic_on_mock_providers() -> None:
    """The end-to-end run is reproducible on mock providers (supports 18.4)."""
    settings = _no_key_settings()
    first, _ = run_plan(_japan_state(), settings=settings)
    second, _ = run_plan(_japan_state(), settings=settings)

    assert [sc.name for sc in first.scored] == [sc.name for sc in second.scored]
    assert [sc.score for sc in first.scored] == [sc.score for sc in second.scored]
    assert first.skeleton is not None and second.skeleton is not None
    assert Decimal(first.skeleton.total_cost) == Decimal(second.skeleton.total_cost)
