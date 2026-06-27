"""Tests for the evaluation harness scoring and thresholds (Task 22.2).

Validates Requirements:
- 14.1: the harness scores each produced plan on all five metrics.
- 14.2: the harness compares each score against the scenario's thresholds.
- 14.3: the harness fails the run and reports the failing metric when a plan
  scores below a defined threshold.

The Japan golden scenario is graded end-to-end on the default mock providers
(no paid API key, Requirement 18.4).
"""

from __future__ import annotations

from copy import deepcopy

from app.config import Settings
from app.eval.harness import (
    EVAL_METRICS,
    HarnessError,
    evaluate_scenario,
    run_harness,
    score_plan,
)
from app.eval.scenarios import GOLDEN_SCENARIOS
from app.eval.scenarios.japan import JAPAN_SCENARIO


def _settings() -> Settings:
    # No credentials => mock providers => deterministic, zero-key run.
    return Settings()


def test_japan_scenario_passes_every_threshold() -> None:
    """The Japan golden scenario clears all five metric thresholds (14.2)."""
    result = evaluate_scenario(JAPAN_SCENARIO, settings=_settings())

    assert result.passed, f"unexpected failing metrics: {result.failing_metrics}"
    assert result.scenario_id == JAPAN_SCENARIO.id
    # Every metric is covered with a score in [0, 1] (Property 29 / Req 14.1).
    assert {o.metric for o in result.outcomes} == set(EVAL_METRICS)
    for outcome in result.outcomes:
        assert 0.0 <= outcome.score <= 1.0
        assert outcome.passed


def test_harness_emits_a_score_for_every_metric() -> None:
    """score_plan produces a score for each of the five metrics (Req 14.1)."""
    settings = _settings()
    # Reuse evaluate_scenario's run by scoring the produced state directly.
    from app.orchestration.graph import run_plan
    from app.orchestration.state import GraphState

    state = GraphState(
        trip_id="eval-test",
        user_id="eval-harness",
        prompt=JAPAN_SCENARIO.prompt,
        constraints=JAPAN_SCENARIO.constraints,
    )
    final_state, _ = run_plan(state, settings=settings)
    scores = score_plan(JAPAN_SCENARIO, final_state)

    assert set(scores.as_dict()) == set(EVAL_METRICS)
    assert all(0.0 <= v <= 1.0 for v in scores.as_dict().values())


def test_run_harness_passes_for_all_golden_scenarios() -> None:
    """Grading the full registry succeeds on mock providers (14.2)."""
    result = run_harness(settings=_settings())
    assert result.passed
    assert len(result.scenarios) == len(GOLDEN_SCENARIOS)


def test_below_threshold_scenario_fails_and_reports_metric() -> None:
    """An unreachable threshold makes the harness fail and name the metric (14.3)."""
    # Clone Japan but demand a personalization score the mock ranking can't hit.
    impossible = deepcopy(JAPAN_SCENARIO)
    impossible = impossible.model_copy(
        update={
            "thresholds": impossible.thresholds.model_copy(
                update={"personalization": 1.0 + 0.0}
            )
        }
    )
    # Force an impossible bar (above the [0,1] scale is clamped by the model, so
    # use exactly 1.0 which the rank-weighted average cannot reach here).
    result = evaluate_scenario(impossible, settings=_settings())
    assert not result.passed
    assert "personalization" in result.failing_metrics

    # run_harness with raise_on_failure surfaces the failing metric (14.3).
    registry = {impossible.id: impossible}
    try:
        run_harness(registry, settings=_settings(), raise_on_failure=True)
    except HarnessError as exc:
        assert "personalization" in str(exc)
    else:  # pragma: no cover - the run must fail
        raise AssertionError("expected HarnessError for below-threshold metric")
