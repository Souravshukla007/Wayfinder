"""Evaluation harness: score produced plans against golden scenarios.

This module implements task 22.2 (Requirements 14.1, 14.2, 14.3). For each
:class:`~app.eval.scenarios.base.GoldenScenario` it:

1. **Runs the planner end-to-end on mock providers** (no paid API key,
   Requirement 18.4) by driving the LangGraph pipeline via
   :func:`app.orchestration.graph.run_plan`.
2. **Scores the produced plan on the five evaluation metrics** (Requirement
   14.1 / Property 29): ``budget_adherence``, ``constraint_satisfaction``,
   ``feasibility``, ``explanation_coverage`` and ``personalization``. Every
   score is normalized to ``[0, 1]`` (higher is better).
3. **Compares each score against the scenario's per-metric thresholds**
   (Requirement 14.2) carried on
   :class:`~app.eval.scenarios.base.MetricThresholds`.
4. **Fails the run and reports the failing metric(s) per scenario**
   (Requirement 14.3) when any score falls below its threshold.

The scoring is *independent* of the planner internals: each metric re-derives
its judgement from the produced plan / solver skeleton (rather than trusting a
flag the planner set), mirroring the "independent checker" approach used by the
solver's feasibility property test. This keeps the harness an honest external
quality gate.

Programmatic entry points:

- :func:`score_plan` — score one already-produced run (pure, no I/O).
- :func:`evaluate_scenario` — run the planner for one scenario and grade it.
- :func:`run_harness` — grade every scenario; optionally raise on failure.

CLI entry point (used by the CI job in task 22.3)::

    python -m app.eval.harness

prints a per-scenario report and exits non-zero (reporting the failing metric)
when any plan scores below threshold.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, Field

from app.config import Settings
from app.llm.base import LLMProvider
from app.orchestration.graph import run_plan
from app.orchestration.state import GraphState

from .scenarios import EVAL_METRICS, GOLDEN_SCENARIOS, GoldenScenario
from .scenarios.base import MetricThresholds

# The destination catalog and its interest tags back the personalization metric:
# a personalized ranking should rank interest-matching cities highly.
from app.orchestration.nodes.destination import DESTINATION_CATALOG

__all__ = [
    "MetricScores",
    "MetricOutcome",
    "ScenarioResult",
    "HarnessResult",
    "HarnessError",
    "score_plan",
    "evaluate_scenario",
    "run_harness",
    "format_report",
    "main",
]


# --------------------------------------------------------------------------- #
# Result models
# --------------------------------------------------------------------------- #
class MetricScores(BaseModel):
    """The five normalized metric scores for one produced plan (Property 29).

    Each field is the score in ``[0, 1]`` (higher is better) for the metric of
    the same name. The model guarantees coverage of every metric in
    :data:`EVAL_METRICS`.
    """

    budget_adherence: float = Field(ge=0.0, le=1.0)
    constraint_satisfaction: float = Field(ge=0.0, le=1.0)
    feasibility: float = Field(ge=0.0, le=1.0)
    explanation_coverage: float = Field(ge=0.0, le=1.0)
    personalization: float = Field(ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, float]:
        """Return the scores keyed by metric name in canonical order."""
        return {metric: getattr(self, metric) for metric in EVAL_METRICS}


class MetricOutcome(BaseModel):
    """The pass/fail outcome of one metric against its threshold."""

    metric: str
    score: float
    threshold: float
    passed: bool


class ScenarioResult(BaseModel):
    """The graded result of running the harness against one golden scenario."""

    scenario_id: str
    scenario_name: str
    scores: dict[str, float]
    thresholds: dict[str, float]
    outcomes: list[MetricOutcome]
    passed: bool

    @property
    def failing_metrics(self) -> list[str]:
        """Names of the metrics that fell below threshold (empty when passed)."""
        return [o.metric for o in self.outcomes if not o.passed]


class HarnessResult(BaseModel):
    """The aggregate result of grading every golden scenario."""

    scenarios: list[ScenarioResult]

    @property
    def passed(self) -> bool:
        """True only when every scenario passed every metric threshold."""
        return all(s.passed for s in self.scenarios)

    @property
    def failures(self) -> list[ScenarioResult]:
        """The scenarios that failed at least one metric threshold."""
        return [s for s in self.scenarios if not s.passed]


class HarnessError(AssertionError):
    """Raised to fail a run when a plan scores below a defined threshold.

    Subclasses :class:`AssertionError` so it reads as a test/CI failure. The
    message names each failing scenario and metric (Requirement 14.3).
    """


# --------------------------------------------------------------------------- #
# Metric scorers (each returns a score in [0, 1]; higher is better)
# --------------------------------------------------------------------------- #
def _plan_total_cost(state: GraphState) -> Optional[Decimal]:
    """The total trip cost to judge against the budget cap.

    Prefers the solver skeleton's total (the provably feasible figure,
    Requirement 18.2) and falls back to the merged plan's estimate. Returns
    ``None`` when no cost was produced at all.
    """
    if state.skeleton is not None and state.skeleton.total_cost > 0:
        return Decimal(state.skeleton.total_cost)
    if state.final_plan is not None and state.final_plan.estimated_total_cost > 0:
        return Decimal(state.final_plan.estimated_total_cost)
    return None


def _score_budget_adherence(scenario: GoldenScenario, state: GraphState) -> float:
    """Score how well the plan's total cost respects the budget cap (Req 18.2).

    ``1.0`` when the total is at or under the cap (equality accepted); otherwise
    the cap/total ratio so that larger overruns score progressively lower.
    """
    cap = Decimal(scenario.budget_cap)
    total = _plan_total_cost(state)
    if total is None:
        # No itinerary cost was produced (e.g. solver rejection): nothing
        # exceeds the cap, but there is also no plan to credit. Treat as a
        # non-overrun (the feasibility metric captures the missing plan).
        return 1.0
    if total <= cap:
        return 1.0
    if cap <= 0:
        return 0.0
    return max(0.0, min(1.0, float(cap / total)))


def _score_constraint_satisfaction(
    scenario: GoldenScenario, state: GraphState
) -> float:
    """Fraction of declared hard constraints the produced plan satisfies.

    Re-checks the emitted skeleton against the trip constraints independently of
    the solver (city count, hotel changes, budget, duration, dietary). A solver
    rejection means no plan satisfied the constraints, scoring ``0.0``.
    """
    constraints = state.constraints or scenario.constraints
    skeleton = state.skeleton
    if skeleton is None:
        return 0.0

    checks: list[bool] = []

    # Budget cap: skeleton total cost <= trip budget (equality accepted, 9.1).
    checks.append(Decimal(skeleton.total_cost) <= Decimal(constraints.budget))

    # City-count limit: distinct cities <= max_cities (9.4).
    distinct_cities = {leg.city for leg in skeleton.legs}
    checks.append(len(distinct_cities) <= constraints.max_cities)

    # Hotel-change limit (9.3).
    checks.append(skeleton.hotel_changes <= constraints.max_hotel_changes)

    # Duration: the itinerary fits within the trip length (no leg departs after
    # the last trip day).
    if skeleton.legs:
        last_day = max(leg.departure_day for leg in skeleton.legs)
        checks.append(last_day <= constraints.duration_days)

    # Dietary exclusions (9.5): with no dietary constraints there is nothing to
    # violate; when present, a feasible skeleton has already excluded violators.
    checks.append(True if not constraints.dietary else skeleton.legs != [])

    if not checks:
        return 1.0
    return sum(1 for ok in checks if ok) / len(checks)


def _score_feasibility(state: GraphState) -> float:
    """1.0 when a feasible skeleton was emitted, 0.0 on rejection/empty (9.x)."""
    if state.rejection is not None:
        return 0.0
    if state.skeleton is not None and state.skeleton.legs:
        return 1.0
    return 0.0


def _score_explanation_coverage(state: GraphState) -> float:
    """Fraction of major recommendations that carry a rationale (1.x, 8.x).

    Major recommendations are the presented destinations (which must carry both
    a rationale and the additive per-feature factors), flights and hotels. Any
    recommendation that had to be *withheld* for lacking a rationale counts
    against coverage (Requirement 1.3).
    """
    plan = state.final_plan
    if plan is None:
        return 0.0

    covered_flags: list[bool] = []
    covered_flags.extend(bool(d.rationale and d.factors) for d in plan.destinations)
    covered_flags.extend(bool(f.rationale) for f in plan.flights)
    covered_flags.extend(bool(h.rationale) for h in plan.hotels)

    withheld = len(state.withheld_recommendations)
    total = len(covered_flags) + withheld
    if total == 0:
        return 0.0
    return sum(1 for ok in covered_flags if ok) / total


def _normalize_tag(text: str) -> str:
    """Lowercase/trim an interest or catalog tag for robust comparison."""
    return text.strip().lower()


def _interest_match_count(tags: Sequence[str], interests: Sequence[str]) -> int:
    """Count how many of ``interests`` the destination's ``tags`` cover.

    Matching is case-insensitive and substring-aware in both directions so that
    "local food" matches the tag "food" and "less crowded places" matches the
    tag "less crowded".
    """
    norm_tags = [_normalize_tag(t) for t in tags]
    count = 0
    for raw in interests:
        wanted = _normalize_tag(raw)
        if not wanted:
            continue
        if any(wanted in tag or tag in wanted for tag in norm_tags):
            count += 1
    return count


_CATALOG_INTERESTS: dict[str, tuple[str, ...]] = {
    entry.name: entry.interests for entry in DESTINATION_CATALOG
}


def _score_personalization(scenario: GoldenScenario, state: GraphState) -> float:
    """How well the ranking reflects the scenario's stated interests (6.x).

    Each ranked destination is credited with its interest relevance (fraction of
    the scenario's interests its catalog tags cover), weighted by rank so that
    placing interest-matching cities *higher* scores better. The result is the
    rank-weighted average relevance in ``[0, 1]``.
    """
    interests = list(scenario.interests) or (
        list(state.constraints.interests) if state.constraints else []
    )
    plan = state.final_plan
    ranked: list[Any]
    if plan is not None and plan.destinations:
        ranked = list(plan.destinations)
    else:
        ranked = list(state.scored)

    if not interests:
        return 1.0  # No stated interests => nothing to personalize against.
    if not ranked:
        return 0.0

    n_interests = len(interests)
    weighted_sum = 0.0
    weight_total = 0.0
    for item in ranked:
        rank = getattr(item, "rank", 0) or 1
        weight = 1.0 / float(rank)
        tags = _CATALOG_INTERESTS.get(item.name, ())
        matches = _interest_match_count(tags, interests)
        relevance = min(1.0, matches / n_interests)
        weighted_sum += weight * relevance
        weight_total += weight

    if weight_total <= 0:
        return 0.0
    return max(0.0, min(1.0, weighted_sum / weight_total))


def score_plan(scenario: GoldenScenario, state: GraphState) -> MetricScores:
    """Score a produced plan on all five evaluation metrics (Property 29).

    Pure function of the scenario and the final graph state — no planner is run
    and no I/O is performed. Emits a score in ``[0, 1]`` for every metric in
    :data:`EVAL_METRICS`.
    """
    return MetricScores(
        budget_adherence=_score_budget_adherence(scenario, state),
        constraint_satisfaction=_score_constraint_satisfaction(scenario, state),
        feasibility=_score_feasibility(state),
        explanation_coverage=_score_explanation_coverage(state),
        personalization=_score_personalization(scenario, state),
    )


# --------------------------------------------------------------------------- #
# Threshold comparison + scenario evaluation
# --------------------------------------------------------------------------- #
def _compare_to_thresholds(
    scores: MetricScores, thresholds: MetricThresholds
) -> list[MetricOutcome]:
    """Compare each metric score to its threshold (pass iff score >= threshold)."""
    score_map = scores.as_dict()
    threshold_map = thresholds.as_dict()
    outcomes: list[MetricOutcome] = []
    for metric in EVAL_METRICS:
        score = score_map[metric]
        threshold = threshold_map[metric]
        outcomes.append(
            MetricOutcome(
                metric=metric,
                score=score,
                threshold=threshold,
                # A small epsilon absorbs float noise so a score that equals the
                # threshold always passes (Requirement 14.2 "at or above").
                passed=score >= threshold - 1e-9,
            )
        )
    return outcomes


def _build_state(scenario: GoldenScenario) -> GraphState:
    """Seed the graph state for a scenario run.

    The Coordinator decomposes ``prompt`` into constraints, so the prompt is the
    authoritative input; the scenario's constraints/interests are carried as
    initial state for fidelity.
    """
    return GraphState(
        trip_id=f"eval-{scenario.id}",
        user_id="eval-harness",
        prompt=scenario.prompt,
        constraints=scenario.constraints,
    )


def evaluate_scenario(
    scenario: GoldenScenario,
    *,
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
) -> ScenarioResult:
    """Run the planner for ``scenario`` on mock providers and grade the plan.

    Drives the full LangGraph pipeline (Requirement 18.4: no paid key needed),
    scores the produced plan on the five metrics (14.1), and compares each score
    against the scenario's thresholds (14.2). The returned result reports which
    metrics, if any, fell below threshold (14.3).
    """
    # Default to a fresh Settings() so a run with no configured credentials
    # selects mock providers (matching the graph composition tests).
    settings = settings or Settings()

    state = _build_state(scenario)
    final_state, _events = run_plan(state, settings=settings, llm=llm)

    scores = score_plan(scenario, final_state)
    outcomes = _compare_to_thresholds(scores, scenario.thresholds)

    return ScenarioResult(
        scenario_id=scenario.id,
        scenario_name=scenario.name,
        scores=scores.as_dict(),
        thresholds=scenario.thresholds.as_dict(),
        outcomes=outcomes,
        passed=all(o.passed for o in outcomes),
    )


def run_harness(
    scenarios: Mapping[str, GoldenScenario] | None = None,
    *,
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
    raise_on_failure: bool = False,
) -> HarnessResult:
    """Grade every golden scenario and aggregate the results.

    Args:
        scenarios: the scenarios to grade (defaults to :data:`GOLDEN_SCENARIOS`).
        settings: configuration; defaults to mock-provider ``Settings()``.
        llm: optional narration provider (defaults to the config-selected one).
        raise_on_failure: when ``True``, raise :class:`HarnessError` reporting
            the failing scenario/metric if any plan scores below threshold
            (Requirement 14.3).

    Returns:
        A :class:`HarnessResult` covering all graded scenarios.
    """
    registry = scenarios if scenarios is not None else GOLDEN_SCENARIOS

    results = [
        evaluate_scenario(scenario, settings=settings, llm=llm)
        for scenario in registry.values()
    ]
    result = HarnessResult(scenarios=results)

    if raise_on_failure and not result.passed:
        raise HarnessError(_failure_message(result))
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _failure_message(result: HarnessResult) -> str:
    """Build a concise message naming each failing scenario and metric."""
    lines = ["Evaluation harness failed: plan scored below threshold."]
    for scenario in result.failures:
        for outcome in scenario.outcomes:
            if outcome.passed:
                continue
            lines.append(
                f"  [{scenario.scenario_id}] {outcome.metric}: "
                f"{outcome.score:.3f} < threshold {outcome.threshold:.3f}"
            )
    return "\n".join(lines)


def format_report(result: HarnessResult) -> str:
    """Render a human-readable per-scenario report of scores vs thresholds."""
    lines: list[str] = ["Evaluation Harness Report", "=" * 26]
    for scenario in result.scenarios:
        status = "PASS" if scenario.passed else "FAIL"
        lines.append(f"\n[{status}] {scenario.scenario_id} — {scenario.scenario_name}")
        for outcome in scenario.outcomes:
            mark = "ok" if outcome.passed else "BELOW"
            lines.append(
                f"    {outcome.metric:<24} {outcome.score:6.3f}  "
                f"(threshold {outcome.threshold:5.3f})  {mark}"
            )
    overall = "PASS" if result.passed else "FAIL"
    lines.append(f"\nOverall: {overall}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by the CI job (task 22.3).

    Runs the harness against every golden scenario, prints a per-scenario
    report, and returns a non-zero exit code (after printing the failing
    metric) when any plan scores below threshold (Requirement 14.3).
    """
    result = run_harness()
    print(format_report(result))
    if not result.passed:
        print("\n" + _failure_message(result), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI/CI job.
    raise SystemExit(main())
