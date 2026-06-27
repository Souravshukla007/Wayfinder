"""Golden scenario fixture types for the evaluation harness.

A *golden scenario* pairs a fixed planning prompt (plus its derived
constraints) with the per-metric thresholds a produced plan must meet. The
evaluation harness (task 22.2) consumes these fixtures: it runs the planner on
**mock providers** (no paid API key), scores the produced plan on the five
metrics, and fails the run when any metric falls below its threshold
(Requirements 14.1, 14.2, 14.3, 18.4).

Metric semantics (all scores are normalized to ``[0, 1]``; higher is better):

- ``budget_adherence``      — how well the plan's total cost respects the
  budget cap; ``1.0`` means total cost ≤ ``budget_cap`` (Requirement 18.2).
- ``constraint_satisfaction`` — fraction of declared trip constraints the plan
  satisfies (city count, hotel changes, dietary, interests, travel time).
- ``feasibility``           — whether the emitted itinerary skeleton is a
  feasible solution per the solver (Requirements 9.x).
- ``explanation_coverage``  — fraction of major recommendations that carry a
  rationale / explainability ledger (Requirements 1.x, 8.x).
- ``personalization``       — how well the ranking reflects the scenario's
  stated interests (Requirement 6.x).
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field, model_validator

from app.models.schemas import TripConstraints

# Canonical ordered list of evaluation metrics (Property 29 / Requirement 14.1).
# The harness must emit a score for each of these for every produced plan.
EVAL_METRICS: tuple[str, ...] = (
    "budget_adherence",
    "constraint_satisfaction",
    "feasibility",
    "explanation_coverage",
    "personalization",
)


class MetricThresholds(BaseModel):
    """Minimum acceptable score (in ``[0, 1]``) for each evaluation metric.

    A produced plan passes a metric when its score is **at or above** the
    corresponding threshold. Each field is bounded to ``[0, 1]`` so thresholds
    are always expressed on the same normalized scale as the metric scores.
    """

    budget_adherence: float = Field(ge=0.0, le=1.0)
    constraint_satisfaction: float = Field(ge=0.0, le=1.0)
    feasibility: float = Field(ge=0.0, le=1.0)
    explanation_coverage: float = Field(ge=0.0, le=1.0)
    personalization: float = Field(ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, float]:
        """Return the thresholds keyed by metric name in canonical order."""
        return {metric: getattr(self, metric) for metric in EVAL_METRICS}


class GoldenScenario(BaseModel):
    """A fixed prompt + constraints + thresholds the harness evaluates against.

    The scenario is provider-agnostic: it carries only the *inputs* (prompt,
    interests, constraints) and the *acceptance bar* (thresholds + budget cap).
    Running it on mock providers must complete without any paid API key
    (Requirement 18.4).
    """

    id: str
    name: str
    prompt: str
    interests: list[str] = Field(default_factory=list)
    constraints: TripConstraints
    thresholds: MetricThresholds
    # Hard budget ceiling for the scenario, surfaced separately from the
    # budget_adherence metric so the harness can assert the itinerary total
    # never exceeds it (Requirement 18.2). Mirrors ``constraints.budget``.
    budget_cap: Decimal

    @model_validator(mode="after")
    def _budget_cap_matches_constraints(self) -> "GoldenScenario":
        if self.budget_cap != self.constraints.budget:
            raise ValueError(
                "budget_cap must match constraints.budget "
                f"({self.budget_cap} != {self.constraints.budget})"
            )
        return self


__all__ = ["EVAL_METRICS", "MetricThresholds", "GoldenScenario"]
