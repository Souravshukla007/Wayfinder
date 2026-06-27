"""Unit tests for the Japan golden scenario fixture (task 22.1).

These assert the fixture encodes the driving example correctly so the
evaluation harness (task 22.2) can consume it: the verbatim prompt, the
₹2 lakh budget cap, the five evaluation-metric thresholds, and registry
exposure. No providers are exercised here (Requirement 18.4).
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from app.eval.scenarios import EVAL_METRICS, GOLDEN_SCENARIOS, JAPAN_SCENARIO
from app.eval.scenarios.base import GoldenScenario, MetricThresholds


def test_prompt_is_the_verbatim_driving_example() -> None:
    # Requirement 18.1: the exact Japan driving prompt.
    assert JAPAN_SCENARIO.prompt == (
        "Plan a 7-day Japan trip in October under \u20b92 lakh. "
        "I like anime, photography, local food, and less crowded places."
    )


def test_interests_match_the_prompt() -> None:
    assert JAPAN_SCENARIO.interests == [
        "anime",
        "photography",
        "local food",
        "less crowded places",
    ]


def test_budget_cap_is_two_lakh_and_matches_constraints() -> None:
    # Requirement 18.2: ₹2,00,000 ceiling, consistent with trip constraints.
    assert JAPAN_SCENARIO.budget_cap == Decimal("200000")
    assert JAPAN_SCENARIO.constraints.budget == Decimal("200000")


def test_constraints_reflect_a_seven_day_trip() -> None:
    c = JAPAN_SCENARIO.constraints
    assert c.duration_days == 7
    assert c.max_cities >= 1
    assert c.max_same_day_travel > timedelta(0)


def test_thresholds_cover_all_five_metrics_in_unit_range() -> None:
    # Property 29 / Requirement 14.1: a threshold for each evaluation metric.
    thresholds = JAPAN_SCENARIO.thresholds.as_dict()
    assert set(thresholds) == set(EVAL_METRICS)
    assert all(0.0 <= v <= 1.0 for v in thresholds.values())


def test_hard_requirements_are_fully_enforced() -> None:
    t = JAPAN_SCENARIO.thresholds
    # Budget, constraints, feasibility, and explanation coverage are
    # definition-of-done guarantees for the driving example.
    assert t.budget_adherence == 1.0
    assert t.constraint_satisfaction == 1.0
    assert t.feasibility == 1.0
    assert t.explanation_coverage == 1.0


def test_scenario_is_registered_in_the_golden_registry() -> None:
    assert JAPAN_SCENARIO.id in GOLDEN_SCENARIOS
    assert GOLDEN_SCENARIOS[JAPAN_SCENARIO.id] is JAPAN_SCENARIO


def test_budget_cap_mismatch_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        GoldenScenario(
            id="bad",
            name="bad",
            prompt="p",
            interests=[],
            constraints=JAPAN_SCENARIO.constraints,
            thresholds=MetricThresholds(
                budget_adherence=1.0,
                constraint_satisfaction=1.0,
                feasibility=1.0,
                explanation_coverage=1.0,
                personalization=0.6,
            ),
            budget_cap=Decimal("1"),  # != constraints.budget
        )
