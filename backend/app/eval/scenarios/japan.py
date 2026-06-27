"""The Japan golden scenario — the v1 driving example (Requirement 18).

Prompt: *"Plan a 7-day Japan trip in October under ₹2 lakh. I like anime,
photography, local food, and less crowded places."*

This fixture encodes that prompt, the interests derived from it, the trip
constraints it implies, and the per-metric thresholds a produced plan must
clear. The evaluation harness (task 22.2) runs it end-to-end on mock providers
with no paid API key (Requirement 18.4) and checks:

- ranked, scored destination options with an ordering explanation (18.1),
- an itinerary skeleton whose total cost does not exceed ₹2,00,000 (18.2),
- an explainability ledger whose contributions sum to the chosen score (18.3).
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from app.models.schemas import TripConstraints

from .base import GoldenScenario, MetricThresholds

# The exact driving prompt from Requirement 18.1 (kept verbatim).
JAPAN_PROMPT = (
    "Plan a 7-day Japan trip in October under \u20b92 lakh. "
    "I like anime, photography, local food, and less crowded places."
)

# Interests parsed from the prompt; drive personalization of the ranking.
JAPAN_INTERESTS = ["anime", "photography", "local food", "less crowded places"]

# ₹2 lakh hard budget ceiling (Requirement 18.2).
JAPAN_BUDGET_CAP = Decimal("200000")

JAPAN_CONSTRAINTS = TripConstraints(
    budget=JAPAN_BUDGET_CAP,
    duration_days=7,
    # "less crowded places" + a 7-day window: keep the route compact.
    max_cities=3,
    max_hotel_changes=2,
    max_same_day_travel=timedelta(hours=4),
    dietary=[],
    interests=JAPAN_INTERESTS,
    solver_timeout=timedelta(seconds=10),
)

# Acceptance bar for the Japan plan. Budget, constraint satisfaction, and
# feasibility are hard requirements of the driving example (18.2) and the
# solver (9.x), so they must be fully met. Explanation coverage is a v1
# definition-of-done guarantee (1.x, 8.x). Personalization is graded more
# leniently because exact ranking depends on mock fixture data.
JAPAN_THRESHOLDS = MetricThresholds(
    budget_adherence=1.0,
    constraint_satisfaction=1.0,
    feasibility=1.0,
    explanation_coverage=1.0,
    personalization=0.6,
)

JAPAN_SCENARIO = GoldenScenario(
    id="japan-7d-october",
    name="Japan 7-day October under \u20b92 lakh",
    prompt=JAPAN_PROMPT,
    interests=JAPAN_INTERESTS,
    constraints=JAPAN_CONSTRAINTS,
    thresholds=JAPAN_THRESHOLDS,
    budget_cap=JAPAN_BUDGET_CAP,
)

__all__ = [
    "JAPAN_PROMPT",
    "JAPAN_INTERESTS",
    "JAPAN_BUDGET_CAP",
    "JAPAN_CONSTRAINTS",
    "JAPAN_THRESHOLDS",
    "JAPAN_SCENARIO",
]
