"""Golden scenario fixtures for the evaluation harness.

Exposes the typed fixture models and the registry of golden scenarios the
harness iterates over. The Japan driving example (Requirement 18) is the v1
golden scenario.
"""

from .base import EVAL_METRICS, GoldenScenario, MetricThresholds
from .japan import JAPAN_SCENARIO

# All golden scenarios the evaluation harness runs in CI. Keyed by scenario id
# so the harness can report results per scenario (Requirements 14.2, 14.3).
GOLDEN_SCENARIOS: dict[str, GoldenScenario] = {
    JAPAN_SCENARIO.id: JAPAN_SCENARIO,
}

__all__ = [
    "EVAL_METRICS",
    "GoldenScenario",
    "MetricThresholds",
    "JAPAN_SCENARIO",
    "GOLDEN_SCENARIOS",
]
