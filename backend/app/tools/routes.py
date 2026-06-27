"""Routes tool (Task 5.5, Requirement 3.5).

Deterministic tool resolving the configured routes provider via the registry.
No LLM reasoning happens here.
"""

from __future__ import annotations

from app.config import Settings
from app.providers.base import DistanceResult, Duration, GeoPoint, TravelMode
from app.providers.registry import get_routes_provider


def get_distance(
    a: GeoPoint, b: GeoPoint, *, settings: Settings | None = None
) -> DistanceResult:
    """Return the distance between two points from the configured provider."""
    return get_routes_provider(settings).distance(a, b)


def get_travel_time(
    a: GeoPoint,
    b: GeoPoint,
    mode: TravelMode = TravelMode.DRIVING,
    *,
    settings: Settings | None = None,
) -> Duration:
    """Return the travel time between two points for ``mode``."""
    return get_routes_provider(settings).travel_time(a, b, mode)
