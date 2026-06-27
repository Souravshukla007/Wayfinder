"""Flight tool (Task 5.5, Requirement 3.5).

A deterministic tool function that resolves the configured flight provider via
the registry and returns its results. No LLM reasoning happens here.
"""

from __future__ import annotations

from app.config import Settings
from app.providers.base import FlightOption, FlightQuery
from app.providers.registry import get_flight_provider


def search_flights(
    query: FlightQuery, *, settings: Settings | None = None
) -> list[FlightOption]:
    """Return flight options for ``query`` from the configured provider."""
    return get_flight_provider(settings).search_flights(query)
