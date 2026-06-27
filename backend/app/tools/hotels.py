"""Hotel tool (Task 5.5, Requirement 3.5).

Deterministic tool resolving the configured hotel provider via the registry.
No LLM reasoning happens here.
"""

from __future__ import annotations

from app.config import Settings
from app.providers.base import HotelOption, HotelQuery
from app.providers.registry import get_hotel_provider


def search_hotels(
    query: HotelQuery, *, settings: Settings | None = None
) -> list[HotelOption]:
    """Return hotel options for ``query`` from the configured provider."""
    return get_hotel_provider(settings).search_hotels(query)
