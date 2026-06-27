"""Events tool (Task 5.5, Requirement 3.5).

Deterministic tool resolving the configured events provider via the registry.
No LLM reasoning happens here.
"""

from __future__ import annotations

from app.config import Settings
from app.providers.base import EventOption, EventQuery
from app.providers.registry import get_events_provider


def search_events(
    query: EventQuery, *, settings: Settings | None = None
) -> list[EventOption]:
    """Return event options for ``query`` from the configured provider."""
    return get_events_provider(settings).search_events(query)
