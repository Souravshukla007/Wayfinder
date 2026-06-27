"""Weather tool (Task 5.5, Requirement 3.5).

Deterministic tool resolving the configured weather provider via the registry.
No LLM reasoning happens here.
"""

from __future__ import annotations

from app.config import Settings
from app.providers.base import DateRange, GeoPoint, WeatherForecast
from app.providers.registry import get_weather_provider


def get_forecast(
    location: GeoPoint, date_range: DateRange, *, settings: Settings | None = None
) -> WeatherForecast:
    """Return a forecast for ``location`` over ``date_range``."""
    return get_weather_provider(settings).forecast(location, date_range)
