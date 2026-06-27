"""Mock provider + tool unit tests (Tasks 5.2 and 5.5).

Verifies the mock providers return fixture data whose shapes mirror the
frontend prototype (Requirements 2.2, 2.3) and that the deterministic tool
functions resolve providers via the registry with no LLM (Requirement 3.5).
"""

from __future__ import annotations

from datetime import date

from app.config import Settings
from app.providers.base import (
    DateRange,
    EventQuery,
    FlightOption,
    FlightQuery,
    GeoPoint,
    HotelQuery,
    TravelMode,
)
from app.tools import budget
from app.tools.events import search_events
from app.tools.flights import search_flights
from app.tools.hotels import search_hotels
from app.tools.routes import get_distance, get_travel_time
from app.tools.weather import get_forecast

MOCK_SETTINGS = Settings(_env_file=None)


def test_flights_tool_returns_kyoto_fixtures() -> None:
    options = search_flights(
        FlightQuery(origin="DEL", destination="Kyoto"), settings=MOCK_SETTINGS
    )
    assert all(isinstance(o, FlightOption) for o in options)
    airlines = {o.airline for o in options}
    assert "ANA NH 829" in airlines  # mirrors mockApi.ts Kyoto fixture
    # Shape parity with the frontend FlightOpt type.
    first = options[0]
    assert set(first.model_dump().keys()) == {"airline", "price", "duration", "tag"}


def test_hotels_tool_returns_osaka_fixtures() -> None:
    options = search_hotels(HotelQuery(city="Osaka"), settings=MOCK_SETTINGS)
    names = {h.name for h in options}
    assert "Cross Hotel Osaka" in names


def test_events_tool_returns_hokkaido_fixtures() -> None:
    options = search_events(EventQuery(city="Hokkaido"), settings=MOCK_SETTINGS)
    names = {e.name for e in options}
    assert "Sounkyo Momiji Festival" in names


def test_unknown_destination_falls_back_to_defaults() -> None:
    options = search_flights(
        FlightQuery(origin="DEL", destination="Atlantis"), settings=MOCK_SETTINGS
    )
    assert len(options) >= 1  # generic default fixtures, still renderable


def test_weather_tool_returns_forecast() -> None:
    forecast = get_forecast(
        GeoPoint(lat=35.0, lon=135.7, name="Kyoto"),
        DateRange(start=date(2024, 10, 18), end=date(2024, 10, 28)),
        settings=MOCK_SETTINGS,
    )
    assert forecast.location == "Kyoto"
    assert 0.0 <= forecast.precipitation_chance <= 1.0


def test_routes_tool_distance_and_travel_time_consistent() -> None:
    a = GeoPoint(lat=35.0116, lon=135.7681, name="Kyoto")
    b = GeoPoint(lat=34.6937, lon=135.5023, name="Osaka")
    dist = get_distance(a, b, settings=MOCK_SETTINGS)
    assert dist.distance_km > 0
    # Faster modes take less time over the same distance.
    drive = get_travel_time(a, b, TravelMode.DRIVING, settings=MOCK_SETTINGS)
    fly = get_travel_time(a, b, TravelMode.FLYING, settings=MOCK_SETTINGS)
    assert fly.seconds < drive.seconds


def test_mock_providers_are_deterministic() -> None:
    q = FlightQuery(origin="DEL", destination="Kyoto")
    assert search_flights(q, settings=MOCK_SETTINGS) == search_flights(
        q, settings=MOCK_SETTINGS
    )


# --- budget tool (pure arithmetic) ------------------------------------------


def test_budget_total_and_remaining() -> None:
    assert budget.total_cost([100, 200.5, "50"]) == budget._to_decimal("350.5")
    assert budget.remaining_budget(1000, 350) == budget._to_decimal("650")


def test_budget_within_budget_equality_allowed() -> None:
    assert budget.within_budget(1000, 1000) is True
    assert budget.within_budget(1000, 1000.01) is False


def test_budget_per_day_and_fit_ratio() -> None:
    assert budget.per_day_budget(700, 7) == budget._to_decimal("100")
    assert budget.budget_fit_ratio(1000, 500) == 0.5


def test_budget_rejects_non_positive_inputs() -> None:
    import pytest

    with pytest.raises(ValueError):
        budget.per_day_budget(700, 0)
    with pytest.raises(ValueError):
        budget.budget_fit_ratio(0, 100)
