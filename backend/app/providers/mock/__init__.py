"""Mock provider implementations (Task 5.2, Requirements 2.2, 2.3).

These are the zero-key default providers. Each implements its domain Protocol
from :mod:`app.providers.base` and returns fixture data whose shapes mirror the
frontend prototype's ``src/lib/mockApi.ts`` so results render identically with
no paid API key. The implementations are deterministic and perform no network
I/O.
"""

from __future__ import annotations

import math

from app.providers.base import (
    DateRange,
    DistanceResult,
    Duration,
    EventOption,
    EventQuery,
    FlightOption,
    FlightQuery,
    GeoPoint,
    HotelOption,
    HotelQuery,
    TravelMode,
    WeatherForecast,
)
from app.providers.mock import fixtures

# Rough average ground speed (km/h) per travel mode for mock travel-time math.
_MODE_SPEED_KMH: dict[TravelMode, float] = {
    TravelMode.WALKING: 5.0,
    TravelMode.CYCLING: 15.0,
    TravelMode.DRIVING: 60.0,
    TravelMode.TRANSIT: 45.0,
    TravelMode.FLYING: 700.0,
}

_EARTH_RADIUS_KM = 6371.0088


def _haversine_km(a: GeoPoint, b: GeoPoint) -> float:
    """Great-circle distance between two geo points in kilometres."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


class MockFlightProvider:
    """Returns fixture flight options for the queried destination."""

    def search_flights(self, query: FlightQuery) -> list[FlightOption]:
        return fixtures.flights_for(query.destination)


class MockHotelProvider:
    """Returns fixture hotel options for the queried city."""

    def search_hotels(self, query: HotelQuery) -> list[HotelOption]:
        return fixtures.hotels_for(query.city)


class MockWeatherProvider:
    """Returns a coarse fixture forecast for the queried location."""

    def forecast(self, location: GeoPoint, date_range: DateRange) -> WeatherForecast:
        key = location.name or f"{location.lat},{location.lon}"
        return fixtures.weather_for(key)


class MockRoutesProvider:
    """Computes deterministic distance/travel-time from great-circle math."""

    def distance(self, a: GeoPoint, b: GeoPoint) -> DistanceResult:
        return DistanceResult(origin=a, destination=b, distance_km=_haversine_km(a, b))

    def travel_time(self, a: GeoPoint, b: GeoPoint, mode: TravelMode) -> Duration:
        km = _haversine_km(a, b)
        speed = _MODE_SPEED_KMH[mode]
        hours = km / speed if speed > 0 else 0.0
        return Duration(seconds=hours * 3600.0)


class MockEventsProvider:
    """Returns fixture event options for the queried city."""

    def search_events(self, query: EventQuery) -> list[EventOption]:
        return fixtures.events_for(query.city)


__all__ = [
    "MockFlightProvider",
    "MockHotelProvider",
    "MockWeatherProvider",
    "MockRoutesProvider",
    "MockEventsProvider",
]
