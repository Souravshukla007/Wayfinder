"""Provider abstraction layer interfaces (Task 5.1, Requirement 2.1).

One :class:`typing.Protocol` per external data domain (flights, hotels,
weather, maps/routes, events) plus the Pydantic v2 query/result models they
exchange. Concrete implementations live in :mod:`app.providers.mock` (fixtures,
the zero-key default) and :mod:`app.providers.real` (thin adapters that would
call Amadeus / OpenWeatherMap / Mapbox / Ticketmaster).

The result models deliberately mirror the shapes the validated frontend
prototype already consumes in ``src/lib/mockApi.ts`` (``FlightOpt``,
``HotelOpt``, ``EventOpt``) so mock results render identically with no paid key
(Requirements 2.2, 2.3).

The protocols are ``@runtime_checkable`` so the registry/integration tests can
assert that a resolved provider structurally implements its domain interface
(Task 5.7).

This module is import-safe: it defines only models and protocols, performs no
I/O, and imports nothing beyond ``pydantic`` and the standard library.
"""

from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Shared value objects
# =============================================================================


class GeoPoint(BaseModel):
    """A geographic point, optionally labelled with a place name."""

    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    name: str | None = None


class DateRange(BaseModel):
    """An inclusive start/end date range."""

    model_config = ConfigDict(extra="forbid")

    start: date
    end: date


class TravelMode(str, Enum):
    """How a leg between two points is travelled."""

    DRIVING = "driving"
    TRANSIT = "transit"
    WALKING = "walking"
    CYCLING = "cycling"
    FLYING = "flying"


class Duration(BaseModel):
    """A travel duration expressed in seconds, with a timedelta convenience."""

    model_config = ConfigDict(extra="forbid")

    seconds: float = Field(..., ge=0.0)

    @classmethod
    def from_timedelta(cls, value: timedelta) -> "Duration":
        return cls(seconds=value.total_seconds())

    @property
    def as_timedelta(self) -> timedelta:
        return timedelta(seconds=self.seconds)


# =============================================================================
# Flights
# =============================================================================


class FlightQuery(BaseModel):
    """A request for flight options between two cities."""

    model_config = ConfigDict(extra="forbid")

    origin: str = Field(..., min_length=1, description="Origin city/airport.")
    destination: str = Field(..., min_length=1, description="Destination city.")
    depart_date: date | None = None
    return_date: date | None = None
    passengers: int = Field(default=1, ge=1)


class FlightOption(BaseModel):
    """A flight option (shape mirrors the prototype ``FlightOpt``)."""

    model_config = ConfigDict(extra="forbid")

    airline: str
    price: str
    duration: str
    tag: str


@runtime_checkable
class FlightProvider(Protocol):
    """Searches for flight options."""

    def search_flights(self, query: FlightQuery) -> list[FlightOption]: ...


# =============================================================================
# Hotels
# =============================================================================


class HotelQuery(BaseModel):
    """A request for hotel options in a city."""

    model_config = ConfigDict(extra="forbid")

    city: str = Field(..., min_length=1)
    check_in: date | None = None
    check_out: date | None = None
    guests: int = Field(default=1, ge=1)


class HotelOption(BaseModel):
    """A hotel option (shape mirrors the prototype ``HotelOpt``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    rating: int = Field(..., ge=0, le=5)
    price: str
    distance: str
    tag: str


@runtime_checkable
class HotelProvider(Protocol):
    """Searches for hotel options."""

    def search_hotels(self, query: HotelQuery) -> list[HotelOption]: ...


# =============================================================================
# Weather
# =============================================================================


class WeatherForecast(BaseModel):
    """A coarse forecast for a location over a date range."""

    model_config = ConfigDict(extra="forbid")

    location: str
    summary: str
    condition: str
    high_c: float
    low_c: float
    precipitation_chance: float = Field(..., ge=0.0, le=1.0)


@runtime_checkable
class WeatherProvider(Protocol):
    """Forecasts weather for a location over a date range."""

    def forecast(self, location: GeoPoint, date_range: DateRange) -> WeatherForecast: ...


# =============================================================================
# Maps / Routes
# =============================================================================


class DistanceResult(BaseModel):
    """The great-circle / routed distance between two points."""

    model_config = ConfigDict(extra="forbid")

    origin: GeoPoint
    destination: GeoPoint
    distance_km: float = Field(..., ge=0.0)


@runtime_checkable
class RoutesProvider(Protocol):
    """Computes distance and travel time between two points."""

    def distance(self, a: GeoPoint, b: GeoPoint) -> DistanceResult: ...

    def travel_time(self, a: GeoPoint, b: GeoPoint, mode: TravelMode) -> Duration: ...


# =============================================================================
# Events
# =============================================================================


class EventQuery(BaseModel):
    """A request for events in a city over an optional date range."""

    model_config = ConfigDict(extra="forbid")

    city: str = Field(..., min_length=1)
    start_date: date | None = None
    end_date: date | None = None


class EventOption(BaseModel):
    """An event/festival option (shape mirrors the prototype ``EventOpt``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    date: str
    type: str


@runtime_checkable
class EventsProvider(Protocol):
    """Searches for events in a city."""

    def search_events(self, query: EventQuery) -> list[EventOption]: ...


__all__ = [
    "GeoPoint",
    "DateRange",
    "TravelMode",
    "Duration",
    "FlightQuery",
    "FlightOption",
    "FlightProvider",
    "HotelQuery",
    "HotelOption",
    "HotelProvider",
    "WeatherForecast",
    "WeatherProvider",
    "DistanceResult",
    "RoutesProvider",
    "EventQuery",
    "EventOption",
    "EventsProvider",
]
