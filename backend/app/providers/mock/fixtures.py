"""Mock fixture data mirroring the frontend prototype (Requirements 2.2, 2.3).

The shapes here mirror ``src/lib/mockApi.ts`` (the Kyoto / Osaka / Hokkaido
fixtures) so that results returned by the mock providers render identically in
the existing frontend with no paid API key.

Fixtures are keyed by a normalized destination/city token. A generic default
set is returned for any unknown destination so the system always produces
plausible, renderable results on mock data.
"""

from __future__ import annotations

from app.providers.base import (
    EventOption,
    FlightOption,
    HotelOption,
    WeatherForecast,
)


def normalize_key(value: str) -> str:
    """Normalize a city/destination string to a lookup key (lowercased word)."""
    return value.strip().lower()


# --- Flights (mirror mockApi.ts FlightOpt fixtures) --------------------------

FLIGHTS: dict[str, list[FlightOption]] = {
    "kyoto": [
        FlightOption(airline="ANA NH 829", price="₹68,400", duration="11h 25m", tag="Best balance"),
        FlightOption(airline="JAL JL 740", price="₹62,200", duration="12h 50m", tag="Cheapest"),
        FlightOption(airline="SQ 11 via SIN", price="₹74,900", duration="10h 10m", tag="Shortest"),
    ],
    "osaka": [
        FlightOption(airline="JAL JL 727", price="₹61,800", duration="13h 05m", tag="Cheapest"),
        FlightOption(airline="Cathay CX 567", price="₹66,500", duration="11h 40m", tag="Best balance"),
    ],
    "hokkaido": [
        FlightOption(airline="ANA NH 60", price="₹71,200", duration="13h 50m", tag="Best balance"),
        FlightOption(airline="Air Japan via NRT", price="₹64,000", duration="15h 20m", tag="Cheapest"),
    ],
}

_DEFAULT_FLIGHTS: list[FlightOption] = [
    FlightOption(airline="IndiGo 6E 1402", price="₹54,900", duration="12h 15m", tag="Best balance"),
    FlightOption(airline="Emirates EK 511", price="₹49,700", duration="14h 05m", tag="Cheapest"),
]


# --- Hotels (mirror mockApi.ts HotelOpt fixtures) ----------------------------

HOTELS: dict[str, list[HotelOption]] = {
    "kyoto": [
        HotelOption(name="Hoshinoya Kyoto", rating=5, price="₹38,000/n", distance="12 min to center", tag="Most photogenic"),
        HotelOption(name="Nazuna Kyoto Gosho", rating=4, price="₹14,500/n", distance="8 min to center", tag="Best value"),
        HotelOption(name="The Thousand Kyoto", rating=5, price="₹22,000/n", distance="0 min · at station", tag="Most convenient"),
    ],
    "osaka": [
        HotelOption(name="Cross Hotel Osaka", rating=4, price="₹9,800/n", distance="5 min to Dotonbori", tag="Best value"),
        HotelOption(name="Conrad Osaka", rating=5, price="₹26,400/n", distance="Skyline views", tag="Splurge"),
    ],
    "hokkaido": [
        HotelOption(name="Sounkyo Onsen Ryokan", rating=4, price="₹16,800/n", distance="Gorge-front", tag="Most scenic"),
        HotelOption(name="JR Tower Hotel Sapporo", rating=4, price="₹12,400/n", distance="At station", tag="Most convenient"),
    ],
}

_DEFAULT_HOTELS: list[HotelOption] = [
    HotelOption(name="Grand Central Hotel", rating=4, price="₹11,000/n", distance="Central", tag="Best value"),
    HotelOption(name="The Skyline Suites", rating=5, price="₹24,000/n", distance="City views", tag="Splurge"),
]


# --- Events (mirror mockApi.ts EventOpt fixtures) ----------------------------

EVENTS: dict[str, list[EventOption]] = {
    "kyoto": [
        EventOption(name="Jidai Matsuri parade", date="Oct 22", type="Festival"),
        EventOption(name="Kurama Fire Festival", date="Oct 22", type="Festival"),
        EventOption(name="Kyoto Photo Biennale", date="Oct 18 – Nov 5", type="Exhibition"),
        EventOption(name="Tofuku-ji night illumination", date="Oct 25", type="Exhibition"),
    ],
    "osaka": [
        EventOption(name="Midosuji Autumn Party", date="Oct 26", type="Festival"),
        EventOption(name="Osaka Asian Film Fest", date="Oct 20 – 28", type="Exhibition"),
    ],
    "hokkaido": [
        EventOption(name="Sounkyo Momiji Festival", date="Oct 19 – 21", type="Festival"),
        EventOption(name="Sapporo Autumn Fest", date="Oct 1 – 31", type="Festival"),
        EventOption(name="PMF chamber concert", date="Oct 23", type="Concert"),
    ],
}

_DEFAULT_EVENTS: list[EventOption] = [
    EventOption(name="City Lights Music Festival", date="Oct 20", type="Festival"),
    EventOption(name="Contemporary Art Biennale", date="Oct 15 – 30", type="Exhibition"),
]


# --- Weather (coarse forecasts per destination) ------------------------------

WEATHER: dict[str, WeatherForecast] = {
    "kyoto": WeatherForecast(location="Kyoto", summary="Mild autumn days, cool evenings", condition="Partly cloudy", high_c=21.0, low_c=11.0, precipitation_chance=0.2),
    "osaka": WeatherForecast(location="Osaka", summary="Warm and pleasant", condition="Sunny", high_c=23.0, low_c=14.0, precipitation_chance=0.15),
    "hokkaido": WeatherForecast(location="Hokkaido", summary="Crisp autumn, brisk winds", condition="Clear", high_c=14.0, low_c=4.0, precipitation_chance=0.25),
}

_DEFAULT_WEATHER = WeatherForecast(
    location="Unknown",
    summary="Seasonal average conditions",
    condition="Partly cloudy",
    high_c=20.0,
    low_c=10.0,
    precipitation_chance=0.3,
)


def flights_for(destination: str) -> list[FlightOption]:
    return list(FLIGHTS.get(normalize_key(destination), _DEFAULT_FLIGHTS))


def hotels_for(city: str) -> list[HotelOption]:
    return list(HOTELS.get(normalize_key(city), _DEFAULT_HOTELS))


def events_for(city: str) -> list[EventOption]:
    return list(EVENTS.get(normalize_key(city), _DEFAULT_EVENTS))


def weather_for(location: str) -> WeatherForecast:
    forecast = WEATHER.get(normalize_key(location), _DEFAULT_WEATHER)
    # Return a copy stamped with the requested location name for default case.
    return forecast.model_copy(update={"location": forecast.location})
