"""Provider registry tests (Tasks 5.4 and 5.7).

Covers:
- **Property 25: Config-driven provider selection** (Task 5.4) - the registry
  selects mock when a domain has no real credentials and routes to the real
  adapter when its credentials are present, with no code changes.
  Validates: Requirements 2.2, 2.4, 2.5.
- **Integration** (Task 5.7) - under the default mock configuration, each domain
  resolves a provider that implements its domain Protocol.
  Validates: Requirement 2.1.

Tagged: Feature: wayfinder-travel-planner, Property 25.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.config import ProviderKind, Settings
from app.providers.base import (
    EventsProvider,
    FlightProvider,
    HotelProvider,
    RoutesProvider,
    WeatherProvider,
)
from app.providers.mock import (
    MockEventsProvider,
    MockFlightProvider,
    MockHotelProvider,
    MockRoutesProvider,
    MockWeatherProvider,
)
from app.providers.real import (
    AmadeusFlightProvider,
    AmadeusHotelProvider,
    MapboxRoutesProvider,
    OpenWeatherMapProvider,
    TicketmasterEventsProvider,
)
from app.providers.registry import (
    get_events_provider,
    get_flight_provider,
    get_hotel_provider,
    get_provider,
    get_routes_provider,
    get_weather_provider,
)

# (domain, registry selector, expected mock class, expected real class).
_DOMAIN_TABLE = [
    ("flights", get_flight_provider, MockFlightProvider, AmadeusFlightProvider),
    ("hotels", get_hotel_provider, MockHotelProvider, AmadeusHotelProvider),
    ("weather", get_weather_provider, MockWeatherProvider, OpenWeatherMapProvider),
    ("routes", get_routes_provider, MockRoutesProvider, MapboxRoutesProvider),
    ("events", get_events_provider, MockEventsProvider, TicketmasterEventsProvider),
]

_PROTOCOLS = {
    "flights": FlightProvider,
    "hotels": HotelProvider,
    "weather": WeatherProvider,
    "routes": RoutesProvider,
    "events": EventsProvider,
}


def _settings_from_flags(
    *,
    amadeus: bool,
    owm: bool,
    mapbox: bool,
    ticketmaster: bool,
) -> Settings:
    """Build Settings with the given real-credential presence flags."""
    kwargs: dict[str, str] = {}
    if amadeus:
        kwargs["amadeus_api_key"] = "key"
        kwargs["amadeus_api_secret"] = "secret"
    if owm:
        kwargs["openweathermap_api_key"] = "owm"
    if mapbox:
        kwargs["mapbox_access_token"] = "mb"
    if ticketmaster:
        kwargs["ticketmaster_api_key"] = "tm"
    return Settings(_env_file=None, **kwargs)


# ---------------------------------------------------------------------------
# Property 25: config-driven provider selection
# Feature: wayfinder-travel-planner, Property 25
# Validates: Requirements 2.2, 2.4, 2.5
# ---------------------------------------------------------------------------


@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    amadeus=st.booleans(),
    owm=st.booleans(),
    mapbox=st.booleans(),
    ticketmaster=st.booleans(),
)
def test_property_25_config_driven_provider_selection(
    amadeus: bool, owm: bool, mapbox: bool, ticketmaster: bool
) -> None:
    settings = _settings_from_flags(
        amadeus=amadeus, owm=owm, mapbox=mapbox, ticketmaster=ticketmaster
    )

    for domain, selector, mock_cls, real_cls in _DOMAIN_TABLE:
        provider = selector(settings)
        kind = settings.provider_kind(domain)
        if kind is ProviderKind.REAL:
            assert isinstance(provider, real_cls), (
                f"{domain}: expected real adapter when credentials present"
            )
        else:
            assert isinstance(provider, mock_cls), (
                f"{domain}: expected mock provider when no credentials"
            )
        # Whichever kind is selected, it must implement the domain Protocol.
        assert isinstance(provider, _PROTOCOLS[domain])


def test_default_settings_select_all_mock() -> None:
    """With no credentials at all, every domain resolves to its mock provider."""
    settings = _settings_from_flags(
        amadeus=False, owm=False, mapbox=False, ticketmaster=False
    )
    for domain, selector, mock_cls, _real in _DOMAIN_TABLE:
        assert isinstance(selector(settings), mock_cls)
        assert settings.uses_mock(domain) is True


# ---------------------------------------------------------------------------
# Task 5.7: integration - each domain resolves a Protocol implementer on mock
# Validates: Requirement 2.1
# ---------------------------------------------------------------------------


def test_each_domain_resolves_protocol_under_mock_config() -> None:
    settings = _settings_from_flags(
        amadeus=False, owm=False, mapbox=False, ticketmaster=False
    )
    for domain, protocol in _PROTOCOLS.items():
        provider = get_provider(domain, settings)
        assert isinstance(provider, protocol), (
            f"{domain} provider does not implement its Protocol"
        )


def test_get_provider_rejects_unknown_domain() -> None:
    import pytest

    with pytest.raises(ValueError):
        get_provider("teleportation")
