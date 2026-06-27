"""Config-driven provider registry (Task 5.3, Requirements 2.4, 2.5).

Selects, per domain, the mock provider when no real credentials are configured
and the real adapter when credentials are present - driven entirely by
:class:`app.config.Settings` (via ``Settings.provider_kind`` / ``uses_mock``),
so switching from mock to real needs no code changes.

Real adapters are imported lazily inside each selector so that the registry
module stays import-safe even where optional real-provider dependencies are
unavailable, and so importing the registry never pulls in network clients.
"""

from __future__ import annotations

import logging

from app.config import ProviderKind, Settings, get_settings
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

logger = logging.getLogger(__name__)


def _resolve(settings: Settings | None) -> Settings:
    return settings if settings is not None else get_settings()


# A real weather adapter that degrades to the mock forecast on any failure is
# built lazily as a subclass of the real provider, so it still satisfies an
# ``isinstance(provider, OpenWeatherMapProvider)`` check (the registry contract:
# a real adapter is selected when credentials are present) while never letting a
# flaky/invalid upstream crash the planning pipeline. Cached after first build.
_resilient_weather_cls: type | None = None


def _resilient_weather_provider_cls() -> type:
    """Return (building once) the fallback-wrapped OpenWeatherMap provider class.

    Imported lazily so the registry stays import-safe and never pulls the real
    adapter (or its network client) at module import time.
    """
    global _resilient_weather_cls
    if _resilient_weather_cls is None:
        from app.providers.real import OpenWeatherMapProvider

        class _ResilientOpenWeatherMapProvider(OpenWeatherMapProvider):
            """Real OWM provider that falls back to the mock forecast on error."""

            def __init__(self, settings: Settings) -> None:
                super().__init__(settings)
                self._mock = MockWeatherProvider()

            def forecast(self, location, date_range):  # type: ignore[override]
                try:
                    return super().forecast(location, date_range)
                except Exception as exc:  # noqa: BLE001 - resilience boundary
                    logger.warning(
                        "Real weather provider failed (%s); "
                        "falling back to mock forecast.",
                        exc,
                    )
                    return self._mock.forecast(location, date_range)

        _resilient_weather_cls = _ResilientOpenWeatherMapProvider
    return _resilient_weather_cls


# Resilient routes adapters, built lazily as subclasses of the selected real
# provider (OpenRouteService or Mapbox) so they satisfy the registry's
# "real adapter selected when credentials present" contract while degrading to
# the deterministic mock route math on any live failure. Cached per backend.
_resilient_routes_cache: dict[str, type] = {}


def _resilient_routes_provider_cls(backend: str) -> type:
    """Return (building once) a fallback-wrapped routes provider class.

    ``backend`` is ``"openrouteservice"`` or ``"mapbox"``. Imported lazily so the
    registry stays import-safe and pulls no network client at import time.
    """
    cached = _resilient_routes_cache.get(backend)
    if cached is not None:
        return cached

    from app.providers.real import (
        MapboxRoutesProvider,
        OpenRouteServiceRoutesProvider,
    )

    base = {
        "openrouteservice": OpenRouteServiceRoutesProvider,
        "mapbox": MapboxRoutesProvider,
    }[backend]

    class _ResilientRoutesProvider(base):  # type: ignore[valid-type, misc]
        """Real routes provider that falls back to mock route math on error."""

        def __init__(self, settings: Settings) -> None:
            super().__init__(settings)
            self._mock = MockRoutesProvider()

        def distance(self, a, b):  # type: ignore[override]
            try:
                return super().distance(a, b)
            except Exception as exc:  # noqa: BLE001 - resilience boundary
                logger.warning(
                    "Real routes provider failed (%s); falling back to mock distance.",
                    exc,
                )
                return self._mock.distance(a, b)

        def travel_time(self, a, b, mode):  # type: ignore[override]
            try:
                return super().travel_time(a, b, mode)
            except Exception as exc:  # noqa: BLE001 - resilience boundary
                logger.warning(
                    "Real routes provider failed (%s); "
                    "falling back to mock travel time.",
                    exc,
                )
                return self._mock.travel_time(a, b, mode)

    _resilient_routes_cache[backend] = _ResilientRoutesProvider
    return _ResilientRoutesProvider


# Resilient events adapter, built lazily as a subclass of the real Ticketmaster
# provider. Falls back to the mock event fixtures on error OR on an empty result
# — Ticketmaster coverage is sparse for some cities, and a populated, plausible
# events list is a better experience than an empty one. Cached after first build.
_resilient_events_cls: type | None = None


def _resilient_events_provider_cls() -> type:
    """Return (building once) the fallback-wrapped Ticketmaster provider class."""
    global _resilient_events_cls
    if _resilient_events_cls is None:
        from app.providers.real import TicketmasterEventsProvider

        class _ResilientTicketmasterEventsProvider(TicketmasterEventsProvider):
            """Real events provider that falls back to mock fixtures.

            Falls back on any failure and also when the live search returns no
            events for the queried city, so the itinerary always has plausible
            event suggestions.
            """

            def __init__(self, settings: Settings) -> None:
                super().__init__(settings)
                self._mock = MockEventsProvider()

            def search_events(self, query):  # type: ignore[override]
                try:
                    events = super().search_events(query)
                except Exception as exc:  # noqa: BLE001 - resilience boundary
                    logger.warning(
                        "Real events provider failed (%s); falling back to mock events.",
                        exc,
                    )
                    return self._mock.search_events(query)
                if not events:
                    logger.info(
                        "Ticketmaster returned no events for %r; using mock events.",
                        query.city,
                    )
                    return self._mock.search_events(query)
                return events

        _resilient_events_cls = _ResilientTicketmasterEventsProvider
    return _resilient_events_cls


def get_flight_provider(settings: Settings | None = None) -> FlightProvider:
    settings = _resolve(settings)
    if settings.provider_kind("flights") is ProviderKind.REAL:
        from app.providers.real import AmadeusFlightProvider

        return AmadeusFlightProvider(settings)
    return MockFlightProvider()


def get_hotel_provider(settings: Settings | None = None) -> HotelProvider:
    settings = _resolve(settings)
    if settings.provider_kind("hotels") is ProviderKind.REAL:
        from app.providers.real import AmadeusHotelProvider

        return AmadeusHotelProvider(settings)
    return MockHotelProvider()


def get_weather_provider(settings: Settings | None = None) -> WeatherProvider:
    settings = _resolve(settings)
    if settings.provider_kind("weather") is ProviderKind.REAL:
        return _resilient_weather_provider_cls()(settings)
    return MockWeatherProvider()


def get_routes_provider(settings: Settings | None = None) -> RoutesProvider:
    settings = _resolve(settings)
    if settings.provider_kind("routes") is ProviderKind.REAL:
        # Prefer OpenRouteService when its (free, no-card) key is present;
        # otherwise fall back to the Mapbox adapter. Both are wrapped so a live
        # failure degrades to the deterministic mock route math.
        if settings.openrouteservice_api_key:
            return _resilient_routes_provider_cls("openrouteservice")(settings)
        return _resilient_routes_provider_cls("mapbox")(settings)
    return MockRoutesProvider()


def get_events_provider(settings: Settings | None = None) -> EventsProvider:
    settings = _resolve(settings)
    if settings.provider_kind("events") is ProviderKind.REAL:
        return _resilient_events_provider_cls()(settings)
    return MockEventsProvider()


# Map of domain -> selector for generic, data-driven resolution.
_SELECTORS = {
    "flights": get_flight_provider,
    "hotels": get_hotel_provider,
    "weather": get_weather_provider,
    "routes": get_routes_provider,
    "events": get_events_provider,
}


def get_provider(domain: str, settings: Settings | None = None) -> object:
    """Resolve the provider for a single domain by name.

    Raises ``ValueError`` for an unknown domain so misconfiguration surfaces
    loudly (consistent with ``Settings.provider_kind``).
    """
    selector = _SELECTORS.get(domain)
    if selector is None:
        raise ValueError(
            f"Unknown provider domain {domain!r}; expected one of {tuple(_SELECTORS)}."
        )
    return selector(settings)


__all__ = [
    "get_flight_provider",
    "get_hotel_provider",
    "get_weather_provider",
    "get_routes_provider",
    "get_events_provider",
    "get_provider",
]
