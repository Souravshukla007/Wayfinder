"""Real provider adapters (Task 5.6, Requirement 2.4).

Thin adapters that implement the domain Protocols from
:mod:`app.providers.base` and would call the real upstream APIs:

- :class:`AmadeusFlightProvider` / :class:`AmadeusHotelProvider` -> Amadeus Self-Service
- :class:`OpenWeatherMapProvider` -> OpenWeatherMap
- :class:`MapboxRoutesProvider` -> Mapbox Directions/Matrix
- :class:`TicketmasterEventsProvider` -> Ticketmaster Discovery

These are routed in by :mod:`app.providers.registry` only when the
corresponding credentials are present in :class:`app.config.Settings`. The HTTP
client is imported lazily inside the request path so the module stays
import-safe even where ``httpx`` is unavailable, and so the test suite never
performs live network calls (constructing an adapter does no I/O).
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.config import Settings

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when a real provider cannot fulfil a request.

    The registry wraps real providers so that a ``ProviderError`` (or any other
    failure) degrades to the deterministic mock provider rather than crashing
    the planning pipeline.
    """


def _http_client(timeout: float = 10.0) -> Any:
    """Lazily construct an httpx client (kept off the import path)."""
    import httpx  # imported lazily so the module is import-safe without httpx

    return httpx.Client(timeout=timeout)


class _AmadeusBase:
    """Shared Amadeus configuration/credentials."""

    BASE_URL = "https://test.api.amadeus.com"

    def __init__(self, settings: "Settings") -> None:
        self._api_key = settings.amadeus_api_key
        self._api_secret = settings.amadeus_api_secret


class AmadeusFlightProvider(_AmadeusBase):
    """Flight search backed by the Amadeus Self-Service API."""

    def search_flights(self, query: FlightQuery) -> list[FlightOption]:
        # Thin adapter: a full implementation obtains an OAuth token and calls
        # GET /v2/shopping/flight-offers, then maps offers to FlightOption.
        raise NotImplementedError(
            "AmadeusFlightProvider requires live Amadeus credentials and network access."
        )


class AmadeusHotelProvider(_AmadeusBase):
    """Hotel search backed by the Amadeus Self-Service API."""

    def search_hotels(self, query: HotelQuery) -> list[HotelOption]:
        raise NotImplementedError(
            "AmadeusHotelProvider requires live Amadeus credentials and network access."
        )


class OpenWeatherMapProvider:
    """Weather forecasts backed by the OpenWeatherMap free-tier APIs.

    Uses two free-tier endpoints (no paid "One Call" subscription required):

    - Geocoding ``/geo/1.0/direct`` to turn a city name into coordinates.
    - 5-day / 3-hour ``/data/2.5/forecast`` for the actual forecast points.

    The 3-hourly points that fall inside the requested ``date_range`` are
    aggregated into the coarse :class:`WeatherForecast` the rest of the system
    consumes: ``high_c``/``low_c`` are the max/min over the window, ``condition``
    is the most frequent weather category, and ``precipitation_chance`` is the
    peak probability of precipitation. If the range lies outside the free 5-day
    window (no points match), all returned points are used as an approximation.
    """

    BASE_URL = "https://api.openweathermap.org"

    def __init__(self, settings: "Settings") -> None:
        self._api_key = settings.openweathermap_api_key

    def forecast(self, location: GeoPoint, date_range: DateRange) -> WeatherForecast:
        if not self._api_key:
            raise ProviderError("OpenWeatherMap API key is not configured.")

        lat, lon, resolved_name = self._resolve_coordinates(location)
        payload = self._get_forecast(lat, lon)
        points = payload.get("list") or []
        if not points:
            raise ProviderError("OpenWeatherMap returned no forecast points.")

        label = resolved_name or (payload.get("city") or {}).get("name") or "Unknown"
        return self._aggregate(points, date_range, label)

    # -- HTTP calls --------------------------------------------------------- #
    def _resolve_coordinates(self, location: GeoPoint) -> tuple[float, float, str | None]:
        """Resolve coordinates: geocode the name, else use the given lat/lon.

        The coordinator passes a labelled point with placeholder (0, 0) coords,
        so a non-empty name is geocoded for an accurate location; otherwise the
        supplied coordinates are used directly.
        """
        if location.name:
            try:
                with _http_client() as client:
                    resp = client.get(
                        f"{self.BASE_URL}/geo/1.0/direct",
                        params={"q": location.name, "limit": 1, "appid": self._api_key},
                    )
                    resp.raise_for_status()
                    hits = resp.json()
                if hits:
                    return float(hits[0]["lat"]), float(hits[0]["lon"]), hits[0].get("name")
            except Exception as exc:  # noqa: BLE001 - degrade to provided coords
                logger.warning("OpenWeatherMap geocoding failed for %r: %s", location.name, exc)
        return location.lat, location.lon, location.name

    def _get_forecast(self, lat: float, lon: float) -> dict[str, Any]:
        try:
            with _http_client() as client:
                resp = client.get(
                    f"{self.BASE_URL}/data/2.5/forecast",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "units": "metric",
                        "appid": self._api_key,
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenWeatherMap forecast request failed: {exc}") from exc

    # -- Response mapping --------------------------------------------------- #
    @staticmethod
    def _point_date(point: dict[str, Any]) -> date | None:
        text = point.get("dt_txt")
        if isinstance(text, str):
            try:
                return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").date()
            except ValueError:
                return None
        dt = point.get("dt")
        if isinstance(dt, (int, float)):
            return datetime.utcfromtimestamp(dt).date()
        return None

    def _aggregate(
        self, points: list[dict[str, Any]], date_range: DateRange, label: str
    ) -> WeatherForecast:
        in_range = [
            p
            for p in points
            if (d := self._point_date(p)) is not None
            and date_range.start <= d <= date_range.end
        ]
        sample = in_range or points

        highs: list[float] = []
        lows: list[float] = []
        conditions: list[str] = []
        pops: list[float] = []
        for p in sample:
            main = p.get("main") or {}
            if "temp_max" in main:
                highs.append(float(main["temp_max"]))
            if "temp_min" in main:
                lows.append(float(main["temp_min"]))
            weather = p.get("weather") or []
            if weather:
                conditions.append(str(weather[0].get("main", "")))
            pop = p.get("pop")
            if isinstance(pop, (int, float)):
                pops.append(float(pop))

        high_c = max(highs) if highs else 0.0
        low_c = min(lows) if lows else 0.0
        condition = Counter(c for c in conditions if c).most_common(1)
        condition_label = condition[0][0] if condition else "Unknown"
        precip = max(pops) if pops else 0.0
        precip = min(1.0, max(0.0, precip))

        summary = (
            f"{condition_label}, highs near {high_c:.0f}°C and lows near {low_c:.0f}°C"
        )
        return WeatherForecast(
            location=label,
            summary=summary,
            condition=condition_label,
            high_c=high_c,
            low_c=low_c,
            precipitation_chance=precip,
        )


class MapboxRoutesProvider:
    """Distance/travel-time backed by the Mapbox Directions/Matrix APIs."""

    BASE_URL = "https://api.mapbox.com"

    def __init__(self, settings: "Settings") -> None:
        self._access_token = settings.mapbox_access_token

    def distance(self, a: GeoPoint, b: GeoPoint) -> DistanceResult:
        raise NotImplementedError(
            "MapboxRoutesProvider requires a live access token and network access."
        )

    def travel_time(self, a: GeoPoint, b: GeoPoint, mode: TravelMode) -> Duration:
        raise NotImplementedError(
            "MapboxRoutesProvider requires a live access token and network access."
        )


class OpenRouteServiceRoutesProvider:
    """Distance/travel-time backed by the OpenRouteService Matrix API.

    OpenRouteService offers a free, no-card API key (openrouteservice.org/dev)
    over OpenStreetMap data. A single Matrix request returns both the routed
    distance (km) and duration (s) between two points, mapping cleanly onto the
    :class:`RoutesProvider` protocol.

    ORS coordinates are ``[lon, lat]`` ordered. Travel modes map to ORS
    *profiles*; ORS has no public-transit or flight routing, so ``TRANSIT`` is
    approximated by the driving profile and ``FLYING`` raises (the registry's
    resilient wrapper then falls back to the mock great-circle estimate).
    """

    BASE_URL = "https://api.openrouteservice.org"

    # TravelMode -> ORS routing profile.
    _PROFILE: dict[TravelMode, str] = {
        TravelMode.DRIVING: "driving-car",
        TravelMode.TRANSIT: "driving-car",  # ORS has no transit; approximate.
        TravelMode.CYCLING: "cycling-regular",
        TravelMode.WALKING: "foot-walking",
    }

    def __init__(self, settings: "Settings") -> None:
        self._api_key = settings.openrouteservice_api_key

    def _matrix(
        self, a: GeoPoint, b: GeoPoint, profile: str, metrics: list[str]
    ) -> dict[str, Any]:
        if not self._api_key:
            raise ProviderError("OpenRouteService API key is not configured.")
        try:
            with _http_client() as client:
                resp = client.post(
                    f"{self.BASE_URL}/v2/matrix/{profile}",
                    headers={
                        "Authorization": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        # ORS expects [lon, lat] pairs.
                        "locations": [[a.lon, a.lat], [b.lon, b.lat]],
                        "metrics": metrics,
                        "units": "km",
                    },
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenRouteService matrix request failed: {exc}") from exc

    def distance(self, a: GeoPoint, b: GeoPoint) -> DistanceResult:
        payload = self._matrix(a, b, "driving-car", ["distance"])
        try:
            distance_km = float(payload["distances"][0][1])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"OpenRouteService returned an unexpected matrix shape: {exc}"
            ) from exc
        return DistanceResult(origin=a, destination=b, distance_km=distance_km)

    def travel_time(self, a: GeoPoint, b: GeoPoint, mode: TravelMode) -> Duration:
        profile = self._PROFILE.get(mode)
        if profile is None:
            # FLYING (or any unmapped mode) has no ORS routing profile.
            raise ProviderError(
                f"OpenRouteService cannot route travel mode {mode.value!r}."
            )
        payload = self._matrix(a, b, profile, ["duration"])
        try:
            seconds = float(payload["durations"][0][1])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError(
                f"OpenRouteService returned an unexpected matrix shape: {exc}"
            ) from exc
        return Duration(seconds=seconds)


class TicketmasterEventsProvider:
    """Event search backed by the Ticketmaster Discovery API (free, no card).

    Calls ``/discovery/v2/events.json`` filtered by city (and the trip date
    window when supplied), mapping each event onto :class:`EventOption`
    (``name`` / ``date`` / ``type``). The event ``type`` is taken from the
    Ticketmaster classification *segment* (e.g. "Music", "Sports",
    "Arts & Theatre"), falling back to the genre or "Event".

    Coverage note: Ticketmaster is strongest in the US/Canada/Europe and may
    return no events for some cities. An empty result is a valid (non-error)
    response; the registry wrapper decides whether to surface it or fall back.
    """

    BASE_URL = "https://app.ticketmaster.com"
    PAGE_SIZE = 10

    def __init__(self, settings: "Settings") -> None:
        self._api_key = settings.ticketmaster_api_key

    def search_events(self, query: EventQuery) -> list[EventOption]:
        if not self._api_key:
            raise ProviderError("Ticketmaster API key is not configured.")

        params: dict[str, Any] = {
            "apikey": self._api_key,
            "city": query.city,
            "size": self.PAGE_SIZE,
            "sort": "date,asc",
        }
        if query.start_date is not None:
            params["startDateTime"] = f"{query.start_date.isoformat()}T00:00:00Z"
        if query.end_date is not None:
            params["endDateTime"] = f"{query.end_date.isoformat()}T23:59:59Z"

        try:
            with _http_client() as client:
                resp = client.get(
                    f"{self.BASE_URL}/discovery/v2/events.json", params=params
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Ticketmaster events request failed: {exc}") from exc

        events = ((payload.get("_embedded") or {}).get("events")) or []
        return [self._map_event(e) for e in events]

    @staticmethod
    def _map_event(event: dict[str, Any]) -> EventOption:
        name = str(event.get("name") or "Event")
        dates = (event.get("dates") or {}).get("start") or {}
        date_text = str(dates.get("localDate") or dates.get("dateTime") or "TBA")

        event_type = "Event"
        classifications = event.get("classifications") or []
        if classifications:
            first = classifications[0] or {}
            segment = (first.get("segment") or {}).get("name")
            genre = (first.get("genre") or {}).get("name")
            event_type = str(segment or genre or "Event")

        return EventOption(name=name, date=date_text, type=event_type)


__all__ = [
    "AmadeusFlightProvider",
    "AmadeusHotelProvider",
    "OpenWeatherMapProvider",
    "MapboxRoutesProvider",
    "OpenRouteServiceRoutesProvider",
    "TicketmasterEventsProvider",
    "ProviderError",
]
