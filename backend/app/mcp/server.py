"""Demonstrative MCP server exposing Maps/Weather tools (Task 16.1).

Exposes three tools over the Model Context Protocol (Requirement 12.1):

* ``forecast``     - weather forecast for a location over a date range.
* ``distance``     - distance between two geo points.
* ``travel_time``  - travel time between two geo points for a travel mode.

Each tool delegates to the deterministic provider layer through the
``app.tools`` functions, which resolve the configured provider via the
registry. With no real-provider credentials configured the registry returns
the mock providers, so this server runs end-to-end with zero paid API keys
(Requirements 2.2, 2.3, 12.2).

The server is built with the official MCP Python SDK's :class:`FastMCP`, which
serves results back to clients over the Model Context Protocol (Requirement
12.2). The module exposes :func:`build_server` (a factory returning a fresh
``FastMCP`` instance) and :func:`main` (a stdio entrypoint) so the server can
be embedded, tested, or launched as a standalone process.
"""

from __future__ import annotations

from datetime import date

from mcp.server.fastmcp import FastMCP

from app.config import Settings
from app.providers.base import DateRange, GeoPoint, TravelMode
from app.tools.routes import get_distance, get_travel_time
from app.tools.weather import get_forecast

SERVER_NAME = "wayfinder-maps-weather"

# Human-readable list of the travel modes accepted by ``travel_time`` so the
# error message stays in sync with the provider-layer enum.
_TRAVEL_MODES = ", ".join(mode.value for mode in TravelMode)


def _parse_date(value: str, field: str) -> date:
    """Parse an ISO-8601 (``YYYY-MM-DD``) date or raise a clear ValueError."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - message is the contract
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD); got {value!r}") from exc


def _parse_mode(value: str) -> TravelMode:
    """Parse a travel-mode string into a :class:`TravelMode`."""
    try:
        return TravelMode(value.lower())
    except ValueError as exc:
        raise ValueError(
            f"mode must be one of: {_TRAVEL_MODES}; got {value!r}"
        ) from exc


def build_server(settings: Settings | None = None) -> FastMCP:
    """Build the MCP server exposing the Maps/Weather tools.

    Tools delegate to the provider layer via ``app.tools``; passing ``settings``
    (or leaving it ``None`` to use the process configuration) controls which
    providers the registry resolves. Mock providers are used by default.
    """
    mcp = FastMCP(SERVER_NAME)

    @mcp.tool()
    def forecast(
        location_name: str,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
    ) -> dict:
        """Return the weather forecast for a location over a date range.

        Args:
            location_name: Human-readable place name (used as the lookup key).
            lat: Latitude in degrees (-90..90).
            lon: Longitude in degrees (-180..180).
            start_date: Inclusive range start as an ISO date (YYYY-MM-DD).
            end_date: Inclusive range end as an ISO date (YYYY-MM-DD).
        """
        location = GeoPoint(lat=lat, lon=lon, name=location_name)
        date_range = DateRange(
            start=_parse_date(start_date, "start_date"),
            end=_parse_date(end_date, "end_date"),
        )
        return get_forecast(location, date_range, settings=settings).model_dump()

    @mcp.tool()
    def distance(
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        origin_name: str | None = None,
        destination_name: str | None = None,
    ) -> dict:
        """Return the distance (km) between two geo points.

        Args:
            origin_lat: Origin latitude in degrees (-90..90).
            origin_lon: Origin longitude in degrees (-180..180).
            destination_lat: Destination latitude in degrees (-90..90).
            destination_lon: Destination longitude in degrees (-180..180).
            origin_name: Optional origin place name.
            destination_name: Optional destination place name.
        """
        a = GeoPoint(lat=origin_lat, lon=origin_lon, name=origin_name)
        b = GeoPoint(lat=destination_lat, lon=destination_lon, name=destination_name)
        return get_distance(a, b, settings=settings).model_dump()

    @mcp.tool()
    def travel_time(
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        mode: str = TravelMode.DRIVING.value,
        origin_name: str | None = None,
        destination_name: str | None = None,
    ) -> dict:
        """Return the travel time (seconds) between two geo points.

        Args:
            origin_lat: Origin latitude in degrees (-90..90).
            origin_lon: Origin longitude in degrees (-180..180).
            destination_lat: Destination latitude in degrees (-90..90).
            destination_lon: Destination longitude in degrees (-180..180).
            mode: Travel mode (one of: driving, transit, walking, cycling, flying).
            origin_name: Optional origin place name.
            destination_name: Optional destination place name.
        """
        a = GeoPoint(lat=origin_lat, lon=origin_lon, name=origin_name)
        b = GeoPoint(lat=destination_lat, lon=destination_lon, name=destination_name)
        duration = get_travel_time(a, b, _parse_mode(mode), settings=settings)
        return duration.model_dump()

    return mcp


# Module-level server instance for the stdio entrypoint and for clients/tests
# that prefer importing a ready-to-use server.
server = build_server()


def main() -> None:
    """Run the MCP server over stdio (the standard MCP transport)."""
    server.run()


if __name__ == "__main__":  # pragma: no cover - process entrypoint
    main()
