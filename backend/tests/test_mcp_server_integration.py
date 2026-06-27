"""Integration test for MCP tool invocation over the protocol (Task 16.2).

Drives the demonstrative MCP server (``app.mcp.server``) through a real MCP
``ClientSession`` connected to the server with the SDK's in-memory transport.
This exercises the full Model Context Protocol round trip - ``initialize``,
``list_tools`` and ``call_tool`` requests/responses - rather than calling the
tool handlers directly, so it verifies that:

* the server advertises the Maps/Weather tools to clients (Requirement 12.1), and
* invoking a tool returns its result back over the protocol (Requirement 12.2).

With no real-provider credentials configured the tools resolve the mock
providers, so the test runs end-to-end with zero paid API keys.
"""

from __future__ import annotations

import json

from mcp.shared.memory import create_connected_server_and_client_session

from app.mcp.server import build_server

# Kyoto and Osaka coordinates, matching the mock-provider fixtures used
# elsewhere in the suite.
_KYOTO = {"lat": 35.0116, "lon": 135.7681}
_OSAKA = {"lat": 34.6937, "lon": 135.5023}


def _tool_payload(result) -> dict:
    """Extract the tool result dict returned over the protocol.

    FastMCP serialises a tool's ``dict`` return value into a text content block
    carrying JSON; parse it back into a dict for assertions.
    """
    assert result.isError is False
    assert result.content, "expected at least one content block in the response"
    return json.loads(result.content[0].text)


async def test_mcp_server_lists_maps_and_weather_tools() -> None:
    """Requirement 12.1: the server exposes forecast/distance/travel_time."""
    async with create_connected_server_and_client_session(build_server()) as session:
        await session.initialize()
        listed = await session.list_tools()

    names = {tool.name for tool in listed.tools}
    assert {"forecast", "distance", "travel_time"} <= names


async def test_mcp_distance_tool_returns_result_over_protocol() -> None:
    """Requirement 12.2: invoking ``distance`` returns its result over MCP."""
    async with create_connected_server_and_client_session(build_server()) as session:
        await session.initialize()
        result = await session.call_tool(
            "distance",
            {
                "origin_lat": _KYOTO["lat"],
                "origin_lon": _KYOTO["lon"],
                "destination_lat": _OSAKA["lat"],
                "destination_lon": _OSAKA["lon"],
            },
        )

    payload = _tool_payload(result)
    assert payload["distance_km"] > 0


async def test_mcp_forecast_tool_returns_result_over_protocol() -> None:
    """Requirement 12.2: invoking ``forecast`` returns its result over MCP."""
    async with create_connected_server_and_client_session(build_server()) as session:
        await session.initialize()
        result = await session.call_tool(
            "forecast",
            {
                "location_name": "Kyoto",
                "lat": _KYOTO["lat"],
                "lon": _KYOTO["lon"],
                "start_date": "2024-10-18",
                "end_date": "2024-10-28",
            },
        )

    payload = _tool_payload(result)
    assert payload["location"] == "Kyoto"
    assert 0.0 <= payload["precipitation_chance"] <= 1.0
