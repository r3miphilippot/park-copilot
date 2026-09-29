import sys
from datetime import datetime

import httpx
import respx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.clock import frozen_now
from app.config import PARIS_TZ
from app.tools import TOOLS
from app.tools.weather import OPEN_METEO_URL
from mcp_server.server import build_server
from tests.test_tools import _hourly

TOOL_NAMES = {fn.__name__ for fn in TOOLS}


async def test_stdio_server_end_to_end():
    """Real protocol: the server runs as a subprocess, like in Claude Desktop / Claude Code.
    Only calls that need no network are made."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        assert init.server_info.name == "park-copilot"
        assert "Never invent a wait time" in init.instructions

        tools = {t.name: t for t in (await session.list_tools()).tools}
        assert set(tools) == TOOL_NAMES
        assert all(t.annotations.read_only_hint for t in tools.values())
        assert tools["search_park_guide"].annotations.open_world_hint is False
        schema = tools["get_live_wait_times"].input_schema["properties"]["park"]
        assert schema["enum"] == ["disneyland_park", "adventure_world"]

        past = await session.call_tool("get_weather", {"day": "2020-01-01"})
        assert past.is_error is True
        assert "in the past" in past.content[0].text  # the reason reaches the client

        invalid = await session.call_tool("get_live_wait_times", {"park": "parc_asterix"})
        assert invalid.is_error is True


@respx.mock
async def test_successful_call_returns_structured_content():
    respx.get(url__startswith=OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=_hourly({14: (1.2, 80)}))
    )
    server = build_server()
    weather = next(t for t in await server.list_tools() if t.name == "get_weather")
    # The advertised output is the forecast itself, not "forecast or error".
    assert "rainy_hours" in weather.output_schema["properties"]

    with frozen_now(datetime(2026, 7, 14, 10, 30, tzinfo=PARIS_TZ)):
        result = await server.call_tool("get_weather", {"day": "2026-07-15"})
    assert result.is_error is False
    assert result.structured_content["rainy_hours"] == ["14:00"]
    assert result.structured_content["source"] == "Open-Meteo.com"
