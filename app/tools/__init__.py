"""The agent's tools, defined once and shared by the LangGraph agent and the MCP server."""

from app.tools.common import ToolError
from app.tools.compare import compare_live_vs_typical
from app.tools.guide import search_park_guide
from app.tools.history import get_typical_wait
from app.tools.live import get_live_wait_times
from app.tools.weather import get_weather

TOOLS = [
    get_live_wait_times,
    get_typical_wait,
    compare_live_vs_typical,
    get_weather,
    search_park_guide,
]

__all__ = [
    "TOOLS",
    "ToolError",
    "compare_live_vs_typical",
    "get_live_wait_times",
    "get_typical_wait",
    "get_weather",
    "search_park_guide",
]
