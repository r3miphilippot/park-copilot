"""The agent's tools, defined once and shared by the LangGraph agent and the MCP server."""

from app.tools.common import ToolError
from app.tools.compare import compare_live_vs_typical
from app.tools.crowds import get_crowd_calendar
from app.tools.guide import search_park_guide
from app.tools.history import get_typical_wait
from app.tools.live import get_live_wait_times
from app.tools.park_hours import get_park_hours
from app.tools.planner import plan_day
from app.tools.rides import list_rides
from app.tools.weather import get_weather

TOOLS = [
    get_live_wait_times,
    get_typical_wait,
    compare_live_vs_typical,
    get_weather,
    get_park_hours,
    get_crowd_calendar,
    search_park_guide,
    list_rides,
    plan_day,
]

__all__ = [
    "TOOLS",
    "ToolError",
    "compare_live_vs_typical",
    "get_crowd_calendar",
    "get_live_wait_times",
    "get_park_hours",
    "get_typical_wait",
    "get_weather",
    "list_rides",
    "plan_day",
    "search_park_guide",
]
