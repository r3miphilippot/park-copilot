"""MCP server exposing the agent's 5 tools (stdio transport).

    uv run python -m mcp_server.server

The tools are imported from app.tools: the exact same functions the LangGraph agent uses.
Built with the official `mcp` SDK; its high-level API (FastMCP) is named MCPServer since v2.
"""

from __future__ import annotations

import functools
import inspect
import logging
import sys
import typing
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as AnticipatedToolError
from mcp.types import ToolAnnotations

from app.tools import TOOLS, ToolError

INSTRUCTIONS = """Tools to plan a day at Disneyland Paris (Disneyland Park and Disney Adventure
World): live and usual wait times, live-vs-usual comparison, hourly weather, and a visit guide.
- Parks: "disneyland_park" or "adventure_world". weekday: 0=Monday ... 6=Sunday. Hours are
  Paris time.
- Live waits only make sense today; to plan a future date use get_typical_wait + get_weather.
- Never invent a wait time, and say how many days of history an estimate relies on.
Wait times: Queue-Times.com (unofficial). Weather: Open-Meteo.com."""

# Human-friendly titles; every tool only reads data (readOnlyHint), and all but the guide
# call external services (openWorldHint).
TITLES = {
    "get_live_wait_times": "Live wait times",
    "get_typical_wait": "Usual wait times (history)",
    "compare_live_vs_typical": "Live vs usual wait times",
    "get_weather": "Hourly weather forecast",
    "search_park_guide": "Search the visit guide",
}


def as_mcp_tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a shared tool to MCP conventions without changing the tool itself.

    Our tools return `Model | ToolError`. In MCP a failure should be an error result
    (isError=true): the SDK's "anticipated failure" exception is raised instead, so its message
    reaches the client, and the advertised output schema is the success model only.
    """
    hints = typing.get_type_hints(fn)  # resolves the string annotations of the real function
    [success] = [t for t in typing.get_args(hints["return"]) if t is not ToolError]
    signature = inspect.signature(fn)
    params = [
        p.replace(annotation=hints.get(p.name, p.annotation)) for p in signature.parameters.values()
    ]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        if isinstance(result, ToolError):
            raise AnticipatedToolError(result.error)
        return result

    wrapper.__signature__ = signature.replace(parameters=params, return_annotation=success)  # type: ignore[attr-defined]
    return wrapper


def build_server() -> MCPServer:
    server = MCPServer(
        "park-copilot",
        title="Park Copilot",
        instructions=INSTRUCTIONS,
        version="0.1.0",
        log_level="WARNING",
    )
    for fn in TOOLS:
        server.add_tool(
            as_mcp_tool(fn),
            name=fn.__name__,
            title=TITLES.get(fn.__name__),
            annotations=ToolAnnotations(
                readOnlyHint=True, openWorldHint=fn.__name__ != "search_park_guide"
            ),
        )
    return server


def main() -> None:
    # stdout carries the MCP protocol: logs must go to stderr only.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    build_server().run("stdio")


if __name__ == "__main__":
    main()
