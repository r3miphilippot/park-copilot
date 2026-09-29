# Park Copilot

AI agent that helps a visitor plan their day at Disneyland Paris (both parks):
LangGraph, RAG, tool calling, MCP, FastAPI, data pipeline, evals, observability.

> Work in progress. Full documentation coming soon.

## Quick start

```bash
uv sync
cp .env.example .env                                 # then fill in MONGODB_URI
uv run python -m collector.collect --dry-run         # fetch wait times without a database
uv run pytest
```

## MCP

The agent's 5 tools are also exposed as an [MCP](https://modelcontextprotocol.io) server
(stdio transport), built with the official `mcp` Python SDK (its high-level API, FastMCP, is
called `MCPServer` since v2). The functions are the same ones the LangGraph agent uses, imported
from `app.tools`: they are defined once.

| Tool | What it returns |
|---|---|
| `get_live_wait_times(park)` | current wait of every ride (Queue-Times, cached 5 min) |
| `get_typical_wait(ride?, weekday?, hour?, park?)` | average / median wait, observations and days of history |
| `compare_live_vs_typical(park)` | live wait vs usual wait at this weekday and hour |
| `get_weather(day)` | hourly forecast at the resort, rainy hours (Open-Meteo, cached 1 h) |
| `search_park_guide(query, k?)` | relevant passages of the visit guide (RAG) |

Every tool is read-only. Failures (API down, date out of range, invalid argument) come back as
MCP error results (`isError: true`) with a readable reason, never as a crash.

The server reads its configuration (`MONGODB_URI`…) from the `.env` file at the repository root,
whatever directory it is started from. Without `MONGODB_URI`, the history tools return an error
and the other three still work.

### Claude Code

The repository ships a project-scoped [`.mcp.json`](.mcp.json): open the project in Claude Code
and approve the `park-copilot` server when prompted. To add it for all your projects instead:

```bash
claude mcp add --scope user park-copilot -- uv --directory /absolute/path/to/park-copilot run python -m mcp_server.server
```

Check it with `claude mcp list`, or `/mcp` inside a session.

### Claude Desktop

Open *Settings → Developer → Edit Config* and add the server to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "park-copilot": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/park-copilot",
        "run", "python", "-m", "mcp_server.server"
      ]
    }
  }
}
```

On Windows, write the path with double backslashes (`"C:\\Users\\me\\park-copilot"`). If Claude
Desktop cannot find `uv`, use its full path as `command` (`which uv` / `where uv`). Restart Claude
Desktop, then ask for example: *"Il va pleuvoir samedi à Disneyland Paris ? Quelles attractions
intérieures faire ?"*

### Debugging

```bash
npx @modelcontextprotocol/inspector uv run python -m mcp_server.server
```

---

Powered by [Queue-Times.com](https://queue-times.com/) · Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0)
