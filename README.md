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

---

Powered by [Queue-Times.com](https://queue-times.com/)
