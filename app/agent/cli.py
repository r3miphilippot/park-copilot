"""Chat with the agent in a terminal (same event stream as the API).

uv run python -m app.agent.cli                      # interactive
uv run python -m app.agent.cli "Je viens samedi, un plan ?"
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid

from langchain_core.messages import AIMessageChunk, HumanMessage

from app.agent.graph import build_graph
from app.config import get_settings
from app.llm import build_router


async def ask(graph, question: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    stream = graph.astream(
        {"messages": [HumanMessage(question)]}, config, stream_mode=["messages", "custom"]
    )
    async for kind, payload in stream:
        if kind == "custom":
            if payload["type"] == "mode":
                print(f"  [mode: {payload['mode']}, date: {payload['target_date']}]")
            elif payload["type"] == "tool_start":
                print(f"  [tool] {payload['name']}({payload['args']})")
            elif payload["type"] == "tool_end":
                status = "ok" if payload["ok"] else "ERROR"
                print(f"  [tool] {payload['name']} -> {status} in {payload['duration_ms']} ms")
            elif payload["type"] == "llm":
                print(f"  [llm] {payload['provider']} fallback={payload['fallback']}")
        else:
            chunk, meta = payload
            # Only the agent node's text is the answer (detect_mode also calls the LLM).
            is_answer = meta.get("langgraph_node") == "agent" and isinstance(chunk, AIMessageChunk)
            if is_answer and isinstance(chunk.content, str) and chunk.content:
                print(chunk.content, end="", flush=True)
    print()


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = get_settings()
    graph = build_graph(build_router(settings), max_iterations=settings.agent_max_iterations)
    thread_id = uuid.uuid4().hex
    if len(sys.argv) > 1:
        await ask(graph, " ".join(sys.argv[1:]), thread_id)
        return
    print("Park Copilot, Ctrl+C to quit")
    while True:
        question = input("\n> ").strip()
        if question:
            await ask(graph, question, thread_id)


if __name__ == "__main__":
    asyncio.run(main())
