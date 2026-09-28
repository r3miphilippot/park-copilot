"""The LangGraph agent, built explicitly node by node.

    START -> detect_mode -> agent --(tool calls?)--> tools -> agent ... -> END

- detect_mode: small structured LLM call (mode + target date), then deterministic guards.
- agent:       the LLM, with only the tools allowed in the detected mode.
- tools:       runs the requested tools in parallel, emits tool_start / tool_end events.

Memory: a checkpointer keyed by `thread_id` keeps each conversation's messages.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.agent.prompts import (
    MODE_DETECTION_PROMPT,
    Mode,
    build_system_prompt,
    calendar,
    format_day,
)
from app.clock import now_paris
from app.llm import LLMRouter, LLMUnavailable
from app.tools import TOOLS, ToolError

log = logging.getLogger(__name__)

# Live tools only make sense today, in the park. Removing them from the planning mode
# enforces "never use live waits to plan a future date" in code, not only in the prompt.
LIVE_TOOLS = {"get_live_wait_times", "compare_live_vs_typical"}
PAST_EXCHANGES_KEPT = 6  # previous question/answer pairs sent back to the LLM

UNAVAILABLE_MESSAGE = (
    "Le service d'IA est momentanément saturé (quota gratuit atteint). "
    "Réessaie dans une minute, désolé !"
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    mode: Mode
    target_date: str | None  # ISO date
    iterations: int  # LLM calls for the current user message


class ModeDecision(BaseModel):
    """Classification of the visitor's request."""

    mode: Mode
    target_date: date | None = Field(default=None, description="YYYY-MM-DD, or null")


def apply_mode_guards(decision: ModeDecision, today: date) -> tuple[Mode, date | None]:
    """The LLM proposes, the code decides: dates always win over the LLM's label."""
    mode, target = decision.mode, decision.target_date
    if target is not None and target > today:
        return "planning", target  # future date: never in-park mode
    if target is not None and target < today:
        return "general", None  # past date: nothing to plan
    if mode == "in_park":
        return "in_park", today
    if mode == "planning":
        return "planning", target or today
    return "general", target


def history_for_llm(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Messages sent to the LLM: the current turn in full, but only the questions and final
    answers of previous turns. Old tool outputs are the biggest token cost, and Groq's free
    tier allows a few thousand tokens per minute."""
    last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    past = [
        m
        for m in messages[:last_human]
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
    ]
    return past[-2 * PAST_EXCHANGES_KEPT :] + messages[last_human:]


def _recent_questions(messages: list[BaseMessage], n: int = 3) -> str:
    questions = [m.content for m in messages if isinstance(m, HumanMessage)][-n:]
    if len(questions) == 1:
        return str(questions[0])
    *previous, last = questions
    return "Messages précédents :\n- " + "\n- ".join(map(str, previous)) + f"\n\nDemande : {last}"


def _emit(event: dict[str, Any]) -> None:
    """Custom stream event (tool_start, tool_end, mode, llm). No-op outside streaming."""
    with contextlib.suppress(Exception):  # not running under astream(stream_mode="custom")
        get_stream_writer()(event)


def build_graph(
    router: LLMRouter,
    *,
    tools: list[Callable] | None = None,
    checkpointer: Any | None = None,
    max_iterations: int = 6,
):
    tools = tools or TOOLS
    registry = {fn.__name__: fn for fn in tools}
    # LangChain reads each function's signature and docstring to build the JSON schema.
    lc_tools = {fn.__name__: StructuredTool.from_function(fn, parse_docstring=True) for fn in tools}

    def tools_for(mode: Mode) -> list[StructuredTool]:
        if mode == "in_park":
            return list(lc_tools.values())
        return [t for name, t in lc_tools.items() if name not in LIVE_TOOLS]

    # ------------------------------------------------------------------ nodes

    async def detect_mode(state: AgentState, config: RunnableConfig) -> dict:
        now = now_paris()
        previous = ""
        if state.get("mode"):
            previous = f"Contexte de la conversation : mode {state['mode']}"
            previous += f", date {state['target_date']}.\n" if state.get("target_date") else ".\n"
        prompt = MODE_DETECTION_PROMPT.format(
            today=format_day(now.date()),
            today_iso=now.date().isoformat(),
            time=f"{now:%H:%M}",
            previous=previous,
            calendar=calendar(now.date()),
        )
        try:
            result = await router.ainvoke(
                lambda m: m.with_structured_output(ModeDecision),
                [SystemMessage(prompt), HumanMessage(_recent_questions(state["messages"]))],
                config,
            )
            decision = result.value
        except Exception as exc:  # LLM down or unparsable output: safest mode, no live data
            log.warning("mode detection failed: %r", exc)
            decision = ModeDecision(mode="general")
        mode, target = apply_mode_guards(decision, now.date())
        _emit({"type": "mode", "mode": mode, "target_date": target and target.isoformat()})
        return {"mode": mode, "target_date": target and target.isoformat(), "iterations": 0}

    async def agent(state: AgentState, config: RunnableConfig) -> dict:
        iterations = state.get("iterations", 0)
        limit_reached = iterations >= max_iterations
        target = date.fromisoformat(state["target_date"]) if state.get("target_date") else None
        system = build_system_prompt(
            now_paris(), state["mode"], target, limit_reached=limit_reached
        )
        bound = [] if limit_reached else tools_for(state["mode"])
        try:
            result = await router.ainvoke(
                lambda m: m.bind_tools(bound) if bound else m,
                [SystemMessage(system), *history_for_llm(state["messages"])],
                config,
            )
        except LLMUnavailable as exc:
            log.error("LLM unavailable: %s", exc)
            _emit({"type": "llm", "provider": None, "fallback": True, "unavailable": True})
            return {"messages": [AIMessage(UNAVAILABLE_MESSAGE)], "iterations": iterations + 1}
        _emit({"type": "llm", "provider": result.provider, "fallback": result.fallback_used})
        return {"messages": [result.value], "iterations": iterations + 1}

    async def run_tools(state: AgentState) -> dict:
        allowed = {t.name for t in tools_for(state["mode"])}

        async def run_one(call: dict) -> ToolMessage:
            name = call["name"]
            _emit({"type": "tool_start", "id": call["id"], "name": name, "args": call["args"]})
            started = time.perf_counter()
            if name not in allowed:
                result: BaseModel = ToolError(tool=name, error=f"{name} is not available now.")
            else:
                # Tools are sync (httpx, pymongo): run them in threads, in parallel.
                # to_thread copies the context, so frozen_now() still applies in evals.
                result = await asyncio.to_thread(registry[name], **call["args"])
            duration_ms = round((time.perf_counter() - started) * 1000)
            ok = not isinstance(result, ToolError)
            _emit({"type": "tool_end", "id": call["id"], "name": name, "ok": ok,
                   "duration_ms": duration_ms})  # fmt: skip
            return ToolMessage(
                content=result.model_dump_json(exclude_none=True),  # compact: fewer tokens
                tool_call_id=call["id"],
                name=name,
                status="success" if ok else "error",
            )

        last = state["messages"][-1]
        return {"messages": await asyncio.gather(*(run_one(c) for c in last.tool_calls))}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    # ------------------------------------------------------------------ graph

    graph = StateGraph(AgentState)
    graph.add_node("detect_mode", detect_mode)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "detect_mode")
    graph.add_edge("detect_mode", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
