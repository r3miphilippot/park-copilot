from datetime import date, datetime

import groq
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from app.agent.graph import (
    UNAVAILABLE_MESSAGE,
    ModeDecision,
    apply_mode_guards,
    build_graph,
    history_for_llm,
)
from app.agent.prompts import build_system_prompt, calendar
from app.clock import frozen_now
from app.config import PARIS_TZ
from app.llm import LLMRouter, Provider
from tests.fakes import ScriptedChatModel, mode_decision, tool_call

TUESDAY_1030 = datetime(2026, 7, 14, 10, 30, tzinfo=PARIS_TZ)
TODAY = TUESDAY_1030.date()


# --------------------------------------------------------------------------- fake tools


class Echo(BaseModel):
    tool: str
    args: dict


def _echo(name, **args):
    return Echo(tool=name, args=args)


def get_live_wait_times(park: str) -> Echo:
    """Live waits.

    Args:
        park: park key.
    """
    return _echo("get_live_wait_times", park=park)


def compare_live_vs_typical(park: str) -> Echo:
    """Compare.

    Args:
        park: park key.
    """
    return _echo("compare_live_vs_typical", park=park)


def get_weather(day: str) -> Echo:
    """Weather.

    Args:
        day: date.
    """
    return _echo("get_weather", day=day)


def search_park_guide(query: str) -> Echo:
    """Guide.

    Args:
        query: text.
    """
    return _echo("search_park_guide", query=query)


FAKE_TOOLS = [get_live_wait_times, compare_live_vs_typical, get_weather, search_park_guide]
LIVE = {"get_live_wait_times", "compare_live_vs_typical"}


def make_graph(responses, max_iterations=6):
    model = ScriptedChatModel(responses=responses)
    graph = build_graph(
        LLMRouter(Provider("fake", model), None), tools=FAKE_TOOLS, max_iterations=max_iterations
    )
    return graph, model


async def run(graph, question, thread="t1"):
    config = {"configurable": {"thread_id": thread}}
    events = []
    with frozen_now(TUESDAY_1030):
        async for event in graph.astream(
            {"messages": [HumanMessage(question)]}, config, stream_mode="custom"
        ):
            events.append(event)
    state = (await graph.aget_state(config)).values
    return state, events


# --------------------------------------------------------------------------- pure helpers


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (ModeDecision(mode="in_park", target_date=date(2026, 7, 18)),
         ("planning", date(2026, 7, 18))),
        (ModeDecision(mode="planning", target_date=date(2026, 7, 1)), ("general", None)),
        (ModeDecision(mode="in_park"), ("in_park", TODAY)),
        (ModeDecision(mode="planning"), ("planning", TODAY)),
        (ModeDecision(mode="general"), ("general", None)),
    ],
)  # fmt: skip
def test_mode_guards(decision, expected):
    """A future date always means planning, whatever label the LLM chose."""
    assert apply_mode_guards(decision, TODAY) == expected


def test_calendar_gives_correct_weekdays():
    lines = calendar(TODAY).splitlines()
    assert lines[0] == "- mardi 14 juillet 2026 (aujourd'hui) = 2026-07-14"
    assert lines[4] == "- samedi 18 juillet 2026 = 2026-07-18"
    assert len(lines) == 15


def test_history_keeps_current_turn_and_drops_old_tool_outputs():
    old_call = tool_call("get_weather", {"day": "2026-07-15"})
    messages = [
        HumanMessage("q1"),
        old_call,
        ToolMessage("{huge json}", tool_call_id="call_1"),
        AIMessage("answer 1"),
        HumanMessage("q2"),
        tool_call("search_park_guide", {"query": "x"}, "call_2"),
        ToolMessage("{current}", tool_call_id="call_2"),
    ]
    kept = history_for_llm(messages)
    assert [m.content for m in kept] == ["q1", "answer 1", "q2", "", "{current}"]


@pytest.mark.parametrize(
    ("lang", "expected"),
    [("en", "Réponds en ANGLAIS"), ("fr", "Réponds en français."), (None, "langue du visiteur")],
)
def test_system_prompt_answer_language(lang, expected):
    assert expected in build_system_prompt(TUESDAY_1030, "general", None, lang=lang)


async def test_english_unavailable_message():
    request = httpx.Request("POST", "https://api.groq.com")
    limited = groq.RateLimitError("limit", response=httpx.Response(429, request=request), body=None)
    graph, _ = make_graph([limited, limited])
    config = {"configurable": {"thread_id": "en"}}
    with frozen_now(TUESDAY_1030):
        state = await graph.ainvoke({"messages": [HumanMessage("Plan?")], "lang": "en"}, config)
    assert state["messages"][-1].content.startswith("The AI service is temporarily overloaded")


def test_system_prompt_planning_mentions_target_weekday():
    prompt = build_system_prompt(TUESDAY_1030, "planning", date(2026, 7, 18))
    assert "mardi 14 juillet 2026, 10:30" in prompt
    assert "PLANIFICATION pour le samedi 18 juillet 2026 (weekday=5)" in prompt
    assert "2026-07-18" in prompt


# --------------------------------------------------------------------------- graph


async def test_planning_mode_hides_live_tools():
    graph, model = make_graph(
        [
            mode_decision("planning", "2026-07-18"),
            tool_call("get_weather", {"day": "2026-07-18"}),
            AIMessage("09:30 – Big Thunder Mountain"),
        ]
    )
    state, events = await run(graph, "Je viens samedi, un plan ?")

    assert (state["mode"], state["target_date"]) == ("planning", "2026-07-18")
    agent_tools = model.bound[1:]  # bound[0] is the structured output of detect_mode
    assert all(LIVE.isdisjoint(names) for names in agent_tools)
    tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert '"day":"2026-07-18"' in tool_msg.content
    assert state["messages"][-1].content == "09:30 – Big Thunder Mountain"

    types = [e["type"] for e in events]
    assert types == ["mode", "llm", "tool_start", "tool_end", "llm"]
    assert events[3]["ok"] is True and events[3]["duration_ms"] >= 0


async def test_in_park_mode_offers_live_tools():
    graph, model = make_graph(
        [
            mode_decision("in_park"),
            tool_call("compare_live_vs_typical", {"park": "disneyland_park"}),
            AIMessage("Va sur Pirates maintenant."),
        ]
    )
    state, _ = await run(graph, "Qu'est-ce qui est calme maintenant ?")
    assert (state["mode"], state["target_date"]) == ("in_park", "2026-07-14")
    assert set(model.bound[1]) >= LIVE


async def test_live_tool_requested_in_planning_mode_is_refused():
    graph, _ = make_graph(
        [
            mode_decision("planning", "2026-07-18"),
            tool_call("get_live_wait_times", {"park": "disneyland_park"}),
            AIMessage("Je n'ai pas de temps live pour samedi."),
        ]
    )
    state, events = await run(graph, "Attente samedi ?")
    tool_msg = next(m for m in state["messages"] if isinstance(m, ToolMessage))
    assert tool_msg.status == "error" and "not available" in tool_msg.content
    assert next(e for e in events if e["type"] == "tool_end")["ok"] is False


async def test_iteration_limit_forces_an_answer():
    graph, model = make_graph(
        [
            mode_decision("general"),
            tool_call("search_park_guide", {"query": "pluie"}),
            AIMessage("Réponse avec ce que j'ai."),
        ],
        max_iterations=1,
    )
    state, _ = await run(graph, "Conseils pluie ?")
    assert state["messages"][-1].content == "Réponse avec ce que j'ai."
    assert len(model.bound) == 2  # 2nd agent call: no tools bound at all
    last_system = model.received[-1][0]
    assert isinstance(last_system, SystemMessage) and "IMPORTANT" in last_system.content


async def test_llm_unavailable_gives_a_clear_message():
    request = httpx.Request("POST", "https://api.groq.com")
    limited = groq.RateLimitError("limit", response=httpx.Response(429, request=request), body=None)
    graph, _ = make_graph([limited, limited])  # detect_mode and agent both hit the limit
    state, events = await run(graph, "Un plan pour samedi ?")
    assert state["mode"] == "general"  # detection failed -> safest mode
    assert state["messages"][-1].content == UNAVAILABLE_MESSAGE
    assert any(e.get("unavailable") for e in events)


async def test_memory_is_kept_per_thread():
    graph, _ = make_graph(
        [
            mode_decision("general"),
            AIMessage("Réponse 1"),
            mode_decision("planning", "2026-07-15"),
            AIMessage("Réponse 2"),
            mode_decision("general"),
            AIMessage("Autre thread"),
        ]
    )
    await run(graph, "Question 1", thread="a")
    state_a, _ = await run(graph, "Et demain ?", thread="a")
    state_b, _ = await run(graph, "Bonjour", thread="b")
    assert [m.content for m in state_a["messages"]] == [
        "Question 1", "Réponse 1", "Et demain ?", "Réponse 2",
    ]  # fmt: skip
    assert len(state_b["messages"]) == 2
