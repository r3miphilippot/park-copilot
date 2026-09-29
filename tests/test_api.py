import base64
import json
import logging
from datetime import date

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.agent.graph import build_graph
from app.api.limits import DailyCap, RateLimiter, ThreadRegistry
from app.api.main import DAILY_CAP_MESSAGE, ERROR_MESSAGE, NOT_CONFIGURED_MESSAGE, create_app
from app.config import Settings
from app.llm import LLMRouter, Provider
from app.observability import JsonFormatter, RequestTrace, _percentile
from tests.fakes import ScriptedChatModel, mode_decision, tool_call
from tests.test_agent import FAKE_TOOLS


def settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def fake_graph(*responses):
    model = ScriptedChatModel(responses=list(responses))
    return build_graph(LLMRouter(Provider("fake", model), None), tools=FAKE_TOOLS)


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def chat(c, message="Un plan pour samedi ?", thread_id=None, ip="1.2.3.4"):
    body = {"message": message} | ({"thread_id": thread_id} if thread_id else {})
    response = await c.post("/chat", json=body, headers={"x-forwarded-for": ip})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


@pytest.fixture(autouse=True)
def _no_mongo(monkeypatch):
    """/metrics asks MongoDB for the size of the history: keep unit tests offline."""
    monkeypatch.setattr("app.api.main._history_summary", _fake_history_summary)


async def _fake_history_summary():
    return {"history_days": 12, "history_snapshots": 34_000, "last_snapshot_at": "2026-07-14"}


# --------------------------------------------------------------------------- chat


async def test_chat_streams_events_in_order():
    graph = fake_graph(
        mode_decision("planning", "2099-01-03"),
        tool_call("get_weather", {"date": "2099-01-03"}),
        AIMessage("09:30 – Big Thunder Mountain"),
    )
    app = create_app(settings=settings(), graph=graph, warmup=False)
    async with client(app) as c:
        events = await chat(c, thread_id="t-1")

    assert [e for e, _ in events] == ["mode", "tool_start", "tool_end", "token", "done"]
    assert events[0][1] == {"mode": "planning", "target_date": "2099-01-03"}
    assert events[1][1]["name"] == "get_weather" and events[2][1]["ok"] is True
    assert events[3][1] == {"text": "09:30 – Big Thunder Mountain"}
    done = events[-1][1]
    assert done["thread_id"] == "t-1" and done["tools"] == ["get_weather"]
    assert done["fallback"] is False and done["latency_ms"] >= 0


async def test_thread_id_is_created_and_memory_reused():
    graph = fake_graph(
        mode_decision("general"), AIMessage("Réponse 1"),
        mode_decision("general"), AIMessage("Réponse 2"),
    )  # fmt: skip
    app = create_app(settings=settings(), graph=graph, warmup=False)
    async with client(app) as c:
        first = await chat(c, "Bonjour")
        thread_id = first[-1][1]["thread_id"]
        await chat(c, "Et ensuite ?", thread_id=thread_id)
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    assert len(thread_id) == 32
    assert [m.content for m in state.values["messages"]] == [
        "Bonjour", "Réponse 1", "Et ensuite ?", "Réponse 2",
    ]  # fmt: skip


async def test_rate_limit_per_ip_answers_with_a_message():
    graph = fake_graph(mode_decision("general"), AIMessage("ok"))
    app = create_app(settings=settings(rate_limit_per_minute=1), graph=graph, warmup=False)
    async with client(app) as c:
        await chat(c, ip="9.9.9.9")
        limited = await chat(c, ip="9.9.9.9")
        metrics = (await c.get("/metrics")).json()
    assert "Trop de messages" in limited[0][1]["text"]
    assert limited[-1] == ("done", {"thread_id": limited[-1][1]["thread_id"],
                                    "limited": "rate_limited"})  # fmt: skip
    assert metrics["rate_limited"] == 1 and metrics["requests"] == 1


async def test_daily_cap_answers_with_a_message():
    graph = fake_graph(mode_decision("general"), AIMessage("ok"))
    app = create_app(settings=settings(daily_request_cap=1), graph=graph, warmup=False)
    async with client(app) as c:
        await chat(c, ip="1.1.1.1")
        capped = await chat(c, ip="2.2.2.2")  # another visitor: the cap is global
    assert capped[0][1]["text"] == DAILY_CAP_MESSAGE
    assert capped[-1][1]["limited"] == "daily_cap"


async def test_language_reaches_the_agent_and_the_limit_messages():
    graph = fake_graph(mode_decision("general"), AIMessage("Hello!"))
    app = create_app(settings=settings(daily_request_cap=1), graph=graph, warmup=False)
    async with client(app) as c:
        response = await c.post("/chat", json={"message": "Hi", "lang": "en", "thread_id": "en-1"})
        capped = parse_sse((await c.post("/chat", json={"message": "Hi", "lang": "en"})).text)
    assert response.status_code == 200
    state = await graph.aget_state({"configurable": {"thread_id": "en-1"}})
    assert state.values["lang"] == "en"
    assert capped[0][1]["text"].startswith("Park Copilot has reached")


async def test_unknown_language_is_rejected():
    app = create_app(settings=settings(), graph=fake_graph(), warmup=False)
    async with client(app) as c:
        assert (await c.post("/chat", json={"message": "Hola", "lang": "es"})).status_code == 422


async def test_missing_llm_configuration_answers_with_a_message():
    app = create_app(settings=settings(), graph=None, warmup=False)
    async with client(app) as c:
        events = await chat(c)
    assert events[0][1]["text"] == NOT_CONFIGURED_MESSAGE


async def test_unexpected_failure_sends_error_event():
    class BrokenGraph:
        checkpointer = None

        def astream(self, *args, **kwargs):
            raise RuntimeError("boom")

    app = create_app(settings=settings(), graph=BrokenGraph(), warmup=False)
    async with client(app) as c:
        events = await chat(c)
        metrics = (await c.get("/metrics")).json()
    assert events == [("error", {"message": ERROR_MESSAGE})]
    assert metrics["error_rate"] == 1.0


@pytest.mark.parametrize(
    "body",
    [{"message": ""}, {"message": "x" * 2001}, {"message": "ok", "thread_id": "bad id!"}],
)
async def test_invalid_requests_are_rejected(body):
    app = create_app(settings=settings(), graph=fake_graph(), warmup=False)
    async with client(app) as c:
        assert (await c.post("/chat", json=body)).status_code == 422


async def test_old_threads_are_evicted_from_memory():
    graph = fake_graph(*[r for _ in range(3) for r in (mode_decision("general"), AIMessage("ok"))])
    app = create_app(settings=settings(max_threads=2), graph=graph, warmup=False)
    async with client(app) as c:
        for thread in ("a", "b", "c"):
            await chat(c, thread_id=thread)
    remaining = {
        t for t in "abc" if (await graph.aget_state({"configurable": {"thread_id": t}})).values
    }
    assert remaining == {"b", "c"}


# --------------------------------------------------------------------------- other routes


async def test_health_and_index():
    app = create_app(settings=settings(), graph=None, warmup=False)
    async with client(app) as c:
        assert (await c.get("/health")).json() == {"status": "ok"}
        page = await c.get("/")
    assert page.status_code == 200 and "Powered by Queue-Times.com" in page.text


async def test_metrics_after_a_request():
    graph = fake_graph(
        mode_decision("general"), tool_call("search_park_guide", {"query": "pluie"}),
        AIMessage("ok"),
    )  # fmt: skip
    app = create_app(settings=settings(), graph=graph, warmup=False)
    async with client(app) as c:
        await chat(c)
        metrics = (await c.get("/metrics")).json()
    assert metrics["requests"] == 1 and metrics["error_rate"] == 0.0
    assert metrics["latency_ms"]["p50"] is not None and metrics["latency_ms"]["samples"] == 1
    assert metrics["tool_calls"] == {"search_park_guide": 1}
    assert metrics["history_days"] == 12 and metrics["history_snapshots"] == 34_000
    assert metrics["llm_fallback_rate"] == 0.0


async def test_prometheus_metrics_after_a_request():
    graph = fake_graph(
        mode_decision("general"), tool_call("search_park_guide", {"query": "pluie"}),
        AIMessage("ok"),
    )  # fmt: skip
    app = create_app(settings=settings(), graph=graph, warmup=False)
    async with client(app) as c:
        await chat(c)
        response = await c.get("/metrics/prometheus")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert 'park_copilot_chat_requests_total{status="ok"} 1.0' in text
    assert 'park_copilot_tool_calls_total{ok="true",tool="search_park_guide"} 1.0' in text
    assert "park_copilot_chat_duration_seconds_bucket" in text


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({}, 401),
        ({"Authorization": "Bearer wrong"}, 401),
        ({"Authorization": "Bearer s3cret"}, 200),
        ({"Authorization": "Basic " + base64.b64encode(b"grafana:s3cret").decode()}, 200),
        ({"Authorization": "Basic not-base64!"}, 401),
    ],
)
async def test_prometheus_metrics_token(headers, status):
    app = create_app(settings=settings(metrics_token="s3cret"), graph=None, warmup=False)
    async with client(app) as c:
        assert (await c.get("/metrics/prometheus", headers=headers)).status_code == status


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [("https://remiphilippot.vercel.app", True), ("https://evil.example", False)],
)
async def test_cors(origin, allowed):
    app = create_app(settings=settings(), graph=None, warmup=False)
    async with client(app) as c:
        response = await c.options(
            "/chat",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
        )
    assert (response.headers.get("access-control-allow-origin") == origin) is allowed


def test_allowed_origins_from_comma_separated_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.com, http://localhost:3000")
    assert settings().allowed_origins == ["https://a.com", "http://localhost:3000"]


# --------------------------------------------------------------------------- limits


def test_rate_limiter_sliding_window():
    limiter = RateLimiter(limit=2, window_s=60)
    assert limiter.hit("ip", now=0)[0] and limiter.hit("ip", now=10)[0]
    assert limiter.hit("ip", now=20) == (False, 40)  # first hit expires at t=60
    assert limiter.hit("other", now=20)[0]  # per client
    assert limiter.hit("ip", now=61)[0]


def test_daily_cap_resets_each_day():
    cap = DailyCap(1)
    assert cap.try_acquire(date(2026, 7, 14))
    assert not cap.try_acquire(date(2026, 7, 14))
    assert cap.try_acquire(date(2026, 7, 15))


def test_thread_registry_evicts_least_recently_used():
    registry = ThreadRegistry(2)
    assert registry.touch("a") == [] and registry.touch("b") == []
    registry.touch("a")  # a is now the most recent
    assert registry.touch("c") == ["b"]


# --------------------------------------------------------------------------- observability


def test_json_log_line_contains_trace_fields():
    trace = RequestTrace("t-9")
    for event in [
        {"type": "mode", "mode": "in_park", "target_date": "2026-07-14"},
        {"type": "tool_start", "name": "get_weather", "id": "1", "args": {}},
        {"type": "tool_end", "name": "get_weather", "id": "1", "ok": True, "duration_ms": 42},
        {"type": "llm", "provider": "groq:a", "fallback": False},
        {"type": "llm", "provider": "groq:b", "fallback": True},
    ]:
        trace.observe(event)
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "chat_request", None, None)
    record.fields = trace.finish().log_fields()
    line = json.loads(JsonFormatter().format(record))
    assert line["thread_id"] == "t-9" and line["mode"] == "in_park"
    assert line["tools"] == [{"name": "get_weather", "duration_ms": 42, "ok": True}]
    assert line["providers"] == ["groq:a", "groq:b"] and line["fallback"] is True


def test_percentiles():
    assert _percentile([], 50) is None
    assert _percentile([100], 95) == 100
    values = list(range(1, 101))
    assert (_percentile(values, 50), _percentile(values, 95)) == (50, 95)
