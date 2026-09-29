"""FastAPI app: streaming chat (SSE), health, metrics.

    uv run uvicorn app.api.main:app --port 7860

SSE events sent by POST /chat, in order:
    mode        {"mode", "target_date"}            detected mode (planning / in_park / general)
    tool_start  {"id", "name", "args"}              a tool is running
    tool_end    {"id", "name", "ok", "duration_ms"}
    token       {"text"}                            a piece of the answer
    done        {"thread_id", "mode", "tools", "latency_ms", "fallback", ...}
    error       {"message"}                         unexpected failure (answer incomplete)
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.api.limits import DailyCap, RateLimiter, ThreadRegistry
from app.clock import now_paris
from app.config import Settings, get_settings
from app.observability import Metrics, RequestTrace, langfuse_callbacks, setup_logging

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

MESSAGES = {
    "rate_limited": {
        "fr": "Doucement 🙂 Trop de messages en une minute. Réessaie dans {seconds} secondes.",
        "en": "Easy 🙂 Too many messages in a minute. Try again in {seconds} seconds.",
    },
    "daily_cap": {
        "fr": "Park Copilot a atteint sa limite de conversations pour aujourd'hui (quota "
        "gratuit). Reviens demain !",
        "en": "Park Copilot has reached its conversation limit for today (free quota). "
        "Come back tomorrow!",
    },
    "not_configured": {
        "fr": "Le service d'IA n'est pas configuré (clé API manquante).",
        "en": "The AI service is not configured (missing API key).",
    },
    "error": {
        "fr": "Une erreur inattendue est survenue. Réessaie dans un instant.",
        "en": "An unexpected error occurred. Please try again in a moment.",
    },
}
# French versions, kept as constants for the tests and the default language.
DAILY_CAP_MESSAGE = MESSAGES["daily_cap"]["fr"]
NOT_CONFIGURED_MESSAGE = MESSAGES["not_configured"]["fr"]
ERROR_MESSAGE = MESSAGES["error"]["fr"]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    # Answer language picked in the interface; omitted = the language of the question.
    lang: Literal["fr", "en"] | None = None

    def text(self, key: str, **values: Any) -> str:
        return MESSAGES[key][self.lang or "fr"].format(**values)


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def client_ip(request: Request) -> str:
    # Behind the Hugging Face proxy the real client is the first X-Forwarded-For entry.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _authorized(request: Request, token: str) -> bool:
    """Bearer <token>, or Basic auth whose password is the token (what scrapers support)."""
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer":
        return hmac.compare_digest(value, token)
    if scheme.lower() == "basic":
        try:
            _, _, password = base64.b64decode(value).decode().partition(":")
        except ValueError:
            return False
        return hmac.compare_digest(password, token)
    return False


def _warmup() -> None:
    """Pay the slow first calls (embedding model load, MongoDB TLS handshake) at startup,
    not during a visitor's first question."""
    from app.rag.index import get_guide_index
    from app.tools.history import get_history_store

    for name, step in [
        ("guide index", get_guide_index),
        ("mongodb", lambda: get_history_store().coverage()),
    ]:
        try:
            step()
            log.info("warmup ok: %s", name)
        except Exception as exc:  # the tools report the problem cleanly later on
            log.warning("warmup failed: %s: %r", name, exc)


def create_app(
    *, settings: Settings | None = None, graph: Any | None = None, warmup: bool = True
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.log_level)
        if app.state.graph is None:
            app.state.graph = _build_default_graph(settings)
        if warmup:
            # Background: the port opens immediately (Hugging Face waits for it).
            app.state.warmup = asyncio.create_task(asyncio.to_thread(_warmup))
        yield

    app = FastAPI(title="Park Copilot", version="0.1.0", lifespan=lifespan)
    app.state.graph = graph
    app.state.metrics = Metrics()
    limiter = RateLimiter(settings.rate_limit_per_minute)
    daily_cap = DailyCap(settings.daily_request_cap)
    threads = ThreadRegistry(settings.max_threads)
    callbacks = langfuse_callbacks(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # ------------------------------------------------------------------ routes

    @app.get("/health")
    async def health() -> dict:
        """Liveness probe and keep-alive target: touches neither the LLM nor MongoDB."""
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict:
        snapshot = app.state.metrics.snapshot()
        snapshot.update(await _history_summary())
        snapshot["llm"] = {
            "primary": f"{settings.llm_provider}:{settings.llm_model}",
            "fallback": f"{settings.fallback_provider}:{settings.fallback_model}",
        }
        return snapshot

    @app.get("/metrics/prometheus", include_in_schema=False)
    async def prometheus_metrics(request: Request) -> Response:
        """Prometheus text format, scraped by Grafana Cloud (Metrics Endpoint integration)."""
        if settings.metrics_token and not _authorized(request, settings.metrics_token):
            return Response(status_code=401, headers={"WWW-Authenticate": "Bearer"})
        return Response(app.state.metrics.prometheus.render(), media_type=PROMETHEUS_CONTENT_TYPE)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        thread_id = body.thread_id or uuid.uuid4().hex
        trace = RequestTrace(thread_id)

        metrics = app.state.metrics
        allowed, retry_in = limiter.hit(client_ip(request))
        if not allowed:
            text = body.text("rate_limited", seconds=retry_in)
            return _message_response(text, trace, "rate_limited", metrics)
        if app.state.graph is None:
            return _message_response(body.text("not_configured"), trace, "error", metrics)
        if not daily_cap.try_acquire(now_paris().date()):
            return _message_response(body.text("daily_cap"), trace, "daily_cap", metrics)

        for old_thread in threads.touch(thread_id):
            app.state.graph.checkpointer.delete_thread(old_thread)

        stream = _stream_agent(app.state.graph, body, trace, callbacks, metrics)
        return StreamingResponse(stream, media_type="text/event-stream", headers=SSE_HEADERS)

    return app


def _build_default_graph(settings: Settings):
    from app.agent.graph import build_graph
    from app.llm import LLMUnavailable, build_router

    try:
        return build_graph(build_router(settings), max_iterations=settings.agent_max_iterations)
    except LLMUnavailable as exc:
        log.error("agent disabled: %s", exc)
        return None


async def _history_summary() -> dict[str, Any]:
    """Size of the collected history: stable numbers, unlike the in-memory request counters
    that restart at 0 on each deploy (shown on the portfolio)."""
    from app.db import get_snapshots_collection
    from app.tools.history import get_history_store

    def read() -> dict[str, Any]:
        collection = get_snapshots_collection()
        last = collection.find_one({}, {"fetched_at": 1}, sort=[("fetched_at", -1)])
        return {
            "history_days": get_history_store().coverage().days,
            "history_snapshots": collection.estimated_document_count(),  # metadata, cheap
            "last_snapshot_at": last["fetched_at"].isoformat() if last else None,
        }

    try:
        # 10 s covers a cold MongoDB connection; once warm (startup warm-up) it takes ~10 ms.
        return await asyncio.wait_for(asyncio.to_thread(read), timeout=10)
    except Exception:
        return {"history_days": None, "history_snapshots": None, "last_snapshot_at": None}


def _message_response(
    text: str, trace: RequestTrace, status: str, metrics: Metrics
) -> StreamingResponse:
    """A limit or configuration problem is answered like a normal chat message (HTTP 200):
    the chat widget just displays it, no special error handling needed."""

    async def stream() -> AsyncIterator[str]:
        yield sse("token", {"text": text})
        trace.finish(status)
        metrics.record(trace)
        yield sse("done", {"thread_id": trace.thread_id, "limited": status})

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


async def _stream_agent(
    graph, body: ChatRequest, trace: RequestTrace, callbacks: list, metrics: Metrics
) -> AsyncIterator[str]:
    config = {
        "configurable": {"thread_id": trace.thread_id},
        "recursion_limit": 30,  # safety net on top of the agent's own iteration limit
        "callbacks": callbacks,
        "metadata": {"langfuse_session_id": trace.thread_id},
    }
    answer: list[str] = []
    status, error = "ok", None
    try:
        stream = graph.astream(
            {"messages": [HumanMessage(body.message)], "lang": body.lang},
            config,
            stream_mode=["messages", "custom"],
        )
        async for kind, payload in stream:
            if kind == "custom":
                trace.observe(payload)
                event = {k: v for k, v in payload.items() if k != "type"}
                if payload["type"] in ("mode", "tool_start", "tool_end"):
                    yield sse(payload["type"], event)
                if payload["type"] == "tool_start":
                    answer.clear()  # text written before calling tools was only a preamble
                continue
            chunk, meta = payload
            # Only the agent node writes the answer (detect_mode also calls the LLM).
            if meta.get("langgraph_node") != "agent" or not isinstance(chunk, AIMessage):
                continue
            if isinstance(chunk.content, str) and chunk.content:
                answer.append(chunk.content)
                yield sse("token", {"text": chunk.content})

        if not answer:
            # Messages created without a streaming LLM call (e.g. "service saturé").
            last = (await graph.aget_state(config)).values["messages"][-1]
            if isinstance(last, AIMessage) and isinstance(last.content, str) and last.content:
                yield sse("token", {"text": last.content})

        trace.finish()
        yield sse(
            "done",
            {
                "thread_id": trace.thread_id,
                "mode": trace.mode,
                "target_date": trace.target_date,
                "tools": [t.name for t in trace.tools],
                "providers": trace.providers,
                "fallback": trace.fallback,
                "latency_ms": trace.latency_ms,
            },
        )
    except asyncio.CancelledError:  # client closed the connection
        status, error = "error", "client disconnected"
        raise
    except Exception as exc:
        log.exception("chat failed")
        status, error = "error", repr(exc)
        yield sse("error", {"message": body.text("error")})
    finally:
        if trace.latency_ms is None or status == "error":
            trace.finish(status, error)
        metrics.record(trace)


app = create_app()
