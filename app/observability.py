"""Observability: JSON logs, in-memory metrics, per-request traces, optional Langfuse.

Everything works without any external service; Langfuse is an optional extra.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.config import Settings

log = logging.getLogger("park_copilot")


# --------------------------------------------------------------------------- JSON logs


class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy to grep locally, and parsed by any log platform."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))  # structured extras, see RequestTrace
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "chromadb", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# --------------------------------------------------------------------------- request trace


@dataclass
class ToolCall:
    name: str
    duration_ms: int | None = None
    ok: bool | None = None


@dataclass
class RequestTrace:
    """Built from the agent's stream events, logged once at the end of each /chat request."""

    thread_id: str
    started: float = field(default_factory=time.perf_counter)
    mode: str | None = None
    target_date: str | None = None
    tools: list[ToolCall] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    fallback: bool = False
    llm_unavailable: bool = False
    status: str = "ok"  # ok | error | rate_limited | daily_cap
    error: str | None = None
    latency_ms: int | None = None

    def observe(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "mode":
            self.mode, self.target_date = event["mode"], event.get("target_date")
        elif kind == "tool_start":
            self.tools.append(ToolCall(event["name"]))
        elif kind == "tool_end":
            pending = next(
                (t for t in self.tools if t.name == event["name"] and t.ok is None), None
            )
            if pending:
                pending.duration_ms, pending.ok = event["duration_ms"], event["ok"]
        elif kind == "llm":
            if event.get("provider") and event["provider"] not in self.providers:
                self.providers.append(event["provider"])
            self.fallback |= bool(event.get("fallback"))
            self.llm_unavailable |= bool(event.get("unavailable"))

    def finish(self, status: str | None = None, error: str | None = None) -> RequestTrace:
        self.latency_ms = round((time.perf_counter() - self.started) * 1000)
        if status:
            self.status = status
        self.error = error
        log.info("chat_request", extra={"fields": self.log_fields()})
        return self

    def log_fields(self) -> dict[str, Any]:
        fields = {k: v for k, v in asdict(self).items() if k != "started"}
        return {"event": "chat_request", **fields}


# --------------------------------------------------------------------------- metrics


class Metrics:
    """Process-wide counters. Reset on restart, which is fine for a demo (no paid backend)."""

    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self.started_at = datetime.now(UTC)
        self.requests = 0
        self.errors = 0
        self.fallbacks = 0
        self.rate_limited = 0
        self.daily_capped = 0
        self.tool_calls: dict[str, int] = {}
        self._latencies: deque[int] = deque(maxlen=window)  # last N requests only

    def record(self, trace: RequestTrace) -> None:
        with self._lock:
            if trace.status == "rate_limited":
                self.rate_limited += 1
                return
            if trace.status == "daily_cap":
                self.daily_capped += 1
                return
            self.requests += 1
            self.errors += trace.status == "error"
            self.fallbacks += trace.fallback
            if trace.latency_ms is not None:
                self._latencies.append(trace.latency_ms)
            for tool in trace.tools:
                self.tool_calls[tool.name] = self.tool_calls.get(tool.name, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self._latencies)
            return {
                "since": self.started_at.isoformat(timespec="seconds"),
                "requests": self.requests,
                "error_rate": round(self.errors / self.requests, 3) if self.requests else 0.0,
                "llm_fallback_rate": round(self.fallbacks / self.requests, 3)
                if self.requests
                else 0.0,
                "latency_ms": {
                    "p50": _percentile(latencies, 50),
                    "p95": _percentile(latencies, 95),
                    "samples": len(latencies),
                },
                "rate_limited": self.rate_limited,
                "daily_cap_reached": self.daily_capped,
                "tool_calls": dict(self.tool_calls),
            }


def _percentile(sorted_values: list[int], pct: int) -> int | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    return round(statistics.quantiles(sorted_values, n=100, method="inclusive")[pct - 1])


# --------------------------------------------------------------------------- Langfuse


def langfuse_callbacks(settings: Settings) -> list:
    """LangChain callbacks sending traces to Langfuse, or [] when not configured."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return []
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # Registers the client globally; the handler finds it through the public key.
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )
        return [CallbackHandler(public_key=settings.langfuse_public_key)]
    except Exception as exc:  # misconfigured Langfuse must never break the app
        log.warning("Langfuse disabled: %r", exc)
        return []
