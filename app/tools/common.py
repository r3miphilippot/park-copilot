"""Shared building blocks for the tools: error contract, guard decorator, TTL cache.

Every tool is a plain typed function returning a Pydantic model. The same functions are
exposed to the LangGraph agent and to the MCP server, so they are defined only once.
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError, validate_call
from pymongo.errors import PyMongoError

from app.db import DatabaseNotConfigured

log = logging.getLogger(__name__)


class ToolError(BaseModel):
    """Returned instead of raising: the agent reads it and tells the user what is missing."""

    status: Literal["error"] = "error"
    tool: str
    error: str


class ToolFailure(Exception):
    """Raised inside a tool for an expected problem; its message is shown to the agent."""


F = TypeVar("F", bound=Callable[..., Any])


def tool_guard(fn: F) -> F:
    """Validate arguments with Pydantic and turn every exception into a ToolError.

    `functools.wraps` keeps the original signature and docstring, which LangChain and
    FastMCP read to build the tool schema shown to the LLM.
    """
    validated = validate_call(fn)

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return validated(*args, **kwargs)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors()
            )
            message = f"Invalid arguments: {problems}"
        except ToolFailure as exc:
            message = str(exc)
        except DatabaseNotConfigured:
            message = "Historical database is not configured, no history available."
        except PyMongoError as exc:
            message = f"Historical database unavailable ({type(exc).__name__})."
        except httpx.HTTPError as exc:
            message = f"External API unavailable ({type(exc).__name__}). Do not guess values."
        except Exception as exc:  # last resort: never let a raw exception reach the agent
            log.exception("tool %s crashed", fn.__name__)
            message = f"Unexpected error ({type(exc).__name__})."
        log.warning("tool %s failed: %s", fn.__name__, message)
        return ToolError(tool=fn.__name__, error=message)

    return wrapper  # type: ignore[return-value]


class TTLCache:
    """Tiny thread-safe in-memory cache with per-entry expiry (no external service needed)."""

    def __init__(self, ttl_s: float) -> None:
        self.ttl_s = ttl_s
        self._data: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None or entry[0] < time.monotonic():
                self._data.pop(key, None)
                return None
            return entry[1]

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl_s, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
