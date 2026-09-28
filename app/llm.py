"""LLM provider abstraction with automatic fallback.

    primary (Groq gpt-oss-120b) --429 / timeout / 5xx--> fallback (Groq gpt-oss-20b or Ollama)

- `retry-after` is respected: a short wait is waited out and the primary retried once; a
  long one puts the primary in "cooldown" so the next requests go straight to the fallback
  instead of hammering a provider that already said no.
- Built-in SDK retries are disabled (max_retries=0) so this module is the only retry logic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import groq
import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig

from app.config import Settings, get_settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """Every provider failed with a transient error (quota, timeout, outage)."""


@dataclass
class Provider:
    name: str  # e.g. "groq:openai/gpt-oss-120b", used in logs and metrics
    model: BaseChatModel


@dataclass
class LLMResult:
    value: Any  # AIMessage, or a Pydantic object for structured output
    provider: str
    fallback_used: bool


def make_provider(provider: str, model: str, settings: Settings) -> Provider:
    if provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise LLMUnavailable("GROQ_API_KEY is not set")
        chat: BaseChatModel = ChatGroq(
            model=model,
            api_key=settings.groq_api_key,
            temperature=0.2,
            timeout=settings.llm_timeout_s,
            max_retries=0,
            # gpt-oss models reason before answering: "low" keeps latency and tokens down.
            reasoning_effort="low" if model.startswith("openai/gpt-oss") else None,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        chat = ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
            reasoning=False,  # qwen3 "thinking" is slow on a laptop CPU
            client_kwargs={"timeout": settings.llm_timeout_s * 4},  # local models are slower
        )
    else:
        raise ValueError(f"unknown LLM provider {provider!r}")
    return Provider(f"{provider}:{model}", chat)


def fallback_reason(exc: BaseException) -> tuple[bool, float | None]:
    """Is this error worth switching provider for? Returns (fallback, retry_after_seconds)."""
    if isinstance(exc, groq.RateLimitError):
        return True, _retry_after(exc.response)
    if isinstance(exc, groq.APIStatusError) and exc.status_code == 413:
        return True, None  # request larger than the per-minute token quota
    if isinstance(exc, groq.APIStatusError | groq.APIConnectionError):
        # APITimeoutError is an APIConnectionError; 5xx are InternalServerError.
        return isinstance(exc, groq.APIConnectionError) or exc.status_code >= 500, None
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError | asyncio.TimeoutError):
        return True, None  # Ollama (httpx-based) down or too slow
    return False, None


def _retry_after(response: httpx.Response | None) -> float | None:
    try:
        return float(response.headers["retry-after"]) if response is not None else None
    except (KeyError, ValueError):
        return None


class LLMRouter:
    def __init__(
        self, primary: Provider, fallback: Provider | None, *, max_wait_s: float = 5.0
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_wait_s = max_wait_s
        self._cooldown_until = 0.0  # time.monotonic() before which the primary is skipped

    async def ainvoke(
        self,
        build: Callable[[BaseChatModel], Runnable],
        messages: list,
        config: RunnableConfig | None = None,
    ) -> LLMResult:
        """Run `build(model)` (e.g. `lambda m: m.bind_tools(tools)`) on the first provider
        that answers."""
        if time.monotonic() >= self._cooldown_until:
            try:
                return await self._call(self.primary, build, messages, config, fallback=False)
            except Exception as exc:
                should_fallback, retry_after = fallback_reason(exc)
                if not should_fallback:
                    raise
                if retry_after is not None and retry_after <= self.max_wait_s:
                    log.info("%s asked to retry in %.1fs, waiting", self.primary.name, retry_after)
                    await asyncio.sleep(retry_after)
                    try:
                        return await self._call(self.primary, build, messages, config, False)
                    except Exception as exc2:
                        should_fallback, retry_after = fallback_reason(exc2)
                        if not should_fallback:
                            raise
                if retry_after:
                    self._cooldown_until = time.monotonic() + retry_after
                log.warning("%s unavailable (%s), falling back", self.primary.name, repr(exc))

        if self.fallback is None:
            raise LLMUnavailable(f"{self.primary.name} unavailable and no fallback configured")
        try:
            return await self._call(self.fallback, build, messages, config, fallback=True)
        except Exception as exc:
            if fallback_reason(exc)[0]:
                raise LLMUnavailable(f"all LLM providers unavailable: {exc!r}") from exc
            raise

    @staticmethod
    async def _call(provider, build, messages, config, fallback: bool) -> LLMResult:
        value = await build(provider.model).ainvoke(messages, config)
        return LLMResult(value=value, provider=provider.name, fallback_used=fallback)


def build_router(settings: Settings | None = None) -> LLMRouter:
    settings = settings or get_settings()
    primary = make_provider(settings.llm_provider, settings.llm_model, settings)
    fallback = None
    if settings.fallback_provider != "none":
        try:
            fallback = make_provider(settings.fallback_provider, settings.fallback_model, settings)
        except LLMUnavailable as exc:
            log.warning("fallback provider disabled: %s", exc)
    return LLMRouter(primary, fallback, max_wait_s=settings.llm_max_retry_wait_s)
