import groq
import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.llm import LLMRouter, LLMUnavailable, Provider, fallback_reason
from tests.fakes import ScriptedChatModel

REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def rate_limited(retry_after: str | None = "30") -> groq.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    return groq.RateLimitError(
        "rate limit", response=httpx.Response(429, headers=headers, request=REQUEST), body=None
    )


def router(primary_responses, fallback_responses=None, max_wait_s=5.0):
    primary = Provider("primary", ScriptedChatModel(responses=primary_responses))
    fallback = None
    if fallback_responses is not None:
        fallback = Provider("fallback", ScriptedChatModel(responses=fallback_responses))
    return LLMRouter(primary, fallback, max_wait_s=max_wait_s)


async def ask(r: LLMRouter):
    return await r.ainvoke(lambda m: m, [HumanMessage("hi")])


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (rate_limited("12"), (True, 12.0)),
        (rate_limited(None), (True, None)),
        (groq.APITimeoutError(request=REQUEST), (True, None)),
        (groq.InternalServerError("boom", response=httpx.Response(503, request=REQUEST),
                                  body=None), (True, None)),
        (groq.BadRequestError("bad", response=httpx.Response(400, request=REQUEST), body=None),
         (False, None)),
        (httpx.ConnectError("ollama down"), (True, None)),
        (ValueError("bug"), (False, None)),
    ],
)  # fmt: skip
def test_fallback_reason(exc, expected):
    assert fallback_reason(exc) == expected


async def test_primary_answers():
    result = await ask(router([AIMessage("hello")], [AIMessage("unused")]))
    assert (result.value.content, result.provider, result.fallback_used) == (
        "hello",
        "primary",
        False,
    )


async def test_long_retry_after_falls_back_and_cools_down():
    r = router([rate_limited("30")], [AIMessage("a"), AIMessage("b")])
    first = await ask(r)
    second = await ask(r)  # primary is in cooldown: not even called
    assert (first.provider, first.fallback_used) == ("fallback", True)
    assert second.provider == "fallback"
    assert len(r.primary.model.received) == 1


async def test_short_retry_after_is_waited_out(monkeypatch):
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr("app.llm.asyncio.sleep", fake_sleep)
    r = router([rate_limited("2"), AIMessage("after wait")], [AIMessage("unused")])
    result = await ask(r)
    assert slept == [2.0]
    assert (result.value.content, result.provider) == ("after wait", "primary")


def invalid_tool_call() -> groq.BadRequestError:
    body = {"error": {"code": "tool_use_failed", "message": "parameters did not match schema"}}
    return groq.BadRequestError(
        "Error code: 400 - " + str(body), response=httpx.Response(400, request=REQUEST), body=body
    )


async def test_invalid_tool_call_is_retried_on_the_same_model():
    r = router([invalid_tool_call(), AIMessage("fixed")], [AIMessage("unused")])
    result = await ask(r)
    assert (result.value.content, result.provider) == ("fixed", "primary")


async def test_repeated_invalid_tool_call_falls_back():
    r = router([invalid_tool_call(), invalid_tool_call()], [AIMessage("backup")])
    assert (await ask(r)).provider == "fallback"


async def test_timeout_falls_back():
    r = router([groq.APITimeoutError(request=REQUEST)], [AIMessage("backup")])
    assert (await ask(r)).provider == "fallback"


async def test_non_transient_error_is_raised():
    r = router([ValueError("bug")], [AIMessage("unused")])
    with pytest.raises(ValueError):
        await ask(r)


async def test_everything_down_raises_llm_unavailable():
    with pytest.raises(LLMUnavailable):
        await ask(router([rate_limited("30")], [rate_limited("30")]))
    with pytest.raises(LLMUnavailable):
        await ask(router([rate_limited("30")], None))  # no fallback configured
