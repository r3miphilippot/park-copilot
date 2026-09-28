"""Scripted chat model: plays back prepared answers, records what it was given."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    responses: list[Any] = Field(default_factory=list)  # AIMessage or Exception, in order
    received: list[list] = Field(default_factory=list)  # messages of each call
    bound: list[list[str]] = Field(default_factory=list)  # tool names bound before each call

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound.append([getattr(t, "name", None) or getattr(t, "__name__", "?") for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.received.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response)])


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


def mode_decision(mode: str, target_date: str | None = None) -> AIMessage:
    """What with_structured_output(ModeDecision) expects from a tool-calling model."""
    return tool_call("ModeDecision", {"mode": mode, "target_date": target_date}, "call_mode")
