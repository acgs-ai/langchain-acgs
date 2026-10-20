"""Offline test doubles: a scripted tool-calling chat model."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field


class ScriptedChatModel(GenericFakeChatModel):
    """Replays scripted ``AIMessage`` objects and records the messages it was given.

    ``create_agent`` binds tools to the model; binding returns the same model so the script keeps playing.
    """

    received: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.received.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    async def _agenerate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.received.append(list(messages))
        return await super()._agenerate(messages, *args, **kwargs)


def tool_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def script(*turns: AIMessage) -> ScriptedChatModel:
    """A model that answers with ``turns`` in order."""
    messages: Iterator[AIMessage] = iter(turns)
    return ScriptedChatModel(messages=messages)


def calls(*tool_calls: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=list(tool_calls))


def text(content: str) -> AIMessage:
    return AIMessage(content=content)
