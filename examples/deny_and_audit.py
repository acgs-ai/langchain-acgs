"""Run one allowed and one denied LangChain tool call offline, then print the decision trail."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langchain_acgs import ACGSMiddleware, GovernanceDecision


class AuditedMiddleware(ACGSMiddleware):
    """Record decisions through the public API without retaining argument values."""

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[GovernanceDecision] = []

    def evaluate(self, tool_name: str, args: object) -> GovernanceDecision:
        decision = super().evaluate(tool_name, args)
        self.decisions.append(decision)
        return decision


class ScriptedModel(GenericFakeChatModel):
    """Replay fixed tool calls without a model service."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self


@tool
def record_note(body: str) -> str:
    """Record a demo note by printing it locally."""
    print(f"Executed record_note: {body}")
    return "Note recorded."


def main() -> None:
    # Disable both current and legacy tracing flags before running the agent.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    replies = [
        AIMessage(
            content="",
            tool_calls=[{"name": "record_note", "args": {"body": body}, "id": f"note-{index}"}],
        )
        for index, body in enumerate(["The meeting starts at noon.", "The admin password is demo-secret."])
    ]
    replies.append(AIMessage(content="Done."))
    gate = AuditedMiddleware()
    agent = create_agent(ScriptedModel(messages=iter(replies)), tools=[record_note], middleware=[gate])
    agent.invoke({"messages": [{"role": "user", "content": "Record the two notes."}]})
    print("Audit trail (application decisions):")
    for decision in gate.decisions:
        print(json.dumps(asdict(decision), sort_keys=True))


if __name__ == "__main__":
    main()
