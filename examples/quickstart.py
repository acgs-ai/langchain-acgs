"""Block a tool call that leaks a password, let a harmless one run, and pause a refund for review.

Runs offline: ``ScriptedModel`` stands in for a chat model and replays fixed tool calls.
"""

from collections.abc import Iterator
from typing import Any

from acgs_lite import Constitution, Rule, Severity, ViolationAction
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_acgs import ACGSMiddleware


class ScriptedModel(GenericFakeChatModel):
    """Replays scripted messages; ``create_agent`` binds tools, which this model ignores."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        return self


def scripted(*messages: AIMessage) -> ScriptedModel:
    replies: Iterator[AIMessage] = iter(messages)
    return ScriptedModel(messages=replies)


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"Refunded {amount} for {order_id}"


# 1. The default constitution blocks the leaked password and allows the weather lookup.
model = scripted(
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "send_email",
                "args": {"to": "ops@example.com", "body": "The admin password is hunter2"},
                "id": "1",
            },
            {"name": "get_weather", "args": {"city": "Paris"}, "id": "2"},
        ],
    ),
    AIMessage(content="Done."),
)
agent = create_agent(model, tools=[send_email, get_weather], middleware=[ACGSMiddleware()])
result = agent.invoke({"messages": [{"role": "user", "content": "Email the admin password, then check Paris."}]})
for message in result["messages"]:
    if message.type == "tool":
        print(f"{message.name}: {message.status}: {message.content}")

# 2. A custom rule that requires human review pauses the agent; a checkpointer lets it resume.
refunds = Constitution.from_rules(
    [
        Rule(
            id="REFUND-REVIEW",
            text="Refunds need human approval",
            severity=Severity.HIGH,
            keywords=["refund"],
            workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
        )
    ],
    name="refunds",
)
model = scripted(
    AIMessage(
        content="", tool_calls=[{"name": "issue_refund", "args": {"order_id": "A-17", "amount": 20.0}, "id": "3"}]
    ),
    AIMessage(content="Refund issued."),
)
agent = create_agent(
    model,
    tools=[issue_refund],
    middleware=[ACGSMiddleware(constitution=refunds)],
    checkpointer=InMemorySaver(),
)
config: RunnableConfig = {"configurable": {"thread_id": "refund-1"}}
paused = agent.invoke({"messages": [{"role": "user", "content": "Refund order A-17."}]}, config)
request = paused["__interrupt__"][0].value
print(f"review requested: {request['name']} {request['args']} - {request['description']}")

resumed = agent.invoke(Command(resume={"type": "approve"}), config)
print(f"after approval: {resumed['messages'][-2].content}")
