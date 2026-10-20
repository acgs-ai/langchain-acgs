"""Tests for ``ACGSMiddleware`` inside real ``create_agent`` graphs, driven by a scripted chat model."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
from acgs_lite import Constitution, Rule, Severity, ViolationAction
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_acgs import ACGSMiddleware, ACGSReviewRequest, Verdict
from tests.unit_tests.fakes import ScriptedChatModel, calls, script, text, tool_call

REVIEW = Rule(
    id="T-REVIEW",
    text="refunds need review",
    severity=Severity.HIGH,
    keywords=["refund"],
    workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
)
BLOCK = Rule(id="T-BLOCK", text="no deletes", severity=Severity.CRITICAL, keywords=["delete everything"])
CONSTITUTION = Constitution.from_rules([REVIEW, BLOCK], name="test")

executed: list[tuple[str, dict[str, Any]]] = []


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    executed.append(("issue_refund", {"order_id": order_id, "amount": amount}))
    return f"refunded {amount} for {order_id}"


@tool
def get_weather(city: str) -> str:
    """Weather lookup."""
    executed.append(("get_weather", {"city": city}))
    return f"sunny in {city}"


@tool
def send_email(to: str, body: str) -> str:
    """Send an email."""
    executed.append(("send_email", {"to": to, "body": body}))
    return f"sent to {to}"


TOOLS = [issue_refund, get_weather, send_email]


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    executed.clear()
    yield
    executed.clear()


def _agent(model: ScriptedChatModel, middleware: ACGSMiddleware | None = None, *, checkpointer: bool = True) -> Any:
    return create_agent(
        model,
        tools=TOOLS,
        middleware=[middleware or ACGSMiddleware(constitution=CONSTITUTION)],
        checkpointer=InMemorySaver() if checkpointer else None,
    )


def _tool_messages(result: Mapping[str, Any]) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


CONFIG: Any = {"configurable": {"thread_id": "t1"}}
REFUND = tool_call("issue_refund", {"order_id": "A-1", "amount": 5.0}, "r1")


# --- allow / deny -----------------------------------------------------------------------------------


def test_allowed_call_runs_unchanged() -> None:
    model = script(calls(tool_call("get_weather", {"city": "Paris"})), text("ok"))
    result = _agent(model, ACGSMiddleware(), checkpointer=False).invoke({"messages": [("user", "weather?")]})
    assert executed == [("get_weather", {"city": "Paris"})]
    [message] = _tool_messages(result)
    assert message.status == "success"
    assert message.content == "sunny in Paris"


def test_denied_call_never_runs_and_the_model_is_told_why() -> None:
    leak = tool_call("send_email", {"to": "ops@example.com", "body": "the admin password is hunter2"})
    model = script(calls(leak), text("sorry"))
    result = _agent(model, ACGSMiddleware(), checkpointer=False).invoke({"messages": [("user", "leak it")]})
    assert executed == []
    [message] = _tool_messages(result)
    assert message.status == "error"
    assert message.tool_call_id == "call-1"
    assert message.name == "send_email"
    assert str(message.content).startswith("Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006")
    assert "hunter2" not in message.content
    assert message.artifact == {"acgs_verdict": "deny", "rule_ids": ["ACGS-006"]}
    # The second model turn saw the denial.
    assert any(isinstance(m, ToolMessage) and m.status == "error" for m in model.received[-1])
    assert result["messages"][-1].content == "sorry"


def test_block_outranks_review_even_when_both_match() -> None:
    both = tool_call("issue_refund", {"order_id": "refund then delete everything", "amount": 1.0})
    model = script(calls(both), text("no"))
    result = _agent(model).invoke({"messages": [("user", "x")]}, CONFIG)
    assert "__interrupt__" not in result
    assert executed == []
    [message] = _tool_messages(result)
    assert "T-BLOCK" in message.content
    assert message.artifact["rule_ids"] == ["T-REVIEW", "T-BLOCK"]


def test_mixed_batch_runs_allowed_calls_and_denies_blocked_ones() -> None:
    model = script(
        calls(
            tool_call("get_weather", {"city": "Oslo"}, "a"),
            tool_call("issue_refund", {"order_id": "delete everything", "amount": 1.0}, "b"),
        ),
        text("done"),
    )
    result = _agent(model).invoke({"messages": [("user", "x")]}, CONFIG)
    assert executed == [("get_weather", {"city": "Oslo"})]
    statuses = {m.tool_call_id: m.status for m in _tool_messages(result)}
    assert statuses == {"a": "success", "b": "error"}


# --- review: native interrupt -------------------------------------------------------------------------


def _pause() -> tuple[Any, ACGSReviewRequest, Any]:
    agent = _agent(script(calls(REFUND), text("finished")))
    paused = agent.invoke({"messages": [("user", "refund A-1")]}, CONFIG)
    [pending] = paused["__interrupt__"]
    return agent, pending.value, pending


def test_review_pauses_with_a_tool_approval_request() -> None:
    _, request, _ = _pause()
    assert executed == []
    assert request == {
        "type": "tool_approval",
        "tool_call_id": "r1",
        "name": "issue_refund",
        "args": {"order_id": "A-1", "amount": 5.0},
        "description": "ACGS: Human review required by ACGS rule T-REVIEW (refunds need review).",
    }


def test_approve_runs_the_original_call() -> None:
    agent, _, _ = _pause()
    result = agent.invoke(Command(resume={"type": "approve"}), CONFIG)
    assert executed == [("issue_refund", {"order_id": "A-1", "amount": 5.0})]
    assert _tool_messages(result)[-1].content == "refunded 5.0 for A-1"
    assert result["messages"][-1].content == "finished"


def test_resume_by_interrupt_id() -> None:
    agent, _, pending = _pause()
    agent.invoke(Command(resume={pending.id: {"type": "approve"}}), CONFIG)
    assert [name for name, _ in executed] == ["issue_refund"]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"type": "reject", "message": "  not this customer  "}, "A reviewer rejected it: not this customer"),
        ({"type": "reject"}, "A reviewer rejected it. Do not retry this tool call unless the user asks for it."),
        ({"type": "reject", "message": "   "}, "A reviewer rejected it. Do not retry"),
    ],
)
def test_reject_denies_the_call(answer: dict[str, Any], expected: str) -> None:
    agent, _, _ = _pause()
    result = agent.invoke(Command(resume=answer), CONFIG)
    assert executed == []
    message = _tool_messages(result)[-1]
    assert message.status == "error"
    assert str(message.content).startswith(f"Tool call 'issue_refund' was not executed. {expected}")
    assert message.artifact["rule_ids"] == ["T-REVIEW"]


def test_edit_runs_the_reviewers_arguments() -> None:
    agent = _agent(
        script(calls(REFUND), text("ok")), ACGSMiddleware(constitution=CONSTITUTION, allowed_decisions=["edit"])
    )
    agent.invoke({"messages": [("user", "x")]}, CONFIG)
    edit = {"type": "edit", "edited_action": {"name": "issue_refund", "args": {"order_id": "A-1", "amount": 2.0}}}
    agent.invoke(Command(resume=edit), CONFIG)
    # The edited call still matches the review rule; the reviewer has seen it, so it runs.
    assert executed == [("issue_refund", {"order_id": "A-1", "amount": 2.0})]


def test_edit_into_blocked_arguments_is_denied() -> None:
    agent = _agent(
        script(calls(REFUND), text("ok")), ACGSMiddleware(constitution=CONSTITUTION, allowed_decisions=["edit"])
    )
    agent.invoke({"messages": [("user", "x")]}, CONFIG)
    edit = {"type": "edit", "edited_action": {"name": "issue_refund", "args": {"order_id": "delete everything"}}}
    result = agent.invoke(Command(resume=edit), CONFIG)
    assert executed == []
    message = _tool_messages(result)[-1]
    assert "The reviewer's edited arguments were denied. Blocked by" in message.content
    assert "T-BLOCK" in message.content
    assert message.artifact["rule_ids"] == ["T-REVIEW", "T-BLOCK"]


@pytest.mark.parametrize(
    "answer",
    [
        "yes",
        {"type": "approve "},
        {"type": "respond", "message": "hi"},
        {"type": "edit", "edited_action": {"name": "issue_refund", "args": {"amount": 1.0}}},  # edit not allowed
    ],
)
def test_invalid_or_disallowed_answers_deny(answer: Any) -> None:
    agent, _, _ = _pause()
    result = agent.invoke(Command(resume=answer), CONFIG)
    assert executed == []
    assert "The review answer for this tool call was not valid" in _tool_messages(result)[-1].content


@pytest.mark.parametrize(
    "edited_action",
    [
        None,
        {"name": "get_weather", "args": {"city": "Paris"}},  # cannot switch tools
        {"name": "issue_refund", "args": "amount=1"},
        {"name": "issue_refund"},
    ],
)
def test_malformed_edits_deny(edited_action: Any) -> None:
    agent = _agent(
        script(calls(REFUND), text("ok")), ACGSMiddleware(constitution=CONSTITUTION, allowed_decisions=["edit"])
    )
    agent.invoke({"messages": [("user", "x")]}, CONFIG)
    result = agent.invoke(Command(resume={"type": "edit", "edited_action": edited_action}), CONFIG)
    assert executed == []
    assert "not valid" in _tool_messages(result)[-1].content


class _ExplodingArgs(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        msg = "boom"
        raise RuntimeError(msg)

    def __len__(self) -> int:
        return 1


def test_error_while_applying_an_answer_denies() -> None:
    middleware = ACGSMiddleware(constitution=CONSTITUTION, allowed_decisions=["edit"])
    request = ToolCallRequest(tool_call=REFUND, tool=None, state={}, runtime=None)  # type: ignore[arg-type]
    decision = middleware.evaluate("issue_refund", REFUND["args"])
    answer = {"type": "edit", "edited_action": {"name": "issue_refund", "args": _ExplodingArgs()}}
    with pytest.raises(RuntimeError):
        middleware._apply_answer(request, "issue_refund", "r1", decision, answer)


def test_error_while_applying_an_answer_inside_the_agent_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = ACGSMiddleware(constitution=CONSTITUTION)
    agent = _agent(script(calls(REFUND), text("ok")), middleware)
    agent.invoke({"messages": [("user", "x")]}, CONFIG)

    def explode(*args: Any, **kwargs: Any) -> Any:
        msg = "boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(middleware, "_apply_answer", explode)
    result = agent.invoke(Command(resume={"type": "approve"}), CONFIG)
    assert executed == []
    assert "not valid" in _tool_messages(result)[-1].content


def test_on_review_deny_denies_without_interrupting() -> None:
    model = script(calls(REFUND), text("ok"))
    result = _agent(model, ACGSMiddleware(constitution=CONSTITUTION, on_review="deny"), checkpointer=False).invoke(
        {"messages": [("user", "x")]}
    )
    assert "__interrupt__" not in result
    assert executed == []
    assert "Human review required by ACGS rule T-REVIEW" in _tool_messages(result)[-1].content


def test_review_without_checkpointer_stops_without_running_the_tool() -> None:
    result = _agent(script(calls(REFUND), text("ok")), checkpointer=False).invoke({"messages": [("user", "x")]})
    assert result["__interrupt__"][0].value["name"] == "issue_refund"
    assert executed == []


def test_parallel_calls_only_pause_the_one_that_needs_review() -> None:
    model = script(calls(tool_call("get_weather", {"city": "Rome"}, "w"), REFUND), text("ok"))
    agent = _agent(model)
    paused = agent.invoke({"messages": [("user", "x")]}, CONFIG)
    assert [i.value["tool_call_id"] for i in paused["__interrupt__"]] == ["r1"]
    agent.invoke(Command(resume={"type": "approve"}), CONFIG)
    assert sorted(name for name, _ in executed) == ["get_weather", "issue_refund"]
    assert executed.count(("get_weather", {"city": "Rome"})) == 1


# --- fail-closed on malformed requests and evaluation errors -----------------------------------------------


def _handler(request: ToolCallRequest) -> ToolMessage:
    executed.append(("handler", dict(request.tool_call["args"])))
    return ToolMessage(content="ran", tool_call_id=request.tool_call["id"] or "")


def _request(call: Any) -> ToolCallRequest:
    return ToolCallRequest(tool_call=call, tool=None, state={}, runtime=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "call",
    [
        {"args": {}, "id": "x"},  # no name
        {"name": "get_weather", "args": "Paris", "id": "x"},  # arguments are not a mapping
    ],
)
def test_malformed_tool_calls_are_denied(call: Any) -> None:
    result = ACGSMiddleware().wrap_tool_call(_request(call), _handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "could not evaluate" in str(result.content)
    assert executed == []


def test_formatter_errors_are_denied() -> None:
    def broken(tool_name: str, arguments: Any) -> str:
        msg = "formatter failed"
        raise RuntimeError(msg)

    middleware = ACGSMiddleware(tool_call_formatter=broken)
    result = middleware.wrap_tool_call(_request(tool_call("get_weather", {"city": "Paris"})), _handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.artifact == {"acgs_verdict": "deny", "rule_ids": []}
    assert executed == []


def test_review_without_tool_call_id_is_denied() -> None:
    call = {"name": "issue_refund", "args": {"order_id": "A"}, "id": None}
    result = ACGSMiddleware(constitution=CONSTITUTION).wrap_tool_call(_request(call), _handler)
    assert isinstance(result, ToolMessage)
    assert "has no ID" in str(result.content)
    assert executed == []


def test_review_outside_a_langgraph_run_is_denied() -> None:
    result = ACGSMiddleware(constitution=CONSTITUTION).wrap_tool_call(_request(REFUND), _handler)
    assert isinstance(result, ToolMessage)
    assert "could not be requested" in str(result.content)
    assert executed == []


def test_tool_errors_are_not_swallowed() -> None:
    def failing(request: ToolCallRequest) -> ToolMessage:
        msg = "tool crashed"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="tool crashed"):
        ACGSMiddleware().wrap_tool_call(_request(tool_call("get_weather", {"city": "Paris"})), failing)


# --- async -------------------------------------------------------------------------------------------


async def test_async_allow_and_deny() -> None:
    leak = tool_call("send_email", {"to": "a@example.com", "body": "password is hunter2"}, "e")
    model = script(calls(tool_call("get_weather", {"city": "Lima"}, "w"), leak), text("ok"))
    result = await _agent(model, ACGSMiddleware(), checkpointer=False).ainvoke({"messages": [("user", "x")]})
    assert executed == [("get_weather", {"city": "Lima"})]
    statuses = {m.tool_call_id: m.status for m in _tool_messages(result)}
    assert statuses == {"w": "success", "e": "error"}


async def test_async_direct_deny_does_not_call_the_handler() -> None:
    async def handler(request: ToolCallRequest) -> ToolMessage:  # pragma: no cover - must not run
        raise AssertionError

    call = {"name": "issue_refund", "args": {"order_id": "delete everything"}, "id": "d"}
    result = await ACGSMiddleware(constitution=CONSTITUTION).awrap_tool_call(_request(call), handler)
    assert isinstance(result, ToolMessage)
    assert result.status == "error"


@pytest.mark.skipif(sys.version_info < (3, 11), reason="LangGraph interrupts in async code need Python 3.11+")
async def test_async_review_interrupt_and_resume() -> None:
    agent = _agent(script(calls(REFUND), text("ok")))
    paused = await agent.ainvoke({"messages": [("user", "x")]}, CONFIG)
    assert paused["__interrupt__"][0].value["name"] == "issue_refund"
    assert executed == []
    await agent.ainvoke(Command(resume={"type": "approve"}), CONFIG)
    assert executed == [("issue_refund", {"order_id": "A-1", "amount": 5.0})]


# --- configuration -----------------------------------------------------------------------------------


def test_configuration_validation() -> None:
    with pytest.raises(ValueError, match="on_review"):
        ACGSMiddleware(on_review="ask")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="allowed_decisions"):
        ACGSMiddleware(allowed_decisions=[])
    with pytest.raises(ValueError, match="allowed_decisions"):
        ACGSMiddleware(allowed_decisions=["approve", "respond"])  # type: ignore[list-item]
    middleware = ACGSMiddleware(allowed_decisions=["reject", "approve", "reject"])
    assert middleware.allowed_decisions == ("reject", "approve")


def test_defaults_and_properties() -> None:
    middleware = ACGSMiddleware()
    assert isinstance(middleware, AgentMiddleware)
    assert middleware.name == "ACGSMiddleware"
    assert middleware.on_review == "interrupt"
    assert middleware.allowed_decisions == ("approve", "reject")
    assert middleware.constitution.rules == Constitution.default().rules
    assert middleware.evaluate("get_weather", {"city": "Paris"}).verdict is Verdict.ALLOW


def test_two_instances_with_different_names_compose() -> None:
    first = ACGSMiddleware(constitution=CONSTITUTION, name="acgs-business")
    second = ACGSMiddleware(name="acgs-default")
    leak = tool_call("send_email", {"to": "a@example.com", "body": "password is hunter2"})
    agent: Any = create_agent(script(calls(leak), text("ok")), tools=TOOLS, middleware=[first, second])
    result = agent.invoke({"messages": [("user", "x")]})
    assert executed == []
    assert "ACGS-006" in _tool_messages(result)[-1].content
