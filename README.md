# langchain-acgs

[![CI](https://github.com/acgs-ai/langchain-acgs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/acgs-ai/langchain-acgs/actions/workflows/ci.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/langchain-acgs.svg)](https://pypi.org/project/langchain-acgs/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/langchain-acgs.svg)](https://pypi.org/project/langchain-acgs/)

`langchain-acgs` checks each tool call a LangChain agent is about to make against the rules of an
[ACGS](https://pypi.org/project/acgs-lite/) constitution (`acgs-lite`). It provides `ACGSMiddleware`, an
`AgentMiddleware` for `create_agent` that implements `wrap_tool_call` and `awrap_tool_call`:

- calls that match no blocking rule run unchanged;
- calls that match a blocking rule do not run; the model receives a `ToolMessage` with `status="error"` that names
  the rule;
- calls that match only review rules pause the agent with a native LangGraph `interrupt`, and run only if a person
  approves them;
- any error while evaluating a call, and any review answer that is not a valid decision, denies the call.

Rule evaluation is deterministic keyword and regex matching done locally by `acgs-lite`; no model or network call is
made.

## Installation

```bash
pip install langchain-acgs
```

Requires Python 3.10+, `langchain` 1.x and `acgs-lite` 3.x. Pausing for review in async code (`ainvoke`, `astream`)
needs Python 3.11+, as for any LangGraph `interrupt`.

## Usage

The example below runs offline. `ScriptedModel` replays fixed tool calls in place of a chat model; in real use, pass
any LangChain chat model that supports tool calling. The same file is
[`examples/quickstart.py`](examples/quickstart.py) and is executed by the test suite.

```python
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
```

Output:

```text
send_email: error: Tool call 'send_email' was not executed. Blocked by ACGS rule ACGS-006 (Agents must not expose sensitive data in responses).
get_weather: success: Sunny in Paris
review requested: issue_refund {'order_id': 'A-17', 'amount': 20.0} - ACGS: Human review required by ACGS rule REFUND-REVIEW (Refunds need human approval).
after approval: Refunded 20.0 for A-17
```

## How a tool call is evaluated

1. `request.tool_call["name"]` and `request.tool_call["args"]` are rendered as text by `format_tool_call`:

   ```python
   from langchain_acgs import format_tool_call

   print(format_tool_call("update_config", {"service": "billing", "change": "disable\nlogging", "dryRun": False}))
   ```

   ```text
   tool: update_config
   arguments:
   change: disable logging
   dryRun: false
   service: billing
   normalized: update config | dry run
   ```

   Each argument leaf is one `path: value` line (nested values use `a.b[0].c` paths), so both argument names and
   values are visible to rules. Whitespace inside values is collapsed so phrases split across lines still match.
   Identifiers written in `snake_case`, `kebab-case` or `camelCase` (the tool name, argument names, and
   single-token string values) are repeated as plain words on the `normalized:` line, so the keyword
   `skip audit` also matches a tool called `skip_audit`. Nothing is truncated.
   Pass `tool_call_formatter=` to use a different representation.

2. The text is validated with `acgs_lite.GovernanceEngine.validate` against the constitution
   (`Constitution.default()` unless you pass one).

3. The result decides what happens to the call:

   | ACGS outcome | Result |
   |---|---|
   | No blocking rule matched (warn-only rules may match) | The next handler runs the tool |
   | Only `require_human_review` / `escalate_to_senior` rules matched | `interrupt` with an `ACGSReviewRequest` (or a denial with `on_review="deny"`) |
   | Any `block`, `block_and_notify` or `halt_and_alert` rule matched | Denied |
   | A rule ID not found in the constitution, or an invalid result without a rule | Denied |
   | Any exception while reading or evaluating the tool call | Denied (fail-closed) |

   A matched blocking rule is never downgraded to review because a review rule matched as well.

A denial is returned in place of the tool result:
`ToolMessage(content="Tool call '<name>' was not executed. <reason>", status="error")`. The reason names the matched
rules but never repeats argument values. `message.artifact` holds `{"acgs_verdict": "deny", "rule_ids": [...]}` for
code that inspects results; it is not sent to the model. Exceptions raised by the tool itself are not handled by this
middleware.

`ACGSMiddleware.evaluate(tool_name, args)` returns the underlying `GovernanceDecision`
(`verdict`, `reason`, `rule_ids`, `warning_rule_ids`, `error`) without running anything.

## Human review

Rules whose workflow action is `require_human_review` or `escalate_to_senior` (in the default constitution:
`ACGS-002` and `ACGS-005`) pause the agent. The pause is a LangGraph `interrupt`, so the agent needs a checkpointer
to resume. Without one, the run ends with the interrupt in `result["__interrupt__"]` and the tool does not run.

The interrupt value is an `ACGSReviewRequest`, with the same keys as the `ToolApprovalRequest` that
`HumanInTheLoopMiddleware` raises in `per_call` mode:

```python
from langchain_acgs import ACGSReviewRequest

request: ACGSReviewRequest = {
    "type": "tool_approval",
    "tool_call_id": "3",
    "name": "issue_refund",
    "args": {"order_id": "A-17", "amount": 20.0},
    "description": "ACGS: Human review required by ACGS rule REFUND-REVIEW (Refunds need human approval).",
}
```

Resume with one decision. `allowed_decisions` (default `("approve", "reject")`) limits which ones are accepted:

```python
from langgraph.types import Command

approve = Command(resume={"type": "approve"})
reject = Command(resume={"type": "reject", "message": "Refunds over 10 need a manager."})
# Needs allowed_decisions to include "edit". The tool name cannot change.
edit = Command(
    resume={"type": "edit", "edited_action": {"name": "issue_refund", "args": {"order_id": "A-17", "amount": 10.0}}}
)
```

- `approve` runs the original call.
- `reject` denies it; the model is told a reviewer rejected it, with the message if one was given.
- `edit` runs the call with the reviewer's arguments after evaluating them again: if a blocking rule matches the
  edited arguments the call is denied; if only review rules match, it runs, because a reviewer has already seen it.
- Any other answer (an unknown or disallowed `type`, a different tool name, arguments that are not a mapping) denies
  the call.

When several tool calls in one model turn need review, each raises its own interrupt; resume them by ID with
`Command(resume={interrupt.id: decision, ...})`. Calls in the same turn that need no review run without waiting.
A call without a tool call ID cannot be matched to an answer, so it is denied instead of paused. If the interrupt
cannot be raised at all (for example when `wrap_tool_call` is called outside a LangGraph run), the call is denied.

Pass `on_review="deny"` to deny review verdicts without pausing.

### Order relative to other middleware

`create_agent` treats the first middleware in the list as the outermost wrapper of tool calls.

- Middleware listed **after** `ACGSMiddleware` runs inside it, so changes it makes to a tool call are not checked.
  List `ACGSMiddleware` after any middleware that rewrites tool calls.
- When a call pauses for review, LangGraph re-runs the tool node on resume. Middleware listed **before**
  `ACGSMiddleware` runs again up to the pause, so it must not have side effects before calling its handler.
  `ACGSMiddleware` itself has none: it evaluates the call again (with the same result) and then applies the answer.

`HumanInTheLoopMiddleware` can be used alongside it, for example to ask about specific tools regardless of ACGS. In
its default `batched` mode it asks before tools run, so a person may be asked about a call that ACGS then denies.

## Custom constitutions

```python
from acgs_lite import Constitution, Rule, Severity, ViolationAction

from langchain_acgs import ACGSMiddleware

constitution = Constitution.from_rules(
    [
        Rule(
            id="NO-DESTRUCTIVE-SQL",
            text="No destructive SQL",
            severity=Severity.CRITICAL,
            keywords=["drop table", "truncate table"],
        ),
        Rule(
            id="REFUND-REVIEW",
            text="Refunds need approval",
            severity=Severity.HIGH,
            keywords=["refund"],
            workflow_action=ViolationAction.REQUIRE_HUMAN_REVIEW,
        ),
    ],
    name="payments",
)
middleware = ACGSMiddleware(constitution=constitution, agent_id="payments-agent", allowed_decisions=["approve", "edit", "reject"])
```

`Constitution.from_yaml("rules.yaml")` works as well. To add several `ACGSMiddleware` instances to one agent, give
each a distinct `name`.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `constitution` | `Constitution.default()` | Rules to enforce |
| `agent_id` | `"langchain-agent"` | Agent identifier recorded by ACGS |
| `on_review` | `"interrupt"` | `"interrupt"` pauses for a person; `"deny"` denies review verdicts |
| `allowed_decisions` | `("approve", "reject")` | Accepted review answers: any of `"approve"`, `"edit"`, `"reject"` |
| `tool_call_formatter` | `format_tool_call` | `(tool_name, args) -> str` used as the evaluated text |
| `name` | `"ACGSMiddleware"` | Middleware name; must be unique per agent |

## Limitations

- Only tool calls are checked. Model input and output, and tool results, are not.
- `Constitution.default()` focuses on governance integrity (self-approval, skipping audits, privilege escalation)
  and on secrets and personal data (`password`, `api_key`, `sk-...` and `ghp_...` tokens, US SSN
  patterns). It does **not** cover destructive operations such as `rm -rf` or `DROP TABLE`. Write rules for the
  tools you expose. `tests/unit_tests/test_default_constitution.py` lists what the default rules do and do not catch.
- Keyword rules in acgs-lite 3.0.1 match literal, case-insensitive substrings. They do not understand negation
  ("never share the password" matches `password`) and do not treat `-`, `_` and spaces as equivalent
  (`api-key` does not match the keyword `api_key`).
- Because argument names are part of the evaluated text, a tool with a parameter named `password` is denied under
  the default constitution whenever it is called.

## Development

```bash
uv sync --all-groups
make format      # ruff format + ruff check --fix
make lint        # import checks, ruff, mypy --strict
make test        # pytest with a 100% branch-coverage gate, network disabled
```

## License

Apache-2.0. See [LICENSE](LICENSE).
