# SPDX-FileCopyrightText: 2026-present ACGS Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACGS governance as LangChain agent middleware (``wrap_tool_call`` / ``awrap_tool_call``)."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from acgs_lite import Constitution
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from langchain_acgs._governance import GovernanceDecision, ToolCallFormatter, ToolCallGovernor, Verdict

logger = logging.getLogger(__name__)

ReviewDecisionType = Literal["approve", "edit", "reject"]
"""Decisions a reviewer can return for an ACGS review request."""

ToolResult = ToolMessage | Command[Any]

_DECISION_TYPES: tuple[ReviewDecisionType, ...] = ("approve", "edit", "reject")
_EVALUATION_FAILED = "ACGS governance could not evaluate this tool call, so it was denied."
_INVALID_ANSWER = "The review answer for this tool call was not valid, so the call was denied."


class ACGSReviewRequest(TypedDict):
    """Interrupt value raised when a tool call needs human review.

    It has the same keys as LangChain's ``ToolApprovalRequest`` (the ``per_call`` interrupt of
    ``HumanInTheLoopMiddleware``), so clients that render those requests can render this one.
    Answer it with ``Command(resume=...)`` and one decision: ``{"type": "approve"}``,
    ``{"type": "reject", "message": "..."}`` (``message`` is optional), or
    ``{"type": "edit", "edited_action": {"name": <same tool>, "args": {...}}}``, limited to
    ``allowed_decisions``.
    """

    type: Literal["tool_approval"]
    tool_call_id: str
    name: str
    args: dict[str, Any]
    description: str


class ACGSMiddleware(AgentMiddleware):
    """Check every tool call against an ACGS constitution before the tool runs.

    Example:
        ```python
        from langchain.agents import create_agent

        from langchain_acgs import ACGSMiddleware

        agent = create_agent(model, tools=[...], middleware=[ACGSMiddleware()])
        ```

    For each tool call, the tool name and arguments are rendered with ``format_tool_call`` (or a
    custom ``tool_call_formatter``) and validated with ``acgs_lite.GovernanceEngine``:

    - ``allow``: the call is passed to the next handler unchanged;
    - ``deny``: the tool does not run; the model receives a ``ToolMessage`` with ``status="error"``
      that names the matched rules;
    - ``review``: with ``on_review="interrupt"`` (the default) the agent pauses with a LangGraph
      ``interrupt`` whose value is an ``ACGSReviewRequest``; with ``on_review="deny"`` the call is
      denied.

    Any error while evaluating the call, and any review answer that is not a valid allowed
    decision, denies the call. Arguments edited by a reviewer are evaluated again, and the call is
    denied if a blocking rule matches them.
    """

    def __init__(
        self,
        *,
        constitution: Constitution | None = None,
        agent_id: str = "langchain-agent",
        on_review: Literal["interrupt", "deny"] = "interrupt",
        allowed_decisions: Sequence[ReviewDecisionType] = ("approve", "reject"),
        tool_call_formatter: ToolCallFormatter | None = None,
        name: str = "ACGSMiddleware",
    ) -> None:
        """Create the middleware.

        Args:
            constitution: The ACGS constitution to enforce. Defaults to
                ``acgs_lite.Constitution.default()``.
            agent_id: Agent identifier recorded by ACGS for each validation.
            on_review: What to do when only review rules match: ``"interrupt"`` pauses the agent
                for a human decision (needs a checkpointer to resume); ``"deny"`` denies the call.
            allowed_decisions: Decisions a reviewer may return. Any of ``"approve"``, ``"edit"``
                and ``"reject"``.
            tool_call_formatter: Optional function ``(tool_name, args) -> str`` that replaces
                ``format_tool_call``.
            name: Middleware name. Use different names to add more than one ``ACGSMiddleware``
                to an agent.

        Raises:
            ValueError: If ``on_review`` or ``allowed_decisions`` is invalid.
        """
        super().__init__()
        if on_review not in ("interrupt", "deny"):
            msg = f"on_review must be 'interrupt' or 'deny', got {on_review!r}"
            raise ValueError(msg)
        decisions = tuple(dict.fromkeys(allowed_decisions))
        if not decisions or any(d not in _DECISION_TYPES for d in decisions):
            msg = f"allowed_decisions must be a non-empty subset of {_DECISION_TYPES}, got {allowed_decisions!r}"
            raise ValueError(msg)
        self.on_review = on_review
        self.allowed_decisions: tuple[ReviewDecisionType, ...] = decisions
        self._name = name
        self._governor = ToolCallGovernor(constitution, agent_id=agent_id, tool_call_formatter=tool_call_formatter)

    @property
    def name(self) -> str:
        """Name of this middleware instance."""
        return self._name

    @property
    def constitution(self) -> Constitution:
        """The constitution being enforced."""
        return self._governor.constitution

    def evaluate(self, tool_name: str, args: object) -> GovernanceDecision:
        """Evaluate a tool call without running anything. Never raises.

        Args:
            tool_name: Name of the tool.
            args: Arguments of the tool call.

        Returns:
            The ACGS governance decision.
        """
        return self._governor.evaluate(tool_name, args)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolResult],
    ) -> ToolResult:
        """Run the tool only if ACGS allows it (or a reviewer approves it).

        Args:
            request: The pending tool call.
            handler: Runs the tool (or the next middleware).

        Returns:
            The tool result, or a ``ToolMessage`` with ``status="error"`` if the call was denied.
        """
        outcome = self._gate(request)
        if isinstance(outcome, ToolMessage):
            return outcome
        return handler(outcome)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolResult]],
    ) -> ToolResult:
        """Async variant of ``wrap_tool_call``.

        Args:
            request: The pending tool call.
            handler: Runs the tool (or the next middleware).

        Returns:
            The tool result, or a ``ToolMessage`` with ``status="error"`` if the call was denied.
        """
        outcome = self._gate(request)
        if isinstance(outcome, ToolMessage):
            return outcome
        return await handler(outcome)

    def _gate(self, request: ToolCallRequest) -> ToolCallRequest | ToolMessage:
        """Return the request to run, or the message to return instead of running the tool."""
        try:
            tool_call = request.tool_call
            tool_name = str(tool_call["name"])
            tool_call_id = tool_call.get("id") or ""
            args = tool_call.get("args", {})
            if not isinstance(args, Mapping):
                msg = f"Tool call arguments must be a mapping, got {type(args).__name__}."
                raise TypeError(msg)
        except Exception:  # noqa: BLE001  # fail-closed: any malformed request is denied
            logger.warning("ACGS could not read the pending tool call; denying it (fail-closed).", exc_info=True)
            return _denied("unknown", "", _EVALUATION_FAILED, None)

        decision = self.evaluate(tool_name, args)
        if decision.verdict is Verdict.ALLOW:
            return request
        if decision.verdict is Verdict.DENY or self.on_review == "deny":
            return _denied(tool_name, tool_call_id, decision.reason, decision)
        return self._review(request, tool_name, tool_call_id, args, decision)

    def _review(
        self,
        request: ToolCallRequest,
        tool_name: str,
        tool_call_id: str,
        args: Mapping[str, Any],
        decision: GovernanceDecision,
    ) -> ToolCallRequest | ToolMessage:
        """Pause for a human decision with a LangGraph interrupt, then apply the answer."""
        if not tool_call_id:
            reason = "Human review is required but the tool call has no ID, so the call was denied."
            return _denied(tool_name, tool_call_id, reason, decision)

        review_request = ACGSReviewRequest(
            type="tool_approval",
            tool_call_id=tool_call_id,
            name=tool_name,
            args=dict(args),
            description=f"ACGS: {decision.reason}",
        )
        try:
            answer = interrupt(review_request)
        except GraphBubbleUp:
            # GraphInterrupt pauses the run on the first pass; LangGraph must receive it unchanged.
            raise
        except Exception:  # noqa: BLE001  # fail-closed: e.g. called outside a LangGraph run
            logger.warning("ACGS could not request review for '%s'; denying it.", tool_name, exc_info=True)
            reason = "Human review is required but could not be requested, so the call was denied."
            return _denied(tool_name, tool_call_id, reason, decision)
        try:
            return self._apply_answer(request, tool_name, tool_call_id, decision, answer)
        except Exception:  # noqa: BLE001  # fail-closed: an answer that cannot be applied is denied
            logger.warning("ACGS could not apply the review answer for '%s'; denying it.", tool_name, exc_info=True)
            return _denied(tool_name, tool_call_id, _INVALID_ANSWER, decision)

    def _apply_answer(
        self,
        request: ToolCallRequest,
        tool_name: str,
        tool_call_id: str,
        decision: GovernanceDecision,
        answer: object,
    ) -> ToolCallRequest | ToolMessage:
        """Turn a reviewer's answer into the request to run, or a denial."""
        if not isinstance(answer, Mapping) or answer.get("type") not in self.allowed_decisions:
            logger.warning("ACGS received an invalid review answer for '%s'.", tool_name)
            return _denied(tool_name, tool_call_id, _INVALID_ANSWER, decision)
        kind = answer["type"]
        if kind == "approve":
            return request
        if kind == "reject":
            message = answer.get("message")
            if isinstance(message, str) and message.strip():
                reason = f"A reviewer rejected it: {message.strip()}"
            else:
                reason = "A reviewer rejected it. Do not retry this tool call unless the user asks for it."
            return _denied(tool_name, tool_call_id, reason, decision)
        edited = answer.get("edited_action")
        if (
            not isinstance(edited, Mapping)
            or edited.get("name") != tool_name
            or not isinstance(edited.get("args"), Mapping)
        ):
            logger.warning("ACGS received an invalid edit for '%s'.", tool_name)
            return _denied(tool_name, tool_call_id, _INVALID_ANSWER, decision)
        new_args = dict(edited["args"])
        recheck = self.evaluate(tool_name, new_args)
        if recheck.verdict is Verdict.DENY:
            return _denied(
                tool_name, tool_call_id, f"The reviewer's edited arguments were denied. {recheck.reason}", recheck
            )
        return request.override(tool_call={**request.tool_call, "args": new_args})


def _denied(tool_name: str, tool_call_id: str, reason: str, decision: GovernanceDecision | None) -> ToolMessage:
    """Build the message the model receives instead of the tool result."""
    artifact = {
        "acgs_verdict": "deny",
        "rule_ids": list(decision.rule_ids) if decision is not None else [],
    }
    return ToolMessage(
        content=f"Tool call '{tool_name}' was not executed. {reason}",
        name=tool_name,
        tool_call_id=tool_call_id,
        status="error",
        artifact=artifact,
    )
