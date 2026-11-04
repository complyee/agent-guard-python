"""CrewAI adapter: wrap any CrewAI tool so it runs only on a verified grant.

    from complyee_agent_guard import create_guard
    from complyee_agent_guard.crewai import GuardedCrewAITool

    guard = create_guard(workspace_url=..., api_key=..., agent="refund-crew")
    refund = GuardedCrewAITool(
        tool=issue_refund_tool,             # any crewai BaseTool, incl. @tool functions
        guard=guard,
        action="refund.issue",
        context=lambda a: {"amount": a["amount"]},
    )
    Agent(role="Support lead", tools=[refund], ...)

The agent sees the wrapped tool's own name, description and argument schema.
When the step does not run, the tool returns a CrewAI ToolFailure (plain text
on CrewAI versions without it) instead of raising: CrewAI retries tools that
raise, and every retry would be a fresh request to Agent Control Room. CrewAI's
own tool_failure_policy ("warn" by default, "raise" to stop the crew) decides
what happens next.

Requires the optional extra: pip install "complyee-agent-guard[crewai]".
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections import OrderedDict
from typing import Any, Callable, Mapping, Optional

from crewai.tools import BaseTool
from pydantic import PrivateAttr

from .core import VerifiedGrant
from .errors import ActionRefused, ApprovalPending, GrantInvalid, GuardError, GuardUnavailable
from .fingerprint import tool_call_payload_hash
from .guard import Guard

try:  # CrewAI versions with structured tool failures
    from crewai.tools.tool_failure import ToolFailure
except ImportError:  # pragma: no cover - exercised by monkeypatching in tests
    ToolFailure = None  # type: ignore[assignment,misc]

__all__ = ["GuardedCrewAITool", "current_grant", "failure_for"]

ArgsFn = Callable[[Mapping[str, Any]], Any]

_current_grant: contextvars.ContextVar[Optional[VerifiedGrant]] = contextvars.ContextVar(
    "complyee_agent_guard_crewai_grant", default=None
)

# How many held steps a single tool remembers so an identical retry resumes them.
MAX_REMEMBERED_APPROVALS = 256


def current_grant() -> Optional[VerifiedGrant]:
    """The verified grant for the step now running, inside the wrapped tool.

    Use grant.decision_id as the idempotency key for the real side effect.
    None outside a guarded call."""
    return _current_grant.get()


def _never_cache(_args: Any = None, _result: Any = None) -> bool:
    return False


def _codes() -> dict:
    return {
        ActionRefused: ("acr_refused", False),
        ApprovalPending: ("acr_approval_pending", True),
        GrantInvalid: ("acr_grant_invalid", False),
        GuardUnavailable: ("acr_unavailable", True),
    }


def _message(error: GuardError) -> str:
    reason = str(error).rstrip(". ")
    if isinstance(error, ApprovalPending):
        return (
            f"This step is waiting for a person to approve it in Agent Control Room "
            f"(approval {error.approval_id or 'unknown'}). The tool did not run. "
            "Calling this tool again later with exactly the same input continues the same "
            "approval request; tell the user it is awaiting approval."
        )
    if isinstance(error, ActionRefused):
        return (
            f"Agent Control Room refused this step: {reason}. The tool did not run. "
            "Do not retry the same request; explain the refusal to the user."
        )
    if isinstance(error, GrantInvalid):
        return (
            f"The permission for this step could not be verified: {reason}. The tool did not run. "
            "Do not retry; report this to the user."
        )
    return (
        f"Agent Control Room could not be reached: {reason}. The tool did not run. "
        "You may try again later."
    )


def failure_for(error: GuardError) -> Any:
    """The value a guarded CrewAI tool returns when its step did not run.

    A ToolFailure carrying a stable code, whether a retry could succeed and the
    decision/approval ids (never the arguments). Plain text when the installed
    CrewAI has no ToolFailure."""
    code, retryable = next(
        (v for cls, v in _codes().items() if isinstance(error, cls)), ("acr_unavailable", True)
    )
    message = _message(error)
    if ToolFailure is None:
        return f"{message} (code: {code})"
    details = {"status": error.status}
    if error.decision_id:
        details["decisionId"] = error.decision_id
    if isinstance(error, ApprovalPending) and error.approval_id:
        details["approvalId"] = error.approval_id
    return ToolFailure(message=message, code=code, retryable=retryable, details=details)


class GuardedCrewAITool(BaseTool):
    """A CrewAI tool that asks Agent Control Room before the wrapped tool runs.

    Options (all keyword): tool, guard, action (default the tool name), name
    (fingerprint and display name, default the tool's), context, resource_ref,
    action_description, parent_decision_id (each a function of the arguments),
    approvals (override the guard's approval settings for this tool) and
    raise_guard_errors (re-raise instead of returning a ToolFailure; for direct
    calls outside a crew, since CrewAI retries tools that raise)."""

    _inner: BaseTool = PrivateAttr()
    _guarded_sync: Callable[..., Any] = PrivateAttr()
    _guarded_async: Callable[..., Any] = PrivateAttr()
    _raise_guard_errors: bool = PrivateAttr(default=False)
    _held: "OrderedDict[str, dict]" = PrivateAttr(default_factory=OrderedDict)
    _held_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _acr_action: str = PrivateAttr()

    def __init__(
        self,
        *,
        tool: BaseTool,
        guard: Guard,
        action: Optional[str] = None,
        name: Optional[str] = None,
        context: Optional[ArgsFn] = None,
        resource_ref: Optional[ArgsFn] = None,
        action_description: Optional[ArgsFn] = None,
        parent_decision_id: Optional[ArgsFn] = None,
        approvals: Optional[Mapping[str, Any]] = None,
        raise_guard_errors: bool = False,
        **fields: Any,
    ) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError("tool must be a crewai BaseTool (wrap plain functions with crewai's @tool first)")
        if not isinstance(guard, Guard):
            raise TypeError("guard must come from complyee_agent_guard.create_guard")
        tool_name = name or tool.name
        defaults = dict(
            name=tool_name,
            description=tool.description,
            args_schema=tool.args_schema,
            result_as_answer=tool.result_as_answer,
            tool_failure_policy=getattr(tool, "tool_failure_policy", None),
            # A governed step must reach Agent Control Room every time: a cached
            # result would skip the decision and a cached refusal would outlive it.
            cache_function=_never_cache,
        )
        if "tool_failure_policy" not in BaseTool.model_fields:  # older CrewAI
            defaults.pop("tool_failure_policy")
        super().__init__(**{**defaults, **fields})

        self._inner = tool
        self._raise_guard_errors = raise_guard_errors
        options = dict(name=tool_name, action=action or tool_name, context=context, resource_ref=resource_ref,
                       action_description=action_description, parent_decision_id=parent_decision_id,
                       approvals=approvals)

        def call_inner(acr_grant: Optional[VerifiedGrant] = None, **arguments: Any) -> Any:
            token = _current_grant.set(acr_grant)
            try:
                return tool.run(**arguments)
            finally:
                _current_grant.reset(token)

        async def call_inner_async(acr_grant: Optional[VerifiedGrant] = None, **arguments: Any) -> Any:
            token = _current_grant.set(acr_grant)
            try:
                if _has_async(tool):
                    return await tool.arun(**arguments)
                # to_thread copies the context, so current_grant() works in sync tools too.
                return await asyncio.to_thread(tool.run, **arguments)
            finally:
                _current_grant.reset(token)

        self._guarded_sync = guard.tool(**options)(call_inner)
        self._guarded_async = guard.tool(**options)(call_inner_async)
        self._acr_action = self._guarded_sync.acr_action  # type: ignore[attr-defined]

    @property
    def wrapped_tool(self) -> BaseTool:
        return self._inner

    @property
    def acr_action(self) -> str:
        return self._acr_action

    # ---- execution ----

    def _arguments(self, args: tuple, kwargs: dict) -> dict:
        if not args:
            return dict(kwargs)
        names = list(self.args_schema.model_fields)
        if len(args) > len(names):
            raise TypeError(f"{self.name} takes at most {len(names)} positional arguments")
        positional = dict(zip(names, args))
        overlap = set(positional) & set(kwargs)
        if overlap:
            raise TypeError(f"{self.name} got multiple values for {sorted(overlap)}")
        return {**positional, **kwargs}

    def _held_key(self, arguments: dict) -> Optional[str]:
        try:
            return tool_call_payload_hash(self.name, arguments)
        except Exception:
            return None  # unfingerprintable: the guard refuses it anyway

    def _take_held(self, key: Optional[str]) -> Optional[dict]:
        if key is None:
            return None
        with self._held_lock:
            return self._held.pop(key, None)

    def _remember_held(self, key: Optional[str], error: GuardError) -> None:
        if key is None or not isinstance(error, ApprovalPending):
            return
        with self._held_lock:
            self._held[key] = error.to_checkpoint()
            self._held.move_to_end(key)
            while len(self._held) > MAX_REMEMBERED_APPROVALS:
                self._held.popitem(last=False)

    def _failed(self, key: Optional[str], error: GuardError) -> Any:
        self._remember_held(key, error)
        if self._raise_guard_errors:
            raise error
        return failure_for(error)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        arguments = self._arguments(args, kwargs)
        key = self._held_key(arguments)
        checkpoint = self._take_held(key)
        try:
            if checkpoint is not None:
                # The same request as the held one: claim that approval, do not open another.
                return self._guarded_sync.resume(checkpoint)  # type: ignore[attr-defined]
            return self._guarded_sync(**arguments)
        except GuardError as error:
            return self._failed(key, error)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        arguments = self._arguments(args, kwargs)
        key = self._held_key(arguments)
        checkpoint = self._take_held(key)
        try:
            if checkpoint is not None:
                return await self._guarded_async.resume(checkpoint)  # type: ignore[attr-defined]
            return await self._guarded_async(**arguments)
        except GuardError as error:
            return self._failed(key, error)


def _has_async(tool: BaseTool) -> bool:
    """True when the tool has its own async implementation."""
    func = getattr(tool, "func", None)
    if func is not None and asyncio.iscoroutinefunction(func):
        return True
    return type(tool)._arun is not BaseTool._arun and not hasattr(tool, "func")
