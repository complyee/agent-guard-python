"""LlamaIndex adapter: wrap any LlamaIndex tool so it runs only on a verified grant.

    from complyee_agent_guard import create_guard
    from complyee_agent_guard.llama_index import GuardedLlamaTool

    guard = create_guard(workspace_url=..., api_key=..., agent="analyst-agent")
    finance_docs = GuardedLlamaTool(
        tool=finance_query_tool,            # QueryEngineTool, RetrieverTool, FunctionTool or any BaseTool
        guard=guard,
        action="knowledge.query",
        resource_ref=lambda a: "kb:finance",
        context=lambda a: {"classification": "confidential"},
    )
    FunctionAgent(tools=[finance_docs], llm=llm)

The same wrapper governs a knowledge lookup (the query, the retrieved
documents and the answer stay in your process; Agent Control Room sees only a
fingerprint of the query and the facts you declare) and an action tool.

The agent sees the wrapped tool's own name, description and argument schema.
When the step does not run, the tool returns a ToolOutput with is_error=True
and a plain explanation instead of raising: LlamaIndex agents read that output
and carry on, while an exception can make the older tool-calling helpers try
again with a fresh request to Agent Control Room.

Requires the optional extra: pip install "complyee-agent-guard[llama-index]".
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import threading
from collections import OrderedDict
from typing import Any, Callable, Mapping, Optional

try:
    from llama_index.core.tools import FunctionTool
    from llama_index.core.tools.types import AsyncBaseTool, BaseTool, ToolMetadata, ToolOutput
except ImportError as missing:  # pragma: no cover - exercised without the extra installed
    raise ImportError(
        'complyee_agent_guard.llama_index needs LlamaIndex: pip install "complyee-agent-guard[llama-index]"'
    ) from missing

from .core import VerifiedGrant
from .errors import ActionRefused, ApprovalPending, GrantInvalid, GuardError, GuardUnavailable
from .fingerprint import tool_call_payload_hash
from .guard import Guard

__all__ = ["GuardedLlamaTool", "current_grant", "output_for"]

ArgsFn = Callable[[Mapping[str, Any]], Any]

_current_grant: contextvars.ContextVar[Optional[VerifiedGrant]] = contextvars.ContextVar(
    "complyee_agent_guard_llama_index_grant", default=None
)
# How many leading arguments the caller passed positionally, so the wrapped
# tool is called exactly as it would have been without the guard.
_positional_count: contextvars.ContextVar[int] = contextvars.ContextVar(
    "complyee_agent_guard_llama_index_positional", default=0
)

# How many held steps a single tool remembers so an identical retry resumes them.
MAX_REMEMBERED_APPROVALS = 256


def current_grant() -> Optional[VerifiedGrant]:
    """The verified grant for the step now running, inside the wrapped tool.

    Use grant.decision_id as the idempotency key for the real side effect.
    None outside a guarded call."""
    return _current_grant.get()


class _InnerFailed(Exception):
    """The wrapped tool ran and failed. Carries the error output so the guard
    records a downstream error and the agent still gets a ToolOutput."""

    def __init__(self, output: ToolOutput):
        super().__init__(output.content)
        self.output = output


_CODES = (
    (ActionRefused, "acr_refused", False),
    (ApprovalPending, "acr_approval_pending", True),
    (GrantInvalid, "acr_grant_invalid", False),
    (GuardUnavailable, "acr_unavailable", True),
)


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


def output_for(error: GuardError, tool_name: str, arguments: Optional[Mapping[str, Any]] = None) -> ToolOutput:
    """The ToolOutput a guarded LlamaIndex tool returns when its step did not run.

    is_error=True, a plain explanation as content (what the model reads, never
    the arguments) and, as raw_output, a stable code, whether a retry could
    succeed and the decision/approval ids. The GuardError is on .exception."""
    code, retryable = next(((c, r) for cls, c, r in _CODES if isinstance(error, cls)), ("acr_unavailable", True))
    details: dict = {"code": code, "retryable": retryable, "status": error.status}
    if error.decision_id:
        details["decisionId"] = error.decision_id
    if isinstance(error, ApprovalPending) and error.approval_id:
        details["approvalId"] = error.approval_id
    return _tool_output(
        content=_message(error),
        tool_name=tool_name,
        raw_input=dict(arguments or {}),
        raw_output=details,
        exception=error,
    )


def _tool_output(*, content: str, tool_name: str, raw_input: dict, raw_output: Any, exception: Exception) -> ToolOutput:
    try:
        return ToolOutput(content=content, tool_name=tool_name, raw_input=raw_input, raw_output=raw_output,
                          is_error=True, exception=exception)
    except TypeError:  # pragma: no cover - LlamaIndex versions before ToolOutput(exception=...)
        return ToolOutput(content=content, tool_name=tool_name, raw_input=raw_input, raw_output=raw_output,
                          is_error=True)


def _field_names(metadata: ToolMetadata) -> list:
    schema = metadata.fn_schema
    if schema is None:
        return ["input"]
    return list(schema.model_fields)


class GuardedLlamaTool(AsyncBaseTool):
    """A LlamaIndex tool that asks Agent Control Room before the wrapped tool runs.

    Options (all keyword): tool, guard, action (default the tool name), name
    (fingerprint and display name, default the tool's), context, resource_ref,
    action_description, parent_decision_id (each a function of the arguments),
    approvals (override the guard's approval settings for this tool) and
    raise_guard_errors (re-raise instead of returning an error ToolOutput; for
    direct calls outside an agent).

    If the wrapped tool itself raises, the agent gets an error ToolOutput
    (the exception is on .exception) and Agent Control Room records a
    downstream error, so no helper retries the step under a second request."""

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
    ) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError("tool must be a llama_index BaseTool (wrap plain functions with FunctionTool.from_defaults first)")
        if not isinstance(guard, Guard):
            raise TypeError("guard must come from complyee_agent_guard.create_guard")
        if isinstance(tool, FunctionTool) and getattr(tool, "requires_context", False):
            raise TypeError(
                "tools that take the workflow Context cannot be guarded: the agent passes the "
                "Context only to unwrapped FunctionTools and it cannot be fingerprinted"
            )
        inner_meta = tool.metadata
        tool_name = name or inner_meta.name
        if not tool_name:
            raise ValueError("the tool has no name; pass name=")

        self._inner = tool
        self._metadata = dataclasses.replace(inner_meta, name=tool_name)
        self._fields = _field_names(inner_meta)
        self._raise_guard_errors = raise_guard_errors
        self._held: "OrderedDict[str, dict]" = OrderedDict()
        self._held_lock = threading.Lock()

        options = dict(name=tool_name, action=action or tool_name, context=context, resource_ref=resource_ref,
                       action_description=action_description, parent_decision_id=parent_decision_id,
                       approvals=approvals)
        fields = self._fields

        def split(arguments: dict) -> tuple:
            count = _positional_count.get()
            names = [n for n in fields[:count] if n in arguments]
            return tuple(arguments[n] for n in names), {k: v for k, v in arguments.items() if k not in names}

        def failed(error: Exception) -> _InnerFailed:
            return _InnerFailed(_tool_output(content=f"Encountered error: {error}", tool_name=tool_name,
                                             raw_input={}, raw_output=str(error), exception=error))

        def checked(output: Any) -> Any:
            if isinstance(output, ToolOutput) and output.is_error:
                raise _InnerFailed(output)
            return output

        def call_inner(acr_grant: Optional[VerifiedGrant] = None, **arguments: Any) -> Any:
            a, kw = split(arguments)
            token = _current_grant.set(acr_grant)
            try:
                output = tool(*a, **kw)
            except Exception as error:
                raise failed(error) from error
            finally:
                _current_grant.reset(token)
            return checked(output)

        async def call_inner_async(acr_grant: Optional[VerifiedGrant] = None, **arguments: Any) -> Any:
            a, kw = split(arguments)
            token = _current_grant.set(acr_grant)
            try:
                if isinstance(tool, AsyncBaseTool):
                    output = await tool.acall(*a, **kw)
                else:
                    # to_thread copies the context, so current_grant() works in sync tools too.
                    output = await asyncio.to_thread(tool, *a, **kw)
            except Exception as error:
                raise failed(error) from error
            finally:
                _current_grant.reset(token)
            return checked(output)

        self._guarded_sync = guard.tool(**options)(call_inner)
        self._guarded_async = guard.tool(**options)(call_inner_async)
        self._acr_action = self._guarded_sync.acr_action  # type: ignore[attr-defined]

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

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
        if len(args) > len(self._fields):
            raise TypeError(f"{self._metadata.name} takes at most {len(self._fields)} positional arguments")
        positional = dict(zip(self._fields, args))
        overlap = set(positional) & set(kwargs)
        if overlap:
            raise TypeError(f"{self._metadata.name} got multiple values for {sorted(overlap)}")
        return {**positional, **kwargs}

    def _held_key(self, arguments: dict) -> Optional[str]:
        try:
            return tool_call_payload_hash(self._metadata.name, arguments)
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

    def _failed(self, key: Optional[str], error: GuardError, arguments: dict) -> ToolOutput:
        self._remember_held(key, error)
        if self._raise_guard_errors:
            raise error
        return output_for(error, self._metadata.name, arguments)

    def _with_input(self, output: _InnerFailed, arguments: dict) -> ToolOutput:
        result = output.output
        if not result.raw_input:
            result.raw_input = dict(arguments)
        return result

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        arguments = self._arguments(args, kwargs)
        key = self._held_key(arguments)
        checkpoint = self._take_held(key)
        token = _positional_count.set(len(args))
        try:
            if checkpoint is not None:
                # The same request as the held one: claim that approval, do not open another.
                return self._guarded_sync.resume(checkpoint)  # type: ignore[attr-defined]
            return self._guarded_sync(**arguments)
        except _InnerFailed as failure:
            return self._with_input(failure, arguments)
        except GuardError as error:
            return self._failed(key, error, arguments)
        finally:
            _positional_count.reset(token)

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        arguments = self._arguments(args, kwargs)
        key = self._held_key(arguments)
        checkpoint = self._take_held(key)
        token = _positional_count.set(len(args))
        try:
            if checkpoint is not None:
                return await self._guarded_async.resume(checkpoint)  # type: ignore[attr-defined]
            return await self._guarded_async(**arguments)
        except _InnerFailed as failure:
            return self._with_input(failure, arguments)
        except GuardError as error:
            return self._failed(key, error, arguments)
        finally:
            _positional_count.reset(token)
