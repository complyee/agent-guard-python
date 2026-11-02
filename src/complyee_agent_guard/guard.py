"""create_guard and the @guard.tool decorator.

    guard = create_guard(workspace_url=..., api_key=..., agent="invoice-agent")

    @guard.tool(action="refund.issue", context=lambda a: {"amount": a["amount"]})
    async def issue_refund(order_id: str, amount: float, acr_grant=None): ...

An async def tool is awaited; a plain def tool runs synchronously. The tool
body runs only after a verified grant for exactly these arguments.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Awaitable, Callable, Mapping, Optional, Union

import httpx

from .config import ApprovalSettings, GuardConfig
from .core import JwksCache, ToolPlan, VerifiedGrant, authorize, report_call
from .drivers import AsyncDriver, SyncDriver, default_async_sleep, default_sync_sleep

GRANT_PARAMETER = "acr_grant"
ArgsFn = Callable[[Mapping[str, Any]], Any]


class Guard:
    def __init__(
        self,
        config: GuardConfig,
        *,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        sleep: Callable[[float], None] = default_sync_sleep,
        async_sleep: Callable[[float], Awaitable[None]] = default_async_sleep,
    ):
        self.config = config
        self.keys = JwksCache(config.jwks_url, config.jwks_cache_ttl_s, config.jwks_min_refresh_s)
        self._sync = SyncDriver(config, http_client, sleep)
        self._async = AsyncDriver(config, async_http_client, async_sleep)

    # ---- decorator ----

    def tool(
        self,
        func: Optional[Callable] = None,
        *,
        name: Optional[str] = None,
        action: Optional[str] = None,
        context: Optional[ArgsFn] = None,
        resource_ref: Optional[ArgsFn] = None,
        action_description: Optional[ArgsFn] = None,
        parent_decision_id: Optional[ArgsFn] = None,
        approvals: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Use as @guard.tool or @guard.tool(action=..., context=...)."""
        if func is not None:
            return self.tool(name=name, action=action, context=context, resource_ref=resource_ref,
                             action_description=action_description, parent_decision_id=parent_decision_id,
                             approvals=approvals)(func)

        settings = self.config.approvals.merged(approvals)

        def decorate(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            act = action or tool_name
            if "|" in act:
                raise ValueError(f'action "{act}" must not contain |')
            signature = inspect.signature(fn)
            wants_grant = GRANT_PARAMETER in signature.parameters

            def bind(args: tuple, kwargs: dict) -> dict:
                kwargs = {k: v for k, v in kwargs.items() if k != GRANT_PARAMETER}
                bound = signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
                arguments.pop(GRANT_PARAMETER, None)
                # **kwargs parameters are flattened so the fingerprint sees the real names.
                for p in signature.parameters.values():
                    if p.kind is inspect.Parameter.VAR_KEYWORD and p.name in arguments:
                        arguments.update(arguments.pop(p.name))
                    elif p.kind is inspect.Parameter.VAR_POSITIONAL and p.name in arguments:
                        arguments[p.name] = list(arguments[p.name])
                return arguments

            def plan_for(arguments: dict) -> ToolPlan:
                return ToolPlan(
                    tool=tool_name,
                    action=act,
                    arguments=arguments,
                    approvals=settings,
                    context=context(arguments) if context else None,
                    resource_ref=resource_ref(arguments) if resource_ref else None,
                    action_description=action_description(arguments) if action_description else None,
                    parent_decision_id=parent_decision_id(arguments) if parent_decision_id else None,
                )

            def call_kwargs(arguments: dict, grant: VerifiedGrant) -> tuple[tuple, dict]:
                values = dict(arguments)
                if wants_grant:
                    values[GRANT_PARAMETER] = grant
                args, kwargs = [], {}
                for p in signature.parameters.values():
                    if p.kind is inspect.Parameter.VAR_POSITIONAL:
                        args.extend(values.pop(p.name, []))
                    elif p.kind is inspect.Parameter.POSITIONAL_ONLY:
                        args.append(values.pop(p.name))
                kwargs.update(values)
                return tuple(args), kwargs

            def check_checkpoint(checkpoint: Mapping[str, Any]) -> tuple[dict, str]:
                if checkpoint.get("tool") != tool_name or not checkpoint.get("idempotencyKey"):
                    raise ValueError(f"checkpoint does not belong to tool '{tool_name}'")
                return dict(checkpoint.get("arguments") or {}), str(checkpoint["idempotencyKey"])

            if inspect.iscoroutinefunction(fn):

                async def run_async(arguments: dict, key: str) -> Any:
                    grant = await self._async.run(authorize(self.config, self.keys, plan_for(arguments), key))
                    a, kw = call_kwargs(arguments, grant)
                    try:
                        result = await fn(*a, **kw)
                    except BaseException:
                        await self._report_async(grant, "downstream_error")
                        raise
                    await self._report_async(grant, "executed")
                    return result

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    # One key per invocation: a retry and the claim after approval reuse it.
                    return await run_async(bind(args, kwargs), self.config.random_id())

                async def resume_async(checkpoint: Mapping[str, Any]) -> Any:
                    return await run_async(*check_checkpoint(checkpoint))

                async_wrapper.resume = resume_async  # type: ignore[attr-defined]
                async_wrapper.acr_action = act  # type: ignore[attr-defined]
                return async_wrapper

            def run_sync(arguments: dict, key: str) -> Any:
                grant = self._sync.run(authorize(self.config, self.keys, plan_for(arguments), key))
                a, kw = call_kwargs(arguments, grant)
                try:
                    result = fn(*a, **kw)
                except BaseException:
                    self._report_sync(grant, "downstream_error")
                    raise
                self._report_sync(grant, "executed")
                return result

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return run_sync(bind(args, kwargs), self.config.random_id())

            def resume_sync(checkpoint: Mapping[str, Any]) -> Any:
                return run_sync(*check_checkpoint(checkpoint))

            sync_wrapper.resume = resume_sync  # type: ignore[attr-defined]
            sync_wrapper.acr_action = act  # type: ignore[attr-defined]
            return sync_wrapper

        return decorate

    # ---- outcome reporting: evidence only, never changes the tool result ----

    def _report_sync(self, grant: VerifiedGrant, outcome: str) -> None:
        try:
            self._sync.call(report_call(grant, outcome, self.config.now()))
        except Exception as error:
            if self.config.on_report_error:
                self.config.on_report_error(error)

    async def _report_async(self, grant: VerifiedGrant, outcome: str) -> None:
        try:
            await self._async.run(_single(report_call(grant, outcome, self.config.now())))
        except Exception as error:
            if self.config.on_report_error:
                self.config.on_report_error(error)


def _single(op: Any):
    result = yield op
    return result


def create_guard(
    *,
    workspace_url: str,
    api_key: str,
    agent: str,
    approvals: Optional[Union[Mapping[str, Any], ApprovalSettings]] = None,
    http_client: Optional[httpx.Client] = None,
    async_http_client: Optional[httpx.AsyncClient] = None,
    sleep: Callable[[float], None] = default_sync_sleep,
    async_sleep: Callable[[float], Awaitable[None]] = default_async_sleep,
    **options: Any,
) -> Guard:
    """Build a guard. options: jwks_url, issuer, jwks_cache_ttl_s, clock_skew_s,
    retries, timeout_s, on_report_error, now, random_id."""
    settings = approvals if isinstance(approvals, ApprovalSettings) else ApprovalSettings().merged(approvals)
    config = GuardConfig(workspace_url=workspace_url, api_key=api_key, agent=agent, approvals=settings, **options)
    return Guard(config, http_client=http_client, async_http_client=async_http_client, sleep=sleep, async_sleep=async_sleep)
