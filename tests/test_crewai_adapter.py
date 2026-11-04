"""GuardedCrewAITool against the real crewai BaseTool and a fake Agent Control Room."""

import asyncio

import pytest

crewai_tools = pytest.importorskip("crewai.tools")

from crewai.tools import BaseTool, tool  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from complyee_agent_guard import ActionRefused, ApprovalPending, tool_call_payload_hash  # noqa: E402
from complyee_agent_guard import crewai as adapter  # noqa: E402
from complyee_agent_guard.crewai import GuardedCrewAITool, current_grant  # noqa: E402

from conftest import APPROVAL_ID, make_guard  # noqa: E402

ToolFailure = adapter.ToolFailure


class RefundArgs(BaseModel):
    payment_id: str
    amount: float


class RefundTool(BaseTool):
    name: str = "issue_refund"
    description: str = "Refund a payment."
    args_schema: type[BaseModel] = RefundArgs
    calls: list = []

    def _run(self, payment_id: str, amount: float) -> str:
        grant = current_grant()
        self.calls.append((payment_id, amount, grant.decision_id if grant else None))
        return f"refunded {amount}"


def wrap(acr, inner=None, **options):
    inner = inner or RefundTool(calls=[])
    guarded = GuardedCrewAITool(tool=inner, guard=make_guard(acr), action="refund.issue", **options)
    return guarded, inner


ARGS = {"payment_id": "pay_1", "amount": 40.0}
HASH = tool_call_payload_hash("issue_refund", ARGS)


def test_agent_sees_the_wrapped_tools_name_description_and_schema(acr):
    guarded, inner = wrap(acr)
    assert guarded.name == inner.name
    assert guarded.description == inner.description
    assert guarded.args_schema is RefundArgs
    assert guarded.acr_action == "refund.issue"


def test_grant_runs_the_wrapped_tool_with_the_grant_available(acr):
    guarded, inner = wrap(acr, context=lambda a: {"amount": a["amount"]})
    assert guarded.run(**ARGS) == "refunded 40.0"
    decide = acr.ops("decideAction")[0]["parameters"]
    assert decide["action"] == "refund.issue" and decide["payloadHash"] == HASH
    assert decide["context"] == {"amount": 40.0}
    assert inner.calls == [("pay_1", 40.0, "00000000-0000-4000-8000-000000000001")]
    assert acr.ops("reportExecutionOutcome")[0]["parameters"]["outcome"] == "executed"
    assert current_grant() is None


def test_async_path_awaits_the_same_gate(acr):
    guarded, inner = wrap(acr)
    assert asyncio.run(guarded.arun(**ARGS)) == "refunded 40.0"
    assert inner.calls[0][2] == "00000000-0000-4000-8000-000000000001"


def test_raw_arguments_never_leave_the_process(acr):
    guarded, _ = wrap(acr)
    guarded.run(payment_id="pay_SECRET_9", amount=40)
    assert all("pay_SECRET_9" not in s["raw"] for s in acr.sent)


def test_refusal_returns_a_tool_failure_and_the_tool_does_not_run(acr):
    acr.effects = ["reject"]
    guarded, inner = wrap(acr)
    result = guarded.run(**ARGS)
    assert inner.calls == []
    assert isinstance(result, ToolFailure)
    assert result.code == "acr_refused" and result.retryable is False
    assert "over limit. The tool did not run." in result.message
    assert result.details == {"status": "refused", "decisionId": "00000000-0000-4000-8000-000000000001"}
    assert "pay_1" not in result.model_dump_json()


def test_unreachable_acr_is_a_retryable_failure(acr):
    acr.gateway_failures = 10
    guarded, inner = wrap(acr)
    result = guarded.run(**ARGS)
    assert inner.calls == [] and result.code == "acr_unavailable" and result.retryable is True


def test_invalid_grant_never_runs_the_tool(acr):
    acr.effects = ["grant-no-jwt"]
    guarded, inner = wrap(acr)
    result = guarded.run(**ARGS)
    assert inner.calls == [] and result.code == "acr_grant_invalid" and result.retryable is False


def test_identical_retry_after_a_hold_resumes_the_same_approval(acr):
    acr.effects = ["pending"]
    guarded, inner = wrap(acr, approvals={"mode": "raise"})
    held = guarded.run(**ARGS)
    assert held.code == "acr_approval_pending" and held.retryable is True
    assert held.details["approvalId"] == APPROVAL_ID and inner.calls == []

    acr.effects = ["grant"]
    assert guarded.run(**ARGS) == "refunded 40.0"
    keys = [s["key"] for s in acr.ops("decideAction")]
    assert len(keys) == 2 and keys[0] == keys[1]  # the same request claimed the approval


def test_different_arguments_after_a_hold_open_a_new_request(acr):
    acr.effects = ["pending"]
    guarded, _ = wrap(acr, approvals={"mode": "raise"})
    guarded.run(**ARGS)
    acr.effects = ["grant"]
    guarded.run(payment_id="pay_1", amount=41)
    keys = [s["key"] for s in acr.ops("decideAction")]
    assert keys[0] != keys[1]


def test_raise_guard_errors_re_raises_for_direct_callers(acr):
    acr.effects = ["reject"]
    guarded, _ = wrap(acr, raise_guard_errors=True)
    with pytest.raises(ActionRefused):
        guarded.run(**ARGS)
    acr.effects = ["pending"]
    guarded, _ = wrap(acr, approvals={"mode": "raise"}, raise_guard_errors=True)
    with pytest.raises(ApprovalPending):
        guarded.run(**ARGS)


def test_results_are_never_cached(acr):
    guarded, _ = wrap(acr)
    assert guarded.cache_function(ARGS, "refunded 40.0") is False


def test_wraps_tools_made_with_crewais_tool_decorator(acr):
    seen = []

    @tool("lookup_order")
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        seen.append((order_id, current_grant().decision_id))
        return "shipped"

    guarded = GuardedCrewAITool(tool=lookup_order, guard=make_guard(acr), action="order.read")
    assert guarded.run(order_id="o-1") == "shipped"
    assert asyncio.run(guarded.arun(order_id="o-2")) == "shipped"
    assert [s[0] for s in seen] == ["o-1", "o-2"] and all(s[1] for s in seen)
    assert acr.ops("decideAction")[0]["parameters"]["payloadHash"] == tool_call_payload_hash(
        "lookup_order", {"order_id": "o-1"})


def test_crewai_structured_tool_path_is_gated(acr):
    acr.effects = ["reject"]
    guarded, inner = wrap(acr)
    structured = guarded.to_structured_tool()
    result = structured.invoke(ARGS)
    assert inner.calls == []
    assert isinstance(result, ToolFailure) and result.code == "acr_refused"


def test_plain_text_on_crewai_without_tool_failure(acr, monkeypatch):
    monkeypatch.setattr(adapter, "ToolFailure", None)
    acr.effects = ["reject"]
    guarded, _ = wrap(acr)
    result = guarded.run(**ARGS)
    assert isinstance(result, str) and result.endswith("(code: acr_refused)")


def test_rejects_things_that_are_not_crewai_tools(acr):
    with pytest.raises(TypeError):
        GuardedCrewAITool(tool=lambda: None, guard=make_guard(acr))
