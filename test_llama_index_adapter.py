"""GuardedLlamaTool against the real llama_index tools and a fake Agent Control Room."""

import asyncio

import pytest

pytest.importorskip("llama_index.core")

from llama_index.core import Document, VectorStoreIndex  # noqa: E402
from llama_index.core.base.llms.types import ChatMessage, MessageRole  # noqa: E402
from llama_index.core.embeddings import MockEmbedding  # noqa: E402
from llama_index.core.llms import MockLLM  # noqa: E402
from llama_index.core.query_engine import CustomQueryEngine  # noqa: E402
from llama_index.core.tools import FunctionTool, QueryEngineTool, RetrieverTool, ToolOutput  # noqa: E402
from llama_index.core.tools.calling import acall_tool, call_tool  # noqa: E402
from llama_index.core.workflow import Context  # noqa: E402

from complyee_agent_guard import ActionRefused, ApprovalPending, tool_call_payload_hash  # noqa: E402
from complyee_agent_guard.llama_index import GuardedLlamaTool, current_grant  # noqa: E402

from conftest import APPROVAL_ID, make_guard  # noqa: E402

try:  # agent workflows and scripted tool-calling models (recent LlamaIndex)
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.base.llms.types import ToolCallBlock
    from llama_index.core.llms.mock import MockFunctionCallingLLM
except ImportError:  # pragma: no cover - older LlamaIndex
    FunctionAgent = None

needs_agents = pytest.mark.skipif(FunctionAgent is None, reason="LlamaIndex without agent workflows")

DECISION_1 = "00000000-0000-4000-8000-000000000001"


def refund_tool(calls):
    def issue_refund(payment_id: str, amount: float) -> str:
        """Refund a payment."""
        grant = current_grant()
        calls.append((payment_id, amount, grant.decision_id if grant else None))
        return f"refunded {amount}"

    return FunctionTool.from_defaults(fn=issue_refund)


def wrap(acr, inner=None, **options):
    calls: list = []
    inner = inner or refund_tool(calls)
    guarded = GuardedLlamaTool(tool=inner, guard=make_guard(acr), action=options.pop("action", "refund.issue"),
                               **options)
    return guarded, calls


ARGS = {"payment_id": "pay_1", "amount": 40.0}
HASH = tool_call_payload_hash("issue_refund", ARGS)


class FinanceEngine(CustomQueryEngine):
    """A query engine standing in for a private knowledge base."""

    asked: list = []

    def custom_query(self, query_str: str) -> str:
        self.asked.append((query_str, current_grant().decision_id if current_grant() else None))
        return "Q3 margin was 41% (confidential)."


def finance_tool():
    engine = FinanceEngine(asked=[])
    tool = QueryEngineTool.from_defaults(query_engine=engine, name="finance_docs",
                                         description="Answer questions from the finance knowledge base.")
    return tool, engine


# ---- what the agent sees ----


def test_agent_sees_the_wrapped_tools_name_description_and_schema(acr):
    calls: list = []
    inner = refund_tool(calls)
    guarded, _ = wrap(acr, inner)
    assert guarded.metadata.name == inner.metadata.name == "issue_refund"
    assert guarded.metadata.description == inner.metadata.description
    assert guarded.metadata.fn_schema is inner.metadata.fn_schema
    assert guarded.metadata.to_openai_tool() == inner.metadata.to_openai_tool()
    assert guarded.acr_action == "refund.issue" and guarded.wrapped_tool is inner


def test_return_direct_and_a_renamed_tool(acr):
    inner = FunctionTool.from_defaults(fn=lambda x: x, name="echo", description="Echo.", return_direct=True)
    guarded = GuardedLlamaTool(tool=inner, guard=make_guard(acr), name="echo_v2")
    assert guarded.metadata.return_direct is True and guarded.metadata.name == "echo_v2"
    assert guarded.acr_action == "echo_v2"
    guarded(x="hi")
    assert acr.ops("decideAction")[0]["parameters"]["payloadHash"] == tool_call_payload_hash("echo_v2", {"x": "hi"})


def test_construction_rejects_what_cannot_be_guarded(acr):
    guard = make_guard(acr)
    with pytest.raises(TypeError, match="BaseTool"):
        GuardedLlamaTool(tool=lambda: None, guard=guard)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="create_guard"):
        GuardedLlamaTool(tool=refund_tool([]), guard=object())  # type: ignore[arg-type]

    async def needs_ctx(ctx: Context, query: str) -> str:
        return query

    with pytest.raises(TypeError, match="workflow Context"):
        GuardedLlamaTool(tool=FunctionTool.from_defaults(async_fn=needs_ctx), guard=guard)


# ---- the gate ----


def test_grant_runs_the_wrapped_tool_with_the_grant_available(acr):
    guarded, calls = wrap(acr, context=lambda a: {"amount": a["amount"]})
    output = guarded(**ARGS)
    assert isinstance(output, ToolOutput) and output.is_error is False
    assert output.content == "refunded 40.0"
    decide = acr.ops("decideAction")[0]["parameters"]
    assert decide["action"] == "refund.issue" and decide["payloadHash"] == HASH
    assert decide["context"] == {"amount": 40.0}
    assert calls == [("pay_1", 40.0, DECISION_1)]
    assert acr.ops("reportExecutionOutcome")[0]["parameters"]["outcome"] == "executed"
    assert current_grant() is None


def test_async_path_awaits_the_same_gate(acr):
    guarded, calls = wrap(acr)
    output = asyncio.run(guarded.acall(**ARGS))
    assert output.content == "refunded 40.0" and calls[0][2] == DECISION_1


def test_positional_and_keyword_calls_fingerprint_identically(acr):
    guarded, _ = wrap(acr)
    guarded("pay_1", amount=40.0)
    assert acr.ops("decideAction")[0]["parameters"]["payloadHash"] == HASH
    with pytest.raises(TypeError, match="multiple values"):
        guarded("pay_1", payment_id="pay_1")


def test_raw_arguments_never_leave_the_process(acr):
    guarded, _ = wrap(acr)
    guarded(payment_id="pay_SECRET_9", amount=40)
    assert all("pay_SECRET_9" not in s["raw"] for s in acr.sent)


def test_refusal_returns_an_error_output_and_the_tool_does_not_run(acr):
    acr.effects = ["reject"]
    guarded, calls = wrap(acr)
    output = guarded(**ARGS)
    assert calls == [] and output.is_error is True
    assert "over limit. The tool did not run." in output.content and "pay_1" not in output.content
    assert output.raw_output == {"code": "acr_refused", "retryable": False, "status": "refused",
                                 "decisionId": DECISION_1}
    assert isinstance(output.exception, ActionRefused)
    assert output.tool_name == "issue_refund"


def test_unreachable_acr_is_a_retryable_error_output(acr):
    acr.gateway_failures = 10
    guarded, calls = wrap(acr)
    output = guarded(**ARGS)
    assert calls == [] and output.raw_output["code"] == "acr_unavailable" and output.raw_output["retryable"] is True


def test_invalid_grant_never_runs_the_tool(acr):
    acr.effects = ["grant-no-jwt"]
    guarded, calls = wrap(acr)
    output = guarded(**ARGS)
    assert calls == [] and output.raw_output["code"] == "acr_grant_invalid"


def test_raise_guard_errors_re_raises_for_direct_callers(acr):
    acr.effects = ["reject"]
    guarded, calls = wrap(acr, raise_guard_errors=True)
    with pytest.raises(ActionRefused):
        guarded(**ARGS)
    assert calls == []


# ---- approvals ----


def test_identical_retry_after_a_hold_resumes_the_same_approval(acr):
    acr.effects = ["pending"]
    guarded, calls = wrap(acr, approvals={"mode": "raise"})
    held = guarded(**ARGS)
    assert held.is_error and held.raw_output["code"] == "acr_approval_pending"
    assert held.raw_output["approvalId"] == APPROVAL_ID and calls == []
    assert isinstance(held.exception, ApprovalPending)

    acr.effects = ["grant"]
    assert guarded(**ARGS).content == "refunded 40.0"
    keys = [s["key"] for s in acr.ops("decideAction")]
    assert len(keys) == 2 and keys[0] == keys[1]  # the same request claimed the approval


def test_identical_async_retry_after_a_hold_resumes_the_same_approval(acr):
    acr.effects = ["pending"]
    guarded, calls = wrap(acr, approvals={"mode": "raise"})
    asyncio.run(guarded.acall(**ARGS))
    acr.effects = ["grant"]
    assert asyncio.run(guarded.acall(**ARGS)).content == "refunded 40.0"
    keys = [s["key"] for s in acr.ops("decideAction")]
    assert keys[0] == keys[1] and len(calls) == 1


def test_different_arguments_after_a_hold_open_a_new_request(acr):
    acr.effects = ["pending"]
    guarded, _ = wrap(acr, approvals={"mode": "raise"})
    guarded(**ARGS)
    acr.effects = ["grant"]
    guarded(payment_id="pay_1", amount=41)
    keys = [s["key"] for s in acr.ops("decideAction")]
    assert keys[0] != keys[1]


def test_approval_granted_while_waiting_runs_once(acr):
    acr.effects = ["pending", "grant"]
    acr.approval_statuses = ["pending", "granted"]
    guarded, calls = wrap(acr)
    assert guarded(**ARGS).content == "refunded 40.0"
    assert len(calls) == 1


# ---- the wrapped tool failing ----


def test_a_failing_tool_becomes_an_error_output_and_a_downstream_error(acr):
    def explode(order_id: str) -> str:
        """Always fails."""
        raise RuntimeError("bank offline")

    guarded = GuardedLlamaTool(tool=FunctionTool.from_defaults(fn=explode), guard=make_guard(acr))
    output = guarded(order_id="o1")
    assert output.is_error and output.content == "Encountered error: bank offline"
    assert isinstance(output.exception, RuntimeError) and output.raw_input == {"order_id": "o1"}
    assert acr.ops("reportExecutionOutcome")[0]["parameters"]["outcome"] == "downstream_error"


def test_an_error_output_from_the_tool_is_recorded_as_a_downstream_error(acr):
    def lookup(input: str) -> str:
        """Look something up."""
        return "unused"

    inner = FunctionTool.from_defaults(fn=lookup)
    inner.call = lambda *a, **k: ToolOutput(content="not found", tool_name="lookup", raw_input={}, raw_output=None,
                                            is_error=True)  # type: ignore[method-assign]
    guarded = GuardedLlamaTool(tool=inner, guard=make_guard(acr))
    output = guarded(input="x")
    assert output.is_error and output.content == "not found"
    assert acr.ops("reportExecutionOutcome")[0]["parameters"]["outcome"] == "downstream_error"


def test_llama_index_call_helpers_never_open_a_second_request(acr):
    """call_tool retries a single-input tool with keyword arguments when the
    first call raises; the wrapper never raises, so there is one decision."""

    tool, _ = finance_tool()
    tool.call = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))  # type: ignore[method-assign]
    guarded = GuardedLlamaTool(tool=tool, guard=make_guard(acr), action="knowledge.query")
    output = call_tool(guarded, {"input": "margin?"})
    assert output.is_error and len(acr.ops("decideAction")) == 1


# ---- data boundaries: knowledge tools ----


def test_query_engine_tool_is_guarded_and_the_query_stays_local(acr):
    tool, engine = finance_tool()
    guarded = GuardedLlamaTool(tool=tool, guard=make_guard(acr), action="knowledge.query",
                               resource_ref=lambda a: "kb:finance",
                               context=lambda a: {"classification": "confidential"})
    output = call_tool(guarded, {"input": "What was the Q3 margin?"})  # positional, as LlamaIndex calls it
    assert output.content == "Q3 margin was 41% (confidential)."
    assert engine.asked == [("What was the Q3 margin?", DECISION_1)]
    decide = acr.ops("decideAction")[0]["parameters"]
    assert decide["resourceRef"] == "kb:finance" and decide["context"] == {"classification": "confidential"}
    assert decide["payloadHash"] == tool_call_payload_hash("finance_docs", {"input": "What was the Q3 margin?"})
    assert all("Q3 margin" not in s["raw"] and "41%" not in s["raw"] for s in acr.sent)


def test_refused_knowledge_query_never_reaches_the_index(acr):
    acr.effects = ["reject"]
    tool, engine = finance_tool()
    guarded = GuardedLlamaTool(tool=tool, guard=make_guard(acr), action="knowledge.query")
    output = asyncio.run(acall_tool(guarded, {"input": "What was the Q3 margin?"}))
    assert output.is_error and engine.asked == []


def test_retriever_tool_over_a_real_index(acr):
    index = VectorStoreIndex.from_documents(
        [Document(text="Salary bands are confidential."), Document(text="The office opens at nine.")],
        embed_model=MockEmbedding(embed_dim=8),
    )
    tool = RetrieverTool.from_defaults(retriever=index.as_retriever(similarity_top_k=1), name="hr_docs",
                                       description="Search HR documents.")
    guarded = GuardedLlamaTool(tool=tool, guard=make_guard(acr), action="knowledge.retrieve")
    output = call_tool(guarded, {"input": "salary"})
    assert output.is_error is False and output.content.strip()
    assert all("Salary" not in s["raw"] for s in acr.sent)


# ---- a real agent run ----


def scripted_llm(tool_name, tool_kwargs, seen):
    def respond(messages, **kwargs):
        tool_results = [m for m in messages if m.role == MessageRole.TOOL]
        if tool_results:
            seen.extend(m.content for m in tool_results)
            return ChatMessage(role=MessageRole.ASSISTANT, content=f"Done: {tool_results[-1].content}")
        return ChatMessage(role=MessageRole.ASSISTANT,
                           blocks=[ToolCallBlock(tool_call_id="call-1", tool_name=tool_name, tool_kwargs=tool_kwargs)])

    return MockFunctionCallingLLM(response_generator=respond, is_chat_model=True)


async def run_agent(guarded, llm):
    agent = FunctionAgent(tools=[guarded], llm=llm, system_prompt="Support agent.")
    return await agent.run(user_msg="Please refund pay_1.")


@needs_agents
def test_function_agent_runs_an_approved_step_once(acr):
    seen: list = []
    guarded, calls = wrap(acr)
    result = asyncio.run(run_agent(guarded, scripted_llm("issue_refund", ARGS, seen)))
    assert calls == [("pay_1", 40.0, DECISION_1)]
    assert seen == ["refunded 40.0"] and "refunded 40.0" in str(result)
    assert len(acr.ops("decideAction")) == 1


@needs_agents
@pytest.mark.parametrize("effect,phrase", [("reject", "refused this step"), ("pending", "waiting for a person")])
def test_function_agent_reads_why_a_step_did_not_run(acr, effect, phrase):
    acr.effects = [effect]
    seen: list = []
    guarded, calls = wrap(acr, approvals={"mode": "raise"})
    result = asyncio.run(run_agent(guarded, scripted_llm("issue_refund", ARGS, seen)))
    assert calls == [] and len(acr.ops("decideAction")) == 1
    assert phrase in seen[0] and "The tool did not run." in str(result)


def test_package_imports_without_llama_index_needed_for_the_core():
    import complyee_agent_guard

    assert "llama_index" not in complyee_agent_guard.__all__
    assert MockLLM  # the optional extra is installed in this environment
