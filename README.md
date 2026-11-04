# complyee-agent-guard

A customer-run companion that puts an Agent Control Room grant check in front of
any Python agent tool — LangGraph, CrewAI, LangChain or plain functions. It does
the repeatable work (fingerprinting, asking, waiting for a person, claiming,
verifying the grant token, reporting the outcome); you supply the action name and
the context facts your rules need.

Status: 0.2 (beta). Python 3.9 or later; depends only on `httpx` and `cryptography`.
The optional CrewAI adapter needs `crewai` 1.0 or later (Python 3.10 or later).

```bash
pip install complyee-agent-guard
```

[Agent Control Room](https://agent-cr.com) is the decision point; this package is
the enforcement point that runs inside your agent process. See
[agent-cr.com/plugins](https://agent-cr.com/plugins) and
[agent-cr.com/enforcement-point](https://agent-cr.com/enforcement-point).

## Use

```python
import os
from complyee_agent_guard import create_guard, ApprovalPending, ActionRefused, VerifiedGrant

guard = create_guard(
    workspace_url="https://acme.agent-cr.com",  # your workspace address (also the JWT audience)
    api_key=os.environ["ACR_API_KEY"],          # scopes: policy.decide, policy.read
    agent="finance-assistant",
    approvals={"mode": "poll", "poll_timeout_s": 30, "poll_interval_s": 2},  # the defaults
)

@guard.tool(action="refund.issue", context=lambda a: {"amount": a["amount"]})
async def issue_refund(payment_id: str, amount: float, acr_grant: VerifiedGrant = None):
    # Runs only after a verified grant for exactly these arguments.
    return await payments.refund(payment_id, amount, idempotency_key=acr_grant.decision_id)
```

A plain `def` tool works the same way, synchronously. Add a parameter named
`acr_grant` to receive the verified grant; it is never part of the fingerprint.

Decorator options: `name` (tool name, default the function name), `action`
(default the tool name), `context`, `resource_ref`, `action_description`,
`parent_decision_id` (each a function of the arguments), and `approvals` to
override the guard's approval settings for one tool.

## CrewAI

```bash
pip install "complyee-agent-guard[crewai]"
```

Wrap any CrewAI tool — a `BaseTool` subclass, a `crewai_tools` tool or a function
made with CrewAI's `@tool` — and give the wrapped tool to your agent:

```python
from crewai import Agent
from crewai.tools import tool
from complyee_agent_guard.crewai import GuardedCrewAITool, current_grant

@tool("issue_refund")
def issue_refund(payment_id: str, amount: float) -> str:
    """Refund a payment."""
    grant = current_grant()  # the verified grant for exactly these arguments
    payments.refund(payment_id, amount, idempotency_key=grant.decision_id)
    return "Refund issued."

refund = GuardedCrewAITool(
    tool=issue_refund,
    guard=guard,                                 # from create_guard(...) above
    action="refund.issue",
    context=lambda a: {"amount": a["amount"]},
)

agent = Agent(role="Support lead", goal="...", backstory="...", tools=[refund])
```

The agent sees the wrapped tool's own name, description and argument schema.
`GuardedCrewAITool` takes the same options as the decorator (`action`, `name`,
`context`, `resource_ref`, `action_description`, `parent_decision_id`,
`approvals`). Inside the wrapped tool, `current_grant()` returns the verified grant.

When the step does not run, the tool returns a CrewAI `ToolFailure` instead of
raising — CrewAI retries tools that raise, and each retry would be a new request.
The failure tells the agent why in plain words, never contains the arguments and
carries a stable code:

| Code | Meaning | `retryable` |
|---|---|---|
| `acr_refused` | policy or a person refused, or the arguments cannot be fingerprinted | no |
| `acr_approval_pending` | a person has not approved yet | yes |
| `acr_grant_invalid` | the grant token is missing or failed verification | no |
| `acr_unavailable` | Agent Control Room could not be reached | yes |

What happens next is CrewAI's own `tool_failure_policy`, set on the tool, task,
agent or crew: `"warn"` (CrewAI's default) lets the agent continue and explain,
`"raise"` stops the crew with `ToolExecutionFailedError`. Either way the failure is
listed in the task output's `tool_failures`.

If a person has not approved yet and the agent later calls the tool again with
exactly the same input, the same approval request continues — no second request
is opened. Results of a guarded tool are never cached by CrewAI, so every call is
decided. Pass `raise_guard_errors=True` only when you call the tool directly
outside a crew and want the exceptions below instead.

## When a person must approve

- `"poll"` (default): waits up to `poll_timeout_s`, then claims the grant with the
  identical request. If nobody answers in time it raises `ApprovalPending`.
- `"raise"`: raises `ApprovalPending` at once — useful when a LangGraph workflow
  should checkpoint and resume later.

```python
try:
    result = await issue_refund("pay_1", amount=40)
except ApprovalPending as held:
    store.save(held.to_checkpoint())                 # JSON-safe; arguments stay in your store
...
result = await issue_refund.resume(store.load())    # same request, same idempotency key
```

## Fail closed

The tool body never runs when any of these is raised:

| Exception | Meaning |
|---|---|
| `ActionRefused` | policy or a person refused, or the arguments cannot be fingerprinted |
| `ApprovalPending` | a person has not approved yet |
| `GrantInvalid` | the grant token is missing or failed verification (no fallback to any other format) |
| `GuardUnavailable` | Agent Control Room could not be reached |

Raw arguments never leave your process: they are hashed locally
(`sha256-canonical-json-v1`, identical to ACR and the TypeScript MCP guard) and
only the hash is sent. A failed outcome report never changes the tool result.

## Tests

```bash
pip install -e ".[dev]" && pytest            # CrewAI tests are skipped
pip install -e ".[dev,crewai]" && pytest     # includes the CrewAI adapter
```

`tests/fixtures/grant.json` and `tests/fixtures/conformance-vectors.json` are
generated from Agent Control Room's own code and are shared with the TypeScript
companion ([complyee/mcp-guard](https://github.com/complyee/mcp-guard)), so all
three hash and verify identically. Do not edit them by hand.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Copyright Complyee AB.
