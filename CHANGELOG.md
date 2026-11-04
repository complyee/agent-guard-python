# Changelog

## 0.2.0 — CrewAI adapter

- `complyee_agent_guard.crewai.GuardedCrewAITool` wraps any CrewAI tool (including `@tool` functions) so it runs only on a verified grant; install with `pip install "complyee-agent-guard[crewai]"`.
- A step that does not run returns a CrewAI `ToolFailure` with a stable code (`acr_refused`, `acr_approval_pending`, `acr_grant_invalid`, `acr_unavailable`) instead of raising, so CrewAI does not retry it as a new request; CrewAI's `tool_failure_policy` decides whether the crew continues.
- An identical call after a hold continues the same approval request; guarded results are never cached.
- `current_grant()` gives the wrapped tool the verified grant (use `decision_id` as the idempotency key).

## 0.1.1

- Registry keywords; no code changes.

## 0.1.0 — first public release

- `create_guard` and the `@guard.tool(...)` decorator for plain and `async def` tools (LangGraph, CrewAI, LangChain, plain Python).
- Local `sha256-canonical-json-v1` fingerprint; raw tool arguments never leave the process.
- `decideAction` with `grantFormat: "jwt"` and one Idempotency-Key per call.
- Approval handling: `poll` (default, 30 s) or `raise`, configurable per tool; `ApprovalPending.to_checkpoint()` and `.resume()` for workflows that pause.
- Grant JWT verified (Ed25519) against the published JWK Set, with caching and key-rotation refetch.
- Fails closed with `ActionRefused`, `ApprovalPending`, `GrantInvalid` or `GuardUnavailable`.
- Outcome reporting (`executed` / `downstream_error`) that never changes the tool result.
