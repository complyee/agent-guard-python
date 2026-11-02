# Changelog

## 0.1.0 — first public release

- `create_guard` and the `@guard.tool(...)` decorator for plain and `async def` tools (LangGraph, CrewAI, LangChain, plain Python).
- Local `sha256-canonical-json-v1` fingerprint; raw tool arguments never leave the process.
- `decideAction` with `grantFormat: "jwt"` and one Idempotency-Key per call.
- Approval handling: `poll` (default, 30 s) or `raise`, configurable per tool; `ApprovalPending.to_checkpoint()` and `.resume()` for workflows that pause.
- Grant JWT verified (Ed25519) against the published JWK Set, with caching and key-rotation refetch.
- Fails closed with `ActionRefused`, `ApprovalPending`, `GrantInvalid` or `GuardUnavailable`.
- Outcome reporting (`executed` / `downstream_error`) that never changes the tool result.
