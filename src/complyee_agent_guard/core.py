"""The guard's whole decision flow, written once with no I/O.

The flow is a generator that yields small operation objects (ask ACR, read an
approval, sleep, fetch the JWK Set, report an outcome) and receives their
results. The sync and async drivers only perform those operations, so the two
modes cannot drift apart.

  ask (decideAction, grantFormat "jwt", payload hashed locally)
  -> a person must decide? poll getApprovalStatus, then claim with the identical request
  -> verify the grant JWT against the published JWK Set (EdDSA)
  -> hand back a VerifiedGrant; the driver runs the tool and reports the outcome

Fail closed throughout. There is no fallback from a rejected token to anything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generator, Mapping, Optional

from .config import ApprovalSettings, GuardConfig
from .errors import ActionRefused, ApprovalPending, GrantInvalid, GuardUnavailable
from .fingerprint import tool_call_payload_hash
from .jwt_verify import ExpectedGrant, TokenRejected, parse_token, verify_grant_token


# ---- operations the drivers perform ---------------------------------------------------------


@dataclass(frozen=True)
class Call:
    """One gateway call. retries applies to network failures and 5xx answers."""

    operation: str
    parameters: Mapping[str, Any]
    idempotency_key: Optional[str] = None
    retries: int = 0


@dataclass(frozen=True)
class Sleep:
    seconds: float


@dataclass(frozen=True)
class FetchJwks:
    url: str


class AcrRequestError(Exception):
    def __init__(self, message: str, status: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.code = code


class JwksUnavailable(Exception):
    pass


Flow = Generator[Any, Any, Any]


# ---- gateway request / response shape -------------------------------------------------------


def gateway_request(config: GuardConfig, call: Call) -> tuple[str, dict, dict]:
    headers = {"content-type": "application/json", "x-api-key": config.api_key}
    if call.idempotency_key:
        headers["idempotency-key"] = call.idempotency_key
    return config.gateway_url, headers, {"operation": call.operation, "parameters": dict(call.parameters)}


def should_retry(status: Optional[int], attempt: int, call: Call) -> bool:
    """status None means the request never got an answer."""
    return attempt < call.retries and (status is None or status >= 500)


def gateway_result(status: int, body: Any) -> dict:
    body = body if isinstance(body, dict) else {}
    if status < 200 or status >= 300:
        err = body.get("error") if isinstance(body.get("error"), dict) else {}
        msg = f"Agent Control Room answered {status}"
        if err.get("message"):
            msg += f": {err['message']}"
        raise AcrRequestError(msg, status, err.get("code"))
    return body


# ---- JWK Set cache (state only; fetching is an operation) -----------------------------------


@dataclass
class JwksCache:
    url: str
    ttl_s: float
    min_refresh_s: float
    keys: Optional[list] = None
    fetched_at: float = 0.0

    def _find(self, kid: str) -> Optional[dict]:
        for k in self.keys or []:
            if isinstance(k, dict) and k.get("kid") == kid:
                return k
        return None

    def _load(self, now: float) -> Flow:
        try:
            body = yield FetchJwks(self.url)
        except Exception as error:
            raise JwksUnavailable(str(error)) from None
        if not isinstance(body, dict) or not isinstance(body.get("keys"), list):
            raise JwksUnavailable("JWK Set has no keys array")
        self.keys = body["keys"]
        self.fetched_at = now

    def get_key(self, kid: str, now: float) -> Flow:
        """The key with this kid; refetches once (rate-limited) when it is unknown, e.g. after rotation."""
        if self.keys is None or now - self.fetched_at >= self.ttl_s:
            yield from self._load(now)
        key = self._find(kid)
        if key is None and now - self.fetched_at >= self.min_refresh_s:
            yield from self._load(now)
            key = self._find(kid)
        return key


# ---- the flow -------------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedGrant:
    """Handed to the tool (parameter acr_grant) once the grant is verified. Pass
    decision_id downstream as an idempotency key so a retry never does the work twice."""

    decision_id: str
    payload_hash: str
    expires_at: str
    claims: dict = field(repr=False)


@dataclass(frozen=True)
class ToolPlan:
    tool: str
    action: str
    arguments: Mapping[str, Any]
    approvals: ApprovalSettings
    context: Optional[Mapping[str, Any]] = None
    resource_ref: Optional[str] = None
    action_description: Optional[str] = None
    parent_decision_id: Optional[str] = None


def _held(plan: ToolPlan, decision: dict, key: str) -> ApprovalPending:
    approval_id = decision.get("approvalId")
    return ApprovalPending(
        f"'{plan.action}' is waiting for a person to approve it (approval {approval_id or 'unknown'}). "
        "The tool did not run. Resume with the same checkpoint within fifteen minutes of approval.",
        tool=plan.tool,
        arguments=plan.arguments,
        idempotency_key=key,
        approval_id=approval_id,
        decision_id=decision.get("decisionId"),
    )


_APPROVAL_WHY = {
    "refused": "a person refused it",
    "expired": "the approval request expired before anyone answered",
    "consumed": "the approval was already used",
}


def authorize(config: GuardConfig, keys: JwksCache, plan: ToolPlan, idempotency_key: str) -> Flow:
    """Yields operations; returns a VerifiedGrant or raises a GuardError."""
    action = plan.action
    try:
        payload_hash = tool_call_payload_hash(plan.tool, plan.arguments)
    except (ValueError, TypeError) as error:
        raise ActionRefused(f"'{action}' was not run: its arguments cannot be fingerprinted ({error}).") from None

    request: dict = {"agent": config.agent, "action": action, "payloadHash": payload_hash, "grantFormat": "jwt"}
    if plan.resource_ref:
        request["resourceRef"] = plan.resource_ref
    if plan.context is not None:
        request["context"] = dict(plan.context)
    if plan.action_description:
        request["actionDescription"] = plan.action_description
    if plan.parent_decision_id:
        request["parentDecisionId"] = plan.parent_decision_id

    def decide() -> Flow:
        # Retrying is safe only because the Idempotency-Key makes a repeat return the same answer.
        body = yield Call("decideAction", request, idempotency_key, config.retries)
        return body.get("decision") if isinstance(body.get("decision"), dict) else {}

    try:
        decision = yield from decide()
        if decision.get("effect") == "pending":
            approval_id = decision.get("approvalId")
            if plan.approvals.mode == "raise" or not approval_id:
                raise _held(plan, decision, idempotency_key)
            deadline = config.now() + plan.approvals.poll_timeout_s
            claimed = False
            while config.now() < deadline:
                yield Sleep(min(plan.approvals.poll_interval_s, max(0.0, deadline - config.now())))
                body = yield Call("getApprovalStatus", {"approvalId": approval_id}, None, config.retries)
                approval = body.get("approval") if isinstance(body.get("approval"), dict) else {}
                status = approval.get("status")
                if status == "pending":
                    continue
                if status == "granted":
                    # The identical request claims the approval and returns the signed grant.
                    decision = yield from decide()
                    claimed = True
                    break
                why = _APPROVAL_WHY.get(status, f"the approval reads {status}")
                raise ActionRefused(f"'{action}' was not run: {why}.", decision_id=decision.get("decisionId"), reason=approval.get("reason"))
            if not claimed or decision.get("effect") == "pending":
                raise _held(plan, decision, idempotency_key)
    except AcrRequestError as error:
        raise GuardUnavailable(f"'{action}' was not run: {error}.") from None

    decision_id = decision.get("decisionId")
    if decision.get("effect") != "grant" or not decision_id:
        raise ActionRefused(
            f"'{action}' was refused by policy: {decision.get('rationale') or 'no rationale given'}.",
            decision_id=decision_id,
            reason=decision.get("error"),
        )
    token = decision.get("grantJwt")
    if not token:
        raise GrantInvalid(f"'{action}' was not run: the grant came without a token to verify.", decision_id=decision_id, reason="missing token")

    try:
        _, _, kid = parse_token(token)
        try:
            jwk = yield from keys.get_key(kid, config.now())
        except JwksUnavailable:
            raise TokenRejected("public keys unavailable") from None
        claims = verify_grant_token(
            token,
            jwk,
            ExpectedGrant(
                issuer=config.issuer,
                audience=config.workspace_url,
                agent=config.agent,
                action=action,
                payload_hash=payload_hash,
                decision_id=decision_id,
                parent_decision_id=plan.parent_decision_id,
                now=config.now(),
                clock_skew_s=config.clock_skew_s,
            ),
        )
    except TokenRejected as rejected:
        raise GrantInvalid(f"'{action}' was not run: grant token rejected ({rejected.reason}).", decision_id=decision_id, reason=rejected.reason) from None

    return VerifiedGrant(decision_id=decision_id, payload_hash=payload_hash, expires_at=decision.get("expiresAt") or "", claims=claims)


def report_call(grant: VerifiedGrant, outcome: str, now: float) -> Call:
    from datetime import datetime, timezone

    occurred = datetime.fromtimestamp(now, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return Call(
        "reportExecutionOutcome",
        {"decisionId": grant.decision_id, "payloadHash": grant.payload_hash, "outcome": outcome, "occurredAt": occurred},
    )
