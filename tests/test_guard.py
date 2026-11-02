"""Behaviour of @guard.tool, each case run for a plain def tool and an async def tool."""

import asyncio
import json

import pytest

from complyee_agent_guard import (
    ActionRefused,
    ApprovalPending,
    GrantInvalid,
    GuardUnavailable,
    VerifiedGrant,
    create_guard,
    tool_call_payload_hash,
)

from conftest import APPROVAL_ID, make_guard

MODES = ["sync", "async"]


def define(guard, mode, ran, **options):
    if mode == "sync":

        @guard.tool(name="issue_refund", action="refund.issue", **options)
        def issue_refund(payment_id: str, amount: float, acr_grant: VerifiedGrant = None):
            ran.append(acr_grant)
            return {"refunded": amount}

    else:

        @guard.tool(name="issue_refund", action="refund.issue", **options)
        async def issue_refund(payment_id: str, amount: float, acr_grant: VerifiedGrant = None):
            ran.append(acr_grant)
            return {"refunded": amount}

    return issue_refund


def call(mode, fn, *args, **kwargs):
    return fn(*args, **kwargs) if mode == "sync" else asyncio.run(fn(*args, **kwargs))


def resume(mode, fn, checkpoint):
    return fn.resume(checkpoint) if mode == "sync" else asyncio.run(fn.resume(checkpoint))


ARGS = {"payment_id": "pay_1", "amount": 40}
HASH = tool_call_payload_hash("issue_refund", ARGS)


@pytest.mark.parametrize("mode", MODES)
def test_verified_grant_runs_the_tool_and_reports(acr, mode):
    ran = []
    fn = define(make_guard(acr), mode, ran, context=lambda a: {"amount": a["amount"]})
    assert call(mode, fn, "pay_1", amount=40) == {"refunded": 40}
    assert ran[0].payload_hash == HASH and ran[0].decision_id
    decide = acr.ops("decideAction")[0]
    assert decide["parameters"] == {"agent": "finance-assistant", "action": "refund.issue",
                                    "payloadHash": HASH, "grantFormat": "jwt", "context": {"amount": 40}}
    assert acr.ops("reportExecutionOutcome")[0]["parameters"]["outcome"] == "executed"


@pytest.mark.parametrize("mode", MODES)
def test_raw_arguments_never_leave_the_process(acr, mode):
    fn = define(make_guard(acr), mode, [])
    call(mode, fn, payment_id="pay_SECRET_1", amount=40)
    assert all("pay_SECRET_1" not in s["raw"] for s in acr.sent)


@pytest.mark.parametrize("mode", MODES)
def test_refusal_blocks_the_tool(acr, mode):
    acr.effects = ["reject"]
    ran = []
    fn = define(make_guard(acr), mode, ran)
    with pytest.raises(ActionRefused, match="over limit"):
        call(mode, fn, **ARGS)
    assert ran == [] and acr.ops("reportExecutionOutcome") == []


@pytest.mark.parametrize("mode", MODES)
def test_poll_waits_then_claims_with_the_same_key(acr, mode):
    acr.effects = ["pending", "grant"]
    acr.approval_statuses = ["pending", "granted"]
    ran = []
    fn = define(make_guard(acr), mode, ran)
    assert call(mode, fn, **ARGS) == {"refunded": 40}
    decides = acr.ops("decideAction")
    assert len(decides) == 2 and decides[0]["key"] == decides[1]["key"]
    assert decides[0]["raw"] == decides[1]["raw"]
    assert len(acr.ops("getApprovalStatus")) == 2


@pytest.mark.parametrize("mode", MODES)
def test_poll_timeout_raises_pending_and_resume_claims(acr, mode):
    acr.effects = ["pending"]
    ran = []
    fn = define(make_guard(acr, approvals={"poll_timeout_s": 5, "poll_interval_s": 2}), mode, ran)
    with pytest.raises(ApprovalPending) as held:
        call(mode, fn, **ARGS)
    assert ran == [] and held.value.approval_id == APPROVAL_ID
    checkpoint = json.loads(json.dumps(held.value.to_checkpoint()))  # survives a checkpoint store
    acr.effects = ["grant"]
    assert resume(mode, fn, checkpoint) == {"refunded": 40}
    decides = acr.ops("decideAction")
    assert decides[-1]["key"] == decides[0]["key"] and decides[-1]["raw"] == decides[0]["raw"]


@pytest.mark.parametrize("mode", MODES)
def test_raise_mode_does_not_poll(acr, mode):
    acr.effects = ["pending"]
    fn = define(make_guard(acr, approvals={"mode": "raise"}), mode, [])
    with pytest.raises(ApprovalPending):
        call(mode, fn, **ARGS)
    assert acr.ops("getApprovalStatus") == []


@pytest.mark.parametrize("mode", MODES)
def test_per_tool_override(acr, mode):
    acr.effects = ["pending"]
    fn = define(make_guard(acr), mode, [], approvals={"mode": "raise"})
    with pytest.raises(ApprovalPending):
        call(mode, fn, **ARGS)
    assert acr.ops("getApprovalStatus") == []


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("status", ["refused", "expired", "consumed"])
def test_refusal_during_poll(acr, mode, status):
    acr.effects = ["pending"]
    acr.approval_statuses = [status]
    ran = []
    fn = define(make_guard(acr), mode, ran)
    with pytest.raises(ActionRefused):
        call(mode, fn, **ARGS)
    assert ran == []


def test_resume_refuses_a_foreign_checkpoint(acr):
    fn = define(make_guard(acr), "sync", [])
    with pytest.raises(ValueError):
        fn.resume({"tool": "other", "arguments": {}, "idempotencyKey": "k"})


TAMPER = [
    ("acr.payloadHash", "0" * 64, "wrong payload"),
    ("sub", "someone-else", "wrong agent"),
    ("acr.action", "refund.other", "wrong action"),
    ("aud", "https://other.agent-cr.com", "wrong audience"),
    ("exp", 0, "expired"),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("claim,value,reason", TAMPER)
def test_tampered_grants_never_run(acr, mode, claim, value, reason):
    acr.mint = {claim: value}
    ran = []
    fn = define(make_guard(acr), mode, ran)
    with pytest.raises(GrantInvalid) as err:
        call(mode, fn, **ARGS)
    assert err.value.reason == reason and ran == []


@pytest.mark.parametrize("mode", MODES)
def test_bad_signature_edited_claims_and_missing_token(acr, mode):
    ran = []
    fn = define(make_guard(acr), mode, ran)
    acr.tamper = lambda t: t[:-4] + ("AAAA" if not t.endswith("AAAA") else "BBBB")
    with pytest.raises(GrantInvalid, match="bad signature"):
        call(mode, fn, **ARGS)
    acr.tamper = lambda t: (lambda h, p, s: f"{h}.{p[:-1]}{'A' if p[-1] != 'A' else 'B'}.{s}")(*t.split("."))
    with pytest.raises(GrantInvalid):
        call(mode, fn, **ARGS)
    acr.tamper, acr.effects = None, ["grant-no-jwt"]
    with pytest.raises(GrantInvalid, match="without a token"):
        call(mode, fn, **ARGS)
    assert ran == []


@pytest.mark.parametrize("mode", MODES)
def test_unavailable_keys_fail_closed(acr, mode):
    acr.jwks_down = True
    ran = []
    fn = define(make_guard(acr), mode, ran)
    with pytest.raises(GrantInvalid, match="public keys unavailable"):
        call(mode, fn, **ARGS)
    assert ran == []


@pytest.mark.parametrize("mode", MODES)
def test_engine_retry_then_unavailable(acr, mode):
    acr.gateway_failures = 1
    fn = define(make_guard(acr), mode, [])
    assert call(mode, fn, **ARGS) == {"refunded": 40}
    first, second = acr.ops("decideAction")[:2]
    assert first["key"] == second["key"]
    acr.gateway_failures = 5
    with pytest.raises(GuardUnavailable):
        call(mode, fn, **ARGS)


@pytest.mark.parametrize("mode", MODES)
def test_separate_calls_get_separate_keys(acr, mode):
    fn = define(make_guard(acr), mode, [])
    call(mode, fn, **ARGS)
    call(mode, fn, **ARGS)
    a, b = acr.ops("decideAction")
    assert a["key"] != b["key"]


@pytest.mark.parametrize("mode", MODES)
def test_keys_are_cached_and_refetched_on_rotation(acr, mode):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    fn = define(make_guard(acr), mode, [])
    call(mode, fn, **ARGS)
    call(mode, fn, **ARGS)
    assert acr.jwks_fetches == 1
    acr.clock[0] += 11
    old = acr.jwk(acr.key, acr.kid)
    acr.key, acr.kid = Ed25519PrivateKey.generate(), "k2"
    acr.published = [old, acr.jwk(acr.key, "k2")]
    call(mode, fn, **ARGS)
    assert acr.jwks_fetches == 2


@pytest.mark.parametrize("mode", MODES)
def test_downstream_error_is_reported_and_reraised(acr, mode):
    guard = make_guard(acr)
    if mode == "sync":

        @guard.tool
        def boom(x: int):
            raise RuntimeError("downstream")

    else:

        @guard.tool
        async def boom(x: int):
            raise RuntimeError("downstream")

    with pytest.raises(RuntimeError, match="downstream"):
        call(mode, boom, 1)
    assert acr.ops("decideAction")[0]["parameters"]["action"] == "boom"  # defaults to the tool name
    assert acr.ops("reportExecutionOutcome")[0]["parameters"]["outcome"] == "downstream_error"


@pytest.mark.parametrize("mode", MODES)
def test_failed_report_never_changes_the_result(acr, mode):
    acr.report_fails = True
    errors = []
    fn = define(make_guard(acr, on_report_error=errors.append), mode, [])
    assert call(mode, fn, **ARGS) == {"refunded": 40}
    assert len(errors) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"workspace_url": "http://acme.agent-cr.com"},
        {"workspace_url": "https://acme.agent-cr.com/path"},
        {"api_key": ""},
        {"agent": "a|b"},
        {"approvals": {"mode": "never"}},
    ],
)
def test_unsafe_config_is_refused(options):
    base = dict(workspace_url="https://acme.agent-cr.com", api_key="k", agent="a")
    base.update(options)
    with pytest.raises(ValueError):
        create_guard(**base)


def test_action_with_pipe_is_refused(acr):
    with pytest.raises(ValueError):
        make_guard(acr).tool(action="a|b")(lambda: None)


def test_unfingerprintable_arguments_are_refused_before_asking(acr):
    fn = define(make_guard(acr), "sync", [])
    with pytest.raises(ActionRefused, match="fingerprinted"):
        fn("p", float("nan"))
    assert acr.sent == []
