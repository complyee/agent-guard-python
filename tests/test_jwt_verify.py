"""Verification against a grant minted by ACR's real signing code (fixtures/grant.json)."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from complyee_agent_guard.jwt_verify import ExpectedGrant, TokenRejected, parse_token, verify_grant_token

FIX = json.loads((Path(__file__).parent / "fixtures/grant.json").read_text("utf-8"))
ISSUED = datetime.fromisoformat(FIX["issuedAt"].replace("Z", "+00:00")).timestamp()


def expected(**over):
    values = dict(issuer="https://agent-cr.com", audience=FIX["workspaceUrl"], agent=FIX["agent"],
                  action=FIX["action"], payload_hash=FIX["payloadHash"], decision_id=FIX["decisionId"],
                  parent_decision_id=None, now=ISSUED + 10, clock_skew_s=30)
    values.update(over)
    return ExpectedGrant(**values)


def key_for(token):
    _, _, kid = parse_token(token)
    return next(k for k in FIX["jwks"]["keys"] if k["kid"] == kid)


def test_accepts_a_grant_minted_by_acr():
    claims = verify_grant_token(FIX["grantJwt"], key_for(FIX["grantJwt"]), expected())
    assert claims["acr"]["payloadHash"] == FIX["payloadHash"]


@pytest.mark.parametrize(
    "over,reason",
    [
        ({"payload_hash": "0" * 64}, "wrong payload"),
        ({"agent": "someone-else"}, "wrong agent"),
        ({"action": "refund.other"}, "wrong action"),
        ({"audience": "https://other.agent-cr.com"}, "wrong audience"),
        ({"issuer": "https://evil.example"}, "wrong issuer"),
        ({"decision_id": "x"}, "wrong decision"),
        ({"parent_decision_id": "p"}, "wrong lineage"),
        ({"now": ISSUED + 300 + 31}, "expired"),
        ({"now": ISSUED - 31}, "not yet valid"),
    ],
)
def test_rejects_mismatches(over, reason):
    with pytest.raises(TokenRejected) as err:
        verify_grant_token(FIX["grantJwt"], key_for(FIX["grantJwt"]), expected(**over))
    assert err.value.reason == reason


def test_rejects_an_edited_claims_segment():
    h, p, s = FIX["grantJwt"].split(".")
    forged = f"{h}.{p[:-2]}{'A' if p[-2] != 'A' else 'B'}{p[-1]}.{s}"
    with pytest.raises(TokenRejected):
        verify_grant_token(forged, key_for(FIX["grantJwt"]), expected())


def test_rejects_unknown_key_and_wrong_algorithm():
    with pytest.raises(TokenRejected, match="unknown key id"):
        verify_grant_token(FIX["grantJwt"], None, expected())
    with pytest.raises(TokenRejected, match="algorithm"):
        parse_token("eyJhbGciOiJub25lIn0.e30.")
    with pytest.raises(TokenRejected, match="malformed"):
        parse_token("abc")
