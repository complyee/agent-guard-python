"""A fake Agent Control Room behind httpx.MockTransport, signing real Ed25519
grant JWTs in the acr-grant-jwt-v1 shape."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from complyee_agent_guard import create_guard

WORKSPACE = "https://acme.agent-cr.com"
GATEWAY = f"{WORKSPACE}/api/public/v1/gateway"
JWKS_URL = "https://agent-cr.com/.well-known/jwks.json"
APPROVAL_ID = "22222222-2222-4222-8222-222222222222"
T0 = 1_793_000_000.0


def b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def seg(obj: Any) -> str:
    return b64u(json.dumps(obj, separators=(",", ":")).encode())


@dataclass
class FakeAcr:
    clock: list = field(default_factory=lambda: [T0])
    key: Ed25519PrivateKey = field(default_factory=Ed25519PrivateKey.generate)
    kid: str = "k1"
    published: Optional[list] = None  # override the published keys
    effects: list = field(default_factory=lambda: ["grant"])
    approval_statuses: list = field(default_factory=list)
    sent: list = field(default_factory=list)
    jwks_fetches: int = 0
    jwks_down: bool = False
    gateway_failures: int = 0
    report_fails: bool = False
    mint: dict = field(default_factory=dict)  # override claims
    tamper: Optional[Callable[[str], str]] = None
    counter: int = 0

    def jwk(self, key: Ed25519PrivateKey, kid: str) -> dict:
        raw = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return {"kty": "OKP", "crv": "Ed25519", "x": b64u(raw), "kid": kid, "alg": "EdDSA", "use": "sig"}

    def sign(self, params: dict, decision_id: str) -> str:
        now = int(self.clock[0])
        claims = {
            "iss": "https://agent-cr.com",
            "sub": params["agent"],
            "aud": WORKSPACE,
            "jti": decision_id,
            "iat": now,
            "nbf": now,
            "exp": now + 300,
            "acr": {
                "ver": "acr-grant-jwt-v1",
                "workspaceId": "tenant-a",
                "decisionId": decision_id,
                "action": params["action"],
                "payloadHash": params["payloadHash"],
                "resourceRef": params.get("resourceRef"),
                "policyVersion": "v1",
                "ruleId": "r-1",
                "lineage": {
                    "parentDecisionId": params.get("parentDecisionId"),
                    "rootDecisionId": params.get("parentDecisionId") or decision_id,
                    "depth": 1 if params.get("parentDecisionId") else 0,
                },
            },
        }
        for k, v in self.mint.items():
            if k.startswith("acr."):
                claims["acr"][k[4:]] = v
            else:
                claims[k] = v
        signing_input = f"{seg({'alg': 'EdDSA', 'typ': 'JWT', 'kid': self.kid})}.{seg(claims)}"
        token = f"{signing_input}.{b64u(self.key.sign(signing_input.encode()))}"
        return self.tamper(token) if self.tamper else token

    def handler(self, request: httpx.Request) -> httpx.Response:
        if str(request.url) == JWKS_URL:
            self.jwks_fetches += 1
            if self.jwks_down:
                return httpx.Response(503)
            return httpx.Response(200, json={"keys": self.published if self.published is not None else [self.jwk(self.key, self.kid)]})
        assert str(request.url) == GATEWAY
        body = json.loads(request.content)
        self.sent.append({"operation": body["operation"], "parameters": body["parameters"],
                          "key": request.headers.get("idempotency-key"), "raw": request.content.decode()})
        if self.gateway_failures:
            self.gateway_failures -= 1
            return httpx.Response(503, json={"error": {"message": "busy"}})
        op, params = body["operation"], body["parameters"]
        if op == "decideAction":
            effect = self.effects.pop(0) if len(self.effects) > 1 else self.effects[0]
            self.counter += 1
            decision_id = f"00000000-0000-4000-8000-{self.counter:012d}"
            if effect == "pending":
                return httpx.Response(200, json={"decision": {"decisionId": decision_id, "effect": "pending", "approvalId": APPROVAL_ID}})
            if effect == "reject":
                return httpx.Response(200, json={"decision": {"decisionId": decision_id, "effect": "reject", "rationale": "over limit"}})
            d = {"decisionId": decision_id, "effect": "grant", "expiresAt": "2026-11-01T12:05:00Z"}
            if effect != "grant-no-jwt":
                d["grantJwt"] = self.sign(params, decision_id)
            return httpx.Response(200, json={"decision": d})
        if op == "getApprovalStatus":
            status = self.approval_statuses.pop(0) if self.approval_statuses else "pending"
            return httpx.Response(200, json={"approval": {"approvalId": params["approvalId"], "status": status}})
        if op == "reportExecutionOutcome":
            return httpx.Response(500 if self.report_fails else 200, json={})
        return httpx.Response(400, json={"error": {"message": "unknown operation"}})

    def ops(self, name: str) -> list:
        return [s for s in self.sent if s["operation"] == name]


@pytest.fixture
def acr() -> FakeAcr:
    return FakeAcr()


def make_guard(acr: FakeAcr, **options: Any):
    transport = httpx.MockTransport(acr.handler)

    def sleep(s: float) -> None:
        acr.clock[0] += s

    async def async_sleep(s: float) -> None:
        acr.clock[0] += s
        await asyncio.sleep(0)

    keys = iter(f"key-{i}" for i in range(1000))
    defaults = dict(workspace_url=WORKSPACE, api_key="test-key", agent="finance-assistant",
                    now=lambda: acr.clock[0], random_id=lambda: next(keys))
    defaults.update(options)
    return create_guard(
        http_client=httpx.Client(transport=transport),
        async_http_client=httpx.AsyncClient(transport=transport),
        sleep=sleep,
        async_sleep=async_sleep,
        **defaults,
    )
