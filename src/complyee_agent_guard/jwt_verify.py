"""Check of an ACR grant JWT (EdDSA / Ed25519, RFC 8037). Every comparison uses
what THIS process expects — its own agent, action and payload hash — never
values read back from the answer. Pure: the key lookup is supplied by the caller."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

GRANT_JWT_VERSION = "acr-grant-jwt-v1"
_B64URL = re.compile(r"^[A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class ExpectedGrant:
    issuer: str
    audience: str
    agent: str
    action: str
    payload_hash: str
    decision_id: str
    parent_decision_id: Optional[str]
    now: float
    clock_skew_s: int


class TokenRejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def b64url_decode(value: str) -> bytes:
    if not _B64URL.match(value):
        raise ValueError("not base64url")
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _decode_json(segment: str) -> Any:
    try:
        return json.loads(b64url_decode(segment))
    except Exception:
        return None


def parse_token(token: str) -> tuple[dict, dict, str]:
    """Returns (header, claims, kid) or raises TokenRejected."""
    parts = token.split(".") if isinstance(token, str) else []
    if len(parts) != 3:
        raise TokenRejected("malformed token")
    header, claims = _decode_json(parts[0]), _decode_json(parts[1])
    if not isinstance(header, dict) or not isinstance(claims, dict):
        raise TokenRejected("malformed token")
    if header.get("alg") != "EdDSA":
        raise TokenRejected("algorithm is not EdDSA")
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise TokenRejected("no key id")
    return header, claims, kid


def verify_grant_token(token: str, jwk: Optional[dict], expected: ExpectedGrant) -> dict:
    """Returns the claims of a valid token or raises TokenRejected."""
    _, claims, _ = parse_token(token)
    if not jwk or not jwk.get("x") or jwk.get("kty") not in (None, "OKP") or jwk.get("crv") not in (None, "Ed25519"):
        raise TokenRejected("unknown key id")
    h, p, s = token.split(".")
    try:
        key = Ed25519PublicKey.from_public_bytes(b64url_decode(jwk["x"]))
        key.verify(b64url_decode(s), f"{h}.{p}".encode("ascii"))
    except (InvalidSignature, ValueError, TypeError):
        raise TokenRejected("bad signature") from None

    now = int(expected.now)
    acr = claims.get("acr")
    exp, nbf = claims.get("exp"), claims.get("nbf")
    if claims.get("iss") != expected.issuer:
        raise TokenRejected("wrong issuer")
    if claims.get("aud") != expected.audience:
        raise TokenRejected("wrong audience")
    if claims.get("sub") != expected.agent:
        raise TokenRejected("wrong agent")
    if not isinstance(exp, (int, float)) or isinstance(exp, bool) or now >= exp + expected.clock_skew_s:
        raise TokenRejected("expired")
    if not isinstance(nbf, (int, float)) or isinstance(nbf, bool) or now + expected.clock_skew_s < nbf:
        raise TokenRejected("not yet valid")
    if not isinstance(acr, dict) or acr.get("ver") != GRANT_JWT_VERSION:
        raise TokenRejected("unknown grant version")
    if acr.get("action") != expected.action:
        raise TokenRejected("wrong action")
    if acr.get("payloadHash") != expected.payload_hash:
        raise TokenRejected("wrong payload")
    if claims.get("jti") != expected.decision_id or acr.get("decisionId") != expected.decision_id:
        raise TokenRejected("wrong decision")
    lineage = acr.get("lineage") if isinstance(acr.get("lineage"), dict) else {}
    if lineage.get("parentDecisionId") != expected.parent_decision_id:
        raise TokenRejected("wrong lineage")
    return claims
