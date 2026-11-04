"""complyee-agent-guard: run an agent tool only on a verified Agent Control Room grant."""

from .config import ACR_ISSUER, ACR_JWKS_URL, ApprovalSettings, GuardConfig
from .core import VerifiedGrant
from .errors import ActionRefused, ApprovalPending, GrantInvalid, GuardError, GuardUnavailable
from .fingerprint import TOOL_CALL_HASH_ALGORITHM, canonical_tool_call, tool_call_payload_hash
from .guard import Guard, create_guard

__all__ = [
    "ACR_ISSUER",
    "ACR_JWKS_URL",
    "ActionRefused",
    "ApprovalPending",
    "ApprovalSettings",
    "GrantInvalid",
    "Guard",
    "GuardConfig",
    "GuardError",
    "GuardUnavailable",
    "TOOL_CALL_HASH_ALGORITHM",
    "VerifiedGrant",
    "canonical_tool_call",
    "create_guard",
    "tool_call_payload_hash",
]
__version__ = "0.3.0"
