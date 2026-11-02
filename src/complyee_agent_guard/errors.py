"""Every way a guarded tool can end without running. The tool body never runs
when one of these is raised."""

from __future__ import annotations

from typing import Any, Mapping, Optional


class GuardError(Exception):
    """Base class: the guarded tool did not run."""

    status = "unavailable"

    def __init__(self, message: str, *, decision_id: Optional[str] = None, reason: Optional[str] = None):
        super().__init__(message)
        self.decision_id = decision_id
        self.reason = reason


class ActionRefused(GuardError):
    """Policy refused the step, a person refused it, or the arguments cannot be fingerprinted."""

    status = "refused"


class GrantInvalid(GuardError):
    """A grant arrived but its token is missing or failed verification. Never retried with another format."""

    status = "unverified"


class GuardUnavailable(GuardError):
    """Agent Control Room could not be reached or answered with an error."""

    status = "unavailable"


class ApprovalPending(GuardError):
    """A person must approve the step first. Save to_checkpoint() and call
    the tool's .resume(checkpoint) later; the identical request then claims the grant."""

    status = "held"

    def __init__(
        self,
        message: str,
        *,
        tool: str,
        arguments: Mapping[str, Any],
        idempotency_key: str,
        approval_id: Optional[str],
        decision_id: Optional[str],
    ):
        super().__init__(message, decision_id=decision_id)
        self.tool = tool
        self.arguments = dict(arguments)
        self.idempotency_key = idempotency_key
        self.approval_id = approval_id

    def to_checkpoint(self) -> dict:
        """JSON-serialisable state for a workflow checkpoint. Arguments stay in your own store."""
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "idempotencyKey": self.idempotency_key,
            "approvalId": self.approval_id,
            "decisionId": self.decision_id,
        }
