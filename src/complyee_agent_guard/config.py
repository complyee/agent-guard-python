"""Guard configuration and its validation."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional

ApprovalMode = Literal["poll", "raise"]

ACR_ISSUER = "https://agent-cr.com"
ACR_JWKS_URL = f"{ACR_ISSUER}/.well-known/jwks.json"


@dataclass(frozen=True)
class ApprovalSettings:
    mode: ApprovalMode = "poll"
    poll_timeout_s: float = 30.0
    poll_interval_s: float = 2.0

    def merged(self, override: Optional[Mapping[str, Any]]) -> "ApprovalSettings":
        if not override:
            return self
        values = {**self.__dict__, **dict(override)}
        settings = ApprovalSettings(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.mode not in ("poll", "raise"):
            raise ValueError('approvals.mode must be "poll" or "raise"')
        if self.poll_timeout_s < 0 or self.poll_interval_s <= 0:
            raise ValueError("approval timings must be positive")


@dataclass
class GuardConfig:
    workspace_url: str
    api_key: str
    agent: str
    approvals: ApprovalSettings = field(default_factory=ApprovalSettings)
    jwks_url: str = ACR_JWKS_URL
    issuer: str = ACR_ISSUER
    jwks_cache_ttl_s: float = 300.0
    jwks_min_refresh_s: float = 10.0
    clock_skew_s: int = 30
    retries: int = 1
    timeout_s: float = 10.0
    on_report_error: Optional[Callable[[BaseException], None]] = None
    # Injection points for tests and unusual runtimes.
    now: Callable[[], float] = time.time
    random_id: Callable[[], str] = lambda: str(uuid.uuid4())

    def __post_init__(self) -> None:
        self.workspace_url = self.workspace_url.rstrip("/")
        if not re.fullmatch(r"https://[^/]+", self.workspace_url):
            raise ValueError("workspace_url must be an https origin, e.g. https://acme.agent-cr.com")
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.agent or "|" in self.agent:
            raise ValueError("agent is required and must not contain |")
        if self.retries < 0:
            raise ValueError("retries must not be negative")
        self.approvals.validate()

    @property
    def gateway_url(self) -> str:
        return f"{self.workspace_url}/api/public/v1/gateway"
