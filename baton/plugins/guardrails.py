"""Baton Guardrails Plugin - Rate limits, approval gates, and audit."""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting."""

    tokens: float
    last_update: float
    max_tokens: float
    refill_rate: float

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        now = time.time()
        elapsed = now - self.last_update

        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


@dataclass
class ApprovalRequest:
    """Pending approval request."""

    request_id: str
    action: str
    details: str
    created_at: datetime
    approved: bool | None = None
    approved_at: datetime | None = None
    approved_by: str | None = None


class BatonGuardrails:
    """Guardrails for AI safety and control."""

    def __init__(self, config: dict[str, Any], logger: Any = None, twilio: Any = None):
        self.config = config
        self.logger = logger
        self.twilio = twilio

        guardrails_config = config.get("guardrails", {})
        self.enabled = guardrails_config.get("enabled", True)
        self.default_rpm = guardrails_config.get("rate_limit_rpm", 100)
        self.require_approval = guardrails_config.get("require_approval", [])
        self.blocked_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in guardrails_config.get("blocked_patterns", [])
        ]

        self._rate_limits: dict[str, RateLimitBucket] = {}
        self._pending_approvals: dict[str, ApprovalRequest] = {}

    def _get_rate_limit(self, zone: str | None) -> RateLimitBucket:
        """Get or create rate limit bucket for zone."""
        key = zone or "default"
        if key not in self._rate_limits:
            zone_config = self.config.get("zones", {}).get(zone, {}) if zone else {}
            rpm = zone_config.get("rate_limit_rpm", self.default_rpm)

            self._rate_limits[key] = RateLimitBucket(
                tokens=rpm,
                last_update=time.time(),
                max_tokens=rpm,
                refill_rate=rpm / 60.0,
            )
        return self._rate_limits[key]

    def check_rate_limit(self, zone: str | None = None) -> tuple[bool, str | None]:
        """Check if request is within rate limits."""
        if not self.enabled:
            return True, None

        bucket = self._get_rate_limit(zone)
        if bucket.consume():
            return True, None

        return False, f"Rate limit exceeded for zone '{zone or 'default'}'"

    def check_content(self, content: str) -> tuple[bool, str | None]:
        """Check content against blocked patterns."""
        if not self.enabled or not self.blocked_patterns:
            return True, None

        for pattern in self.blocked_patterns:
            if pattern.search(content):
                return False, f"Content blocked: matches pattern '{pattern.pattern}'"

        return True, None

    def check_action(self, action: str) -> tuple[bool, str | None]:
        """Check if action requires approval."""
        if not self.enabled:
            return True, None

        for approval_action in self.require_approval:
            if action == approval_action or action.startswith(f"{approval_action}:"):
                return False, approval_action

        return True, None

    async def request_approval(
        self,
        request_id: str,
        action: str,
        details: str,
        timeout_seconds: int = 300,
    ) -> bool:
        """Request approval for an action via SMS."""
        approval = ApprovalRequest(
            request_id=request_id,
            action=action,
            details=details,
            created_at=datetime.now(),
        )
        self._pending_approvals[request_id] = approval

        if self.twilio:
            message = f"Baton approval request:\n{action}\n{details}\n\nReply YES to approve, NO to deny."
            await self.twilio.send_sms(message)

        start = time.time()
        while time.time() - start < timeout_seconds:
            if approval.approved is not None:
                return approval.approved
            await asyncio.sleep(1)

        approval.approved = False
        return False

    def process_approval_response(self, request_id: str, approved: bool, approved_by: str) -> bool:
        """Process an approval response."""
        if request_id not in self._pending_approvals:
            return False

        approval = self._pending_approvals[request_id]
        approval.approved = approved
        approval.approved_at = datetime.now()
        approval.approved_by = approved_by
        return True

    async def validate_request(
        self,
        messages: list[dict[str, Any]],
        zone: str | None = None,
        action: str | None = None,
    ) -> tuple[bool, str | None]:
        """Validate a request against all guardrails."""
        if not self.enabled:
            return True, None

        allowed, error = self.check_rate_limit(zone)
        if not allowed:
            return False, error

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                allowed, error = self.check_content(content)
                if not allowed:
                    return False, error

        if action:
            allowed, action_type = self.check_action(action)
            if not allowed:
                return False, f"Action '{action_type}' requires approval"

        return True, None

    def get_stats(self) -> dict[str, Any]:
        """Get guardrails statistics."""
        return {
            "enabled": self.enabled,
            "blocked_patterns": len(self.blocked_patterns),
            "require_approval": self.require_approval,
            "rate_limits": {
                zone: {
                    "tokens": bucket.tokens,
                    "max": bucket.max_tokens,
                }
                for zone, bucket in self._rate_limits.items()
            },
            "pending_approvals": len(self._pending_approvals),
        }
