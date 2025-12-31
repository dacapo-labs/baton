"""Baton Rate Limit Tracking - Per provider/auth/plan rate limiting."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a provider/auth combination."""

    requests_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "rpm": self.requests_per_minute,
            "rpd": self.requests_per_day,
            "tpm": self.tokens_per_minute,
            "tpd": self.tokens_per_day,
        }


# Known rate limits per provider/auth/plan combination
# Format: "provider/auth_method" or "provider/auth_method/plan"
DEFAULT_RATE_LIMITS: dict[str, RateLimitConfig] = {
    # Anthropic
    "anthropic/api": RateLimitConfig(
        requests_per_minute=60, requests_per_day=10000, tokens_per_minute=100000
    ),
    "anthropic/oauth/pro": RateLimitConfig(requests_per_minute=50),
    "anthropic/oauth/max": RateLimitConfig(requests_per_minute=100),
    "anthropic/oauth/teams": RateLimitConfig(requests_per_minute=100),
    "anthropic/oauth/enterprise": RateLimitConfig(requests_per_minute=200),
    # OpenAI
    "openai/api": RateLimitConfig(
        requests_per_minute=60, requests_per_day=10000, tokens_per_minute=150000
    ),
    "openai/api/tier1": RateLimitConfig(
        requests_per_minute=60, requests_per_day=10000, tokens_per_minute=60000
    ),
    "openai/api/tier2": RateLimitConfig(
        requests_per_minute=60, requests_per_day=10000, tokens_per_minute=80000
    ),
    "openai/api/tier3": RateLimitConfig(
        requests_per_minute=60, requests_per_day=10000, tokens_per_minute=150000
    ),
    "openai/codex/plus": RateLimitConfig(requests_per_minute=30, requests_per_day=500),
    "openai/codex/pro": RateLimitConfig(requests_per_minute=60, requests_per_day=1000),
    "openai/codex/enterprise": RateLimitConfig(requests_per_minute=100),
    # Google
    "google/api/free": RateLimitConfig(requests_per_minute=15, requests_per_day=1500),
    "google/api/paid": RateLimitConfig(requests_per_minute=1000, tokens_per_minute=4000000),
    "google/oauth/pro": RateLimitConfig(requests_per_minute=60, requests_per_day=1000),
    "google/oauth/ultra": RateLimitConfig(requests_per_minute=100, requests_per_day=2000),
    "google/vertex": RateLimitConfig(requests_per_minute=300),
    # AWS Bedrock (varies by model, these are approximate)
    "aws/bedrock": RateLimitConfig(requests_per_minute=60),
    # Local
    "ollama/local": RateLimitConfig(),  # No limits
}


@dataclass
class UsageWindow:
    """Track usage within a time window."""

    timestamps: list[float] = field(default_factory=list)
    token_count: int = 0


class RateLimitTracker:
    """Track and enforce rate limits per auth key."""

    def __init__(self, custom_limits: dict[str, dict] | None = None):
        self._lock = Lock()
        self._minute_windows: dict[str, UsageWindow] = defaultdict(UsageWindow)
        self._day_counts: dict[str, int] = defaultdict(int)
        self._day_tokens: dict[str, int] = defaultdict(int)
        self._day_start: dict[str, float] = {}

        # Merge default limits with custom limits
        self._limits: dict[str, RateLimitConfig] = DEFAULT_RATE_LIMITS.copy()
        if custom_limits:
            for key, config in custom_limits.items():
                self._limits[key] = RateLimitConfig(**config)

    def _get_limit_config(self, auth_key: str) -> RateLimitConfig:
        """Get rate limit config, falling back to less specific keys."""
        # Try exact match first
        if auth_key in self._limits:
            return self._limits[auth_key]

        # Try without plan (e.g., "anthropic/oauth" from "anthropic/oauth/pro")
        parts = auth_key.split("/")
        if len(parts) >= 3:
            base_key = "/".join(parts[:2])
            if base_key in self._limits:
                return self._limits[base_key]

        # Try just provider
        if len(parts) >= 1:
            provider_key = parts[0]
            if provider_key in self._limits:
                return self._limits[provider_key]

        # Default: no limits
        return RateLimitConfig()

    def _clean_minute_window(self, auth_key: str, now: float) -> None:
        """Remove timestamps older than 1 minute."""
        minute_ago = now - 60
        window = self._minute_windows[auth_key]
        window.timestamps = [t for t in window.timestamps if t > minute_ago]

    def _check_day_reset(self, auth_key: str, now: float) -> None:
        """Reset daily counters if a new day has started."""
        day_start = self._day_start.get(auth_key)
        if day_start is None or now - day_start >= 86400:  # 24 hours
            self._day_counts[auth_key] = 0
            self._day_tokens[auth_key] = 0
            self._day_start[auth_key] = now

    def check_limit(
        self, auth_key: str, estimated_tokens: int = 0
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Check if a request is within rate limits.

        Returns:
            (allowed, reason, status) - Whether request is allowed, reason if not,
            and current status dict.
        """
        config = self._get_limit_config(auth_key)
        now = time.time()

        with self._lock:
            self._clean_minute_window(auth_key, now)
            self._check_day_reset(auth_key, now)

            window = self._minute_windows[auth_key]
            status = {
                "auth_key": auth_key,
                "rpm_used": len(window.timestamps),
                "rpm_limit": config.requests_per_minute,
                "tpm_used": window.token_count,
                "tpm_limit": config.tokens_per_minute,
                "rpd_used": self._day_counts[auth_key],
                "rpd_limit": config.requests_per_day,
                "tpd_used": self._day_tokens[auth_key],
                "tpd_limit": config.tokens_per_day,
            }

            # Check RPM
            if config.requests_per_minute is not None:
                if len(window.timestamps) >= config.requests_per_minute:
                    wait_time = 60 - (now - window.timestamps[0])
                    return (
                        False,
                        f"RPM limit ({config.requests_per_minute}) reached, wait {wait_time:.1f}s",
                        status,
                    )

            # Check TPM
            if config.tokens_per_minute is not None and estimated_tokens > 0:
                if window.token_count + estimated_tokens > config.tokens_per_minute:
                    return (
                        False,
                        f"TPM limit ({config.tokens_per_minute}) would be exceeded",
                        status,
                    )

            # Check RPD
            if config.requests_per_day is not None:
                if self._day_counts[auth_key] >= config.requests_per_day:
                    return (
                        False,
                        f"Daily request limit ({config.requests_per_day}) reached",
                        status,
                    )

            # Check TPD
            if config.tokens_per_day is not None and estimated_tokens > 0:
                if self._day_tokens[auth_key] + estimated_tokens > config.tokens_per_day:
                    return (
                        False,
                        f"Daily token limit ({config.tokens_per_day}) would be exceeded",
                        status,
                    )

            return True, None, status

    def record_request(self, auth_key: str, tokens_used: int = 0) -> None:
        """Record a completed request."""
        now = time.time()

        with self._lock:
            self._check_day_reset(auth_key, now)

            # Update minute window
            window = self._minute_windows[auth_key]
            window.timestamps.append(now)
            window.token_count += tokens_used

            # Clean old tokens from minute window
            minute_ago = now - 60
            if window.timestamps and window.timestamps[0] < minute_ago:
                # Simple approach: reset token count when cleaning old timestamps
                # A more accurate approach would track tokens per timestamp
                old_count = len([t for t in window.timestamps if t <= minute_ago])
                if old_count > 0:
                    # Approximate: assume even distribution
                    total = len(window.timestamps)
                    if total > 0:
                        window.token_count = int(
                            window.token_count * (total - old_count) / total
                        )

            self._clean_minute_window(auth_key, now)

            # Update daily counters
            self._day_counts[auth_key] += 1
            self._day_tokens[auth_key] += tokens_used

    def get_status(self, auth_key: str) -> dict[str, Any]:
        """Get current rate limit status for an auth key."""
        config = self._get_limit_config(auth_key)
        now = time.time()

        with self._lock:
            self._clean_minute_window(auth_key, now)
            self._check_day_reset(auth_key, now)

            window = self._minute_windows[auth_key]

            return {
                "auth_key": auth_key,
                "limits": config.to_dict(),
                "usage": {
                    "rpm_used": len(window.timestamps),
                    "tpm_used": window.token_count,
                    "rpd_used": self._day_counts[auth_key],
                    "tpd_used": self._day_tokens[auth_key],
                },
                "remaining": {
                    "rpm": (
                        config.requests_per_minute - len(window.timestamps)
                        if config.requests_per_minute
                        else None
                    ),
                    "tpm": (
                        config.tokens_per_minute - window.token_count
                        if config.tokens_per_minute
                        else None
                    ),
                    "rpd": (
                        config.requests_per_day - self._day_counts[auth_key]
                        if config.requests_per_day
                        else None
                    ),
                    "tpd": (
                        config.tokens_per_day - self._day_tokens[auth_key]
                        if config.tokens_per_day
                        else None
                    ),
                },
            }

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status for all tracked auth keys."""
        with self._lock:
            keys = set(self._minute_windows.keys()) | set(self._day_counts.keys())

        return {key: self.get_status(key) for key in sorted(keys)}

    def set_limit(self, auth_key: str, config: RateLimitConfig) -> None:
        """Set or update rate limit for an auth key."""
        self._limits[auth_key] = config

    def reset(self, auth_key: str | None = None) -> None:
        """Reset usage counters for an auth key or all keys."""
        with self._lock:
            if auth_key:
                self._minute_windows.pop(auth_key, None)
                self._day_counts.pop(auth_key, None)
                self._day_tokens.pop(auth_key, None)
                self._day_start.pop(auth_key, None)
            else:
                self._minute_windows.clear()
                self._day_counts.clear()
                self._day_tokens.clear()
                self._day_start.clear()


# Global instance for use across the application
_rate_limiter: RateLimitTracker | None = None


def get_rate_limiter() -> RateLimitTracker:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimitTracker()
    return _rate_limiter


def init_rate_limiter(custom_limits: dict[str, dict] | None = None) -> RateLimitTracker:
    """Initialize the global rate limiter with custom limits."""
    global _rate_limiter
    _rate_limiter = RateLimitTracker(custom_limits)
    return _rate_limiter
