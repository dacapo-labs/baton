"""Tests for rate limit tracking."""

import time

import pytest

from baton.plugins.rate_limits import (
    RateLimitConfig,
    RateLimitTracker,
    DEFAULT_RATE_LIMITS,
    init_rate_limiter,
    get_rate_limiter,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_config_defaults(self):
        """Test default values."""
        config = RateLimitConfig()
        assert config.requests_per_minute is None
        assert config.requests_per_day is None
        assert config.tokens_per_minute is None
        assert config.tokens_per_day is None

    def test_config_with_values(self):
        """Test config with all values set."""
        config = RateLimitConfig(
            requests_per_minute=60,
            requests_per_day=10000,
            tokens_per_minute=100000,
            tokens_per_day=1000000,
        )
        assert config.requests_per_minute == 60
        assert config.requests_per_day == 10000
        assert config.tokens_per_minute == 100000
        assert config.tokens_per_day == 1000000

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = RateLimitConfig(requests_per_minute=60, requests_per_day=10000)
        result = config.to_dict()
        assert result == {
            "rpm": 60,
            "rpd": 10000,
            "tpm": None,
            "tpd": None,
        }


class TestRateLimitTracker:
    """Tests for RateLimitTracker."""

    def test_init_default_limits(self):
        """Test tracker initializes with default limits."""
        tracker = RateLimitTracker()
        # Should have default limits loaded
        assert "anthropic/api" in DEFAULT_RATE_LIMITS

    def test_init_custom_limits(self):
        """Test tracker with custom limits."""
        custom = {
            "custom/provider": {"requests_per_minute": 100, "requests_per_day": 5000}
        }
        tracker = RateLimitTracker(custom_limits=custom)

        # Check custom limit is available
        allowed, reason, status = tracker.check_limit("custom/provider")
        assert allowed is True
        assert status["rpm_limit"] == 100

    def test_check_limit_no_limits(self):
        """Test checking limit for provider with no limits."""
        tracker = RateLimitTracker()
        allowed, reason, status = tracker.check_limit("unknown/provider")

        assert allowed is True
        assert reason is None

    def test_check_limit_within_rpm(self):
        """Test requests within RPM limit."""
        tracker = RateLimitTracker()
        auth_key = "anthropic/api"  # Has RPM limit of 60

        # First request should be allowed
        allowed, reason, status = tracker.check_limit(auth_key)
        assert allowed is True
        assert status["rpm_used"] == 0

    def test_check_limit_exceeds_rpm(self):
        """Test requests exceeding RPM limit."""
        custom = {"test/provider": {"requests_per_minute": 2}}
        tracker = RateLimitTracker(custom_limits=custom)

        # Record 2 requests
        tracker.record_request("test/provider")
        tracker.record_request("test/provider")

        # Third request should be blocked
        allowed, reason, status = tracker.check_limit("test/provider")
        assert allowed is False
        assert "RPM limit" in reason
        assert status["rpm_used"] == 2

    def test_check_limit_exceeds_rpd(self):
        """Test requests exceeding daily limit."""
        custom = {"test/provider": {"requests_per_day": 3}}
        tracker = RateLimitTracker(custom_limits=custom)

        # Record 3 requests
        for _ in range(3):
            tracker.record_request("test/provider")

        # Fourth request should be blocked
        allowed, reason, status = tracker.check_limit("test/provider")
        assert allowed is False
        assert "Daily request limit" in reason

    def test_check_limit_tpm(self):
        """Test token per minute limit."""
        custom = {"test/provider": {"tokens_per_minute": 1000}}
        tracker = RateLimitTracker(custom_limits=custom)

        # Record request with 900 tokens
        tracker.record_request("test/provider", tokens_used=900)

        # Request estimating 200 more tokens should be blocked
        allowed, reason, status = tracker.check_limit("test/provider", estimated_tokens=200)
        assert allowed is False
        assert "TPM limit" in reason

    def test_record_request(self):
        """Test recording a request."""
        tracker = RateLimitTracker()
        auth_key = "test/key"

        tracker.record_request(auth_key, tokens_used=100)

        status = tracker.get_status(auth_key)
        assert status["usage"]["rpm_used"] == 1
        assert status["usage"]["rpd_used"] == 1
        assert status["usage"]["tpd_used"] == 100

    def test_get_status(self):
        """Test getting status for an auth key."""
        custom = {"test/provider": {"requests_per_minute": 60, "requests_per_day": 1000}}
        tracker = RateLimitTracker(custom_limits=custom)

        tracker.record_request("test/provider", tokens_used=50)
        tracker.record_request("test/provider", tokens_used=30)

        status = tracker.get_status("test/provider")

        assert status["auth_key"] == "test/provider"
        assert status["limits"]["rpm"] == 60
        assert status["limits"]["rpd"] == 1000
        assert status["usage"]["rpm_used"] == 2
        assert status["usage"]["rpd_used"] == 2
        assert status["remaining"]["rpm"] == 58
        assert status["remaining"]["rpd"] == 998

    def test_get_all_status(self):
        """Test getting status for all tracked keys."""
        tracker = RateLimitTracker()

        tracker.record_request("provider1/api")
        tracker.record_request("provider2/oauth")

        all_status = tracker.get_all_status()

        assert "provider1/api" in all_status
        assert "provider2/oauth" in all_status

    def test_set_limit(self):
        """Test setting a new limit."""
        tracker = RateLimitTracker()

        new_config = RateLimitConfig(requests_per_minute=100)
        tracker.set_limit("new/provider", new_config)

        allowed, reason, status = tracker.check_limit("new/provider")
        assert status["rpm_limit"] == 100

    def test_reset_single_key(self):
        """Test resetting a single auth key."""
        tracker = RateLimitTracker()

        tracker.record_request("test/key1")
        tracker.record_request("test/key2")

        tracker.reset("test/key1")

        status1 = tracker.get_status("test/key1")
        status2 = tracker.get_status("test/key2")

        assert status1["usage"]["rpm_used"] == 0
        assert status2["usage"]["rpm_used"] == 1

    def test_reset_all_keys(self):
        """Test resetting all auth keys."""
        tracker = RateLimitTracker()

        tracker.record_request("test/key1")
        tracker.record_request("test/key2")

        tracker.reset()

        all_status = tracker.get_all_status()
        assert len(all_status) == 0

    def test_fallback_to_base_key(self):
        """Test fallback from specific to base key for limits."""
        tracker = RateLimitTracker()

        # anthropic/oauth/pro should fall back to anthropic/oauth
        allowed, reason, status = tracker.check_limit("anthropic/oauth/pro")
        assert allowed is True
        # Should find the limit from DEFAULT_RATE_LIMITS

    def test_minute_window_cleanup(self):
        """Test that old timestamps are cleaned up."""
        custom = {"test/provider": {"requests_per_minute": 100}}
        tracker = RateLimitTracker(custom_limits=custom)

        # Record a request
        tracker.record_request("test/provider")

        # Manually set old timestamp
        with tracker._lock:
            tracker._minute_windows["test/provider"].timestamps = [time.time() - 120]

        # Check limit should clean up old timestamps
        allowed, reason, status = tracker.check_limit("test/provider")
        assert status["rpm_used"] == 0


class TestGlobalRateLimiter:
    """Tests for global rate limiter functions."""

    def test_init_rate_limiter(self):
        """Test initializing global rate limiter."""
        limiter = init_rate_limiter({"custom/key": {"requests_per_minute": 50}})

        assert limiter is not None
        assert get_rate_limiter() is limiter

    def test_get_rate_limiter_creates_default(self):
        """Test get_rate_limiter creates default if not initialized."""
        # Reset global
        import baton.plugins.rate_limits as rl_module
        rl_module._rate_limiter = None

        limiter = get_rate_limiter()
        assert limiter is not None
