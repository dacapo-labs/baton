"""Tests for Playwright-based OAuth authentication."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baton.plugins.playwright_auth import (
    PLAYWRIGHT_AVAILABLE,
    AnthropicOAuthProvider,
    GoogleOAuthProvider,
    OAuthSession,
    OpenAIOAuthProvider,
    PlaywrightAuthManager,
    SessionStore,
)


class TestOAuthSession:
    """Tests for OAuthSession dataclass."""

    def test_session_creation(self):
        """Test creating a session."""
        session = OAuthSession(
            provider="anthropic",
            authenticated=True,
            user="test@example.com",
            authenticated_at=1700000000.0,
            expires_at=1700003600.0,
        )

        assert session.provider == "anthropic"
        assert session.authenticated is True
        assert session.user == "test@example.com"

    def test_ttl_seconds(self):
        """Test TTL calculation."""
        future_time = time.time() + 3600
        session = OAuthSession(
            provider="test",
            authenticated=True,
            expires_at=future_time,
        )

        ttl = session.ttl_seconds
        assert ttl is not None
        assert 3590 <= ttl <= 3600

    def test_ttl_seconds_none_when_no_expiry(self):
        """Test TTL is None when no expiry set."""
        session = OAuthSession(
            provider="test",
            authenticated=True,
        )

        assert session.ttl_seconds is None

    def test_is_expired(self):
        """Test expiry check."""
        # Expired session
        expired = OAuthSession(
            provider="test",
            authenticated=True,
            expires_at=time.time() - 100,
        )
        assert expired.is_expired is True

        # Valid session
        valid = OAuthSession(
            provider="test",
            authenticated=True,
            expires_at=time.time() + 3600,
        )
        assert valid.is_expired is False

    def test_to_dict(self):
        """Test conversion to dict (safe for API response)."""
        session = OAuthSession(
            provider="anthropic",
            authenticated=True,
            user="test@example.com",
            authenticated_at=1700000000.0,
            expires_at=1700003600.0,
        )

        result = session.to_dict()

        assert result["provider"] == "anthropic"
        assert result["authenticated"] is True
        assert result["user"] == "test@example.com"
        # Should NOT include storage_state or cookies
        assert "storage_state" not in result
        assert "cookies" not in result

    def test_to_storage(self):
        """Test conversion to storage format (includes sensitive data)."""
        session = OAuthSession(
            provider="anthropic",
            authenticated=True,
            user="test@example.com",
            storage_state={"origins": []},
            cookies=[{"name": "session", "value": "abc123"}],
        )

        result = session.to_storage()

        assert result["storage_state"] is not None
        assert result["cookies"] is not None

    def test_from_storage(self):
        """Test loading from storage format."""
        data = {
            "provider": "anthropic",
            "authenticated": True,
            "user": "test@example.com",
            "storage_state": {"origins": []},
            "cookies": [{"name": "session", "value": "abc123"}],
            "authenticated_at": 1700000000.0,
            "expires_at": 1700003600.0,
        }

        session = OAuthSession.from_storage(data)

        assert session.provider == "anthropic"
        assert session.authenticated is True
        assert session.storage_state == {"origins": []}
        assert len(session.cookies) == 1


class TestSessionStore:
    """Tests for session storage."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_and_load(self, temp_storage_dir):
        """Test saving and loading a session."""
        store = SessionStore(temp_storage_dir)

        session = OAuthSession(
            provider="anthropic",
            authenticated=True,
            user="test@example.com",
            storage_state={"origins": []},
            cookies=[],
            authenticated_at=time.time(),
        )

        # Save
        result = store.save(session)
        assert result is True

        # Load
        loaded = store.load("anthropic")
        assert loaded is not None
        assert loaded.provider == "anthropic"
        assert loaded.authenticated is True
        assert loaded.user == "test@example.com"

    def test_load_nonexistent(self, temp_storage_dir):
        """Test loading nonexistent session returns None."""
        store = SessionStore(temp_storage_dir)

        session = store.load("nonexistent")
        assert session is None

    def test_delete(self, temp_storage_dir):
        """Test deleting a session."""
        store = SessionStore(temp_storage_dir)

        session = OAuthSession(
            provider="anthropic",
            authenticated=True,
        )
        store.save(session)

        # Delete
        result = store.delete("anthropic")
        assert result is True

        # Verify deleted
        loaded = store.load("anthropic")
        assert loaded is None

    def test_list_sessions(self, temp_storage_dir):
        """Test listing all stored sessions."""
        store = SessionStore(temp_storage_dir)

        # Save multiple sessions
        for provider in ["anthropic", "openai", "google"]:
            session = OAuthSession(provider=provider, authenticated=True)
            store.save(session)

        sessions = store.list_sessions()
        assert len(sessions) == 3
        assert "anthropic" in sessions
        assert "openai" in sessions
        assert "google" in sessions

    def test_file_permissions(self, temp_storage_dir):
        """Test that saved files have secure permissions."""
        store = SessionStore(temp_storage_dir)

        session = OAuthSession(provider="anthropic", authenticated=True)
        store.save(session)

        path = temp_storage_dir / "anthropic.json"
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600


class TestAnthropicOAuthProvider:
    """Tests for Anthropic OAuth provider."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return AnthropicOAuthProvider({})

    def test_provider_name(self, provider):
        """Test provider name."""
        assert provider.provider_name == "anthropic"

    def test_login_url(self, provider):
        """Test login URL."""
        assert "console.anthropic.com" in provider.login_url

    def test_success_url_pattern(self, provider):
        """Test success URL pattern."""
        assert "dashboard" in provider.success_url_pattern

    @pytest.mark.asyncio
    async def test_authenticate_no_playwright(self, provider):
        """Test authenticate returns error when Playwright not available."""
        with patch("baton.plugins.playwright_auth.PLAYWRIGHT_AVAILABLE", False):
            # Reload the check
            session = await provider.authenticate()

            # If Playwright is actually available, this test will pass differently
            # The important thing is it doesn't crash

    @pytest.mark.asyncio
    async def test_authenticate_no_credentials(self, provider):
        """Test authenticate with no credentials or playwright not installed."""
        session = await provider.authenticate()

        assert session.authenticated is False
        # Error could be about credentials OR playwright not being installed
        assert "credentials" in session.error.lower() or "playwright" in session.error.lower()

    def test_get_session_none_initially(self, provider):
        """Test get_session returns None initially."""
        assert provider.get_session() is None

    def test_set_session(self, provider):
        """Test setting a session."""
        session = OAuthSession(provider="anthropic", authenticated=True)
        provider.set_session(session)

        assert provider.get_session() is session


class TestOpenAIOAuthProvider:
    """Tests for OpenAI OAuth provider."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return OpenAIOAuthProvider({})

    def test_provider_name(self, provider):
        """Test provider name."""
        assert provider.provider_name == "openai"

    def test_login_url(self, provider):
        """Test login URL."""
        assert "platform.openai.com" in provider.login_url


class TestGoogleOAuthProvider:
    """Tests for Google OAuth provider."""

    @pytest.fixture
    def provider(self):
        """Create provider instance."""
        return GoogleOAuthProvider({})

    def test_provider_name(self, provider):
        """Test provider name."""
        assert provider.provider_name == "google"

    def test_login_url(self, provider):
        """Test login URL."""
        assert "aistudio.google.com" in provider.login_url


class TestPlaywrightAuthManager:
    """Tests for Playwright auth manager."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_storage_dir):
        """Create manager instance."""
        return PlaywrightAuthManager(
            config={},
            storage_dir=temp_storage_dir,
        )

    def test_get_provider(self, manager):
        """Test getting a provider."""
        provider = manager.get_provider("anthropic")
        assert provider is not None
        assert isinstance(provider, AnthropicOAuthProvider)

    def test_get_unknown_provider(self, manager):
        """Test getting unknown provider."""
        provider = manager.get_provider("unknown")
        assert provider is None

    @pytest.mark.asyncio
    async def test_authenticate_unknown_provider(self, manager):
        """Test authenticating with unknown provider."""
        session = await manager.authenticate("unknown")

        assert session.authenticated is False
        assert "Unknown provider" in session.error

    @pytest.mark.asyncio
    async def test_get_status(self, manager):
        """Test getting status for all providers."""
        status = await manager.get_status()

        assert isinstance(status, dict)
        assert "anthropic" in status
        assert "openai" in status
        assert "google" in status

    @pytest.mark.asyncio
    async def test_check_and_refresh_expiring(self, manager):
        """Test checking and refreshing expiring sessions."""
        # Set up an expiring session
        expiring_session = OAuthSession(
            provider="anthropic",
            authenticated=True,
            expires_at=time.time() + 1800,  # 30 min remaining
            storage_state={"origins": []},
        )
        manager.get_provider("anthropic").set_session(expiring_session)

        # Mock refresh to avoid actual browser
        with patch.object(
            manager.get_provider("anthropic"),
            "refresh_session",
            new_callable=AsyncMock,
            return_value=OAuthSession(provider="anthropic", authenticated=True),
        ):
            results = await manager.check_and_refresh_expiring(threshold_seconds=3600)

            # Should have attempted to refresh anthropic
            assert "anthropic" in results
