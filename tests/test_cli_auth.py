"""Tests for CLI-based OAuth authentication."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baton.plugins.cli_auth import (
    AuthStatus,
    CLIAuthManager,
    ClaudeCliAuth,
    CodexCliAuth,
    CompletionResult,
    GeminiCliAuth,
)


class TestAuthStatus:
    """Tests for AuthStatus dataclass."""

    def test_authenticated_status(self):
        """Test authenticated status."""
        status = AuthStatus(
            authenticated=True,
            user="test@example.com",
            plan="pro",
            ttl_seconds=3600,
        )

        assert status.authenticated is True
        assert status.user == "test@example.com"
        assert status.plan == "pro"
        assert status.ttl_seconds == 3600

    def test_unauthenticated_status(self):
        """Test unauthenticated status with error."""
        status = AuthStatus(
            authenticated=False,
            error="CLI not installed",
        )

        assert status.authenticated is False
        assert status.error == "CLI not installed"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        status = AuthStatus(
            authenticated=True,
            user="test@example.com",
            plan="pro",
            ttl_seconds=3600,
            expires_at=1700000000.0,
        )

        result = status.to_dict()

        assert result["authenticated"] is True
        assert result["user"] == "test@example.com"
        assert result["plan"] == "pro"
        assert result["ttl_seconds"] == 3600
        assert result["expires_at"] == 1700000000.0


class TestCompletionResult:
    """Tests for CompletionResult dataclass."""

    def test_successful_completion(self):
        """Test successful completion result."""
        result = CompletionResult(
            success=True,
            content="Hello, I'm doing well!",
            model="claude-3-opus",
            tokens_in=10,
            tokens_out=20,
        )

        assert result.success is True
        assert result.content == "Hello, I'm doing well!"
        assert result.model == "claude-3-opus"

    def test_failed_completion(self):
        """Test failed completion result."""
        result = CompletionResult(
            success=False,
            error="Rate limit exceeded",
        )

        assert result.success is False
        assert result.error == "Rate limit exceeded"


class TestClaudeCliAuth:
    """Tests for Claude CLI authentication."""

    @pytest.fixture
    def claude_auth(self):
        """Create Claude CLI auth instance."""
        return ClaudeCliAuth({})

    def test_provider_name(self, claude_auth):
        """Test provider name."""
        assert claude_auth.provider_name == "anthropic"

    def test_cli_name(self, claude_auth):
        """Test CLI name."""
        assert claude_auth.cli_name == "claude"

    def test_auth_file_paths(self, claude_auth):
        """Test auth file paths."""
        paths = claude_auth.get_auth_file_paths()

        assert len(paths) == 3
        assert any(".claude" in str(p) for p in paths)

    @pytest.mark.asyncio
    async def test_check_auth_not_installed(self, claude_auth):
        """Test check_auth when CLI is not installed."""
        with patch.object(claude_auth, "is_installed", return_value=False):
            status = await claude_auth.check_auth()

            assert status.authenticated is False
            assert "not installed" in status.error

    @pytest.mark.asyncio
    async def test_check_auth_no_auth_file(self, claude_auth):
        """Test check_auth when no auth file exists."""
        with patch.object(claude_auth, "is_installed", return_value=True):
            with patch.object(
                claude_auth, "_run_command_async", return_value=(0, "1.0.0", "")
            ):
                with patch.object(claude_auth, "_get_auth_data", return_value=None):
                    status = await claude_auth.check_auth()

                    assert status.authenticated is False
                    assert "No auth file" in status.error

    @pytest.mark.asyncio
    async def test_check_auth_success(self, claude_auth):
        """Test successful auth check."""
        auth_data = {
            "email": "test@example.com",
            "plan": "pro",
            "expiresAt": "2025-01-01T00:00:00Z",
        }

        with patch.object(claude_auth, "is_installed", return_value=True):
            with patch.object(
                claude_auth, "_run_command_async", return_value=(0, "1.0.0", "")
            ):
                with patch.object(claude_auth, "_get_auth_data", return_value=auth_data):
                    status = await claude_auth.check_auth()

                    assert status.authenticated is True
                    assert status.user == "test@example.com"
                    assert status.plan == "pro"

    @pytest.mark.asyncio
    async def test_login_headless_not_supported(self, claude_auth):
        """Test that headless login returns error."""
        status = await claude_auth.login(headless=True)

        assert status.authenticated is False
        assert "Headless" in status.error

    @pytest.mark.asyncio
    async def test_complete_success(self, claude_auth):
        """Test successful completion."""
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(
            claude_auth,
            "_run_command_async",
            return_value=(0, "Hello there!", ""),
        ):
            result = await claude_auth.complete(messages, "claude-3-opus")

            assert result.success is True
            assert result.content == "Hello there!"
            assert result.model == "claude-3-opus"

    @pytest.mark.asyncio
    async def test_complete_failure(self, claude_auth):
        """Test failed completion."""
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(
            claude_auth,
            "_run_command_async",
            return_value=(1, "", "Error: Rate limit"),
        ):
            result = await claude_auth.complete(messages, "claude-3-opus")

            assert result.success is False
            assert "Rate limit" in result.error


class TestCodexCliAuth:
    """Tests for Codex CLI authentication."""

    @pytest.fixture
    def codex_auth(self):
        """Create Codex CLI auth instance."""
        return CodexCliAuth({})

    def test_provider_name(self, codex_auth):
        """Test provider name."""
        assert codex_auth.provider_name == "openai"

    def test_cli_name(self, codex_auth):
        """Test CLI name."""
        assert codex_auth.cli_name == "codex"

    @pytest.mark.asyncio
    async def test_check_auth_not_installed(self, codex_auth):
        """Test check_auth when CLI is not installed."""
        with patch.object(codex_auth, "is_installed", return_value=False):
            status = await codex_auth.check_auth()

            assert status.authenticated is False
            assert "not installed" in status.error

    @pytest.mark.asyncio
    async def test_check_auth_success(self, codex_auth):
        """Test successful auth check."""
        with patch.object(codex_auth, "is_installed", return_value=True):
            with patch.object(
                codex_auth,
                "_run_command_async",
                return_value=(0, "email: test@example.com\nplan: plus", ""),
            ):
                status = await codex_auth.check_auth()

                assert status.authenticated is True
                assert status.user == "test@example.com"
                assert status.plan == "plus"


class TestGeminiCliAuth:
    """Tests for Gemini CLI authentication."""

    @pytest.fixture
    def gemini_auth(self):
        """Create Gemini CLI auth instance."""
        return GeminiCliAuth({})

    def test_provider_name(self, gemini_auth):
        """Test provider name."""
        assert gemini_auth.provider_name == "google"

    def test_cli_name(self, gemini_auth):
        """Test CLI name."""
        assert gemini_auth.cli_name == "gemini"

    @pytest.mark.asyncio
    async def test_check_auth_not_installed(self, gemini_auth):
        """Test check_auth when CLI is not installed."""
        with patch.object(gemini_auth, "is_installed", return_value=False):
            status = await gemini_auth.check_auth()

            assert status.authenticated is False
            assert "not installed" in status.error


class TestCLIAuthManager:
    """Tests for CLI auth manager."""

    @pytest.fixture
    def manager(self):
        """Create CLI auth manager instance."""
        return CLIAuthManager({})

    def test_list_providers(self, manager):
        """Test listing all providers."""
        providers = manager.list_providers()

        assert "anthropic" in providers
        assert "openai" in providers
        assert "google" in providers

    def test_get_provider(self, manager):
        """Test getting a specific provider."""
        provider = manager.get_provider("anthropic")

        assert provider is not None
        assert isinstance(provider, ClaudeCliAuth)

    def test_get_unknown_provider(self, manager):
        """Test getting unknown provider returns None."""
        provider = manager.get_provider("unknown")

        assert provider is None

    @pytest.mark.asyncio
    async def test_check_all_auth(self, manager):
        """Test checking auth for all providers."""
        # Mock all providers as not installed
        for provider in manager._providers.values():
            provider.is_installed = MagicMock(return_value=False)

        results = await manager.check_all_auth()

        assert "anthropic" in results
        assert "openai" in results
        assert "google" in results
        assert all(not status.authenticated for status in results.values())

    @pytest.mark.asyncio
    async def test_get_status(self, manager):
        """Test getting status as dicts."""
        # Mock all providers as not installed
        for provider in manager._providers.values():
            provider.is_installed = MagicMock(return_value=False)

        status = await manager.get_status()

        assert isinstance(status, dict)
        assert all(isinstance(v, dict) for v in status.values())
