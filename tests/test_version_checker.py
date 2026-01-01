"""Tests for version checker."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from baton.plugins.version_checker import (
    VersionInfo,
    VersionChecker,
    VersionCheckerConfig,
    CLI_TOOLS,
    get_version_checker,
    init_version_checker,
    start_version_checker,
    stop_version_checker,
)


class TestVersionInfo:
    """Tests for VersionInfo dataclass."""

    def test_basic_version_info(self):
        """Test basic version info creation."""
        info = VersionInfo(
            name="claude",
            installed="1.0.0",
            latest="1.1.0",
            is_outdated=True,
        )

        assert info.name == "claude"
        assert info.installed == "1.0.0"
        assert info.latest == "1.1.0"
        assert info.is_outdated is True

    def test_version_info_with_error(self):
        """Test version info with error."""
        info = VersionInfo(
            name="unknown-tool",
            error="Command not found",
        )

        assert info.name == "unknown-tool"
        assert info.installed is None
        assert info.error == "Command not found"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = VersionInfo(
            name="aws",
            installed="2.15.0",
            latest="2.16.0",
            is_outdated=True,
            update_command="pip install --upgrade awscli",
        )

        result = info.to_dict()

        assert result["name"] == "aws"
        assert result["installed"] == "2.15.0"
        assert result["latest"] == "2.16.0"
        assert result["is_outdated"] is True
        assert result["update_command"] == "pip install --upgrade awscli"
        assert "checked_at" in result


class TestVersionCheckerConfig:
    """Tests for VersionCheckerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VersionCheckerConfig()

        assert config.check_interval == 86400
        assert config.check_cli_tools is True
        assert config.check_python_deps is True
        assert config.notify_callback is None

    def test_custom_config(self):
        """Test custom configuration."""
        callback = MagicMock()
        config = VersionCheckerConfig(
            check_interval=3600,
            check_cli_tools=True,
            check_python_deps=False,
            notify_callback=callback,
        )

        assert config.check_interval == 3600
        assert config.check_python_deps is False
        assert config.notify_callback is callback


class TestVersionChecker:
    """Tests for VersionChecker."""

    @pytest.fixture
    def checker(self):
        """Create version checker instance."""
        config = VersionCheckerConfig(
            check_cli_tools=True,
            check_python_deps=False,
        )
        return VersionChecker(config)

    def test_compare_versions_outdated(self, checker):
        """Test version comparison when outdated."""
        assert checker._compare_versions("1.0.0", "1.1.0") is True
        assert checker._compare_versions("1.0.0", "2.0.0") is True
        assert checker._compare_versions("1.9.9", "2.0.0") is True

    def test_compare_versions_current(self, checker):
        """Test version comparison when current."""
        assert checker._compare_versions("1.1.0", "1.1.0") is False
        assert checker._compare_versions("2.0.0", "1.9.9") is False

    def test_compare_versions_partial(self, checker):
        """Test version comparison with partial versions."""
        assert checker._compare_versions("1.0", "1.1.0") is True
        assert checker._compare_versions("1", "2") is True

    def test_compare_versions_invalid(self, checker):
        """Test version comparison with invalid versions."""
        assert checker._compare_versions("invalid", "1.0.0") is False
        assert checker._compare_versions("1.0.0", "invalid") is False

    def test_parse_version(self, checker):
        """Test version parsing from output."""
        # Claude style
        output = "claude version 1.2.3"
        assert checker._parse_version(output, r"(\d+\.\d+\.\d+)") == "1.2.3"

        # AWS CLI style
        output = "aws-cli/2.15.30 Python/3.11.0"
        assert checker._parse_version(output, r"aws-cli/(\d+\.\d+\.\d+)") == "2.15.30"

        # No match
        output = "no version here"
        assert checker._parse_version(output, r"(\d+\.\d+\.\d+)") is None

    def test_run_command_success(self, checker):
        """Test running a successful command."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="1.0.0",
                stderr="",
            )

            success, output = checker._run_command(["echo", "test"])

            assert success is True
            assert output == "1.0.0"

    def test_run_command_not_found(self, checker):
        """Test running command that doesn't exist."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            success, output = checker._run_command(["nonexistent"])

            assert success is False
            assert "not found" in output.lower()

    def test_run_command_timeout(self, checker):
        """Test command timeout."""
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 30)):
            success, output = checker._run_command(["slow-command"])

            assert success is False
            assert "timed out" in output.lower()

    @pytest.mark.asyncio
    async def test_check_cli_tool_not_installed(self, checker):
        """Test checking a tool that's not installed."""
        with patch.object(checker, "_run_command", return_value=(False, "not found")):
            info = await checker.check_cli_tool("claude")

            assert info.name == "claude"
            assert info.installed is None
            assert "not installed" in info.error.lower()

    @pytest.mark.asyncio
    async def test_check_cli_tool_installed(self, checker):
        """Test checking an installed tool."""
        def mock_run_command(cmd):
            if "--version" in cmd:
                return True, "claude version 1.0.0"
            elif "npm" in cmd:
                return True, "1.1.0"
            return False, ""

        with patch.object(checker, "_run_command", side_effect=mock_run_command):
            info = await checker.check_cli_tool("claude")

            assert info.name == "claude"
            assert info.installed == "1.0.0"
            assert info.latest == "1.1.0"
            assert info.is_outdated is True

    @pytest.mark.asyncio
    async def test_check_cli_tool_up_to_date(self, checker):
        """Test checking a tool that's up to date."""
        def mock_run_command(cmd):
            if "--version" in cmd:
                return True, "claude version 1.1.0"
            elif "npm" in cmd:
                return True, "1.1.0"
            return False, ""

        with patch.object(checker, "_run_command", side_effect=mock_run_command):
            info = await checker.check_cli_tool("claude")

            assert info.installed == "1.1.0"
            assert info.latest == "1.1.0"
            assert info.is_outdated is False

    @pytest.mark.asyncio
    async def test_check_cli_tool_unknown(self, checker):
        """Test checking an unknown tool."""
        info = await checker.check_cli_tool("unknown-tool")

        assert info.name == "unknown-tool"
        assert "Unknown tool" in info.error

    @pytest.mark.asyncio
    async def test_check_python_package_installed(self, checker):
        """Test checking an installed Python package."""
        def mock_run_command(cmd):
            if "pip" in cmd and "show" in cmd:
                return True, "Name: requests\nVersion: 2.31.0"
            elif "pip" in cmd and "index" in cmd:
                return True, "requests (2.32.0, 2.31.0)"
            return False, ""

        with patch.object(checker, "_run_command", side_effect=mock_run_command):
            info = await checker.check_python_package("requests")

            assert info.name == "requests"
            assert info.installed == "2.31.0"
            assert info.latest == "2.32.0"
            assert info.is_outdated is True

    @pytest.mark.asyncio
    async def test_check_python_package_not_installed(self, checker):
        """Test checking a package that's not installed."""
        with patch.object(checker, "_run_command", return_value=(False, "not found")):
            info = await checker.check_python_package("nonexistent-package")

            assert info.name == "nonexistent-package"
            assert info.error == "Not installed"

    @pytest.mark.asyncio
    async def test_check_all_cli_tools(self, checker):
        """Test checking all CLI tools."""
        mock_info = VersionInfo(name="test", installed="1.0.0")

        with patch.object(checker, "check_cli_tool", return_value=mock_info):
            results = await checker.check_all_cli_tools()

            assert len(results) == len(CLI_TOOLS)

    @pytest.mark.asyncio
    async def test_check_all_with_cache(self, checker):
        """Test caching behavior."""
        mock_results = {"cli_tools": {}, "python_deps": {}}

        with patch.object(checker, "check_all_cli_tools", return_value={}):
            # First call
            await checker.check_all(use_cache=False)

            # Second call should use cache
            await checker.check_all(use_cache=True, cache_ttl=3600)

    def test_get_outdated(self, checker):
        """Test getting outdated tools."""
        checker._cache = {
            "tool1": VersionInfo(name="tool1", is_outdated=True),
            "tool2": VersionInfo(name="tool2", is_outdated=False),
            "tool3": VersionInfo(name="tool3", is_outdated=True),
        }

        outdated = checker.get_outdated()

        assert len(outdated) == 2
        assert all(v.is_outdated for v in outdated)

    def test_get_summary(self, checker):
        """Test getting summary."""
        checker._cache = {
            "tool1": VersionInfo(name="tool1", is_outdated=True),
            "tool2": VersionInfo(name="tool2", is_outdated=False),
        }
        checker._last_check = 12345.0

        summary = checker.get_summary()

        assert summary["total_checked"] == 2
        assert summary["outdated_count"] == 1
        assert summary["last_check"] == 12345.0
        assert len(summary["outdated"]) == 1

    @pytest.mark.asyncio
    async def test_start_stop(self, checker):
        """Test starting and stopping the checker."""
        assert checker._running is False

        checker.start()
        assert checker._running is True
        assert checker._task is not None

        await asyncio.sleep(0.1)

        checker.stop()
        assert checker._running is False

    @pytest.mark.asyncio
    async def test_notify_callback_on_outdated(self, checker):
        """Test notification callback is called for outdated tools."""
        notified = []

        def callback(info):
            notified.append(info)

        checker.config.notify_callback = callback

        # Mock an outdated tool check
        outdated_info = VersionInfo(
            name="claude",
            installed="1.0.0",
            latest="2.0.0",
            is_outdated=True,
        )

        with patch.object(checker, "check_cli_tool", return_value=outdated_info):
            await checker.check_all_cli_tools()

        assert len(notified) == len(CLI_TOOLS)


class TestCliToolDefinitions:
    """Tests for CLI tool definitions."""

    def test_all_tools_have_required_fields(self):
        """Test all CLI tools have required fields."""
        required = ["version_cmd", "version_pattern"]

        for name, tool in CLI_TOOLS.items():
            for field in required:
                assert field in tool, f"{name} missing {field}"

    def test_tool_names(self):
        """Test expected tools are defined."""
        expected = ["claude", "codex", "gemini", "gcloud", "aws", "litellm", "playwright"]

        for name in expected:
            assert name in CLI_TOOLS, f"Missing tool: {name}"


class TestGlobalVersionChecker:
    """Tests for global version checker functions."""

    def test_init_version_checker_with_config(self):
        """Test initializing global checker with config."""
        config = VersionCheckerConfig(
            check_interval=1800,
            check_cli_tools=True,
            check_python_deps=False,
        )
        checker = init_version_checker(config)

        assert checker is not None
        assert checker.config.check_interval == 1800
        assert get_version_checker() is checker

    def test_init_version_checker_with_dict(self):
        """Test initializing global checker with dict config."""
        checker = init_version_checker({
            "check_interval": 900,
            "check_cli_tools": False,
        })

        assert checker is not None
        assert checker.config.check_interval == 900

    def test_get_version_checker_none(self):
        """Test get_version_checker when not initialized."""
        import baton.plugins.version_checker as vc_module

        vc_module._checker = None
        assert get_version_checker() is None

    @pytest.mark.asyncio
    async def test_start_stop_version_checker(self):
        """Test start and stop functions."""
        checker = init_version_checker({
            "check_cli_tools": False,
            "check_python_deps": False,
        })

        start_version_checker()
        assert checker._running is True

        await asyncio.sleep(0.1)

        stop_version_checker()
        assert checker._running is False

    def test_start_version_checker_not_initialized(self):
        """Test starting checker when not initialized."""
        import baton.plugins.version_checker as vc_module

        vc_module._checker = None

        with pytest.raises(RuntimeError, match="not initialized"):
            start_version_checker()
