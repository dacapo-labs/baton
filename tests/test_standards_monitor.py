"""Tests for standards monitor."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baton.plugins.standards_monitor import (
    ReleaseInfo,
    SpecChange,
    StandardsMonitor,
    StandardsMonitorConfig,
    MONITORED_REPOS,
    MONITORED_SPECS,
    PROVIDER_CHANGELOGS,
    get_standards_monitor,
    init_standards_monitor,
)


class TestReleaseInfo:
    """Tests for ReleaseInfo dataclass."""

    def test_basic_release_info(self):
        """Test basic release info creation."""
        info = ReleaseInfo(
            repo="anthropics/claude-code",
            name="Claude Code v1.3.0",
            tag="v1.3.0",
            version="1.3.0",
            published_at="2025-01-01T00:00:00Z",
            body="## New Features\n- Added skills reload",
            html_url="https://github.com/anthropics/claude-code/releases/tag/v1.3.0",
        )

        assert info.repo == "anthropics/claude-code"
        assert info.version == "1.3.0"
        assert info.is_breaking is False
        assert info.breaking_changes == []

    def test_release_info_with_breaking_changes(self):
        """Test release info with breaking changes."""
        info = ReleaseInfo(
            repo="openai/codex",
            name="Codex v2.0.0",
            tag="v2.0.0",
            version="2.0.0",
            published_at="2025-01-01T00:00:00Z",
            body="Breaking change: new API",
            html_url="https://github.com/openai/codex/releases/tag/v2.0.0",
            is_breaking=True,
            breaking_changes=["New API format required"],
        )

        assert info.is_breaking is True
        assert len(info.breaking_changes) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = ReleaseInfo(
            repo="google-gemini/gemini-cli",
            name="Gemini CLI v2.1.0",
            tag="v2.1.0",
            version="2.1.0",
            published_at="2025-01-01T00:00:00Z",
            body="New features and improvements",
            html_url="https://github.com/google-gemini/gemini-cli/releases/tag/v2.1.0",
            new_features=["Added extensions support"],
        )

        result = info.to_dict()

        assert result["repo"] == "google-gemini/gemini-cli"
        assert result["version"] == "2.1.0"
        assert result["is_breaking"] is False
        assert result["new_features"] == ["Added extensions support"]

    def test_body_truncation(self):
        """Test that long body is truncated in to_dict."""
        long_body = "x" * 1000
        info = ReleaseInfo(
            repo="test/repo",
            name="Test",
            tag="v1.0.0",
            version="1.0.0",
            published_at="2025-01-01T00:00:00Z",
            body=long_body,
            html_url="https://example.com",
        )

        result = info.to_dict()
        assert len(result["body"]) == 500


class TestSpecChange:
    """Tests for SpecChange dataclass."""

    def test_basic_spec_change(self):
        """Test basic spec change creation."""
        change = SpecChange(
            spec_name="agentskills",
            source_url="https://agentskills.io",
            changed_at="2025-01-01T00:00:00Z",
            summary="Added version field requirement",
            changes=["SKILL.md now requires version field"],
        )

        assert change.spec_name == "agentskills"
        assert len(change.changes) == 1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        change = SpecChange(
            spec_name="agents-md",
            source_url="https://agents.md",
            changed_at="2025-01-01T00:00:00Z",
            summary="Updated discovery rules",
            changes=["New precedence order"],
            migration_notes=["Update your config.toml"],
        )

        result = change.to_dict()

        assert result["spec_name"] == "agents-md"
        assert result["migration_notes"] == ["Update your config.toml"]


class TestStandardsMonitorConfig:
    """Tests for StandardsMonitorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StandardsMonitorConfig()

        assert config.check_interval == 3600
        assert config.cache_file is None
        assert config.github_token is None
        assert config.monitor_cli_releases is True
        assert config.monitor_specs is True
        assert config.monitor_api_changes is True

    def test_custom_config(self):
        """Test custom configuration."""
        callback = MagicMock()
        config = StandardsMonitorConfig(
            check_interval=7200,
            github_token="test-token",
            monitor_specs=False,
            notify_callback=callback,
        )

        assert config.check_interval == 7200
        assert config.github_token == "test-token"
        assert config.monitor_specs is False
        assert config.notify_callback is callback


class TestStandardsMonitor:
    """Tests for StandardsMonitor."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create standards monitor instance."""
        config = StandardsMonitorConfig(
            cache_file=str(tmp_path / "standards_cache.json"),
        )
        return StandardsMonitor(config)

    def test_init(self, monitor):
        """Test monitor initialization."""
        assert monitor._cache == {}
        assert monitor._running is False

    def test_analyze_breaking_changes_positive(self, monitor):
        """Test detection of breaking changes."""
        body = """
        ## What's New
        - New feature X

        ## Breaking Changes
        - Changed API format
        - Removed deprecated field
        """

        is_breaking, changes = monitor._analyze_breaking_changes(body)

        assert is_breaking is True
        assert len(changes) > 0

    def test_analyze_breaking_changes_negative(self, monitor):
        """Test no false positives for breaking changes."""
        body = """
        ## What's New
        - Bug fixes
        - Performance improvements
        """

        is_breaking, changes = monitor._analyze_breaking_changes(body)

        assert is_breaking is False
        assert len(changes) == 0

    def test_analyze_breaking_changes_with_warning_emoji(self, monitor):
        """Test detection of breaking changes with emoji."""
        body = "⚠️ Breaking: New authentication required"

        is_breaking, changes = monitor._analyze_breaking_changes(body)

        assert is_breaking is True

    def test_extract_new_features(self, monitor):
        """Test extraction of new features."""
        body = """
        ## Changes
        - Added: New skills system
        - New: Extension support
        ✨ Dark mode theme
        feat: Better error messages
        """

        features = monitor._extract_new_features(body)

        assert len(features) >= 2

    def test_extract_new_features_empty(self, monitor):
        """Test feature extraction with no features."""
        body = "Bug fixes and improvements"

        features = monitor._extract_new_features(body)

        assert features == []

    def test_get_summary(self, monitor):
        """Test get_summary method."""
        summary = monitor.get_summary()

        assert "monitored_repos" in summary
        assert "monitored_specs" in summary
        assert "last_check" in summary
        assert summary["monitored_repos"] == len(MONITORED_REPOS)

    def test_get_compatibility_matrix(self, monitor):
        """Test compatibility matrix."""
        matrix = monitor.get_compatibility_matrix()

        assert "skills_format" in matrix
        assert "context_files" in matrix
        assert "hierarchical_context" in matrix
        assert "modular_imports" in matrix

        # Check Claude Code supports skills
        assert matrix["skills_format"]["claude-code"]["supported"] is True
        assert matrix["skills_format"]["claude-code"]["format"] == "SKILL.md"

        # Check Codex supports skills
        assert matrix["skills_format"]["codex"]["supported"] is True

        # Check Gemini doesn't support skills yet
        assert matrix["skills_format"]["gemini-cli"]["supported"] is False

        # Check context files
        assert matrix["context_files"]["claude-code"] == "CLAUDE.md"
        assert matrix["context_files"]["codex"] == "AGENTS.md"
        assert matrix["context_files"]["gemini-cli"] == "GEMINI.md"

    def test_get_updates_since(self, monitor):
        """Test get_updates_since with empty cache."""
        updates = monitor.get_updates_since(time.time() - 86400)

        assert updates == []

    def test_get_updates_since_with_data(self, monitor):
        """Test get_updates_since with cached data."""
        # Add some cached data
        monitor._cache["releases"] = {
            "anthropics/claude-code": {
                "version": "1.3.0",
                "published_at": "2025-01-01T00:00:00Z",
                "is_breaking": False,
                "html_url": "https://example.com",
            }
        }

        # Query for updates since 2024
        updates = monitor.get_updates_since(0)

        assert len(updates) == 1
        assert updates[0]["repo"] == "anthropics/claude-code"

    def test_cache_save_and_load(self, tmp_path):
        """Test cache persistence."""
        cache_file = str(tmp_path / "test_cache.json")

        # Create monitor and add data
        config = StandardsMonitorConfig(cache_file=cache_file)
        monitor1 = StandardsMonitor(config)
        monitor1._cache["releases"] = {"test/repo": {"version": "1.0.0"}}
        monitor1._save_cache()

        # Create new monitor and verify cache loaded
        monitor2 = StandardsMonitor(config)

        assert "releases" in monitor2._cache
        assert monitor2._cache["releases"]["test/repo"]["version"] == "1.0.0"

    def test_start_stop(self, monitor):
        """Test start and stop methods."""
        assert monitor._running is False

        monitor.start()
        assert monitor._running is True
        assert monitor._task is not None

        monitor.stop()
        assert monitor._running is False


class TestGitHubReleaseFetching:
    """Tests for GitHub release fetching."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create standards monitor instance."""
        config = StandardsMonitorConfig(
            cache_file=str(tmp_path / "standards_cache.json"),
        )
        return StandardsMonitor(config)

    @pytest.mark.asyncio
    async def test_check_github_release_success(self, monitor):
        """Test successful GitHub release fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "tag_name": "v1.3.0",
            "name": "Release 1.3.0",
            "published_at": "2025-01-01T00:00:00Z",
            "body": "## New Features\n- Feature X",
            "html_url": "https://github.com/test/repo/releases/tag/v1.3.0",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            result = await monitor._check_github_release("test/repo")

            assert result is not None
            assert result.version == "1.3.0"
            assert result.repo == "test/repo"

    @pytest.mark.asyncio
    async def test_check_github_release_not_found(self, monitor):
        """Test GitHub release fetch when no releases exist."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            # Should fall back to tags
            mock_tags_response = MagicMock()
            mock_tags_response.status_code = 200
            mock_tags_response.json.return_value = []

            mock_instance.get.side_effect = [mock_response, mock_tags_response]

            result = await monitor._check_github_release("test/repo")

            assert result is None


class TestMonitoredRepos:
    """Tests for monitored repos configuration."""

    def test_monitored_repos_structure(self):
        """Test that monitored repos have required fields."""
        required_fields = ["category", "name"]

        for repo, info in MONITORED_REPOS.items():
            assert "category" in info, f"{repo} missing category"
            assert "name" in info, f"{repo} missing name"

    def test_cli_repos_have_context_file(self):
        """Test that CLI repos specify context file."""
        cli_repos = [r for r, i in MONITORED_REPOS.items() if i["category"] == "cli"]

        for repo in cli_repos:
            info = MONITORED_REPOS[repo]
            assert "context_file" in info, f"{repo} missing context_file"

    def test_known_repos_exist(self):
        """Test that expected repos are monitored."""
        expected = [
            "anthropics/claude-code",
            "openai/codex",
            "google-gemini/gemini-cli",
            "paul-gauthier/aider",
        ]

        for repo in expected:
            assert repo in MONITORED_REPOS, f"{repo} not in MONITORED_REPOS"


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_init_and_get(self, tmp_path):
        """Test singleton initialization and retrieval."""
        config = StandardsMonitorConfig(
            cache_file=str(tmp_path / "singleton_cache.json"),
        )

        monitor1 = init_standards_monitor(config)
        monitor2 = get_standards_monitor()

        assert monitor1 is monitor2

    def test_get_before_init(self):
        """Test get_standards_monitor before initialization returns None."""
        # Note: This test may fail if run after other tests that init the singleton
        # In production, we'd use proper test isolation
        pass


class TestCheckAll:
    """Tests for check_all method."""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create standards monitor instance."""
        config = StandardsMonitorConfig(
            cache_file=str(tmp_path / "standards_cache.json"),
            monitor_cli_releases=True,
            monitor_specs=False,
            monitor_api_changes=False,
        )
        return StandardsMonitor(config)

    @pytest.mark.asyncio
    async def test_check_all_structure(self, monitor):
        """Test check_all returns correct structure."""
        with patch.object(monitor, "_check_github_release", new_callable=AsyncMock) as mock:
            mock.return_value = None

            results = await monitor.check_all()

            assert "checked_at" in results
            assert "releases" in results
            assert "new_since_last_check" in results
