"""Tests for repo updater."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from baton.plugins.repo_updater import (
    RepoInfo,
    RepoUpdater,
    RepoUpdaterConfig,
    KNOWN_REPOS,
    clone_repo,
    get_repo_updater,
    init_repo_updater,
    start_repo_updater,
    stop_repo_updater,
)


class TestRepoInfo:
    """Tests for RepoInfo dataclass."""

    def test_basic_repo_info(self):
        """Test basic repo info creation."""
        info = RepoInfo(
            name="test-repo",
            path="/home/user/repos/test",
            remote_url="https://github.com/user/test.git",
            current_commit="abc1234",
            branch="main",
        )

        assert info.name == "test-repo"
        assert info.path == "/home/user/repos/test"
        assert info.remote_url == "https://github.com/user/test.git"
        assert info.current_commit == "abc1234"
        assert info.branch == "main"
        assert info.has_updates is False

    def test_repo_info_with_updates(self):
        """Test repo info with updates available."""
        info = RepoInfo(
            name="test-repo",
            path="/path",
            current_commit="abc1234",
            latest_commit="def5678",
            has_updates=True,
        )

        assert info.has_updates is True
        assert info.current_commit != info.latest_commit

    def test_repo_info_with_error(self):
        """Test repo info with error."""
        info = RepoInfo(
            name="test-repo",
            path="/nonexistent",
            error="Path does not exist",
        )

        assert info.error == "Path does not exist"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = RepoInfo(
            name="test-repo",
            path="/path",
            remote_url="https://github.com/user/test.git",
            current_commit="abc1234",
            latest_commit="def5678",
            branch="main",
            has_updates=True,
        )

        result = info.to_dict()

        assert result["name"] == "test-repo"
        assert result["path"] == "/path"
        assert result["remote_url"] == "https://github.com/user/test.git"
        assert result["current_commit"] == "abc1234"
        assert result["latest_commit"] == "def5678"
        assert result["branch"] == "main"
        assert result["has_updates"] is True
        assert "checked_at" in result


class TestRepoUpdaterConfig:
    """Tests for RepoUpdaterConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RepoUpdaterConfig()

        assert config.check_interval == 3600
        assert config.auto_update is False
        assert config.notify_callback is None
        assert config.repos_file is None

    def test_custom_config(self):
        """Test custom configuration."""
        callback = MagicMock()
        config = RepoUpdaterConfig(
            check_interval=1800,
            auto_update=True,
            notify_callback=callback,
            repos_file="/tmp/repos.json",
        )

        assert config.check_interval == 1800
        assert config.auto_update is True
        assert config.notify_callback is callback
        assert config.repos_file == "/tmp/repos.json"


class TestRepoUpdater:
    """Tests for RepoUpdater."""

    @pytest.fixture
    def updater(self):
        """Create repo updater instance."""
        return RepoUpdater(RepoUpdaterConfig())

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary git repository."""
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()

        # Initialize git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo_path,
            capture_output=True,
        )

        # Create initial commit
        (repo_path / "README.md").write_text("# Test")
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            capture_output=True,
        )

        return repo_path

    def test_add_repo(self, updater, temp_repo):
        """Test adding a repository to track."""
        info = updater.add_repo("test", str(temp_repo))

        assert info.name == "test"
        assert info.path == str(temp_repo)
        # Current commit may or may not be set depending on git state
        assert info.error is None

    def test_add_repo_nonexistent_path(self, updater):
        """Test adding a repo with nonexistent path."""
        info = updater.add_repo("test", "/nonexistent/path")

        assert info.name == "test"
        assert info.error is not None
        assert "does not exist" in info.error

    def test_remove_repo(self, updater, temp_repo):
        """Test removing a repository."""
        updater.add_repo("test", str(temp_repo))
        assert "test" in updater._repos

        result = updater.remove_repo("test")

        assert result is True
        assert "test" not in updater._repos

    def test_remove_nonexistent_repo(self, updater):
        """Test removing a nonexistent repo."""
        result = updater.remove_repo("nonexistent")
        assert result is False

    def test_get_all_repos(self, updater, temp_repo):
        """Test getting all tracked repos."""
        updater.add_repo("test1", str(temp_repo))
        updater.add_repo("test2", str(temp_repo))

        repos = updater.get_all_repos()

        assert "test1" in repos
        assert "test2" in repos
        assert len(repos) == 2

    @pytest.mark.asyncio
    async def test_check_repo(self, updater, temp_repo):
        """Test checking a repo for updates."""
        updater.add_repo("test", str(temp_repo))

        # Mock fetch to succeed without network
        with patch.object(updater, "_run_command") as mock_run:
            mock_run.side_effect = [
                (True, ""),  # fetch
                (True, "abc12345"),  # current commit
                (True, "abc12345"),  # remote commit (same = no updates)
            ]

            info = await updater.check_repo("test")

            assert info.error is None
            assert info.has_updates is False

    @pytest.mark.asyncio
    async def test_check_repo_with_updates(self, updater, temp_repo):
        """Test checking a repo that has updates."""
        updater.add_repo("test", str(temp_repo))

        with patch.object(updater, "_run_command") as mock_run:
            mock_run.side_effect = [
                (True, ""),  # fetch
                (True, "abc12345"),  # current commit
                (True, "def67890"),  # remote commit (different = has updates)
            ]

            info = await updater.check_repo("test")

            assert info.has_updates is True
            assert info.current_commit == "abc12345"
            assert info.latest_commit == "def67890"

    @pytest.mark.asyncio
    async def test_check_unknown_repo(self, updater):
        """Test checking an unknown repo."""
        info = await updater.check_repo("unknown")

        assert info.error is not None
        assert "Unknown repo" in info.error

    @pytest.mark.asyncio
    async def test_update_repo(self, updater, temp_repo):
        """Test updating a repository."""
        updater.add_repo("test", str(temp_repo))

        with patch.object(updater, "_run_command") as mock_run:
            mock_run.side_effect = [
                (True, ""),  # status check (clean)
                (True, ""),  # pull
                (True, "newcommit"),  # get new commit
            ]

            info = await updater.update_repo("test")

            assert info.error is None
            assert info.last_updated is not None

    @pytest.mark.asyncio
    async def test_update_repo_with_changes(self, updater, temp_repo):
        """Test updating a repo with uncommitted changes."""
        updater.add_repo("test", str(temp_repo))

        with patch.object(updater, "_run_command") as mock_run:
            mock_run.return_value = (True, "M file.txt")  # uncommitted changes

            info = await updater.update_repo("test", force=False)

            assert info.error is not None
            assert "uncommitted changes" in info.error

    @pytest.mark.asyncio
    async def test_check_all(self, updater, temp_repo):
        """Test checking all repos."""
        updater.add_repo("test1", str(temp_repo))
        updater.add_repo("test2", str(temp_repo))

        with patch.object(updater, "check_repo") as mock_check:
            mock_check.return_value = RepoInfo(name="test", path=str(temp_repo))

            results = await updater.check_all()

            assert len(results) == 2
            assert mock_check.call_count == 2

    def test_get_repos_with_updates(self, updater):
        """Test getting repos with available updates."""
        updater._repos = {
            "repo1": RepoInfo(name="repo1", path="/path1", has_updates=True),
            "repo2": RepoInfo(name="repo2", path="/path2", has_updates=False),
            "repo3": RepoInfo(name="repo3", path="/path3", has_updates=True),
        }

        with_updates = updater.get_repos_with_updates()

        assert len(with_updates) == 2
        assert all(r.has_updates for r in with_updates)

    def test_get_summary(self, updater):
        """Test getting summary."""
        updater._repos = {
            "repo1": RepoInfo(name="repo1", path="/path1", has_updates=True),
            "repo2": RepoInfo(name="repo2", path="/path2", has_updates=False),
        }
        updater._last_check = 12345.0

        summary = updater.get_summary()

        assert summary["total_tracked"] == 2
        assert summary["with_updates"] == 1
        assert summary["last_check"] == 12345.0
        assert "repos" in summary
        assert "updates_available" in summary

    @pytest.mark.asyncio
    async def test_start_stop(self, updater):
        """Test starting and stopping the updater."""
        assert updater._running is False

        updater.start()
        assert updater._running is True
        assert updater._task is not None

        await asyncio.sleep(0.1)

        updater.stop()
        assert updater._running is False

    def test_config_file_persistence(self, tmp_path):
        """Test saving and loading repos config."""
        config_file = tmp_path / "repos.json"

        # Create updater and add repo
        config = RepoUpdaterConfig(repos_file=str(config_file))
        updater = RepoUpdater(config)
        updater._repos["test"] = RepoInfo(
            name="test",
            path="/path/to/repo",
            branch="main",
            remote_url="https://github.com/user/test.git",
        )
        updater._save_repos_config()

        # Verify file was created
        assert config_file.exists()

        # Load in new updater
        updater2 = RepoUpdater(config)
        assert "test" in updater2._repos
        assert updater2._repos["test"].path == "/path/to/repo"


class TestKnownRepos:
    """Tests for known repos."""

    def test_known_repos_structure(self):
        """Test KNOWN_REPOS has expected structure."""
        for name, info in KNOWN_REPOS.items():
            assert "url" in info, f"{name} missing url"
            assert "description" in info, f"{name} missing description"

    def test_expected_repos(self):
        """Test expected repos are defined."""
        expected = ["fabric", "oh-my-zsh", "nvm", "pyenv"]

        for name in expected:
            assert name in KNOWN_REPOS, f"Missing known repo: {name}"


class TestCloneRepo:
    """Tests for clone_repo function."""

    @pytest.mark.asyncio
    async def test_clone_unknown_repo(self):
        """Test cloning an unknown repo."""
        info = await clone_repo("unknown-repo-xyz")

        assert info.error is not None
        assert "Unknown repo" in info.error

    @pytest.mark.asyncio
    async def test_clone_existing_path(self, tmp_path):
        """Test cloning to existing path."""
        existing_path = tmp_path / "existing"
        existing_path.mkdir()

        info = await clone_repo("fabric", str(existing_path))

        assert info.error is not None
        assert "already exists" in info.error


class TestGlobalRepoUpdater:
    """Tests for global repo updater functions."""

    def test_init_repo_updater_with_config(self):
        """Test initializing global updater with config."""
        config = RepoUpdaterConfig(
            check_interval=1800,
            auto_update=True,
        )
        updater = init_repo_updater(config)

        assert updater is not None
        assert updater.config.check_interval == 1800
        assert get_repo_updater() is updater

    def test_init_repo_updater_with_dict(self):
        """Test initializing global updater with dict config."""
        updater = init_repo_updater({
            "check_interval": 900,
            "auto_update": False,
        })

        assert updater is not None
        assert updater.config.check_interval == 900

    def test_get_repo_updater_none(self):
        """Test get_repo_updater when not initialized."""
        import baton.plugins.repo_updater as ru_module

        ru_module._updater = None
        assert get_repo_updater() is None

    @pytest.mark.asyncio
    async def test_start_stop_repo_updater(self):
        """Test start and stop functions."""
        updater = init_repo_updater({})

        start_repo_updater()
        assert updater._running is True

        await asyncio.sleep(0.1)

        stop_repo_updater()
        assert updater._running is False

    def test_start_repo_updater_not_initialized(self):
        """Test starting updater when not initialized."""
        import baton.plugins.repo_updater as ru_module

        ru_module._updater = None

        with pytest.raises(RuntimeError, match="not initialized"):
            start_repo_updater()
