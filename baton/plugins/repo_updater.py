"""Baton Repo Updater - Track and update cloned GitHub repositories."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class RepoInfo:
    """Information about a tracked repository."""

    name: str
    path: str
    remote_url: str | None = None
    current_commit: str | None = None
    latest_commit: str | None = None
    branch: str = "main"
    has_updates: bool = False
    last_updated: float | None = None
    error: str | None = None
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "remote_url": self.remote_url,
            "current_commit": self.current_commit,
            "latest_commit": self.latest_commit,
            "branch": self.branch,
            "has_updates": self.has_updates,
            "last_updated": self.last_updated,
            "error": self.error,
            "checked_at": self.checked_at,
        }


@dataclass
class RepoUpdaterConfig:
    """Configuration for repo updater."""

    check_interval: int = 3600  # Check hourly
    auto_update: bool = False  # Auto-pull updates
    notify_callback: Callable[[RepoInfo], None] | None = None
    repos_file: str | None = None  # Path to repos config file


# Well-known repos that can be auto-discovered or configured
KNOWN_REPOS = {
    # AI Tools
    "pai": {
        "url": "https://github.com/anthropics/anthropic-quickstarts",
        "description": "Anthropic quickstarts and examples",
    },
    "claude-code-skills": {
        "url": "https://github.com/anthropics/claude-code-skills",
        "description": "Claude Code skills library",
    },
    "fabric": {
        "url": "https://github.com/danielmiessler/fabric",
        "description": "AI augmentation framework",
    },
    "aider": {
        "url": "https://github.com/paul-gauthier/aider",
        "description": "AI pair programming tool",
    },
    # Dev Tools
    "oh-my-zsh": {
        "url": "https://github.com/ohmyzsh/ohmyzsh",
        "description": "Zsh configuration framework",
    },
    "nvm": {
        "url": "https://github.com/nvm-sh/nvm",
        "description": "Node Version Manager",
    },
    "pyenv": {
        "url": "https://github.com/pyenv/pyenv",
        "description": "Python Version Manager",
    },
}


class RepoUpdater:
    """Track and update cloned GitHub repositories."""

    def __init__(self, config: RepoUpdaterConfig | None = None):
        self.config = config or RepoUpdaterConfig()
        self._repos: dict[str, RepoInfo] = {}
        self._last_check: float = 0
        self._running: bool = False
        self._task: asyncio.Task | None = None

        # Load repos from config file if specified
        if self.config.repos_file:
            self._load_repos_config()

    def _run_command(
        self, cmd: list[str], cwd: str | None = None, timeout: int = 60
    ) -> tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output.strip()
        except FileNotFoundError:
            return False, "Command not found"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def _load_repos_config(self) -> None:
        """Load repos from config file."""
        if not self.config.repos_file:
            return

        config_path = Path(self.config.repos_file)
        if not config_path.exists():
            return

        try:
            with open(config_path) as f:
                data = json.load(f)

            for repo_data in data.get("repos", []):
                name = repo_data.get("name")
                path = repo_data.get("path")
                if name and path:
                    self._repos[name] = RepoInfo(
                        name=name,
                        path=path,
                        branch=repo_data.get("branch", "main"),
                    )

            log.info(f"Loaded {len(self._repos)} repos from config")

        except Exception as e:
            log.error(f"Failed to load repos config: {e}")

    def _save_repos_config(self) -> None:
        """Save repos to config file."""
        if not self.config.repos_file:
            return

        try:
            config_path = Path(self.config.repos_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "repos": [
                    {
                        "name": info.name,
                        "path": info.path,
                        "branch": info.branch,
                        "remote_url": info.remote_url,
                    }
                    for info in self._repos.values()
                ]
            }

            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            log.error(f"Failed to save repos config: {e}")

    def add_repo(
        self,
        name: str,
        path: str,
        branch: str = "main",
    ) -> RepoInfo:
        """Add a repository to track."""
        path = os.path.expanduser(path)

        info = RepoInfo(
            name=name,
            path=path,
            branch=branch,
        )

        # Get repo info
        if os.path.isdir(path):
            # Get remote URL
            success, output = self._run_command(
                ["git", "remote", "get-url", "origin"],
                cwd=path,
            )
            if success:
                info.remote_url = output.strip()

            # Get current commit
            success, output = self._run_command(
                ["git", "rev-parse", "HEAD"],
                cwd=path,
            )
            if success:
                info.current_commit = output.strip()[:8]

            # Get current branch
            success, output = self._run_command(
                ["git", "branch", "--show-current"],
                cwd=path,
            )
            if success and output:
                info.branch = output.strip()

        else:
            info.error = f"Path does not exist: {path}"

        self._repos[name] = info
        self._save_repos_config()
        return info

    def remove_repo(self, name: str) -> bool:
        """Remove a repository from tracking."""
        if name in self._repos:
            del self._repos[name]
            self._save_repos_config()
            return True
        return False

    def discover_repos(self, search_paths: list[str] | None = None) -> list[RepoInfo]:
        """Discover git repositories in common locations."""
        discovered = []

        if search_paths is None:
            search_paths = [
                "~/.config",
                "~/repos",
                "~/projects",
                "~/code",
                "~/src",
                "~/.local/share",
            ]

        for search_path in search_paths:
            path = Path(os.path.expanduser(search_path))
            if not path.exists():
                continue

            # Look for .git directories
            for git_dir in path.glob("*/.git"):
                repo_path = git_dir.parent
                repo_name = repo_path.name

                # Skip if already tracked
                if repo_name in self._repos:
                    continue

                # Check if it's a known repo
                success, output = self._run_command(
                    ["git", "remote", "get-url", "origin"],
                    cwd=str(repo_path),
                )

                if success:
                    remote_url = output.strip()

                    # Check against known repos
                    for known_name, known_info in KNOWN_REPOS.items():
                        if known_info["url"] in remote_url or remote_url in known_info["url"]:
                            info = self.add_repo(known_name, str(repo_path))
                            discovered.append(info)
                            break
                    else:
                        # Add as generic repo
                        info = self.add_repo(repo_name, str(repo_path))
                        discovered.append(info)

        return discovered

    async def check_repo(self, name: str) -> RepoInfo:
        """Check a repository for updates."""
        if name not in self._repos:
            return RepoInfo(name=name, path="", error=f"Unknown repo: {name}")

        info = self._repos[name]

        if not os.path.isdir(info.path):
            info.error = f"Path does not exist: {info.path}"
            return info

        # Fetch from remote
        success, output = self._run_command(
            ["git", "fetch", "origin", info.branch],
            cwd=info.path,
        )

        if not success:
            info.error = f"Failed to fetch: {output}"
            return info

        # Get current commit
        success, output = self._run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=info.path,
        )
        if success:
            info.current_commit = output.strip()[:8]

        # Get latest remote commit
        success, output = self._run_command(
            ["git", "rev-parse", f"origin/{info.branch}"],
            cwd=info.path,
        )
        if success:
            info.latest_commit = output.strip()[:8]

        # Check if updates available
        if info.current_commit and info.latest_commit:
            info.has_updates = info.current_commit != info.latest_commit

        info.checked_at = time.time()
        info.error = None

        return info

    async def update_repo(self, name: str, force: bool = False) -> RepoInfo:
        """Update a repository by pulling latest changes."""
        if name not in self._repos:
            return RepoInfo(name=name, path="", error=f"Unknown repo: {name}")

        info = self._repos[name]

        if not os.path.isdir(info.path):
            info.error = f"Path does not exist: {info.path}"
            return info

        # Check for uncommitted changes
        success, output = self._run_command(
            ["git", "status", "--porcelain"],
            cwd=info.path,
        )

        if success and output and not force:
            info.error = "Repository has uncommitted changes. Use force=True to override."
            return info

        # Stash changes if force
        if force and output:
            self._run_command(["git", "stash"], cwd=info.path)

        # Pull latest
        success, output = self._run_command(
            ["git", "pull", "origin", info.branch],
            cwd=info.path,
        )

        if not success:
            info.error = f"Failed to pull: {output}"
            # Restore stash if we stashed
            if force:
                self._run_command(["git", "stash", "pop"], cwd=info.path)
            return info

        # Update commit info
        success, output = self._run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=info.path,
        )
        if success:
            info.current_commit = output.strip()[:8]
            info.latest_commit = info.current_commit

        info.has_updates = False
        info.last_updated = time.time()
        info.error = None

        log.info(f"Updated {name} to {info.current_commit}")

        return info

    async def check_all(self) -> dict[str, RepoInfo]:
        """Check all tracked repositories for updates."""
        results = {}

        for name in self._repos:
            results[name] = await self.check_repo(name)

            # Notify if updates available
            if results[name].has_updates and self.config.notify_callback:
                try:
                    self.config.notify_callback(results[name])
                except Exception as e:
                    log.error(f"Notification callback failed: {e}")

        self._last_check = time.time()
        return results

    async def update_all(self, force: bool = False) -> dict[str, RepoInfo]:
        """Update all tracked repositories."""
        # First check for updates
        await self.check_all()

        results = {}
        for name, info in self._repos.items():
            if info.has_updates or force:
                results[name] = await self.update_repo(name, force=force)
            else:
                results[name] = info

        return results

    def get_repos_with_updates(self) -> list[RepoInfo]:
        """Get list of repos with available updates."""
        return [info for info in self._repos.values() if info.has_updates]

    def get_summary(self) -> dict[str, Any]:
        """Get summary of repo status."""
        with_updates = self.get_repos_with_updates()
        return {
            "total_tracked": len(self._repos),
            "with_updates": len(with_updates),
            "repos": {name: info.to_dict() for name, info in self._repos.items()},
            "updates_available": [info.to_dict() for info in with_updates],
            "last_check": self._last_check,
        }

    def get_all_repos(self) -> dict[str, RepoInfo]:
        """Get all tracked repos."""
        return self._repos.copy()

    def start(self) -> None:
        """Start periodic update checking."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        log.info("Repo updater started")

    def stop(self) -> None:
        """Stop periodic update checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        log.info("Repo updater stopped")

    async def _run_loop(self) -> None:
        """Main checking loop."""
        while self._running:
            try:
                await self.check_all()

                # Auto-update if configured
                if self.config.auto_update:
                    for name, info in self._repos.items():
                        if info.has_updates:
                            await self.update_repo(name)

            except Exception as e:
                log.error(f"Repo update check failed: {e}")

            await asyncio.sleep(self.config.check_interval)


# Convenience function to clone a known repo
async def clone_repo(
    name: str,
    target_path: str | None = None,
    branch: str = "main",
) -> RepoInfo:
    """Clone a known repository."""
    if name not in KNOWN_REPOS:
        return RepoInfo(name=name, path="", error=f"Unknown repo: {name}")

    repo_info = KNOWN_REPOS[name]
    url = repo_info["url"]

    if target_path is None:
        target_path = os.path.expanduser(f"~/repos/{name}")

    target_path = os.path.expanduser(target_path)

    # Check if already exists
    if os.path.exists(target_path):
        return RepoInfo(
            name=name,
            path=target_path,
            error=f"Path already exists: {target_path}",
        )

    # Clone
    result = subprocess.run(
        ["git", "clone", "--branch", branch, url, target_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return RepoInfo(
            name=name,
            path=target_path,
            error=f"Clone failed: {result.stderr}",
        )

    return RepoInfo(
        name=name,
        path=target_path,
        remote_url=url,
        branch=branch,
    )


# Singleton instance
_updater: RepoUpdater | None = None


def get_repo_updater() -> RepoUpdater | None:
    """Get the global repo updater instance."""
    return _updater


def init_repo_updater(
    config: RepoUpdaterConfig | dict[str, Any] | None = None
) -> RepoUpdater:
    """Initialize the global repo updater."""
    global _updater

    if isinstance(config, dict):
        config = RepoUpdaterConfig(**config)

    _updater = RepoUpdater(config)
    return _updater


def start_repo_updater() -> None:
    """Start the global repo updater."""
    if _updater:
        _updater.start()
    else:
        raise RuntimeError("Repo updater not initialized")


def stop_repo_updater() -> None:
    """Stop the global repo updater."""
    if _updater:
        _updater.stop()
