"""Baton Standards Monitor - Track AI tooling standards and changes.

Monitors:
- CLI tool releases (claude-code, codex, gemini-cli, aider)
- Spec changes (agentskills.io, AGENTS.md)
- Provider API changes and new models
- Context file format changes
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx

log = logging.getLogger(__name__)

# Singleton instance
_monitor: StandardsMonitor | None = None


@dataclass
class ReleaseInfo:
    """Information about a release."""

    repo: str
    name: str
    tag: str
    version: str
    published_at: str
    body: str  # Release notes / changelog
    html_url: str
    is_breaking: bool = False
    breaking_changes: list[str] = field(default_factory=list)
    new_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "name": self.name,
            "tag": self.tag,
            "version": self.version,
            "published_at": self.published_at,
            "body": self.body[:500] if self.body else "",  # Truncate
            "html_url": self.html_url,
            "is_breaking": self.is_breaking,
            "breaking_changes": self.breaking_changes,
            "new_features": self.new_features,
        }


@dataclass
class SpecChange:
    """Information about a spec change."""

    spec_name: str
    source_url: str
    changed_at: str
    summary: str
    changes: list[str] = field(default_factory=list)
    migration_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "source_url": self.source_url,
            "changed_at": self.changed_at,
            "summary": self.summary,
            "changes": self.changes,
            "migration_notes": self.migration_notes,
        }


@dataclass
class StandardsMonitorConfig:
    """Configuration for standards monitor."""

    check_interval: int = 3600  # Check hourly
    cache_file: str | None = None
    github_token: str | None = None  # For higher rate limits
    notify_callback: Callable[[ReleaseInfo | SpecChange], None] | None = None

    # What to monitor
    monitor_cli_releases: bool = True
    monitor_specs: bool = True
    monitor_api_changes: bool = True


# GitHub repos to monitor for releases
MONITORED_REPOS = {
    # AI CLI tools
    "anthropics/claude-code": {
        "category": "cli",
        "name": "Claude Code",
        "context_file": "CLAUDE.md",
        "skills_dir": ".claude/skills",
    },
    "openai/codex": {
        "category": "cli",
        "name": "Codex CLI",
        "context_file": "AGENTS.md",
        "skills_dir": ".codex/skills",
    },
    "google-gemini/gemini-cli": {
        "category": "cli",
        "name": "Gemini CLI",
        "context_file": "GEMINI.md",
        "skills_dir": None,  # Not yet supported
    },
    "paul-gauthier/aider": {
        "category": "cli",
        "name": "Aider",
        "context_file": "CONVENTIONS.md",
        "skills_dir": None,
    },
    # Spec repos
    "anthropics/agentskills": {
        "category": "spec",
        "name": "Agent Skills Spec",
        "spec_url": "https://agentskills.io",
    },
}

# Specs to monitor (web pages)
MONITORED_SPECS = {
    "agentskills": {
        "name": "Agent Skills Spec",
        "url": "https://agentskills.io",
        "version_selector": None,  # Would need scraping
    },
    "agents-md": {
        "name": "AGENTS.md Spec",
        "url": "https://agents.md",
        "version_selector": None,
    },
}

# Provider API changelogs
PROVIDER_CHANGELOGS = {
    "anthropic": {
        "name": "Anthropic",
        "changelog_url": "https://docs.anthropic.com/en/api/changelog",
        "models_url": "https://docs.anthropic.com/en/docs/about-claude/models",
    },
    "openai": {
        "name": "OpenAI",
        "changelog_url": "https://platform.openai.com/docs/changelog",
        "models_url": "https://platform.openai.com/docs/models",
    },
    "google": {
        "name": "Google AI",
        "changelog_url": "https://ai.google.dev/changelog",
        "models_url": "https://ai.google.dev/gemini-api/docs/models/gemini",
    },
}


class StandardsMonitor:
    """Monitor AI standards and tooling changes."""

    def __init__(self, config: StandardsMonitorConfig):
        self.config = config
        self._cache: dict[str, Any] = {}
        self._last_check: float = 0
        self._running = False
        self._task: asyncio.Task | None = None

        # Load cache
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached data from file."""
        if not self.config.cache_file:
            self.config.cache_file = str(
                Path.home() / ".baton" / "standards_cache.json"
            )

        cache_path = Path(self.config.cache_file)
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
                log.info(f"Loaded standards cache from {cache_path}")
            except Exception as e:
                log.warning(f"Failed to load standards cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to file."""
        if not self.config.cache_file:
            return

        cache_path = Path(self.config.cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cache_path.write_text(json.dumps(self._cache, indent=2))
        except Exception as e:
            log.warning(f"Failed to save standards cache: {e}")

    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        log.info("Standards monitor started")

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        log.info("Standards monitor stopped")

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await self.check_all()
            except Exception as e:
                log.error(f"Standards check failed: {e}")

            await asyncio.sleep(self.config.check_interval)

    async def check_all(self) -> dict[str, Any]:
        """Check all monitored sources for updates."""
        results = {
            "checked_at": datetime.utcnow().isoformat(),
            "releases": [],
            "spec_changes": [],
            "new_since_last_check": [],
        }

        last_check_time = self._cache.get("last_check_time", 0)

        # Check GitHub releases
        if self.config.monitor_cli_releases:
            for repo, info in MONITORED_REPOS.items():
                try:
                    release = await self._check_github_release(repo)
                    if release:
                        results["releases"].append(release.to_dict())

                        # Check if this is new since last check
                        if release.published_at:
                            release_time = datetime.fromisoformat(
                                release.published_at.replace("Z", "+00:00")
                            ).timestamp()
                            if release_time > last_check_time:
                                results["new_since_last_check"].append({
                                    "type": "release",
                                    "repo": repo,
                                    "version": release.version,
                                })

                                # Notify if callback configured
                                if self.config.notify_callback:
                                    self.config.notify_callback(release)

                except Exception as e:
                    log.warning(f"Failed to check {repo}: {e}")

        # Update cache
        self._cache["last_check_time"] = time.time()
        self._cache["releases"] = {
            r["repo"]: r for r in results["releases"]
        }
        self._last_check = time.time()
        self._save_cache()

        return results

    async def _check_github_release(self, repo: str) -> ReleaseInfo | None:
        """Check GitHub for latest release."""
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.config.github_token:
            headers["Authorization"] = f"token {self.config.github_token}"

        url = f"https://api.github.com/repos/{repo}/releases/latest"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, timeout=10)
                if response.status_code == 404:
                    # No releases, try tags
                    return await self._check_github_tags(repo, headers)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                log.debug(f"Failed to fetch release for {repo}: {e}")
                return None

        # Parse version from tag
        tag = data.get("tag_name", "")
        version = re.sub(r"^v", "", tag)

        # Analyze release notes for breaking changes
        body = data.get("body", "") or ""
        is_breaking, breaking_changes = self._analyze_breaking_changes(body)
        new_features = self._extract_new_features(body)

        return ReleaseInfo(
            repo=repo,
            name=data.get("name", ""),
            tag=tag,
            version=version,
            published_at=data.get("published_at", ""),
            body=body,
            html_url=data.get("html_url", ""),
            is_breaking=is_breaking,
            breaking_changes=breaking_changes,
            new_features=new_features,
        )

    async def _check_github_tags(
        self, repo: str, headers: dict
    ) -> ReleaseInfo | None:
        """Fallback: check GitHub tags if no releases."""
        url = f"https://api.github.com/repos/{repo}/tags"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                tags = response.json()
            except Exception:
                return None

        if not tags:
            return None

        tag = tags[0]
        version = re.sub(r"^v", "", tag.get("name", ""))

        return ReleaseInfo(
            repo=repo,
            name=tag.get("name", ""),
            tag=tag.get("name", ""),
            version=version,
            published_at="",
            body="",
            html_url=f"https://github.com/{repo}/releases/tag/{tag.get('name', '')}",
            is_breaking=False,
            breaking_changes=[],
            new_features=[],
        )

    def _analyze_breaking_changes(self, body: str) -> tuple[bool, list[str]]:
        """Analyze release notes for breaking changes."""
        breaking_patterns = [
            r"(?i)breaking\s*change",
            r"(?i)BREAKING:",
            r"(?i)⚠️.*breaking",
            r"(?i)migration\s*required",
            r"(?i)deprecated.*removed",
        ]

        changes = []
        is_breaking = False

        for pattern in breaking_patterns:
            matches = re.findall(f".*{pattern}.*", body, re.MULTILINE)
            if matches:
                is_breaking = True
                changes.extend(m.strip() for m in matches[:3])  # Limit to 3

        return is_breaking, changes

    def _extract_new_features(self, body: str) -> list[str]:
        """Extract new features from release notes."""
        feature_patterns = [
            r"(?i)(?:new|added|introducing):\s*(.+)",
            r"(?i)✨\s*(.+)",
            r"(?i)feat:\s*(.+)",
        ]

        features = []
        for pattern in feature_patterns:
            matches = re.findall(pattern, body)
            features.extend(m.strip() for m in matches[:5])  # Limit to 5

        return features

    def get_summary(self) -> dict[str, Any]:
        """Get summary of monitored standards."""
        releases = self._cache.get("releases", {})

        return {
            "monitored_repos": len(MONITORED_REPOS),
            "monitored_specs": len(MONITORED_SPECS),
            "last_check": self._last_check,
            "latest_versions": {
                repo: info.get("version", "unknown")
                for repo, info in releases.items()
            },
            "breaking_changes": [
                {"repo": repo, "version": info.get("version")}
                for repo, info in releases.items()
                if info.get("is_breaking")
            ],
        }

    def get_updates_since(self, since_timestamp: float) -> list[dict[str, Any]]:
        """Get updates since a given timestamp."""
        updates = []
        releases = self._cache.get("releases", {})

        for repo, info in releases.items():
            if info.get("published_at"):
                try:
                    release_time = datetime.fromisoformat(
                        info["published_at"].replace("Z", "+00:00")
                    ).timestamp()
                    if release_time > since_timestamp:
                        updates.append({
                            "type": "release",
                            "repo": repo,
                            "name": MONITORED_REPOS.get(repo, {}).get("name", repo),
                            "version": info.get("version"),
                            "is_breaking": info.get("is_breaking", False),
                            "published_at": info.get("published_at"),
                            "html_url": info.get("html_url"),
                        })
                except Exception:
                    pass

        return sorted(updates, key=lambda x: x.get("published_at", ""), reverse=True)

    def get_repo_info(self, repo: str) -> dict[str, Any] | None:
        """Get cached info for a specific repo."""
        releases = self._cache.get("releases", {})
        return releases.get(repo)

    def get_all_releases(self) -> dict[str, Any]:
        """Get all cached releases."""
        return self._cache.get("releases", {})

    def get_compatibility_matrix(self) -> dict[str, Any]:
        """Get compatibility info across CLIs."""
        return {
            "skills_format": {
                "claude-code": {
                    "supported": True,
                    "format": "SKILL.md",
                    "spec": "agentskills.io",
                    "dir": ".claude/skills",
                },
                "codex": {
                    "supported": True,
                    "format": "SKILL.md",
                    "spec": "agentskills.io",
                    "dir": ".codex/skills",
                },
                "gemini-cli": {
                    "supported": False,
                    "format": None,
                    "spec": None,
                    "dir": None,
                    "notes": "Requested in issue #12890",
                },
                "aider": {
                    "supported": False,
                    "format": None,
                    "spec": None,
                    "dir": None,
                },
            },
            "context_files": {
                "claude-code": "CLAUDE.md",
                "codex": "AGENTS.md",
                "gemini-cli": "GEMINI.md",
                "aider": "CONVENTIONS.md",
            },
            "hierarchical_context": {
                "claude-code": True,
                "codex": True,
                "gemini-cli": True,
                "aider": False,
            },
            "modular_imports": {
                "claude-code": True,
                "codex": True,
                "gemini-cli": True,  # @file.md syntax
                "aider": False,
            },
        }


def init_standards_monitor(
    config: StandardsMonitorConfig | None = None,
) -> StandardsMonitor:
    """Initialize the standards monitor singleton."""
    global _monitor
    if config is None:
        config = StandardsMonitorConfig()
    _monitor = StandardsMonitor(config)
    return _monitor


def get_standards_monitor() -> StandardsMonitor | None:
    """Get the standards monitor singleton."""
    return _monitor
