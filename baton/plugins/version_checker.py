"""Baton Version Checker - Monitor CLI tools and dependencies for updates."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class VersionInfo:
    """Information about a tool's version."""

    name: str
    category: str | None = None
    installed: str | None = None
    latest: str | None = None
    is_outdated: bool = False
    update_command: str | None = None
    error: str | None = None
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "installed": self.installed,
            "latest": self.latest,
            "is_outdated": self.is_outdated,
            "update_command": self.update_command,
            "error": self.error,
            "checked_at": self.checked_at,
        }


@dataclass
class VersionCheckerConfig:
    """Configuration for version checker."""

    check_interval: int = 86400  # Check daily
    check_cli_tools: bool = True
    check_python_deps: bool = True
    notify_callback: Callable[[VersionInfo], None] | None = None
    pyproject_path: str | None = None


# CLI tool definitions with version commands and update instructions
# Organized by category for clarity
CLI_TOOLS = {
    # ==========================================================================
    # AI/LLM CLI Tools (baton core)
    # ==========================================================================
    "claude": {
        "category": "ai",
        "version_cmd": ["claude", "--version"],
        "version_pattern": r"(\d+\.\d+\.\d+)",
        "latest_cmd": ["npm", "view", "@anthropic-ai/claude-code", "version"],
        "update_cmd": "npm update -g @anthropic-ai/claude-code",
        "package_name": "@anthropic-ai/claude-code",
    },
    "codex": {
        "category": "ai",
        "version_cmd": ["codex", "--version"],
        "version_pattern": r"(\d+\.\d+\.\d+)",
        "latest_cmd": ["npm", "view", "@openai/codex", "version"],
        "update_cmd": "npm update -g @openai/codex",
        "package_name": "@openai/codex",
    },
    "gemini": {
        "category": "ai",
        "version_cmd": ["gemini", "--version"],
        "version_pattern": r"(\d+\.\d+\.\d+)",
        "latest_cmd": ["npm", "view", "@google/gemini-cli", "version"],
        "update_cmd": "npm update -g @google/gemini-cli",
        "package_name": "@google/gemini-cli",
    },
    "fabric": {
        "category": "ai",
        "version_cmd": ["fabric", "--version"],
        "version_pattern": r"(\d+\.\d+\.\d+)",
        "update_cmd": "go install github.com/danielmiessler/fabric@latest",
        "latest_check": "github_release",
        "github_repo": "danielmiessler/fabric",
    },
    "litellm": {
        "category": "ai",
        "version_cmd": ["litellm", "--version"],
        "version_pattern": r"(\d+\.\d+\.\d+)",
        "latest_cmd": ["pip", "index", "versions", "litellm"],
        "update_cmd": "pip install --upgrade litellm",
        "package_name": "litellm",
    },
    # ==========================================================================
    # Cloud CLI Tools
    # ==========================================================================
    "gcloud": {
        "category": "cloud",
        "version_cmd": ["gcloud", "version", "--format=json"],
        "version_pattern": r'"Google Cloud SDK": "(\d+\.\d+\.\d+)"',
        "latest_check": "gcloud_components",
        "update_cmd": "gcloud components update",
    },
    "aws": {
        "category": "cloud",
        "version_cmd": ["aws", "--version"],
        "version_pattern": r"aws-cli/(\d+\.\d+\.\d+)",
        "latest_cmd": ["pip", "index", "versions", "awscli"],
        "update_cmd": "pip install --upgrade awscli",
        "package_name": "awscli",
    },
    "az": {
        "category": "cloud",
        "version_cmd": ["az", "version", "-o", "json"],
        "version_pattern": r'"azure-cli": "(\d+\.\d+\.\d+)"',
        "latest_cmd": ["pip", "index", "versions", "azure-cli"],
        "update_cmd": "pip install --upgrade azure-cli",
        "package_name": "azure-cli",
    },
    # ==========================================================================
    # Developer Tools (from maestro)
    # ==========================================================================
    "gh": {
        "category": "dev",
        "version_cmd": ["gh", "--version"],
        "version_pattern": r"gh version (\d+\.\d+\.\d+)",
        "update_cmd": "brew upgrade gh || gh upgrade",
        "latest_check": "github_release",
        "github_repo": "cli/cli",
    },
    "git": {
        "category": "dev",
        "version_cmd": ["git", "--version"],
        "version_pattern": r"git version (\d+\.\d+\.\d+)",
        "update_cmd": "brew upgrade git || apt-get update && apt-get install git",
    },
    "delta": {
        "category": "dev",
        "version_cmd": ["delta", "--version"],
        "version_pattern": r"delta (\d+\.\d+\.\d+)",
        "update_cmd": "brew upgrade git-delta || cargo install git-delta",
        "latest_check": "github_release",
        "github_repo": "dandavison/delta",
    },
    "fzf": {
        "category": "dev",
        "version_cmd": ["fzf", "--version"],
        "version_pattern": r"(\d+\.\d+\.\d+)",
        "update_cmd": "brew upgrade fzf || go install github.com/junegunn/fzf@latest",
        "latest_check": "github_release",
        "github_repo": "junegunn/fzf",
    },
    # ==========================================================================
    # Data/JSON Tools (from maestro)
    # ==========================================================================
    "jq": {
        "category": "data",
        "version_cmd": ["jq", "--version"],
        "version_pattern": r"jq-(\d+\.\d+)",
        "update_cmd": "brew upgrade jq || apt-get update && apt-get install jq",
        "latest_check": "github_release",
        "github_repo": "jqlang/jq",
    },
    "dasel": {
        "category": "data",
        "version_cmd": ["dasel", "version"],
        "version_pattern": r"Version:\s*(\d+\.\d+\.\d+)",
        "update_cmd": "brew upgrade dasel || go install github.com/tomwright/dasel/v2/cmd/dasel@latest",
        "latest_check": "github_release",
        "github_repo": "TomWright/dasel",
    },
    "yq": {
        "category": "data",
        "version_cmd": ["yq", "--version"],
        "version_pattern": r"version v?(\d+\.\d+\.\d+)",
        "update_cmd": "brew upgrade yq || go install github.com/mikefarah/yq/v4@latest",
        "latest_check": "github_release",
        "github_repo": "mikefarah/yq",
    },
    # ==========================================================================
    # Security Tools (from maestro)
    # ==========================================================================
    "bw": {
        "category": "security",
        "version_cmd": ["bw", "--version"],
        "version_pattern": r"(\d+\.\d+\.\d+)",
        "latest_cmd": ["npm", "view", "@bitwarden/cli", "version"],
        "update_cmd": "npm update -g @bitwarden/cli",
        "package_name": "@bitwarden/cli",
    },
    # ==========================================================================
    # Runtime/Package Managers
    # ==========================================================================
    "node": {
        "category": "runtime",
        "version_cmd": ["node", "--version"],
        "version_pattern": r"v(\d+\.\d+\.\d+)",
        "update_cmd": "nvm install node || brew upgrade node",
    },
    "npm": {
        "category": "runtime",
        "version_cmd": ["npm", "--version"],
        "version_pattern": r"(\d+\.\d+\.\d+)",
        "update_cmd": "npm install -g npm@latest",
    },
    "bun": {
        "category": "runtime",
        "version_cmd": ["bun", "--version"],
        "version_pattern": r"(\d+\.\d+\.\d+)",
        "update_cmd": "bun upgrade",
        "latest_check": "github_release",
        "github_repo": "oven-sh/bun",
    },
    "python": {
        "category": "runtime",
        "version_cmd": ["python3", "--version"],
        "version_pattern": r"Python (\d+\.\d+\.\d+)",
        "update_cmd": "brew upgrade python || apt-get update && apt-get install python3",
    },
    "go": {
        "category": "runtime",
        "version_cmd": ["go", "version"],
        "version_pattern": r"go(\d+\.\d+\.\d+)",
        "update_cmd": "brew upgrade go || apt-get update && apt-get install golang",
    },
    # ==========================================================================
    # Testing/Automation Tools
    # ==========================================================================
    "playwright": {
        "category": "testing",
        "version_cmd": ["playwright", "--version"],
        "version_pattern": r"(\d+\.\d+\.\d+)",
        "latest_cmd": ["pip", "index", "versions", "playwright"],
        "update_cmd": "pip install --upgrade playwright && playwright install",
        "package_name": "playwright",
    },
    # ==========================================================================
    # Communication Tools (from maestro)
    # ==========================================================================
    "himalaya": {
        "category": "comm",
        "version_cmd": ["himalaya", "--version"],
        "version_pattern": r"himalaya (\d+\.\d+\.\d+)",
        "update_cmd": "brew upgrade himalaya || cargo install himalaya",
        "latest_check": "github_release",
        "github_repo": "pimalaya/himalaya",
    },
}


class VersionChecker:
    """Check versions of CLI tools and dependencies."""

    def __init__(self, config: VersionCheckerConfig | None = None):
        self.config = config or VersionCheckerConfig()
        self._cache: dict[str, VersionInfo] = {}
        self._last_check: float = 0
        self._running: bool = False
        self._task: asyncio.Task | None = None

    def _run_command(
        self, cmd: list[str], timeout: int = 30
    ) -> tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output.strip()
        except FileNotFoundError:
            return False, "Command not found"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def _parse_version(self, output: str, pattern: str) -> str | None:
        """Extract version from command output using pattern."""
        match = re.search(pattern, output)
        return match.group(1) if match else None

    def _compare_versions(self, installed: str, latest: str) -> bool:
        """Compare versions. Returns True if installed < latest."""
        try:
            inst_parts = [int(x) for x in installed.split(".")[:3]]
            latest_parts = [int(x) for x in latest.split(".")[:3]]

            # Pad to equal length
            while len(inst_parts) < 3:
                inst_parts.append(0)
            while len(latest_parts) < 3:
                latest_parts.append(0)

            return inst_parts < latest_parts
        except (ValueError, AttributeError):
            return False

    def _get_github_latest_version(self, repo: str) -> str | None:
        """Get latest version from GitHub releases API."""
        try:
            import urllib.request
            import urllib.error

            url = f"https://api.github.com/repos/{repo}/releases/latest"
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/vnd.github.v3+json")
            req.add_header("User-Agent", "baton-version-checker")

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                tag = data.get("tag_name", "")
                # Strip leading 'v' if present
                return tag.lstrip("v") if tag else None

        except urllib.error.HTTPError as e:
            if e.code == 403:
                log.warning(f"GitHub rate limit hit for {repo}")
            return None
        except Exception as e:
            log.debug(f"Failed to get GitHub release for {repo}: {e}")
            return None

    async def check_cli_tool(self, name: str) -> VersionInfo:
        """Check version of a CLI tool."""
        if name not in CLI_TOOLS:
            return VersionInfo(
                name=name,
                error=f"Unknown tool: {name}",
            )

        tool = CLI_TOOLS[name]
        info = VersionInfo(
            name=name,
            category=tool.get("category"),
            update_command=tool.get("update_cmd"),
        )

        # Get installed version
        success, output = self._run_command(tool["version_cmd"])
        if not success:
            info.error = f"Not installed or not accessible: {output}"
            return info

        info.installed = self._parse_version(output, tool["version_pattern"])
        if not info.installed:
            info.error = f"Could not parse version from: {output[:100]}"
            return info

        # Get latest version based on method
        if "latest_cmd" in tool:
            success, output = self._run_command(tool["latest_cmd"])
            if success:
                # Handle different output formats
                if "pip index" in " ".join(tool["latest_cmd"]):
                    # pip index versions output: "package (x.y.z)"
                    match = re.search(r"\(([^)]+)\)", output)
                    if match:
                        versions = match.group(1).split(", ")
                        info.latest = versions[0] if versions else None
                else:
                    # npm view output: just the version
                    info.latest = output.strip()

        elif tool.get("latest_check") == "gcloud_components":
            # Special handling for gcloud
            success, output = self._run_command(
                ["gcloud", "components", "list", "--format=json"]
            )
            if success:
                try:
                    components = json.loads(output)
                    for comp in components:
                        if comp.get("id") == "core":
                            current = comp.get("current_version_string")
                            latest = comp.get("latest_version_string")
                            if current and latest:
                                info.installed = current
                                info.latest = latest
                            break
                except json.JSONDecodeError:
                    pass

        elif tool.get("latest_check") == "github_release":
            # Get latest from GitHub releases
            repo = tool.get("github_repo")
            if repo:
                info.latest = self._get_github_latest_version(repo)

        # Check if outdated
        if info.installed and info.latest:
            info.is_outdated = self._compare_versions(info.installed, info.latest)

        return info

    async def check_python_package(self, package: str) -> VersionInfo:
        """Check version of a Python package."""
        info = VersionInfo(
            name=package,
            update_command=f"pip install --upgrade {package}",
        )

        # Get installed version
        success, output = self._run_command(
            ["pip", "show", package]
        )
        if not success:
            info.error = "Not installed"
            return info

        # Parse installed version
        match = re.search(r"Version:\s*(\S+)", output)
        if match:
            info.installed = match.group(1)

        # Get latest version
        success, output = self._run_command(
            ["pip", "index", "versions", package]
        )
        if success:
            match = re.search(r"\(([^)]+)\)", output)
            if match:
                versions = match.group(1).split(", ")
                info.latest = versions[0] if versions else None

        # Check if outdated
        if info.installed and info.latest:
            info.is_outdated = self._compare_versions(info.installed, info.latest)

        return info

    async def check_all_cli_tools(self) -> dict[str, VersionInfo]:
        """Check all CLI tools."""
        results = {}
        for name in CLI_TOOLS:
            results[name] = await self.check_cli_tool(name)
            self._cache[name] = results[name]

            # Notify if outdated
            if results[name].is_outdated and self.config.notify_callback:
                try:
                    self.config.notify_callback(results[name])
                except Exception as e:
                    log.error(f"Notification callback failed: {e}")

        return results

    async def check_pyproject_deps(self) -> dict[str, VersionInfo]:
        """Check dependencies from pyproject.toml."""
        results = {}

        # Find pyproject.toml
        pyproject_path = self.config.pyproject_path
        if not pyproject_path:
            # Try to find it relative to this file
            current = Path(__file__).parent.parent.parent
            pyproject_path = current / "pyproject.toml"

        if not Path(pyproject_path).exists():
            log.warning(f"pyproject.toml not found at {pyproject_path}")
            return results

        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                log.warning("tomllib/tomli not available for parsing pyproject.toml")
                return results

        try:
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)

            deps = pyproject.get("project", {}).get("dependencies", [])

            for dep in deps:
                # Parse package name from dependency string
                match = re.match(r"([a-zA-Z0-9_-]+)", dep)
                if match:
                    package = match.group(1)
                    results[package] = await self.check_python_package(package)
                    self._cache[f"pip:{package}"] = results[package]

        except Exception as e:
            log.error(f"Failed to parse pyproject.toml: {e}")

        return results

    async def check_all(
        self, use_cache: bool = True, cache_ttl: int = 3600
    ) -> dict[str, dict[str, VersionInfo]]:
        """Check all tools and dependencies."""
        now = time.time()

        # Return cached results if valid
        if use_cache and self._cache and (now - self._last_check) < cache_ttl:
            return {
                "cli_tools": {
                    k: v for k, v in self._cache.items() if not k.startswith("pip:")
                },
                "python_deps": {
                    k.replace("pip:", ""): v
                    for k, v in self._cache.items()
                    if k.startswith("pip:")
                },
            }

        results = {"cli_tools": {}, "python_deps": {}}

        if self.config.check_cli_tools:
            results["cli_tools"] = await self.check_all_cli_tools()

        if self.config.check_python_deps:
            results["python_deps"] = await self.check_pyproject_deps()

        self._last_check = now
        return results

    async def check_category(self, category: str) -> dict[str, VersionInfo]:
        """Check all tools in a specific category."""
        results = {}
        for name, tool in CLI_TOOLS.items():
            if tool.get("category") == category:
                results[name] = await self.check_cli_tool(name)
                self._cache[name] = results[name]
        return results

    def get_tools_by_category(self) -> dict[str, list[str]]:
        """Get tool names grouped by category."""
        categories: dict[str, list[str]] = {}
        for name, tool in CLI_TOOLS.items():
            cat = tool.get("category", "other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(name)
        return categories

    def get_outdated(self) -> list[VersionInfo]:
        """Get list of outdated tools."""
        return [v for v in self._cache.values() if v.is_outdated]

    def get_outdated_by_category(self) -> dict[str, list[VersionInfo]]:
        """Get outdated tools grouped by category."""
        outdated: dict[str, list[VersionInfo]] = {}
        for v in self._cache.values():
            if v.is_outdated:
                cat = v.category or "other"
                if cat not in outdated:
                    outdated[cat] = []
                outdated[cat].append(v)
        return outdated

    def get_summary(self) -> dict[str, Any]:
        """Get summary of version status."""
        outdated = self.get_outdated()
        by_category = self.get_outdated_by_category()
        return {
            "total_checked": len(self._cache),
            "outdated_count": len(outdated),
            "outdated": [v.to_dict() for v in outdated],
            "outdated_by_category": {
                cat: [v.to_dict() for v in tools]
                for cat, tools in by_category.items()
            },
            "last_check": self._last_check,
            "categories": self.get_tools_by_category(),
        }

    def start(self) -> None:
        """Start periodic version checking."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        log.info("Version checker started")

    def stop(self) -> None:
        """Stop periodic version checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        log.info("Version checker stopped")

    async def _run_loop(self) -> None:
        """Main checking loop."""
        while self._running:
            try:
                await self.check_all(use_cache=False)
            except Exception as e:
                log.error(f"Version check failed: {e}")

            await asyncio.sleep(self.config.check_interval)


# Singleton instance
_checker: VersionChecker | None = None


def get_version_checker() -> VersionChecker | None:
    """Get the global version checker instance."""
    return _checker


def init_version_checker(
    config: VersionCheckerConfig | dict[str, Any] | None = None
) -> VersionChecker:
    """Initialize the global version checker."""
    global _checker

    if isinstance(config, dict):
        config = VersionCheckerConfig(**config)

    _checker = VersionChecker(config)
    return _checker


def start_version_checker() -> None:
    """Start the global version checker."""
    if _checker:
        _checker.start()
    else:
        raise RuntimeError("Version checker not initialized")


def stop_version_checker() -> None:
    """Stop the global version checker."""
    if _checker:
        _checker.stop()
