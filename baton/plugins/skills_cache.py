"""Skills and MCP cache for baton.

Caches skill definitions and MCP tool schemas to reduce token usage
and startup latency. Skills/MCP tools often load 5000+ tokens of
definitions in every conversation - caching avoids this overhead.

Features:
- Discover and index skills from configured locations
- Cache parsed SKILL.md content
- Cache MCP server tool definitions
- Track versions and detect updates
- Provide unified API for skill/MCP management
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Default locations to search for skills
DEFAULT_SKILL_PATHS = [
    "~/.claude/skills",
    "~/maestro/.claude/skills",
    "~/libretto/.claude/skills",
]

# Default MCP config locations
DEFAULT_MCP_CONFIGS = [
    "~/.claude/mcp.json",
    "~/maestro/.claude/mcp.json",
]


@dataclass
class SkillInfo:
    """Information about a cached skill."""

    name: str
    path: str
    content: str  # SKILL.md content
    content_hash: str
    last_updated: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "content": self.content,
            "content_hash": self.content_hash,
            "last_updated": self.last_updated,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillInfo:
        return cls(
            name=data["name"],
            path=data["path"],
            content=data["content"],
            content_hash=data["content_hash"],
            last_updated=data["last_updated"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class MCPToolInfo:
    """Information about a cached MCP tool."""

    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]
    last_updated: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_name": self.server_name,
            "tool_name": self.tool_name,
            "description": self.description,
            "input_schema": self.input_schema,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPToolInfo:
        return cls(
            server_name=data["server_name"],
            tool_name=data["tool_name"],
            description=data["description"],
            input_schema=data["input_schema"],
            last_updated=data["last_updated"],
        )


@dataclass
class MCPServerInfo:
    """Information about an MCP server."""

    name: str
    server_type: str  # "stdio" or "http"
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""
    tools: list[MCPToolInfo] = field(default_factory=list)
    last_updated: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "server_type": self.server_type,
            "url": self.url,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "description": self.description,
            "tools": [t.to_dict() for t in self.tools],
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPServerInfo:
        return cls(
            name=data["name"],
            server_type=data["server_type"],
            url=data.get("url"),
            command=data.get("command"),
            args=data.get("args", []),
            env=data.get("env", {}),
            description=data.get("description", ""),
            tools=[MCPToolInfo.from_dict(t) for t in data.get("tools", [])],
            last_updated=data.get("last_updated", 0),
        )


@dataclass
class SkillsCacheConfig:
    """Configuration for skills cache."""

    skill_paths: list[str] = field(default_factory=lambda: DEFAULT_SKILL_PATHS.copy())
    mcp_configs: list[str] = field(default_factory=lambda: DEFAULT_MCP_CONFIGS.copy())
    cache_file: str | None = None
    auto_refresh: bool = True
    refresh_interval: int = 3600  # 1 hour
    notify_callback: Callable[[str, Any], None] | None = None


class SkillsCache:
    """Cache for skills and MCP tool definitions."""

    def __init__(self, config: SkillsCacheConfig | None = None):
        self.config = config or SkillsCacheConfig()
        self._skills: dict[str, SkillInfo] = {}
        self._mcp_servers: dict[str, MCPServerInfo] = {}
        self._running = False
        self._task: asyncio.Task | None = None

        # Load from cache file if exists
        if self.config.cache_file:
            self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from file."""
        if not self.config.cache_file:
            return

        cache_path = Path(self.config.cache_file).expanduser()
        if not cache_path.exists():
            return

        try:
            with open(cache_path) as f:
                data = json.load(f)

            for skill_data in data.get("skills", []):
                skill = SkillInfo.from_dict(skill_data)
                self._skills[skill.name] = skill

            for server_data in data.get("mcp_servers", []):
                server = MCPServerInfo.from_dict(server_data)
                self._mcp_servers[server.name] = server

        except (json.JSONDecodeError, KeyError, OSError):
            pass  # Start fresh if cache is corrupted

    def _save_cache(self) -> None:
        """Save cache to file."""
        if not self.config.cache_file:
            return

        cache_path = Path(self.config.cache_file).expanduser()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "skills": [s.to_dict() for s in self._skills.values()],
            "mcp_servers": [s.to_dict() for s in self._mcp_servers.values()],
            "updated": time.time(),
        }

        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

    def _hash_content(self, content: str) -> str:
        """Generate hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def discover_skills(self) -> list[SkillInfo]:
        """Discover skills from configured paths."""
        discovered = []

        for skill_path in self.config.skill_paths:
            path = Path(skill_path).expanduser()
            if not path.exists():
                continue

            for skill_dir in path.iterdir():
                if not skill_dir.is_dir():
                    continue

                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue

                try:
                    content = skill_md.read_text()
                    content_hash = self._hash_content(content)

                    # Parse metadata from SKILL.md frontmatter if present
                    metadata = self._parse_skill_metadata(content)

                    skill = SkillInfo(
                        name=skill_dir.name,
                        path=str(skill_dir),
                        content=content,
                        content_hash=content_hash,
                        last_updated=time.time(),
                        metadata=metadata,
                    )
                    discovered.append(skill)

                except OSError:
                    continue

        return discovered

    def _parse_skill_metadata(self, content: str) -> dict[str, Any]:
        """Parse metadata from SKILL.md content."""
        metadata = {}

        # Check for YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                # Simple key: value parsing
                for line in parts[1].strip().split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"\'')

        # Extract title from first heading
        for line in content.split("\n"):
            if line.startswith("# "):
                metadata["title"] = line[2:].strip()
                break

        return metadata

    def refresh_skills(self) -> dict[str, str]:
        """Refresh skill cache, returns changes."""
        changes = {}
        discovered = self.discover_skills()

        for skill in discovered:
            existing = self._skills.get(skill.name)
            if existing is None:
                changes[skill.name] = "added"
                self._skills[skill.name] = skill
            elif existing.content_hash != skill.content_hash:
                changes[skill.name] = "updated"
                self._skills[skill.name] = skill

        # Check for removed skills
        discovered_names = {s.name for s in discovered}
        for name in list(self._skills.keys()):
            if name not in discovered_names:
                changes[name] = "removed"
                del self._skills[name]

        if changes:
            self._save_cache()
            if self.config.notify_callback:
                self.config.notify_callback("skills_changed", changes)

        return changes

    def get_skill(self, name: str) -> SkillInfo | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_all_skills(self) -> list[SkillInfo]:
        """Get all cached skills."""
        return list(self._skills.values())

    def search_skills(self, query: str) -> list[SkillInfo]:
        """Search skills by name or content."""
        query = query.lower()
        results = []

        for skill in self._skills.values():
            if query in skill.name.lower() or query in skill.content.lower():
                results.append(skill)

        return results

    def load_mcp_configs(self) -> dict[str, MCPServerInfo]:
        """Load MCP server configurations."""
        servers = {}

        for config_path in self.config.mcp_configs:
            path = Path(config_path).expanduser()
            if not path.exists():
                continue

            try:
                with open(path) as f:
                    data = json.load(f)

                for name, config in data.get("mcpServers", {}).items():
                    # Skip example configs
                    if name.startswith("_"):
                        continue

                    server = MCPServerInfo(
                        name=name,
                        server_type=config.get("type", "stdio"),
                        url=config.get("url"),
                        command=config.get("command"),
                        args=config.get("args", []),
                        env=config.get("env", {}),
                        description=config.get("description", ""),
                        last_updated=time.time(),
                    )
                    servers[name] = server

            except (json.JSONDecodeError, OSError):
                continue

        return servers

    def refresh_mcp_servers(self) -> dict[str, str]:
        """Refresh MCP server cache, returns changes."""
        changes = {}
        discovered = self.load_mcp_configs()

        for name, server in discovered.items():
            existing = self._mcp_servers.get(name)
            if existing is None:
                changes[name] = "added"
                self._mcp_servers[name] = server
            # Note: We don't compare tools here, just server config

        # Check for removed servers
        for name in list(self._mcp_servers.keys()):
            if name not in discovered:
                changes[name] = "removed"
                del self._mcp_servers[name]

        if changes:
            self._save_cache()
            if self.config.notify_callback:
                self.config.notify_callback("mcp_changed", changes)

        return changes

    async def fetch_mcp_tools(self, server_name: str) -> list[MCPToolInfo]:
        """Fetch tool definitions from an MCP server."""
        server = self._mcp_servers.get(server_name)
        if not server:
            return []

        tools = []

        if server.server_type == "http" and server.url:
            # HTTP MCP server - fetch via API
            tools = await self._fetch_http_mcp_tools(server)
        elif server.server_type == "stdio" and server.command:
            # Stdio MCP server - spawn and query
            tools = await self._fetch_stdio_mcp_tools(server)

        server.tools = tools
        server.last_updated = time.time()
        self._save_cache()

        return tools

    async def _fetch_http_mcp_tools(self, server: MCPServerInfo) -> list[MCPToolInfo]:
        """Fetch tools from HTTP MCP server."""
        # HTTP MCP servers typically expose /tools endpoint
        # This is a placeholder - actual implementation depends on MCP spec
        return []

    async def _fetch_stdio_mcp_tools(self, server: MCPServerInfo) -> list[MCPToolInfo]:
        """Fetch tools from stdio MCP server."""
        # Stdio servers need to be spawned and queried via JSON-RPC
        # This is a placeholder - actual implementation depends on MCP spec
        return []

    def get_mcp_server(self, name: str) -> MCPServerInfo | None:
        """Get an MCP server by name."""
        return self._mcp_servers.get(name)

    def get_all_mcp_servers(self) -> list[MCPServerInfo]:
        """Get all configured MCP servers."""
        return list(self._mcp_servers.values())

    def get_mcp_tools_summary(self) -> str:
        """Get a summary of all cached MCP tools (for token estimation)."""
        lines = []
        total_tools = 0

        for server in self._mcp_servers.values():
            tool_count = len(server.tools)
            total_tools += tool_count
            lines.append(f"{server.name}: {tool_count} tools")

        lines.append(f"Total: {total_tools} tools")
        return "\n".join(lines)

    def get_skills_summary(self) -> dict[str, Any]:
        """Get summary of cached skills."""
        by_prefix = {}
        for skill in self._skills.values():
            prefix = skill.name.split("-")[0] if "-" in skill.name else "local"
            by_prefix[prefix] = by_prefix.get(prefix, 0) + 1

        return {
            "total": len(self._skills),
            "by_source": by_prefix,
            "skills": [s.name for s in self._skills.values()],
        }

    async def refresh_all(self) -> dict[str, Any]:
        """Refresh both skills and MCP server caches."""
        skill_changes = self.refresh_skills()
        mcp_changes = self.refresh_mcp_servers()

        return {
            "skills": skill_changes,
            "mcp_servers": mcp_changes,
        }

    async def start(self) -> None:
        """Start background refresh task."""
        if self._running:
            return

        self._running = True

        async def refresh_loop():
            while self._running:
                await self.refresh_all()
                await asyncio.sleep(self.config.refresh_interval)

        self._task = asyncio.create_task(refresh_loop())

    async def stop(self) -> None:
        """Stop background refresh task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


# Global instance
_cache: SkillsCache | None = None


def init_skills_cache(config: SkillsCacheConfig | dict[str, Any] | None = None) -> SkillsCache:
    """Initialize the global skills cache."""
    global _cache

    if isinstance(config, dict):
        config = SkillsCacheConfig(**config)

    _cache = SkillsCache(config)
    return _cache


def get_skills_cache() -> SkillsCache | None:
    """Get the global skills cache instance."""
    return _cache


async def start_skills_cache() -> None:
    """Start the global skills cache background task."""
    if _cache:
        await _cache.start()


async def stop_skills_cache() -> None:
    """Stop the global skills cache background task."""
    if _cache:
        await _cache.stop()
