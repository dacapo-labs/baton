"""Tests for skills_cache module."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from baton.plugins.skills_cache import (
    SkillsCache,
    SkillsCacheConfig,
    SkillInfo,
    MCPServerInfo,
    MCPToolInfo,
    get_skills_cache,
    init_skills_cache,
    start_skills_cache,
    stop_skills_cache,
    DEFAULT_SKILL_PATHS,
    DEFAULT_MCP_CONFIGS,
)


class TestSkillInfo:
    """Tests for SkillInfo dataclass."""

    def test_basic_skill_info(self):
        """Test creating basic skill info."""
        skill = SkillInfo(
            name="test-skill",
            path="/path/to/skill",
            content="# Test Skill\nThis is a test.",
            content_hash="abc123",
            last_updated=1000.0,
        )

        assert skill.name == "test-skill"
        assert skill.path == "/path/to/skill"
        assert skill.content_hash == "abc123"
        assert skill.metadata == {}

    def test_skill_info_with_metadata(self):
        """Test skill info with metadata."""
        skill = SkillInfo(
            name="test-skill",
            path="/path/to/skill",
            content="# Test",
            content_hash="abc",
            last_updated=1000.0,
            metadata={"title": "Test Skill", "author": "test"},
        )

        assert skill.metadata["title"] == "Test Skill"
        assert skill.metadata["author"] == "test"

    def test_to_dict(self):
        """Test converting to dict."""
        skill = SkillInfo(
            name="test",
            path="/path",
            content="content",
            content_hash="hash",
            last_updated=1000.0,
            metadata={"key": "value"},
        )

        data = skill.to_dict()
        assert data["name"] == "test"
        assert data["content"] == "content"
        assert data["metadata"]["key"] == "value"

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "name": "test",
            "path": "/path",
            "content": "content",
            "content_hash": "hash",
            "last_updated": 1000.0,
            "metadata": {"key": "value"},
        }

        skill = SkillInfo.from_dict(data)
        assert skill.name == "test"
        assert skill.metadata["key"] == "value"


class TestMCPToolInfo:
    """Tests for MCPToolInfo dataclass."""

    def test_basic_tool_info(self):
        """Test creating basic tool info."""
        tool = MCPToolInfo(
            server_name="github",
            tool_name="list_repos",
            description="List repositories",
            input_schema={"type": "object"},
            last_updated=1000.0,
        )

        assert tool.server_name == "github"
        assert tool.tool_name == "list_repos"

    def test_to_dict(self):
        """Test converting to dict."""
        tool = MCPToolInfo(
            server_name="github",
            tool_name="list_repos",
            description="List repos",
            input_schema={"type": "object"},
            last_updated=1000.0,
        )

        data = tool.to_dict()
        assert data["server_name"] == "github"
        assert data["input_schema"]["type"] == "object"

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "server_name": "github",
            "tool_name": "list_repos",
            "description": "List repos",
            "input_schema": {"type": "object"},
            "last_updated": 1000.0,
        }

        tool = MCPToolInfo.from_dict(data)
        assert tool.server_name == "github"


class TestMCPServerInfo:
    """Tests for MCPServerInfo dataclass."""

    def test_http_server(self):
        """Test HTTP MCP server."""
        server = MCPServerInfo(
            name="github",
            server_type="http",
            url="https://api.github.com/mcp",
            description="GitHub integration",
        )

        assert server.name == "github"
        assert server.server_type == "http"
        assert server.url == "https://api.github.com/mcp"

    def test_stdio_server(self):
        """Test stdio MCP server."""
        server = MCPServerInfo(
            name="filesystem",
            server_type="stdio",
            command="npx",
            args=["-y", "@anthropic/mcp-server-filesystem"],
        )

        assert server.server_type == "stdio"
        assert server.command == "npx"

    def test_to_dict(self):
        """Test converting to dict."""
        server = MCPServerInfo(
            name="test",
            server_type="http",
            url="https://test.com",
            tools=[
                MCPToolInfo(
                    server_name="test",
                    tool_name="tool1",
                    description="Test tool",
                    input_schema={},
                    last_updated=1000.0,
                )
            ],
        )

        data = server.to_dict()
        assert data["name"] == "test"
        assert len(data["tools"]) == 1

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "name": "test",
            "server_type": "http",
            "url": "https://test.com",
            "tools": [
                {
                    "server_name": "test",
                    "tool_name": "tool1",
                    "description": "Test",
                    "input_schema": {},
                    "last_updated": 1000.0,
                }
            ],
        }

        server = MCPServerInfo.from_dict(data)
        assert server.name == "test"
        assert len(server.tools) == 1


class TestSkillsCacheConfig:
    """Tests for SkillsCacheConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SkillsCacheConfig()

        assert config.skill_paths == DEFAULT_SKILL_PATHS
        assert config.mcp_configs == DEFAULT_MCP_CONFIGS
        assert config.auto_refresh is True
        assert config.refresh_interval == 3600

    def test_custom_config(self):
        """Test custom configuration."""
        config = SkillsCacheConfig(
            skill_paths=["/custom/skills"],
            mcp_configs=["/custom/mcp.json"],
            refresh_interval=1800,
        )

        assert config.skill_paths == ["/custom/skills"]
        assert config.refresh_interval == 1800


class TestSkillsCache:
    """Tests for SkillsCache class."""

    def test_init(self):
        """Test cache initialization."""
        cache = SkillsCache()

        assert cache._skills == {}
        assert cache._mcp_servers == {}
        assert cache._running is False

    def test_init_with_config(self):
        """Test cache initialization with config."""
        config = SkillsCacheConfig(refresh_interval=1800)
        cache = SkillsCache(config)

        assert cache.config.refresh_interval == 1800

    def test_hash_content(self):
        """Test content hashing."""
        cache = SkillsCache()

        hash1 = cache._hash_content("test content")
        hash2 = cache._hash_content("test content")
        hash3 = cache._hash_content("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16

    def test_parse_skill_metadata_with_frontmatter(self):
        """Test parsing skill metadata from frontmatter."""
        cache = SkillsCache()

        content = """---
title: Test Skill
author: Test Author
---
# Test Skill

This is a test skill.
"""
        metadata = cache._parse_skill_metadata(content)

        assert metadata["title"] == "Test Skill"
        assert metadata["author"] == "Test Author"

    def test_parse_skill_metadata_heading_only(self):
        """Test parsing skill metadata from heading."""
        cache = SkillsCache()

        content = """# My Awesome Skill

This is a test skill.
"""
        metadata = cache._parse_skill_metadata(content)

        assert metadata["title"] == "My Awesome Skill"

    def test_discover_skills(self):
        """Test skill discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a skill directory
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# Test Skill\nThis is a test.")

            config = SkillsCacheConfig(skill_paths=[tmpdir])
            cache = SkillsCache(config)

            discovered = cache.discover_skills()

            assert len(discovered) == 1
            assert discovered[0].name == "test-skill"
            assert "# Test Skill" in discovered[0].content

    def test_discover_skills_ignores_non_skill_dirs(self):
        """Test that non-skill directories are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory without SKILL.md
            non_skill = Path(tmpdir) / "not-a-skill"
            non_skill.mkdir()
            (non_skill / "README.md").write_text("Not a skill")

            config = SkillsCacheConfig(skill_paths=[tmpdir])
            cache = SkillsCache(config)

            discovered = cache.discover_skills()

            assert len(discovered) == 0

    def test_refresh_skills(self):
        """Test refreshing skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a skill
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# Test")

            config = SkillsCacheConfig(skill_paths=[tmpdir])
            cache = SkillsCache(config)

            changes = cache.refresh_skills()

            assert "test-skill" in changes
            assert changes["test-skill"] == "added"
            assert len(cache._skills) == 1

    def test_refresh_skills_detects_updates(self):
        """Test that skill updates are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text("# Test v1")

            config = SkillsCacheConfig(skill_paths=[tmpdir])
            cache = SkillsCache(config)

            cache.refresh_skills()

            # Update the skill
            skill_file.write_text("# Test v2")
            changes = cache.refresh_skills()

            assert changes["test-skill"] == "updated"

    def test_refresh_skills_detects_removal(self):
        """Test that skill removal is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# Test")

            config = SkillsCacheConfig(skill_paths=[tmpdir])
            cache = SkillsCache(config)

            cache.refresh_skills()

            # Remove the skill
            (skill_dir / "SKILL.md").unlink()
            skill_dir.rmdir()

            changes = cache.refresh_skills()

            assert changes["test-skill"] == "removed"
            assert len(cache._skills) == 0

    def test_get_skill(self):
        """Test getting a skill by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# Test")

            config = SkillsCacheConfig(skill_paths=[tmpdir])
            cache = SkillsCache(config)
            cache.refresh_skills()

            skill = cache.get_skill("test-skill")
            assert skill is not None
            assert skill.name == "test-skill"

            assert cache.get_skill("nonexistent") is None

    def test_get_all_skills(self):
        """Test getting all skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["skill1", "skill2", "skill3"]:
                skill_dir = Path(tmpdir) / name
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text(f"# {name}")

            config = SkillsCacheConfig(skill_paths=[tmpdir])
            cache = SkillsCache(config)
            cache.refresh_skills()

            skills = cache.get_all_skills()
            assert len(skills) == 3

    def test_search_skills(self):
        """Test searching skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, content in [
                ("code-review", "# Code Review\nReview code for bugs"),
                ("test-runner", "# Test Runner\nRun tests"),
                ("deploy", "# Deploy\nDeploy code to production"),
            ]:
                skill_dir = Path(tmpdir) / name
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text(content)

            config = SkillsCacheConfig(skill_paths=[tmpdir])
            cache = SkillsCache(config)
            cache.refresh_skills()

            # Search by name
            results = cache.search_skills("code")
            assert len(results) == 2  # code-review and deploy (code to production)

            # Search by content
            results = cache.search_skills("bugs")
            assert len(results) == 1
            assert results[0].name == "code-review"

    def test_load_mcp_configs(self):
        """Test loading MCP configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp_config = Path(tmpdir) / "mcp.json"
            mcp_config.write_text(json.dumps({
                "mcpServers": {
                    "github": {
                        "type": "http",
                        "url": "https://api.github.com/mcp",
                        "description": "GitHub integration",
                    },
                    "filesystem": {
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "@anthropic/mcp-server-filesystem"],
                    },
                    "_example": {
                        "type": "http",
                        "url": "https://example.com",
                    },
                }
            }))

            config = SkillsCacheConfig(mcp_configs=[str(mcp_config)])
            cache = SkillsCache(config)

            servers = cache.load_mcp_configs()

            # Should have 2 servers (not _example)
            assert len(servers) == 2
            assert "github" in servers
            assert "filesystem" in servers
            assert "_example" not in servers

    def test_refresh_mcp_servers(self):
        """Test refreshing MCP servers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp_config = Path(tmpdir) / "mcp.json"
            mcp_config.write_text(json.dumps({
                "mcpServers": {
                    "github": {
                        "type": "http",
                        "url": "https://api.github.com/mcp",
                    },
                }
            }))

            config = SkillsCacheConfig(mcp_configs=[str(mcp_config)])
            cache = SkillsCache(config)

            changes = cache.refresh_mcp_servers()

            assert "github" in changes
            assert changes["github"] == "added"

    def test_get_mcp_server(self):
        """Test getting MCP server by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp_config = Path(tmpdir) / "mcp.json"
            mcp_config.write_text(json.dumps({
                "mcpServers": {
                    "github": {"type": "http", "url": "https://test.com"},
                }
            }))

            config = SkillsCacheConfig(mcp_configs=[str(mcp_config)])
            cache = SkillsCache(config)
            cache.refresh_mcp_servers()

            server = cache.get_mcp_server("github")
            assert server is not None
            assert server.name == "github"

            assert cache.get_mcp_server("nonexistent") is None

    def test_get_all_mcp_servers(self):
        """Test getting all MCP servers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mcp_config = Path(tmpdir) / "mcp.json"
            mcp_config.write_text(json.dumps({
                "mcpServers": {
                    "server1": {"type": "http", "url": "https://test1.com"},
                    "server2": {"type": "http", "url": "https://test2.com"},
                }
            }))

            config = SkillsCacheConfig(mcp_configs=[str(mcp_config)])
            cache = SkillsCache(config)
            cache.refresh_mcp_servers()

            servers = cache.get_all_mcp_servers()
            assert len(servers) == 2

    def test_get_skills_summary(self):
        """Test getting skills summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["pai-fabric", "pai-code", "anthropic-review", "local"]:
                skill_dir = Path(tmpdir) / name
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text(f"# {name}")

            config = SkillsCacheConfig(skill_paths=[tmpdir])
            cache = SkillsCache(config)
            cache.refresh_skills()

            summary = cache.get_skills_summary()

            assert summary["total"] == 4
            assert summary["by_source"]["pai"] == 2
            assert summary["by_source"]["anthropic"] == 1
            assert summary["by_source"]["local"] == 1

    def test_cache_persistence(self):
        """Test cache file persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"
            skill_dir = Path(tmpdir) / "skills" / "test-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("# Test")

            # Create cache and save
            config = SkillsCacheConfig(
                skill_paths=[str(Path(tmpdir) / "skills")],
                cache_file=str(cache_file),
            )
            cache1 = SkillsCache(config)
            cache1.refresh_skills()

            assert cache_file.exists()

            # Create new cache and load
            cache2 = SkillsCache(config)
            assert len(cache2._skills) == 1
            assert "test-skill" in cache2._skills

    @pytest.mark.asyncio
    async def test_refresh_all(self):
        """Test refreshing all caches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "skills" / "test-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("# Test")

            mcp_config = Path(tmpdir) / "mcp.json"
            mcp_config.write_text(json.dumps({
                "mcpServers": {
                    "test": {"type": "http", "url": "https://test.com"},
                }
            }))

            config = SkillsCacheConfig(
                skill_paths=[str(Path(tmpdir) / "skills")],
                mcp_configs=[str(mcp_config)],
            )
            cache = SkillsCache(config)

            result = await cache.refresh_all()

            assert "test-skill" in result["skills"]
            assert "test" in result["mcp_servers"]

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping background task."""
        config = SkillsCacheConfig(refresh_interval=1)
        cache = SkillsCache(config)

        await cache.start()
        assert cache._running is True
        assert cache._task is not None

        await asyncio.sleep(0.1)

        await cache.stop()
        assert cache._running is False

    def test_notify_callback(self):
        """Test notification callback."""
        notifications = []

        def callback(event_type, data):
            notifications.append((event_type, data))

        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "test-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# Test")

            config = SkillsCacheConfig(
                skill_paths=[tmpdir],
                notify_callback=callback,
            )
            cache = SkillsCache(config)
            cache.refresh_skills()

            assert len(notifications) == 1
            assert notifications[0][0] == "skills_changed"


class TestGlobalSkillsCache:
    """Tests for global skills cache functions."""

    def test_init_skills_cache_with_config(self):
        """Test initializing with SkillsCacheConfig."""
        config = SkillsCacheConfig(refresh_interval=1800)
        cache = init_skills_cache(config)

        assert cache.config.refresh_interval == 1800
        assert get_skills_cache() is cache

    def test_init_skills_cache_with_dict(self):
        """Test initializing with dict config."""
        cache = init_skills_cache({"refresh_interval": 1800})

        assert cache.config.refresh_interval == 1800

    def test_get_skills_cache_none(self):
        """Test getting cache when not initialized."""
        import baton.plugins.skills_cache as module

        module._cache = None
        assert get_skills_cache() is None

    @pytest.mark.asyncio
    async def test_start_stop_skills_cache(self):
        """Test global start/stop functions."""
        config = SkillsCacheConfig(refresh_interval=1)
        init_skills_cache(config)

        await start_skills_cache()
        cache = get_skills_cache()
        assert cache._running is True

        await stop_skills_cache()
        assert cache._running is False

    @pytest.mark.asyncio
    async def test_start_skills_cache_not_initialized(self):
        """Test starting when not initialized."""
        import baton.plugins.skills_cache as module

        module._cache = None
        # Should not raise
        await start_skills_cache()
