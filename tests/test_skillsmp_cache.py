"""Tests for skillsmp_cache module."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baton.plugins.skillsmp_cache import (
    SkillsMPCache,
    SkillsMPCacheConfig,
    SkillsMPSkill,
    get_skillsmp_cache,
    init_skillsmp_cache,
    start_skillsmp_cache,
    stop_skillsmp_cache,
    SKILLSMP_API_BASE,
)


class TestSkillsMPSkill:
    """Tests for SkillsMPSkill dataclass."""

    def test_basic_skill(self):
        """Test creating basic skill."""
        skill = SkillsMPSkill(
            id="skill-123",
            name="Test Skill",
            description="A test skill",
        )

        assert skill.id == "skill-123"
        assert skill.name == "Test Skill"
        assert skill.stars == 0
        assert skill.tags == []

    def test_skill_with_all_fields(self):
        """Test skill with all fields."""
        skill = SkillsMPSkill(
            id="skill-123",
            name="Test Skill",
            description="A test skill",
            author="test-author",
            repo_url="https://github.com/test/skill",
            stars=100,
            category="development",
            tags=["python", "testing"],
            created_at="2024-01-01",
            updated_at="2024-06-01",
        )

        assert skill.author == "test-author"
        assert skill.stars == 100
        assert "python" in skill.tags

    def test_to_dict(self):
        """Test converting to dict."""
        skill = SkillsMPSkill(
            id="skill-123",
            name="Test",
            description="Test",
            stars=50,
        )

        data = skill.to_dict()
        assert data["id"] == "skill-123"
        assert data["stars"] == 50

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "id": "skill-123",
            "name": "Test",
            "description": "Test desc",
            "stars": 50,
            "tags": ["tag1"],
        }

        skill = SkillsMPSkill.from_dict(data)
        assert skill.id == "skill-123"
        assert skill.stars == 50

    def test_from_api_response(self):
        """Test creating from API response."""
        data = {
            "id": "skill-123",
            "name": "Test Skill",
            "description": "A test",
            "author": "someone",
            "stars": 25,
            "tags": ["ai"],
        }

        skill = SkillsMPSkill.from_api_response(data)
        assert skill.id == "skill-123"
        assert skill.name == "Test Skill"

    def test_from_api_response_alternate_fields(self):
        """Test creating from API with alternate field names."""
        data = {
            "_id": "alt-123",
            "title": "Alt Skill",
            "summary": "Alt description",
            "owner": "owner-name",
            "repository": "https://github.com/test",
            "starCount": 100,
            "type": "utility",
            "keywords": ["keyword1"],
        }

        skill = SkillsMPSkill.from_api_response(data)
        assert skill.id == "alt-123"
        assert skill.name == "Alt Skill"
        assert skill.description == "Alt description"
        assert skill.author == "owner-name"
        assert skill.stars == 100


class TestSkillsMPCacheConfig:
    """Tests for SkillsMPCacheConfig."""

    def test_config_with_api_key(self):
        """Test config with API key."""
        config = SkillsMPCacheConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.refresh_interval == 86400
        assert config.page_size == 100

    def test_config_custom_values(self):
        """Test config with custom values."""
        config = SkillsMPCacheConfig(
            api_key="test-key",
            cache_file="/custom/cache.json",
            refresh_interval=43200,
        )

        assert config.cache_file == "/custom/cache.json"
        assert config.refresh_interval == 43200


class TestSkillsMPCache:
    """Tests for SkillsMPCache class."""

    def test_init(self):
        """Test cache initialization."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        assert cache._skills == {}
        assert cache._running is False

    def test_get_headers(self):
        """Test API headers."""
        config = SkillsMPCacheConfig(api_key="test-key-123")
        cache = SkillsMPCache(config)

        headers = cache._get_headers()
        assert headers["Authorization"] == "Bearer test-key-123"

    def test_cache_persistence(self):
        """Test cache file persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"

            config = SkillsMPCacheConfig(
                api_key="test-key",
                cache_file=str(cache_file),
            )

            # Create cache and add skill
            cache1 = SkillsMPCache(config)
            cache1._skills["skill-1"] = SkillsMPSkill(
                id="skill-1",
                name="Test",
                description="Test",
            )
            cache1._last_updated = 1000.0
            cache1._save_cache()

            assert cache_file.exists()

            # Create new cache and load
            cache2 = SkillsMPCache(config)
            assert len(cache2._skills) == 1
            assert "skill-1" in cache2._skills
            assert cache2._last_updated == 1000.0

    def test_get_skill(self):
        """Test getting skill by ID."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        cache._skills["skill-1"] = SkillsMPSkill(
            id="skill-1", name="Test", description="Test"
        )

        assert cache.get_skill("skill-1") is not None
        assert cache.get_skill("nonexistent") is None

    def test_get_all_skills(self):
        """Test getting all skills."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        for i in range(5):
            cache._skills[f"skill-{i}"] = SkillsMPSkill(
                id=f"skill-{i}", name=f"Skill {i}", description="Test"
            )

        skills = cache.get_all_skills()
        assert len(skills) == 5

    def test_get_by_category(self):
        """Test getting skills by category."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        cache._skills["s1"] = SkillsMPSkill(
            id="s1", name="S1", description="", category="dev"
        )
        cache._skills["s2"] = SkillsMPSkill(
            id="s2", name="S2", description="", category="dev"
        )
        cache._skills["s3"] = SkillsMPSkill(
            id="s3", name="S3", description="", category="ops"
        )

        dev_skills = cache.get_by_category("dev")
        assert len(dev_skills) == 2

    def test_get_by_author(self):
        """Test getting skills by author."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        cache._skills["s1"] = SkillsMPSkill(
            id="s1", name="S1", description="", author="alice"
        )
        cache._skills["s2"] = SkillsMPSkill(
            id="s2", name="S2", description="", author="bob"
        )

        alice_skills = cache.get_by_author("alice")
        assert len(alice_skills) == 1
        assert alice_skills[0].id == "s1"

    def test_get_top_skills(self):
        """Test getting top skills by stars."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        cache._skills["s1"] = SkillsMPSkill(
            id="s1", name="S1", description="", stars=10
        )
        cache._skills["s2"] = SkillsMPSkill(
            id="s2", name="S2", description="", stars=100
        )
        cache._skills["s3"] = SkillsMPSkill(
            id="s3", name="S3", description="", stars=50
        )

        top = cache.get_top_skills(limit=2)
        assert len(top) == 2
        assert top[0].stars == 100
        assert top[1].stars == 50

    def test_get_categories(self):
        """Test getting category counts."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        cache._skills["s1"] = SkillsMPSkill(
            id="s1", name="S1", description="", category="dev"
        )
        cache._skills["s2"] = SkillsMPSkill(
            id="s2", name="S2", description="", category="dev"
        )
        cache._skills["s3"] = SkillsMPSkill(
            id="s3", name="S3", description="", category="ops"
        )

        categories = cache.get_categories()
        assert categories["dev"] == 2
        assert categories["ops"] == 1

    def test_get_summary(self):
        """Test getting cache summary."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        cache._skills["s1"] = SkillsMPSkill(
            id="s1", name="S1", description="", category="dev"
        )
        cache._last_updated = 1000.0

        summary = cache.get_summary()
        assert summary["total_skills"] == 1
        assert summary["last_updated"] == 1000.0

    def test_needs_refresh(self):
        """Test refresh check."""
        config = SkillsMPCacheConfig(api_key="test-key", refresh_interval=3600)
        cache = SkillsMPCache(config)

        # Never updated
        assert cache.needs_refresh() is True

        # Recently updated
        import time
        cache._last_updated = time.time()
        assert cache.needs_refresh() is False

        # Updated long ago
        cache._last_updated = time.time() - 7200
        assert cache.needs_refresh() is True

    @pytest.mark.asyncio
    async def test_search(self):
        """Test local search."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        cache._skills["s1"] = SkillsMPSkill(
            id="s1",
            name="Code Review Helper",
            description="Helps review code",
            tags=["review", "quality"],
        )
        cache._skills["s2"] = SkillsMPSkill(
            id="s2",
            name="Test Runner",
            description="Runs tests",
            tags=["testing"],
        )
        cache._skills["s3"] = SkillsMPSkill(
            id="s3",
            name="Deploy Tool",
            description="Deploy code to production",
            tags=["deploy"],
        )

        # Search by name
        results = await cache.search("code")
        assert len(results) == 2  # Code Review Helper and Deploy (deploy code)

        # Search by tag
        results = await cache.search("testing")
        assert len(results) == 1
        assert results[0].id == "s2"

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping background task."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        await cache.start()
        assert cache._running is True

        await cache.stop()
        assert cache._running is False


class TestGlobalSkillsMPCache:
    """Tests for global cache functions."""

    def test_init_skillsmp_cache_with_api_key(self):
        """Test initializing with API key."""
        cache = init_skillsmp_cache(api_key="test-key")

        assert cache.config.api_key == "test-key"
        assert get_skillsmp_cache() is cache

    def test_init_skillsmp_cache_with_config(self):
        """Test initializing with config."""
        config = SkillsMPCacheConfig(api_key="test-key", refresh_interval=1800)
        cache = init_skillsmp_cache(config=config)

        assert cache.config.refresh_interval == 1800

    def test_init_skillsmp_cache_with_dict(self):
        """Test initializing with dict."""
        cache = init_skillsmp_cache(config={
            "api_key": "test-key",
            "refresh_interval": 1800,
        })

        assert cache.config.api_key == "test-key"
        assert cache.config.refresh_interval == 1800

    def test_init_without_api_key_raises(self):
        """Test that init without API key raises."""
        with pytest.raises(ValueError, match="api_key required"):
            init_skillsmp_cache()

    def test_get_skillsmp_cache_none(self):
        """Test getting cache when not initialized."""
        import baton.plugins.skillsmp_cache as module

        module._cache = None
        assert get_skillsmp_cache() is None

    @pytest.mark.asyncio
    async def test_start_stop_skillsmp_cache(self):
        """Test global start/stop functions."""
        init_skillsmp_cache(api_key="test-key")

        await start_skillsmp_cache()
        cache = get_skillsmp_cache()
        assert cache._running is True

        await stop_skillsmp_cache()
        assert cache._running is False

    @pytest.mark.asyncio
    async def test_start_skillsmp_cache_not_initialized(self):
        """Test starting when not initialized."""
        import baton.plugins.skillsmp_cache as module

        module._cache = None
        # Should not raise
        await start_skillsmp_cache()


class TestSkillsMPCacheAPI:
    """Tests for API interactions (mocked)."""

    @pytest.mark.asyncio
    async def test_fetch_page(self):
        """Test fetching a page from API."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "skills": [
                    {"id": "s1", "name": "Skill 1", "description": "Test"},
                    {"id": "s2", "name": "Skill 2", "description": "Test"},
                ],
                "total": 100,
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(cache, "_ensure_client") as mock_client:
            mock_http_client = AsyncMock()
            mock_http_client.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http_client

            skills, total = await cache._fetch_page(query="test", page=1)

            assert len(skills) == 2
            assert total == 100

    @pytest.mark.asyncio
    async def test_ai_search(self):
        """Test AI semantic search."""
        config = SkillsMPCacheConfig(api_key="test-key")
        cache = SkillsMPCache(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {
                "skills": [
                    {"id": "s1", "name": "AI Skill", "description": "AI-powered"},
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(cache, "_ensure_client") as mock_client:
            mock_http_client = AsyncMock()
            mock_http_client.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_http_client

            results = await cache.ai_search("find AI tools")

            assert len(results) == 1
            assert results[0].name == "AI Skill"
