"""SkillsMP marketplace cache for baton.

Caches the SkillsMP directory of 38k+ skills locally for fast search
and discovery. Updates daily to stay current with new skills.

API Docs: https://skillsmp.com/docs/api
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx


SKILLSMP_API_BASE = "https://skillsmp.com/api/v1"
DEFAULT_CACHE_FILE = "~/.baton/skillsmp_cache.json"


@dataclass
class SkillsMPSkill:
    """A skill from SkillsMP marketplace."""

    id: str
    name: str
    description: str
    author: str | None = None
    repo_url: str | None = None
    stars: int = 0
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "repo_url": self.repo_url,
            "stars": self.stars,
            "category": self.category,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "raw_data": self.raw_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillsMPSkill:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            author=data.get("author"),
            repo_url=data.get("repo_url"),
            stars=data.get("stars", 0),
            category=data.get("category"),
            tags=data.get("tags", []),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            raw_data=data.get("raw_data", {}),
        )

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> SkillsMPSkill:
        """Create from raw API response data."""
        return cls(
            id=data.get("id", data.get("_id", "")),
            name=data.get("name", data.get("title", "")),
            description=data.get("description", data.get("summary", "")),
            author=data.get("author", data.get("owner", data.get("creator"))),
            repo_url=data.get("repo_url", data.get("repository", data.get("url"))),
            stars=data.get("stars", data.get("starCount", 0)),
            category=data.get("category", data.get("type")),
            tags=data.get("tags", data.get("keywords", [])),
            created_at=data.get("created_at", data.get("createdAt")),
            updated_at=data.get("updated_at", data.get("updatedAt")),
            raw_data=data,
        )


@dataclass
class SkillsMPCacheConfig:
    """Configuration for SkillsMP cache."""

    api_key: str
    cache_file: str = DEFAULT_CACHE_FILE
    refresh_interval: int = 86400  # 24 hours
    page_size: int = 100  # Max allowed by API
    notify_callback: Callable[[str, Any], None] | None = None


class SkillsMPCache:
    """Cache for SkillsMP marketplace skills."""

    def __init__(self, config: SkillsMPCacheConfig):
        self.config = config
        self._skills: dict[str, SkillsMPSkill] = {}
        self._last_updated: float = 0
        self._running = False
        self._task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None

        # Load from cache file if exists
        self._load_cache()

    def _get_headers(self) -> dict[str, str]:
        """Get API request headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _load_cache(self) -> None:
        """Load cache from file."""
        cache_path = Path(self.config.cache_file).expanduser()
        if not cache_path.exists():
            return

        try:
            with open(cache_path) as f:
                data = json.load(f)

            self._last_updated = data.get("last_updated", 0)
            for skill_data in data.get("skills", []):
                skill = SkillsMPSkill.from_dict(skill_data)
                self._skills[skill.id] = skill

        except (json.JSONDecodeError, KeyError, OSError):
            pass  # Start fresh if cache is corrupted

    def _save_cache(self) -> None:
        """Save cache to file."""
        cache_path = Path(self.config.cache_file).expanduser()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "skills": [s.to_dict() for s in self._skills.values()],
            "last_updated": self._last_updated,
            "skill_count": len(self._skills),
        }

        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=SKILLSMP_API_BASE,
                headers=self._get_headers(),
                timeout=30.0,
            )
        return self._client

    async def _fetch_page(
        self,
        query: str = "",
        page: int = 1,
        sort_by: str = "recent",
    ) -> tuple[list[dict], int]:
        """Fetch a page of skills from the API."""
        client = await self._ensure_client()

        params = {
            "q": query or "*",  # Use wildcard for all
            "page": page,
            "limit": self.config.page_size,
            "sortBy": sort_by,
        }

        response = await client.get("/skills/search", params=params)
        response.raise_for_status()

        data = response.json()

        # Handle different response formats
        if "data" in data:
            skills = data["data"].get("skills", [])
            total = data["data"].get("total", len(skills))
        else:
            skills = data.get("skills", [])
            total = data.get("total", len(skills))

        return skills, total

    async def fetch_all_skills(self) -> int:
        """Fetch all skills from SkillsMP API."""
        all_skills = []
        page = 1
        total = None

        while True:
            try:
                skills, total_count = await self._fetch_page(page=page)

                if total is None:
                    total = total_count

                if not skills:
                    break

                all_skills.extend(skills)

                # Check if we've fetched all pages
                if len(all_skills) >= total:
                    break

                page += 1

                # Small delay to be respectful
                await asyncio.sleep(0.1)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise ValueError("Invalid SkillsMP API key")
                raise

        # Update cache
        new_count = 0
        updated_count = 0

        for skill_data in all_skills:
            skill = SkillsMPSkill.from_api_response(skill_data)
            if skill.id not in self._skills:
                new_count += 1
            else:
                updated_count += 1
            self._skills[skill.id] = skill

        self._last_updated = time.time()
        self._save_cache()

        if self.config.notify_callback:
            self.config.notify_callback("skillsmp_updated", {
                "total": len(self._skills),
                "new": new_count,
                "updated": updated_count,
            })

        return len(self._skills)

    async def search(self, query: str, limit: int = 20) -> list[SkillsMPSkill]:
        """Search cached skills locally."""
        query = query.lower()
        results = []

        for skill in self._skills.values():
            score = 0

            # Name match (highest weight)
            if query in skill.name.lower():
                score += 10

            # Description match
            if query in skill.description.lower():
                score += 5

            # Tag match
            for tag in skill.tags:
                if query in tag.lower():
                    score += 3

            # Category match
            if skill.category and query in skill.category.lower():
                score += 2

            if score > 0:
                results.append((score, skill))

        # Sort by score descending, then by stars
        results.sort(key=lambda x: (x[0], x[1].stars), reverse=True)

        return [skill for _, skill in results[:limit]]

    async def ai_search(self, query: str) -> list[SkillsMPSkill]:
        """Use SkillsMP AI semantic search."""
        client = await self._ensure_client()

        response = await client.get("/skills/ai-search", params={"q": query})
        response.raise_for_status()

        data = response.json()

        if "data" in data:
            skills_data = data["data"].get("skills", [])
        else:
            skills_data = data.get("skills", [])

        return [SkillsMPSkill.from_api_response(s) for s in skills_data]

    def get_skill(self, skill_id: str) -> SkillsMPSkill | None:
        """Get a skill by ID."""
        return self._skills.get(skill_id)

    def get_all_skills(self) -> list[SkillsMPSkill]:
        """Get all cached skills."""
        return list(self._skills.values())

    def get_by_category(self, category: str) -> list[SkillsMPSkill]:
        """Get skills by category."""
        return [
            s for s in self._skills.values()
            if s.category and s.category.lower() == category.lower()
        ]

    def get_by_author(self, author: str) -> list[SkillsMPSkill]:
        """Get skills by author."""
        return [
            s for s in self._skills.values()
            if s.author and s.author.lower() == author.lower()
        ]

    def get_top_skills(self, limit: int = 100) -> list[SkillsMPSkill]:
        """Get top skills by stars."""
        sorted_skills = sorted(
            self._skills.values(),
            key=lambda s: s.stars,
            reverse=True,
        )
        return sorted_skills[:limit]

    def get_recent_skills(self, limit: int = 100) -> list[SkillsMPSkill]:
        """Get most recently updated skills."""
        sorted_skills = sorted(
            self._skills.values(),
            key=lambda s: s.updated_at or s.created_at or "",
            reverse=True,
        )
        return sorted_skills[:limit]

    def get_categories(self) -> dict[str, int]:
        """Get all categories with counts."""
        categories: dict[str, int] = {}
        for skill in self._skills.values():
            if skill.category:
                categories[skill.category] = categories.get(skill.category, 0) + 1
        return dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))

    def get_summary(self) -> dict[str, Any]:
        """Get cache summary."""
        return {
            "total_skills": len(self._skills),
            "last_updated": self._last_updated,
            "categories": len(self.get_categories()),
            "cache_age_hours": (time.time() - self._last_updated) / 3600 if self._last_updated else None,
        }

    def needs_refresh(self) -> bool:
        """Check if cache needs refresh."""
        if not self._last_updated:
            return True
        return (time.time() - self._last_updated) > self.config.refresh_interval

    async def refresh_if_needed(self) -> bool:
        """Refresh cache if needed, returns True if refreshed."""
        if self.needs_refresh():
            await self.fetch_all_skills()
            return True
        return False

    async def start(self) -> None:
        """Start background refresh task."""
        if self._running:
            return

        self._running = True

        async def refresh_loop():
            while self._running:
                try:
                    await self.refresh_if_needed()
                except Exception:
                    pass  # Log in production
                await asyncio.sleep(3600)  # Check hourly

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

        if self._client:
            await self._client.aclose()
            self._client = None


# Global instance
_cache: SkillsMPCache | None = None


def init_skillsmp_cache(
    api_key: str | None = None,
    config: SkillsMPCacheConfig | dict[str, Any] | None = None,
) -> SkillsMPCache:
    """Initialize the global SkillsMP cache."""
    global _cache

    if isinstance(config, dict):
        config = SkillsMPCacheConfig(**config)
    elif config is None:
        if not api_key:
            raise ValueError("api_key required when config not provided")
        config = SkillsMPCacheConfig(api_key=api_key)

    _cache = SkillsMPCache(config)
    return _cache


def get_skillsmp_cache() -> SkillsMPCache | None:
    """Get the global SkillsMP cache instance."""
    return _cache


async def start_skillsmp_cache() -> None:
    """Start the global SkillsMP cache background task."""
    if _cache:
        await _cache.start()


async def stop_skillsmp_cache() -> None:
    """Stop the global SkillsMP cache background task."""
    if _cache:
        await _cache.stop()
