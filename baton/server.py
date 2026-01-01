"""Baton Server - LiteLLM proxy with custom plugins."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import litellm
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import load_config
from .plugins import BatonAuth, BatonFanout, BatonJudge, BatonLogger, BatonZones
from .plugins.fanout import FanoutMode
from .plugins.skills_cache import (
    SkillsCache,
    SkillsCacheConfig,
    init_skills_cache,
    get_skills_cache,
    start_skills_cache,
    stop_skills_cache,
)
from .plugins.skillsmp_cache import (
    SkillsMPCache,
    SkillsMPCacheConfig,
    init_skillsmp_cache,
    get_skillsmp_cache,
    start_skillsmp_cache,
    stop_skillsmp_cache,
)


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str | None = None
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    fanout: str | None = None
    models: list[str] | None = None


config: dict[str, Any] = {}
auth: BatonAuth | None = None
logger: BatonLogger | None = None
judge: BatonJudge | None = None
fanout: BatonFanout | None = None
zones: BatonZones | None = None
skills_cache: SkillsCache | None = None
skillsmp_cache: SkillsMPCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global config, auth, logger, judge, fanout, zones, skills_cache, skillsmp_cache

    config = load_config()

    auth = BatonAuth(config)
    logger = BatonLogger(config)
    judge = BatonJudge(config, logger)
    fanout = BatonFanout(config, judge)
    zones = BatonZones(config)

    env_vars = auth.export_env_vars()
    for key, value in env_vars.items():
        os.environ[key] = value

    litellm.drop_params = True
    litellm.set_verbose = config.get("debug", False)

    # Initialize skills cache (local skills)
    skills_config = config.get("skills", {})
    if skills_config.get("enabled", True):
        skills_cache = init_skills_cache(SkillsCacheConfig(
            skill_paths=skills_config.get("skill_paths", []),
            cache_file=skills_config.get("cache_file"),
            refresh_interval=skills_config.get("refresh_interval", 3600),
        ))
        await start_skills_cache()

    # Initialize SkillsMP cache (marketplace)
    skillsmp_config = config.get("skillsmp", {})
    if skillsmp_config.get("enabled", True):
        # Get API key from Bitwarden
        skillsmp_api_key = auth.get_api_key("skillsmp")
        if skillsmp_api_key:
            skillsmp_cache = init_skillsmp_cache(SkillsMPCacheConfig(
                api_key=skillsmp_api_key,
                cache_file=skillsmp_config.get("cache_file", "~/.baton/skillsmp_cache.json"),
                refresh_interval=skillsmp_config.get("refresh_interval", 86400),
                trusted_authors=set(skillsmp_config.get("trusted_authors", [])),
                known_orgs=set(skillsmp_config.get("known_orgs", [])),
            ))
            await start_skillsmp_cache()

    yield

    # Cleanup
    await stop_skills_cache()
    await stop_skillsmp_cache()


app = FastAPI(
    title="Baton",
    description="AI proxy gateway with multi-model fan-out and zone-aware routing",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint with provider status."""
    provider_status = {}

    providers = ["anthropic", "openai", "google"]
    for provider in providers:
        key = auth.get_api_key(provider) if auth else None
        provider_status[provider] = "configured" if key else "not_configured"

    zone = zones.get_current_zone() if zones else None

    return {
        "status": "healthy",
        "zone": zone,
        "session": zones.get_current_session() if zones else None,
        "providers": provider_status,
        "log_stats": logger.get_log_stats() if logger else None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, raw_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    request_id = str(uuid4())
    start_time = time.perf_counter()

    zone = raw_request.headers.get("X-Maestro-Zone") or (zones.get_current_zone() if zones else None)
    session = raw_request.headers.get("X-Maestro-Session") or (zones.get_current_session() if zones else None)

    model, params = zones.apply_zone_defaults(
        request.model,
        {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        },
        zone,
    ) if zones else (request.model or "claude-sonnet-4-20250514", {})

    params = {k: v for k, v in params.items() if v is not None}

    if logger:
        await logger.log_request(
            request_id=request_id,
            model=model,
            messages=request.messages,
            params=params,
            zone=zone,
            session=session,
            fanout_mode=request.fanout,
        )

    try:
        if request.fanout or request.models:
            fanout_mode = FanoutMode(request.fanout) if request.fanout else FanoutMode.FIRST
            models = request.models or fanout.resolve_alias(model) if fanout else [model]

            if zones:
                models = zones.filter_models(models, zone)

            if not models:
                raise HTTPException(
                    status_code=400,
                    detail="No allowed models for this zone",
                )

            result = await fanout.execute(
                models=models,
                messages=request.messages,
                params=params,
                mode=fanout_mode,
                request_id=request_id,
            )

            if logger:
                await logger.log_fanout(
                    request_id=request_id,
                    mode=fanout_mode.value,
                    models=models,
                    results=result.get("all_results", []),
                    selected_model=result.get("selected_model"),
                )

            if "response" in result:
                return result["response"]
            elif "responses" in result:
                return {
                    "id": request_id,
                    "object": "chat.completion.fanout",
                    "responses": result["responses"],
                }
            else:
                return result

        if request.stream:
            async def generate():
                async for chunk in await litellm.acompletion(
                    model=model,
                    messages=request.messages,
                    stream=True,
                    **params,
                ):
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
            )

        response = await litellm.acompletion(
            model=model,
            messages=request.messages,
            **params,
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        if logger:
            await logger.log_response(
                request_id=request_id,
                model=model,
                response=response.model_dump() if hasattr(response, "model_dump") else dict(response),
                latency_ms=latency_ms,
                tokens_in=response.usage.prompt_tokens if response.usage else None,
                tokens_out=response.usage.completion_tokens if response.usage else None,
            )

        return response

    except Exception as e:
        if logger:
            await logger.log_error(
                request_id=request_id,
                error_type=type(e).__name__,
                error_message=str(e),
                model=model,
            )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: Request):
    """Legacy completions endpoint - redirect to chat."""
    body = await request.json()
    chat_request = ChatRequest(
        model=body.get("model"),
        messages=[{"role": "user", "content": body.get("prompt", "")}],
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
        stream=body.get("stream", False),
    )
    return await chat_completions(chat_request, request)


@app.get("/v1/models")
async def list_models():
    """List available models."""
    aliases = config.get("aliases", {})

    models = []
    for alias, model_list in aliases.items():
        models.append({
            "id": alias,
            "object": "model",
            "owned_by": "baton",
            "type": "alias",
            "models": model_list,
        })

    return {"object": "list", "data": models}


@app.post("/auth/refresh")
async def refresh_auth():
    """Refresh all provider credentials from Bitwarden."""
    if not auth:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    results = await auth.refresh_all_keys()
    return {"refreshed": results}


@app.post("/auth/save")
async def save_auth():
    """Save BW session for later restore."""
    if not auth:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    success = auth.save_session()
    return {"saved": success}


@app.post("/auth/restore")
async def restore_auth():
    """Restore BW session from saved file."""
    if not auth:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    success = auth.restore_session()
    if success:
        env_vars = auth.export_env_vars()
        for key, value in env_vars.items():
            os.environ[key] = value

    return {"restored": success}


# =========================================================================
# Skills Endpoints
# =========================================================================


@app.get("/skills")
async def list_skills():
    """List all locally installed skills."""
    cache = get_skills_cache()
    if not cache:
        return {"skills": [], "total": 0}

    skills = cache.get_all_skills()
    return {
        "skills": [
            {
                "name": s.name,
                "path": s.path,
                "title": s.metadata.get("title", s.name),
            }
            for s in skills
        ],
        "total": len(skills),
    }


@app.get("/skills/search")
async def search_skills(q: str, limit: int = 20):
    """Search locally installed skills."""
    cache = get_skills_cache()
    if not cache:
        return {"results": [], "query": q}

    results = cache.search_skills(q)[:limit]
    return {
        "results": [
            {
                "name": s.name,
                "path": s.path,
                "title": s.metadata.get("title", s.name),
            }
            for s in results
        ],
        "query": q,
    }


@app.get("/skills/{skill_name}")
async def get_skill(skill_name: str):
    """Get details for a specific local skill."""
    cache = get_skills_cache()
    if not cache:
        raise HTTPException(status_code=404, detail="Skills cache not initialized")

    skill = cache.get_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

    return {
        "name": skill.name,
        "path": skill.path,
        "content": skill.content,
        "metadata": skill.metadata,
        "content_hash": skill.content_hash,
    }


@app.post("/skills/refresh")
async def refresh_skills():
    """Refresh the local skills cache."""
    cache = get_skills_cache()
    if not cache:
        raise HTTPException(status_code=500, detail="Skills cache not initialized")

    changes = await cache.refresh()
    return {"changes": changes}


# =========================================================================
# SkillsMP Marketplace Endpoints
# =========================================================================


@app.get("/skillsmp")
async def skillsmp_summary():
    """Get SkillsMP cache summary."""
    cache = get_skillsmp_cache()
    if not cache:
        return {"enabled": False, "message": "SkillsMP not configured (API key missing)"}

    return {
        "enabled": True,
        "summary": cache.get_summary(),
        "categories": cache.get_categories(),
    }


@app.get("/skillsmp/search")
async def skillsmp_search(q: str, limit: int = 20, use_ai: bool = False):
    """Search SkillsMP marketplace.

    Args:
        q: Search query
        limit: Max results to return
        use_ai: Use AI semantic search (requires API call) vs local keyword search
    """
    cache = get_skillsmp_cache()
    if not cache:
        raise HTTPException(status_code=503, detail="SkillsMP not configured")

    if use_ai:
        results = await cache.ai_search(q)
    else:
        results = await cache.search(q, limit=limit)

    return {
        "results": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "author": s.author,
                "stars": s.stars,
                "category": s.category,
                "repo_url": s.repo_url,
                "score": cache.score_skill(s),
            }
            for s in results[:limit]
        ],
        "query": q,
        "mode": "ai" if use_ai else "keyword",
    }


@app.get("/skillsmp/best")
async def skillsmp_best(limit: int = 50):
    """Get top-rated skills from SkillsMP by quality score."""
    cache = get_skillsmp_cache()
    if not cache:
        raise HTTPException(status_code=503, detail="SkillsMP not configured")

    best = cache.get_best_skills(limit=limit)
    return {
        "skills": [
            {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "author": skill.author,
                "stars": skill.stars,
                "category": skill.category,
                "repo_url": skill.repo_url,
                "score": score,
            }
            for skill, score in best
        ],
        "total": len(best),
    }


@app.get("/skillsmp/skill/{skill_id}")
async def skillsmp_get_skill(skill_id: str):
    """Get details for a specific SkillsMP skill."""
    cache = get_skillsmp_cache()
    if not cache:
        raise HTTPException(status_code=503, detail="SkillsMP not configured")

    skill = cache.get_skill(skill_id)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")

    return {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "author": skill.author,
        "stars": skill.stars,
        "category": skill.category,
        "tags": skill.tags,
        "repo_url": skill.repo_url,
        "created_at": skill.created_at,
        "updated_at": skill.updated_at,
        "score": cache.score_skill(skill),
    }


@app.post("/skillsmp/refresh")
async def skillsmp_refresh():
    """Manually refresh SkillsMP cache."""
    cache = get_skillsmp_cache()
    if not cache:
        raise HTTPException(status_code=503, detail="SkillsMP not configured")

    count = await cache.fetch_all_skills()
    return {"refreshed": True, "total_skills": count}


def run():
    """Run the server."""
    import uvicorn

    server_config = config.get("server", {})
    uvicorn.run(
        "baton.server:app",
        host=server_config.get("host", "127.0.0.1"),
        port=server_config.get("port", 4000),
        workers=server_config.get("workers", 1),
        reload=config.get("debug", False),
    )


if __name__ == "__main__":
    run()
