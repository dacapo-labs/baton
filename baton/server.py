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
from .plugins.keepalive import (
    KeepaliveDaemon,
    init_keepalive_daemon,
    get_keepalive_daemon,
)
from .plugins.cli_auth import CLIAuthManager


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
keepalive: KeepaliveDaemon | None = None
cli_auth: CLIAuthManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global config, auth, logger, judge, fanout, zones, skills_cache, skillsmp_cache, keepalive, cli_auth

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

    # Initialize CLI auth manager (claude, codex, gemini CLIs)
    cli_auth = CLIAuthManager(config.get("providers", {}))

    # Initialize keepalive daemon for credential monitoring
    keepalive_config = config.get("keepalive", {})
    if keepalive_config.get("enabled", True):
        keepalive = init_keepalive_daemon(
            keepalive_config,
            cli_auth=cli_auth,
        )
        keepalive.start()

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
    if keepalive:
        keepalive.stop()
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


class AuthSetupRequest(BaseModel):
    """Request for auth setup."""

    provider: str
    auth_method: str = "api"  # api, oauth, bedrock, vertex, etc.


@app.post("/auth/setup")
async def setup_auth(request: AuthSetupRequest):
    """Setup authentication for a provider and return environment variables.

    This endpoint returns the environment variables needed to use a specific
    provider with a specific auth method. Maestro can eval this output.

    Returns:
        env_vars: Dict of environment variables to set
        export_cmd: Shell command to export the variables
    """
    if not auth:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    provider = request.provider.lower()
    method = request.auth_method.lower()

    env_vars = {}
    errors = []

    # Map providers to their environment variables
    provider_env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    if method == "api":
        # Get API key from Bitwarden
        api_key = auth.get_api_key(provider)
        if api_key:
            env_var = provider_env_map.get(provider, f"{provider.upper()}_API_KEY")
            env_vars[env_var] = api_key
        else:
            errors.append(f"No API key found for {provider}")

    elif method == "oauth":
        # Check CLI OAuth status
        if cli_auth:
            cli_provider = cli_auth.get_provider(provider)
            if cli_provider and cli_provider.is_installed():
                status = await cli_provider.check_auth()
                if status and status.authenticated:
                    env_vars["_AUTH_METHOD"] = "oauth"
                    env_vars["_AUTH_PROVIDER"] = provider
                    env_vars["_AUTH_USER"] = status.user or ""
                else:
                    errors.append(f"OAuth not authenticated for {provider}")
            else:
                errors.append(f"CLI not installed for {provider}")
        else:
            errors.append("CLI auth not initialized")

    elif method in ("bedrock", "vertex", "azure"):
        # Cloud provider auth - check credentials
        if method == "bedrock":
            aws_creds = auth.get_aws_credentials()
            if aws_creds:
                if aws_creds.get("access_key_id"):
                    env_vars["AWS_ACCESS_KEY_ID"] = aws_creds["access_key_id"]
                if aws_creds.get("secret_access_key"):
                    env_vars["AWS_SECRET_ACCESS_KEY"] = aws_creds["secret_access_key"]
                if aws_creds.get("session_token"):
                    env_vars["AWS_SESSION_TOKEN"] = aws_creds["session_token"]
                env_vars["AWS_DEFAULT_REGION"] = config.get("providers", {}).get("aws", {}).get("bedrock", {}).get("region", "us-east-1")
            else:
                errors.append("No AWS credentials found")

        elif method == "vertex":
            gcp_creds = auth.get_gcp_credentials()
            if gcp_creds:
                env_vars["GOOGLE_CLOUD_PROJECT"] = config.get("providers", {}).get("google", {}).get("vertex", {}).get("project", "")
                env_vars["GOOGLE_CLOUD_REGION"] = config.get("providers", {}).get("google", {}).get("vertex", {}).get("region", "us-central1")
            else:
                errors.append("No GCP credentials found")

    else:
        errors.append(f"Unknown auth method: {method}")

    # Generate shell export command
    export_lines = [f'export {k}="{v}"' for k, v in env_vars.items()]
    export_cmd = "\n".join(export_lines)

    return {
        "success": len(errors) == 0,
        "provider": provider,
        "auth_method": method,
        "env_vars": env_vars,
        "export_cmd": export_cmd,
        "errors": errors,
    }


@app.get("/auth/status/{provider}")
async def get_auth_status(provider: str):
    """Get authentication status for a specific provider.

    Returns status for all available auth methods (api, oauth, cloud).
    """
    if not auth:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    provider = provider.lower()
    statuses = {}

    # Check API key
    api_key = auth.get_api_key(provider)
    statuses["api"] = {
        "available": api_key is not None,
        "configured": api_key is not None,
    }

    # Check OAuth via CLI
    if cli_auth:
        try:
            cli_provider = cli_auth.get_provider(provider)
            if cli_provider and cli_provider.is_installed():
                oauth_status = await cli_provider.check_auth()
                statuses["oauth"] = {
                    "available": oauth_status.authenticated,
                    "user": oauth_status.user,
                    "plan": oauth_status.plan,
                    "ttl_seconds": oauth_status.ttl_seconds,
                }
            else:
                statuses["oauth"] = {"available": False, "installed": False}
        except Exception:
            statuses["oauth"] = {"available": False, "error": "Check failed"}

    # Check cloud credentials
    if provider in ("anthropic", "amazon"):
        aws_creds = auth.get_aws_credentials()
        statuses["bedrock"] = {
            "available": aws_creds is not None,
        }

    if provider in ("google", "gemini"):
        gcp_creds = auth.get_gcp_credentials()
        statuses["vertex"] = {
            "available": gcp_creds is not None,
        }

    return {
        "provider": provider,
        "methods": statuses,
        "recommended": _get_recommended_method(statuses),
    }


def _get_recommended_method(statuses: dict) -> str:
    """Get recommended auth method based on availability."""
    # Prefer OAuth (uses subscription), then API, then cloud
    if statuses.get("oauth", {}).get("available"):
        return "oauth"
    if statuses.get("api", {}).get("available"):
        return "api"
    if statuses.get("bedrock", {}).get("available"):
        return "bedrock"
    if statuses.get("vertex", {}).get("available"):
        return "vertex"
    return "none"


@app.get("/auth/env")
async def get_auth_env(provider: str | None = None):
    """Get all authentication environment variables.

    If provider is specified, only return vars for that provider.
    Otherwise, return all configured provider env vars.
    """
    if not auth:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    if provider:
        # Get env vars for specific provider
        api_key = auth.get_api_key(provider.lower())
        if not api_key:
            return {"env_vars": {}, "provider": provider}

        provider_env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        env_var = provider_env_map.get(provider.lower(), f"{provider.upper()}_API_KEY")
        return {"env_vars": {env_var: api_key}, "provider": provider}

    # Return all configured env vars
    env_vars = auth.export_env_vars()
    return {"env_vars": env_vars}


# =========================================================================
# Health Check Endpoints
# =========================================================================


@app.get("/healthz")
async def healthz():
    """Simple health check for liveness probe."""
    return {"status": "ok"}


@app.get("/healthz/creds")
async def healthz_creds(provider: str | None = None):
    """Check if credentials are ready for use.

    Returns 200 if credentials are available and valid.
    Returns 503 if credentials are expired or unavailable.
    """
    if not keepalive:
        # No keepalive, just check basic auth
        if provider:
            api_key = auth.get_api_key(provider) if auth else None
            if api_key:
                return {"ready": True, "provider": provider, "method": "api"}
            return {"ready": False, "provider": provider, "error": "No credentials"}
        return {"ready": True, "message": "Keepalive not enabled"}

    # Get keepalive status
    health = keepalive.get_health()

    if provider:
        # Check specific provider
        statuses = keepalive.get_status().get("credentials", {})
        provider_statuses = {k: v for k, v in statuses.items() if k.startswith(provider)}

        if not provider_statuses:
            # Check API key as fallback
            api_key = auth.get_api_key(provider) if auth else None
            if api_key:
                return {"ready": True, "provider": provider, "method": "api"}
            raise HTTPException(status_code=503, detail=f"No credentials for {provider}")

        # Check if any method is healthy
        for key, status in provider_statuses.items():
            if status.get("status") in ("valid", "expiring"):
                return {
                    "ready": True,
                    "provider": provider,
                    "method": key.split("/")[-1],
                    "ttl_seconds": status.get("ttl_seconds"),
                }

        raise HTTPException(status_code=503, detail=f"Credentials expired for {provider}")

    # Check overall health
    if health["healthy"]:
        return {"ready": True, "healthy_count": health["healthy_count"]}

    raise HTTPException(
        status_code=503,
        detail=f"Credentials unhealthy: {health['unhealthy']}",
    )


# =========================================================================
# Keepalive Endpoints
# =========================================================================


@app.get("/keepalive/status")
async def keepalive_status():
    """Get status of all monitored credentials."""
    if not keepalive:
        return {"enabled": False, "message": "Keepalive not configured"}

    return {
        "enabled": True,
        **keepalive.get_status(),
    }


@app.get("/keepalive/health")
async def keepalive_health():
    """Get health summary of credentials."""
    if not keepalive:
        return {"enabled": False, "healthy": True}

    return {
        "enabled": True,
        **keepalive.get_health(),
    }


@app.post("/keepalive/check")
async def keepalive_check():
    """Force a credential check."""
    if not keepalive:
        raise HTTPException(status_code=503, detail="Keepalive not configured")

    statuses = await keepalive.check_all()
    return {
        "checked": True,
        "credentials": {k: v.to_dict() for k, v in statuses.items()},
    }


@app.post("/keepalive/refresh")
async def keepalive_refresh(provider: str | None = None):
    """Force refresh of expiring credentials."""
    if not keepalive:
        raise HTTPException(status_code=503, detail="Keepalive not configured")

    # First check all to update statuses
    await keepalive.check_all()

    # Then refresh expiring ones
    results = await keepalive.refresh_expiring()

    if provider:
        # Filter to specific provider
        results = {k: v for k, v in results.items() if k.startswith(provider)}

    return {
        "refreshed": results,
        "success": all(results.values()) if results else True,
    }


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
