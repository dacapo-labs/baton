"""Baton Server - LiteLLM proxy with custom plugins."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import litellm
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .config import load_config
from .plugins import BatonAuth, BatonFanout, BatonJudge, BatonLogger, BatonZones
from .plugins.fanout import FanoutMode
from .plugins.router import BatonRouter
from .plugins.guardrails import BatonGuardrails
from .plugins.rate_limits import get_rate_limiter, init_rate_limiter, RateLimitTracker
from .plugins.model_checker import get_model_checker, init_model_checker, ModelAvailabilityChecker
from .plugins.model_monitor import get_model_monitor, init_model_monitor, ModelMonitor, MonitorConfig
from .plugins.version_checker import (
    VersionChecker,
    VersionCheckerConfig,
    init_version_checker,
    get_version_checker,
)
from .plugins.repo_updater import (
    RepoUpdater,
    RepoUpdaterConfig,
    init_repo_updater,
    get_repo_updater,
    KNOWN_REPOS,
)
from .plugins.twilio import BatonTwilio
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
from .plugins.standards_monitor import (
    StandardsMonitor,
    StandardsMonitorConfig,
    init_standards_monitor,
    get_standards_monitor,
    MONITORED_REPOS,
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
router: BatonRouter | None = None
guardrails: BatonGuardrails | None = None
rate_limiter: RateLimitTracker | None = None
model_checker: ModelAvailabilityChecker | None = None
model_monitor: ModelMonitor | None = None
version_checker: VersionChecker | None = None
repo_updater: RepoUpdater | None = None
twilio: BatonTwilio | None = None
skills_cache: SkillsCache | None = None
skillsmp_cache: SkillsMPCache | None = None
keepalive: KeepaliveDaemon | None = None
cli_auth: CLIAuthManager | None = None
standards_monitor: StandardsMonitor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global config, auth, logger, judge, fanout, zones, router, guardrails, rate_limiter
    global model_checker, model_monitor, version_checker, repo_updater, twilio
    global skills_cache, skillsmp_cache, keepalive, cli_auth, standards_monitor

    config = load_config()

    auth = BatonAuth(config)
    logger = BatonLogger(config)
    judge = BatonJudge(config, logger)
    fanout = BatonFanout(config, judge)
    zones = BatonZones(config)
    router = BatonRouter(config)
    guardrails = BatonGuardrails(config, logger)
    rate_limiter = init_rate_limiter(config.get("rate_limits"))
    model_checker = init_model_checker(config.get("model_checker", {}))

    # Initialize model monitor
    monitor_config = config.get("model_monitor", {})
    if monitor_config.get("enabled", False):
        model_monitor = init_model_monitor(MonitorConfig(
            check_interval=monitor_config.get("check_interval", 3600),
            state_file=monitor_config.get("state_file"),
            check_bedrock=monitor_config.get("check_bedrock", True),
            check_vertex=monitor_config.get("check_vertex", True),
        ))
        model_monitor.start()

    # Initialize version checker
    version_config = config.get("version_checker", {})
    if version_config.get("enabled", False):
        version_checker = init_version_checker(VersionCheckerConfig(
            check_interval=version_config.get("check_interval", 86400),
            check_cli_tools=version_config.get("check_cli_tools", True),
            check_python_deps=version_config.get("check_python_deps", True),
        ))
        version_checker.start()

    # Initialize repo updater
    repo_config = config.get("repo_updater", {})
    if repo_config.get("enabled", False):
        repo_updater = init_repo_updater(RepoUpdaterConfig(
            check_interval=repo_config.get("check_interval", 3600),
            auto_update=repo_config.get("auto_update", False),
            repos_file=repo_config.get("repos_file"),
        ))
        repo_updater.start()

    # Initialize Twilio for SMS notifications
    twilio = BatonTwilio(config)

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

    # Initialize standards monitor (AI tooling standards tracker)
    standards_config = config.get("standards_monitor", {})
    if standards_config.get("enabled", True):
        github_token = auth.get_api_key("github") if auth else None
        standards_monitor = init_standards_monitor(StandardsMonitorConfig(
            check_interval=standards_config.get("check_interval", 3600),
            cache_file=standards_config.get("cache_file"),
            github_token=github_token,
            monitor_cli_releases=standards_config.get("monitor_cli_releases", True),
            monitor_specs=standards_config.get("monitor_specs", True),
            monitor_api_changes=standards_config.get("monitor_api_changes", True),
        ))
        standards_monitor.start()

    yield

    # Cleanup
    if keepalive:
        keepalive.stop()
    if model_monitor:
        model_monitor.stop()
    if version_checker:
        version_checker.stop()
    if repo_updater:
        repo_updater.stop()
    if standards_monitor:
        standards_monitor.stop()
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


# =========================================================================
# Unified Status & Config Endpoints (P5)
# =========================================================================


@app.get("/status")
async def unified_status():
    """Unified status dashboard - single endpoint for all system status.

    Returns zone, credentials, rate limits, updates, and plugin status
    in a single call. Ideal for CLI tools and dashboards.
    """
    status = {
        "version": "0.1.0",
        "uptime": None,  # Would need app start time
    }

    # Zone info
    if zones:
        status["zone"] = {
            "current": zones.get_current_zone(),
            "session": zones.get_current_session(),
        }
    else:
        status["zone"] = {"current": None, "session": None}

    # Credentials health
    if keepalive:
        health = keepalive.get_health()
        cred_statuses = keepalive.get_status().get("credentials", {})
        status["credentials"] = {
            "healthy": health["healthy"],
            "healthy_count": health["healthy_count"],
            "unhealthy_count": health["unhealthy_count"],
            "expiring_soon": [
                k for k, v in cred_statuses.items()
                if v.get("status") == "expiring"
            ],
        }
    else:
        # Fallback: check API keys
        creds = {}
        for provider in ["anthropic", "openai", "google"]:
            key = auth.get_api_key(provider) if auth else None
            creds[provider] = "configured" if key else "missing"
        status["credentials"] = {"providers": creds}

    # Rate limits summary
    if rate_limiter:
        all_status = rate_limiter.get_all_status()
        near_limit = [
            k for k, v in all_status.items()
            if v.get("rpm_usage_pct", 0) > 80 or v.get("tokens_usage_pct", 0) > 80
        ]
        status["rate_limits"] = {
            "tracked_keys": len(all_status),
            "near_limit": near_limit,
        }
    else:
        status["rate_limits"] = {"enabled": False}

    # Updates summary
    if version_checker:
        summary = version_checker.get_summary()
        status["updates"] = {
            "outdated_count": summary.get("outdated_count", 0),
            "outdated_tools": summary.get("outdated_tools", []),
        }
    else:
        status["updates"] = {"enabled": False}

    # Repos with pending updates
    if repo_updater:
        repos_summary = repo_updater.get_summary()
        status["repos"] = {
            "tracked": repos_summary.get("total", 0),
            "with_updates": repos_summary.get("with_updates", 0),
        }
    else:
        status["repos"] = {"enabled": False}

    # Skills summary
    local_count = 0
    if skills_cache:
        local_count = len(skills_cache.get_all_skills())

    mp_count = 0
    if skillsmp_cache:
        mp_summary = skillsmp_cache.get_summary()
        mp_count = mp_summary.get("total", 0)

    status["skills"] = {
        "local": local_count,
        "marketplace": mp_count,
    }

    # Plugin status
    status["plugins"] = {
        "auth": auth is not None,
        "logger": logger is not None,
        "judge": judge is not None,
        "fanout": fanout is not None,
        "zones": zones is not None,
        "router": router is not None,
        "guardrails": guardrails is not None,
        "rate_limiter": rate_limiter is not None,
        "model_checker": model_checker is not None,
        "model_monitor": model_monitor is not None,
        "version_checker": version_checker is not None,
        "repo_updater": repo_updater is not None,
        "twilio": twilio is not None and twilio.enabled,
        "keepalive": keepalive is not None,
        "skills_cache": skills_cache is not None,
        "skillsmp_cache": skillsmp_cache is not None,
        "cli_auth": cli_auth is not None,
        "standards_monitor": standards_monitor is not None,
    }

    return status


@app.get("/config")
async def get_config():
    """Get full baton configuration as JSON.

    Returns the complete config (minus secrets) for inspection
    or for CLI tools that need to read settings.
    """
    # Return config with secrets redacted
    safe_config = _redact_secrets(dict(config))
    return {"config": safe_config}


@app.get("/config/{key:path}")
async def get_config_key(key: str):
    """Get a specific config value by dot-notation path.

    Examples:
        /config/ai.default_provider
        /config/zones.personal.git.email
        /config/keepalive.enabled
    """
    parts = key.split(".")
    value = config

    for part in parts:
        if isinstance(value, dict):
            if part not in value:
                raise HTTPException(status_code=404, detail=f"Config key '{key}' not found")
            value = value[part]
        else:
            raise HTTPException(status_code=404, detail=f"Config key '{key}' not found")

    # Redact if it looks like a secret
    if _is_secret_key(key):
        value = "[REDACTED]"

    return {"key": key, "value": value}


def _redact_secrets(obj: dict, path: str = "") -> dict:
    """Recursively redact secret values from config."""
    result = {}
    for k, v in obj.items():
        full_key = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            result[k] = _redact_secrets(v, full_key)
        elif _is_secret_key(full_key) or _is_secret_key(k):
            result[k] = "[REDACTED]"
        else:
            result[k] = v
    return result


def _is_secret_key(key: str) -> bool:
    """Check if a key looks like it contains secrets."""
    secret_patterns = [
        "api_key", "apikey", "secret", "token", "password",
        "auth_token", "access_key", "private_key", "credential",
        "account_sid", "auth_token",
    ]
    key_lower = key.lower()
    return any(p in key_lower for p in secret_patterns)


@app.get("/doctor")
async def doctor_health_check():
    """Comprehensive health check - checks all systems and returns issues.

    Like `maestro doctor` but as an API endpoint. Returns structured
    results suitable for both human display and automation.
    """
    checks = []
    errors = 0
    warnings = 0

    # Check required dependencies (via checking plugin availability)
    checks.append({
        "name": "Core Plugins",
        "status": "ok" if auth and logger and zones else "error",
        "details": {
            "auth": "ok" if auth else "missing",
            "logger": "ok" if logger else "missing",
            "zones": "ok" if zones else "missing",
        }
    })
    if not auth or not logger or not zones:
        errors += 1

    # Check credentials
    cred_check = {"name": "Credentials", "status": "ok", "details": {}}
    if keepalive:
        health = keepalive.get_health()
        if not health["healthy"]:
            cred_check["status"] = "warning" if health["healthy_count"] > 0 else "error"
            cred_check["details"]["unhealthy"] = health.get("unhealthy", [])
            if health["healthy_count"] == 0:
                errors += 1
            else:
                warnings += 1
        else:
            cred_check["details"]["healthy_count"] = health["healthy_count"]
    else:
        # Check API keys directly
        for provider in ["anthropic", "openai", "google"]:
            key = auth.get_api_key(provider) if auth else None
            cred_check["details"][provider] = "configured" if key else "missing"
    checks.append(cred_check)

    # Check CLI auth (OAuth)
    cli_check = {"name": "CLI OAuth", "status": "ok", "details": {}}
    if cli_auth:
        for provider_name in ["anthropic", "openai", "google"]:
            try:
                cli_provider = cli_auth.get_provider(provider_name)
                if cli_provider:
                    if cli_provider.is_installed():
                        status = await cli_provider.check_auth()
                        if status and status.authenticated:
                            cli_check["details"][provider_name] = {
                                "authenticated": True,
                                "user": status.user,
                            }
                        else:
                            cli_check["details"][provider_name] = {"authenticated": False}
                    else:
                        cli_check["details"][provider_name] = {"installed": False}
            except Exception as e:
                cli_check["details"][provider_name] = {"error": str(e)}
    else:
        cli_check["status"] = "skipped"
        cli_check["details"]["message"] = "CLI auth not initialized"
    checks.append(cli_check)

    # Check rate limits
    rate_check = {"name": "Rate Limits", "status": "ok", "details": {}}
    if rate_limiter:
        all_status = rate_limiter.get_all_status()
        near_limit = []
        for k, v in all_status.items():
            rpm_pct = v.get("rpm_usage_pct", 0)
            tok_pct = v.get("tokens_usage_pct", 0)
            if rpm_pct > 90 or tok_pct > 90:
                near_limit.append(k)
        if near_limit:
            rate_check["status"] = "warning"
            rate_check["details"]["near_limit"] = near_limit
            warnings += 1
        rate_check["details"]["tracked"] = len(all_status)
    else:
        rate_check["status"] = "skipped"
    checks.append(rate_check)

    # Check version updates
    update_check = {"name": "Tool Updates", "status": "ok", "details": {}}
    if version_checker:
        outdated = version_checker.get_outdated()
        if outdated:
            update_check["status"] = "info"
            update_check["details"]["outdated"] = [v.name for v in outdated[:5]]
            update_check["details"]["outdated_count"] = len(outdated)
    else:
        update_check["status"] = "skipped"
    checks.append(update_check)

    # Check repo updates
    repo_check = {"name": "Repository Updates", "status": "ok", "details": {}}
    if repo_updater:
        summary = repo_updater.get_summary()
        if summary.get("with_updates", 0) > 0:
            repo_check["status"] = "info"
            repo_check["details"]["with_updates"] = summary["with_updates"]
    else:
        repo_check["status"] = "skipped"
    checks.append(repo_check)

    # Check skills
    skills_check = {"name": "Skills", "status": "ok", "details": {}}
    if skills_cache:
        skills_check["details"]["local"] = len(skills_cache.get_all_skills())
    if skillsmp_cache:
        mp_summary = skillsmp_cache.get_summary()
        skills_check["details"]["marketplace"] = mp_summary.get("total", 0)
    if not skills_cache and not skillsmp_cache:
        skills_check["status"] = "skipped"
    checks.append(skills_check)

    # Check Twilio (notifications)
    notify_check = {"name": "Notifications (Twilio)", "status": "ok", "details": {}}
    if twilio:
        if twilio.enabled:
            notify_check["details"]["configured"] = True
            notify_check["details"]["to_configured"] = twilio.is_configured()
        else:
            notify_check["status"] = "skipped"
            notify_check["details"]["message"] = "Not configured"
    else:
        notify_check["status"] = "skipped"
    checks.append(notify_check)

    # Overall status
    if errors > 0:
        overall = "unhealthy"
    elif warnings > 0:
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
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


@app.get("/auth/matrix")
async def get_auth_matrix():
    """Get comprehensive authentication status for ALL providers and methods.

    Returns a matrix showing:
    - All providers (anthropic, openai, google, aws, gcp)
    - All auth methods per provider (api, oauth, cloud)
    - Status of each (available, configured, authenticated)
    """
    if not auth:
        raise HTTPException(status_code=500, detail="Auth not initialized")

    providers = ["anthropic", "openai", "google", "aws", "gcp"]
    matrix = {}

    for provider in providers:
        provider_status = {
            "api": {"available": False, "configured": False},
            "oauth": {"available": False, "authenticated": False, "user": None},
            "cloud": {"available": False, "configured": False},
        }

        # Check API key (from Bitwarden)
        api_key = auth.get_api_key(provider)
        if api_key:
            provider_status["api"]["available"] = True
            provider_status["api"]["configured"] = True

        # Check OAuth via CLI auth
        if cli_auth:
            try:
                cli_provider = cli_auth.get_provider(provider)
                if cli_provider and cli_provider.is_installed():
                    provider_status["oauth"]["available"] = True
                    status = await cli_provider.check_auth()
                    if status and status.authenticated:
                        provider_status["oauth"]["authenticated"] = True
                        provider_status["oauth"]["user"] = status.user
                        if status.ttl_seconds:
                            provider_status["oauth"]["ttl_seconds"] = status.ttl_seconds
            except Exception:
                pass

        # Check cloud provider auth
        if provider == "aws":
            # Check AWS credentials
            import os
            if os.environ.get("AWS_PROFILE") or os.environ.get("AWS_ACCESS_KEY_ID"):
                provider_status["cloud"]["available"] = True
                provider_status["cloud"]["configured"] = True
                provider_status["cloud"]["method"] = "profile" if os.environ.get("AWS_PROFILE") else "keys"
        elif provider == "gcp":
            # Check GCP ADC
            adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
            if adc_path.exists():
                provider_status["cloud"]["available"] = True
                provider_status["cloud"]["configured"] = True
                provider_status["cloud"]["method"] = "adc"

        matrix[provider] = provider_status

    # Add Bitwarden status
    bw_session = auth._get_bw_session() if hasattr(auth, '_get_bw_session') else None
    bw_status = {
        "unlocked": bw_session is not None,
        "session_available": bw_session is not None,
    }

    # Add keepalive status if available
    keepalive_status = None
    if keepalive:
        health = keepalive.get_health()
        keepalive_status = {
            "healthy": health["healthy"],
            "healthy_count": health["healthy_count"],
            "unhealthy_count": health["unhealthy_count"],
        }

    return {
        "providers": matrix,
        "bitwarden": bw_status,
        "keepalive": keepalive_status,
        "summary": {
            "providers_with_api": sum(1 for p in matrix.values() if p["api"]["configured"]),
            "providers_with_oauth": sum(1 for p in matrix.values() if p["oauth"]["authenticated"]),
            "providers_with_cloud": sum(1 for p in matrix.values() if p["cloud"]["configured"]),
        },
    }


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


# =========================================================================
# Zones Endpoints
# =========================================================================


@app.get("/zones")
async def list_zones():
    """List all configured zones."""
    if not zones:
        return {"zones": [], "current": None}

    zone_list = []
    for name, zone_config in zones.zones.items():
        zone_list.append({
            "name": name,
            "default_model": zone_config.get("default_model"),
            "default_alias": zone_config.get("default_alias", "smart"),
            "default_temperature": zone_config.get("default_temperature", 0.7),
            "preferred_auth": zone_config.get("preferred_auth"),
            "blocked_providers": zone_config.get("blocked_providers", []),
        })

    return {
        "zones": zone_list,
        "current": zones.get_current_zone(),
        "session": zones.get_current_session(),
    }


@app.get("/zones/current")
async def get_current_zone():
    """Get current zone from environment."""
    if not zones:
        return {"zone": None, "session": None, "config": {}}

    current = zones.get_current_zone()
    return {
        "zone": current,
        "session": zones.get_current_session(),
        "config": zones.get_zone_config(current) if current else {},
    }


@app.get("/zones/{zone_name}")
async def get_zone(zone_name: str):
    """Get configuration for a specific zone."""
    if not zones:
        raise HTTPException(status_code=503, detail="Zones not configured")

    zone_config = zones.get_zone_config(zone_name)
    if not zone_config:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_name}' not found")

    return {
        "name": zone_name,
        "config": zone_config,
        "default_model": zones.get_default_model(zone_name),
        "default_alias": zones.get_default_alias(zone_name),
        "allowed_providers": zones.get_allowed_providers(zone_name),
        "blocked_providers": zones.get_blocked_providers(zone_name),
        "cost_limit": zones.get_cost_limit(zone_name),
        "rate_limit_rpm": zones.get_rate_limit(zone_name),
    }


class ZoneSwitchRequest(BaseModel):
    """Request to switch zones."""

    zone: str


@app.post("/zones/switch")
async def switch_zone(request: ZoneSwitchRequest):
    """Generate environment variables to switch to a zone.

    Returns shell commands that can be eval'd to switch zones.
    Note: This doesn't actually change the server's zone - it returns
    env vars for the client to set.
    """
    if not zones:
        raise HTTPException(status_code=503, detail="Zones not configured")

    zone_name = request.zone
    zone_config = zones.get_zone_config(zone_name)
    if not zone_config:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_name}' not found")

    # Build environment variables to set
    env_vars = {
        "MAESTRO_ZONE": zone_name,
    }

    # Add zone-specific env vars if configured
    zone_env = zone_config.get("env", {})
    env_vars.update(zone_env)

    # Generate shell export command
    export_lines = [f'export {k}="{v}"' for k, v in env_vars.items()]
    export_cmd = "\n".join(export_lines)

    return {
        "zone": zone_name,
        "env_vars": env_vars,
        "export_cmd": export_cmd,
        "config": {
            "default_model": zones.get_default_model(zone_name),
            "default_alias": zones.get_default_alias(zone_name),
            "preferred_auth": zone_config.get("preferred_auth"),
        },
    }


# =========================================================================
# Fanout Endpoints
# =========================================================================


class FanoutRequest(BaseModel):
    """Request for fan-out query."""

    models: list[str]
    messages: list[dict[str, Any]]
    mode: str = "first"  # first, all, race, vote, judge
    temperature: float | None = None
    max_tokens: int | None = None


@app.post("/fanout")
async def fanout_query(request: FanoutRequest):
    """Execute a fan-out query across multiple models."""
    if not fanout:
        raise HTTPException(status_code=503, detail="Fanout not initialized")

    try:
        mode = FanoutMode(request.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

    params = {}
    if request.temperature is not None:
        params["temperature"] = request.temperature
    if request.max_tokens is not None:
        params["max_tokens"] = request.max_tokens

    result = await fanout.execute(
        models=request.models,
        messages=request.messages,
        params=params,
        mode=mode,
    )

    return result


@app.get("/fanout/modes")
async def fanout_modes():
    """List available fan-out modes."""
    return {
        "modes": [
            {"name": "first", "description": "Return first successful response"},
            {"name": "all", "description": "Return all responses"},
            {"name": "race", "description": "Return fastest response"},
            {"name": "vote", "description": "Majority vote (for classification)"},
            {"name": "judge", "description": "Use judge model to pick best"},
        ]
    }


@app.get("/fanout/aliases")
async def fanout_aliases():
    """List configured model aliases for fan-out."""
    aliases = config.get("aliases", {})
    return {"aliases": aliases}


# =========================================================================
# Router Endpoints
# =========================================================================


@app.get("/router/stats")
async def router_stats():
    """Get routing statistics based on learned judge patterns."""
    if not router:
        return {"enabled": False, "message": "Router not initialized"}

    return {
        "enabled": True,
        "stats": router.get_routing_stats(),
    }


@app.get("/router/best")
async def router_best_model(
    query_type: str,
    models: str | None = None,
    fallback: str | None = None,
):
    """Get the best model for a query type based on learned patterns.

    Args:
        query_type: Type of query (code, explanation, creative, analysis, etc.)
        models: Comma-separated list of available models
        fallback: Fallback model if no data available
    """
    if not router:
        raise HTTPException(status_code=503, detail="Router not initialized")

    available = models.split(",") if models else []
    best = router.get_best_model(query_type, available, fallback)

    return {
        "query_type": query_type,
        "best_model": best,
        "has_data": router.should_use_adaptive(query_type),
    }


@app.get("/router/suggest")
async def router_suggest_models(
    query_type: str,
    models: str | None = None,
    top_n: int = 3,
):
    """Suggest top models for fan-out based on win rates."""
    if not router:
        raise HTTPException(status_code=503, detail="Router not initialized")

    available = models.split(",") if models else []
    suggested = router.suggest_fanout_models(query_type, available, top_n)

    return {
        "query_type": query_type,
        "suggested_models": suggested,
        "top_n": top_n,
    }


@app.post("/router/refresh")
async def router_refresh():
    """Force refresh of routing statistics from logs."""
    if not router:
        raise HTTPException(status_code=503, detail="Router not initialized")

    router.refresh_win_rates(force=True)
    return {"refreshed": True, "stats": router.get_routing_stats()}


# =========================================================================
# Guardrails Endpoints
# =========================================================================


@app.get("/guardrails/stats")
async def guardrails_stats():
    """Get guardrails statistics."""
    if not guardrails:
        return {"enabled": False}

    return {
        "enabled": True,
        "stats": guardrails.get_stats(),
    }


class GuardrailsValidateRequest(BaseModel):
    """Request to validate against guardrails."""

    messages: list[dict[str, Any]]
    zone: str | None = None
    action: str | None = None


@app.post("/guardrails/validate")
async def guardrails_validate(request: GuardrailsValidateRequest):
    """Validate a request against guardrails.

    Checks rate limits, content filters, and approval requirements.
    """
    if not guardrails:
        return {"valid": True, "message": "Guardrails not enabled"}

    valid, error = await guardrails.validate_request(
        messages=request.messages,
        zone=request.zone,
        action=request.action,
    )

    return {
        "valid": valid,
        "error": error,
    }


@app.get("/guardrails/rate-limit")
async def guardrails_rate_limit_check(zone: str | None = None):
    """Check rate limit status for a zone."""
    if not guardrails:
        return {"allowed": True, "message": "Guardrails not enabled"}

    allowed, error = guardrails.check_rate_limit(zone)
    return {
        "allowed": allowed,
        "error": error,
        "zone": zone or "default",
    }


# =========================================================================
# Judge Endpoints
# =========================================================================


class JudgeRequest(BaseModel):
    """Request for judge to select best response."""

    messages: list[dict[str, Any]]
    candidates: list[dict[str, str]]  # [{model: str, response: str}, ...]


@app.post("/judge/select")
async def judge_select_best(request: JudgeRequest):
    """Use judge model to select best response from candidates."""
    if not judge:
        raise HTTPException(status_code=503, detail="Judge not initialized")

    result = await judge.select_best(
        messages=request.messages,
        candidates=request.candidates,
    )

    return result


class JudgeClassifyRequest(BaseModel):
    """Request to classify a query."""

    messages: list[dict[str, Any]]


@app.post("/judge/classify")
async def judge_classify_query(request: JudgeClassifyRequest):
    """Classify a query type for routing decisions."""
    if not judge:
        raise HTTPException(status_code=503, detail="Judge not initialized")

    query_type = await judge.classify_query(request.messages)
    return {"query_type": query_type}


@app.get("/judge/config")
async def judge_config():
    """Get judge configuration."""
    if not judge:
        return {"enabled": False}

    return {
        "enabled": True,
        "judge_model": judge.judge_model,
    }


# =========================================================================
# Rate Limits Endpoints
# =========================================================================


@app.get("/rate-limits")
async def rate_limits_status():
    """Get rate limit status for all tracked auth keys."""
    if not rate_limiter:
        return {"enabled": False}

    return {
        "enabled": True,
        "statuses": rate_limiter.get_all_status(),
    }


@app.get("/rate-limits/{auth_key:path}")
async def rate_limits_for_key(auth_key: str):
    """Get rate limit status for a specific auth key.

    Auth key format: provider/method or provider/method/plan
    e.g., anthropic/oauth/pro, openai/api/tier3
    """
    if not rate_limiter:
        raise HTTPException(status_code=503, detail="Rate limiter not initialized")

    return rate_limiter.get_status(auth_key)


class RateLimitCheckRequest(BaseModel):
    """Request to check rate limit."""

    auth_key: str
    estimated_tokens: int = 0


@app.post("/rate-limits/check")
async def rate_limits_check(request: RateLimitCheckRequest):
    """Check if a request would be within rate limits."""
    if not rate_limiter:
        return {"allowed": True, "message": "Rate limiter not enabled"}

    allowed, reason, status = rate_limiter.check_limit(
        request.auth_key, request.estimated_tokens
    )

    return {
        "allowed": allowed,
        "reason": reason,
        "status": status,
    }


class RateLimitRecordRequest(BaseModel):
    """Request to record a completed request."""

    auth_key: str
    tokens_used: int = 0


@app.post("/rate-limits/record")
async def rate_limits_record(request: RateLimitRecordRequest):
    """Record a completed request for rate limiting."""
    if not rate_limiter:
        return {"recorded": False, "message": "Rate limiter not enabled"}

    rate_limiter.record_request(request.auth_key, request.tokens_used)
    return {"recorded": True}


@app.post("/rate-limits/reset")
async def rate_limits_reset(auth_key: str | None = None):
    """Reset rate limit counters."""
    if not rate_limiter:
        raise HTTPException(status_code=503, detail="Rate limiter not initialized")

    rate_limiter.reset(auth_key)
    return {"reset": True, "auth_key": auth_key or "all"}


# =========================================================================
# Model Checker Endpoints
# =========================================================================


@app.get("/models/check")
async def models_check(check_access: bool = False):
    """Check model availability across all providers.

    Args:
        check_access: If true, actually test model access (slower)
    """
    if not model_checker:
        raise HTTPException(status_code=503, detail="Model checker not initialized")

    results = await model_checker.check_all(check_access=check_access)

    # Convert to serializable format
    serialized = {}
    for platform, providers in results.items():
        serialized[platform] = {}
        for provider, models in providers.items():
            serialized[platform][provider] = [m.to_dict() for m in models]

    return {"results": serialized}


@app.get("/models/accessible")
async def models_accessible():
    """Get list of accessible models by provider."""
    if not model_checker:
        raise HTTPException(status_code=503, detail="Model checker not initialized")

    accessible = await model_checker.get_accessible_models()
    return {"accessible": accessible}


@app.get("/models/summary")
async def models_summary():
    """Get summary of model availability."""
    if not model_checker:
        raise HTTPException(status_code=503, detail="Model checker not initialized")

    summary = await model_checker.get_summary()
    return {"summary": summary}


@app.get("/models/litellm")
async def models_litellm_list():
    """Get models in LiteLLM format."""
    if not model_checker:
        raise HTTPException(status_code=503, detail="Model checker not initialized")

    models = model_checker.get_litellm_model_list()
    return {"models": models}


# =========================================================================
# Model Monitor Endpoints
# =========================================================================


@app.get("/models/monitor")
async def models_monitor_status():
    """Get model monitor status."""
    if not model_monitor:
        return {"enabled": False, "message": "Model monitor not enabled"}

    return {
        "enabled": True,
        "status": model_monitor.get_status(),
    }


@app.get("/models/monitor/changes")
async def models_monitor_changes(hours: int = 24):
    """Get recent model changes."""
    if not model_monitor:
        raise HTTPException(status_code=503, detail="Model monitor not enabled")

    changes = model_monitor.get_recent_changes(hours)
    return {
        "changes": [c.to_dict() for c in changes],
        "hours": hours,
    }


@app.get("/models/monitor/known")
async def models_monitor_known():
    """Get all known models being monitored."""
    if not model_monitor:
        raise HTTPException(status_code=503, detail="Model monitor not enabled")

    return {"known_models": model_monitor.get_all_known_models()}


@app.post("/models/monitor/check")
async def models_monitor_check_now():
    """Trigger immediate check for new models."""
    if not model_monitor:
        raise HTTPException(status_code=503, detail="Model monitor not enabled")

    changes = await model_monitor.check_for_new_models()
    return {
        "checked": True,
        "changes": [c.to_dict() for c in changes],
    }


# =========================================================================
# Version Checker Endpoints
# =========================================================================


@app.get("/updates")
async def updates_summary():
    """Get summary of version status for all tools."""
    if not version_checker:
        return {"enabled": False, "message": "Version checker not enabled"}

    return {
        "enabled": True,
        "summary": version_checker.get_summary(),
    }


@app.get("/updates/tools")
async def updates_list_tools():
    """List all tools by category."""
    if not version_checker:
        return {"enabled": False, "categories": {}}

    return {
        "enabled": True,
        "categories": version_checker.get_tools_by_category(),
    }


@app.get("/updates/check")
async def updates_check_all(use_cache: bool = True):
    """Check all tools for updates."""
    if not version_checker:
        raise HTTPException(status_code=503, detail="Version checker not enabled")

    results = await version_checker.check_all(use_cache=use_cache)
    return {
        "cli_tools": {k: v.to_dict() for k, v in results.get("cli_tools", {}).items()},
        "python_deps": {k: v.to_dict() for k, v in results.get("python_deps", {}).items()},
    }


@app.get("/updates/check/{tool_name}")
async def updates_check_tool(tool_name: str):
    """Check a specific tool for updates."""
    if not version_checker:
        raise HTTPException(status_code=503, detail="Version checker not enabled")

    result = await version_checker.check_cli_tool(tool_name)
    return result.to_dict()


@app.get("/updates/category/{category}")
async def updates_check_category(category: str):
    """Check all tools in a category for updates."""
    if not version_checker:
        raise HTTPException(status_code=503, detail="Version checker not enabled")

    results = await version_checker.check_category(category)
    return {
        "category": category,
        "tools": {k: v.to_dict() for k, v in results.items()},
    }


@app.get("/updates/outdated")
async def updates_outdated():
    """Get list of outdated tools."""
    if not version_checker:
        return {"enabled": False, "outdated": []}

    outdated = version_checker.get_outdated()
    by_category = version_checker.get_outdated_by_category()

    return {
        "enabled": True,
        "outdated": [v.to_dict() for v in outdated],
        "by_category": {
            cat: [v.to_dict() for v in tools]
            for cat, tools in by_category.items()
        },
    }


@app.post("/updates/refresh")
async def updates_refresh():
    """Force refresh version checks."""
    if not version_checker:
        raise HTTPException(status_code=503, detail="Version checker not enabled")

    results = await version_checker.check_all(use_cache=False)
    return {
        "refreshed": True,
        "cli_tools": {k: v.to_dict() for k, v in results.get("cli_tools", {}).items()},
        "python_deps": {k: v.to_dict() for k, v in results.get("python_deps", {}).items()},
    }


# =========================================================================
# Repo Updater Endpoints
# =========================================================================


@app.get("/repos")
async def repos_summary():
    """Get summary of tracked repositories."""
    if not repo_updater:
        return {"enabled": False, "message": "Repo updater not enabled"}

    return {
        "enabled": True,
        "summary": repo_updater.get_summary(),
    }


@app.get("/repos/list")
async def repos_list():
    """List all tracked repositories."""
    if not repo_updater:
        return {"enabled": False, "repos": []}

    repos = repo_updater.get_all_repos()
    return {
        "enabled": True,
        "repos": {name: info.to_dict() for name, info in repos.items()},
    }


@app.get("/repos/known")
async def repos_known():
    """List known repositories that can be cloned."""
    return {"known_repos": KNOWN_REPOS}


class RepoAddRequest(BaseModel):
    """Request to add a repository."""

    name: str
    path: str
    branch: str = "main"


@app.post("/repos/add")
async def repos_add(request: RepoAddRequest):
    """Add a repository to track."""
    if not repo_updater:
        raise HTTPException(status_code=503, detail="Repo updater not enabled")

    info = repo_updater.add_repo(request.name, request.path, request.branch)
    return info.to_dict()


@app.delete("/repos/{name}")
async def repos_remove(name: str):
    """Remove a repository from tracking."""
    if not repo_updater:
        raise HTTPException(status_code=503, detail="Repo updater not enabled")

    removed = repo_updater.remove_repo(name)
    return {"removed": removed, "name": name}


class RepoDiscoverRequest(BaseModel):
    """Request to discover repositories."""

    search_paths: list[str] | None = None


@app.post("/repos/discover")
async def repos_discover(request: RepoDiscoverRequest | None = None):
    """Discover git repositories in common locations."""
    if not repo_updater:
        raise HTTPException(status_code=503, detail="Repo updater not enabled")

    search_paths = request.search_paths if request else None
    discovered = repo_updater.discover_repos(search_paths)
    return {
        "discovered": [info.to_dict() for info in discovered],
        "count": len(discovered),
    }


@app.get("/repos/check")
async def repos_check_all():
    """Check all tracked repositories for updates."""
    if not repo_updater:
        raise HTTPException(status_code=503, detail="Repo updater not enabled")

    results = await repo_updater.check_all()
    return {
        "repos": {name: info.to_dict() for name, info in results.items()},
        "with_updates": [
            info.to_dict() for info in results.values() if info.has_updates
        ],
    }


@app.get("/repos/check/{name}")
async def repos_check_one(name: str):
    """Check a specific repository for updates."""
    if not repo_updater:
        raise HTTPException(status_code=503, detail="Repo updater not enabled")

    info = await repo_updater.check_repo(name)
    return info.to_dict()


# NOTE: Filesystem-modifying endpoints (update, update-all, clone) removed.
# These operations belong in CLI tools (maestro vendor) not in the daemon.
# Baton only tracks/monitors repos, doesn't modify them.


# =========================================================================
# Standards Monitor Endpoints
# =========================================================================


@app.get("/standards")
async def standards_summary():
    """Get summary of monitored AI standards and tooling.

    Tracks releases from Claude Code, Codex, Gemini CLI, and spec changes.
    """
    if not standards_monitor:
        return {"enabled": False, "message": "Standards monitor not enabled"}

    return {
        "enabled": True,
        "summary": standards_monitor.get_summary(),
        "compatibility": standards_monitor.get_compatibility_matrix(),
    }


@app.get("/standards/releases")
async def standards_releases():
    """Get all cached release information."""
    if not standards_monitor:
        raise HTTPException(status_code=503, detail="Standards monitor not enabled")

    return {
        "releases": standards_monitor.get_all_releases(),
        "monitored_repos": list(MONITORED_REPOS.keys()),
    }


@app.get("/standards/releases/{repo:path}")
async def standards_release_info(repo: str):
    """Get release info for a specific repo.

    Example: /standards/releases/anthropics/claude-code
    """
    if not standards_monitor:
        raise HTTPException(status_code=503, detail="Standards monitor not enabled")

    info = standards_monitor.get_repo_info(repo)
    if not info:
        raise HTTPException(status_code=404, detail=f"No info for repo: {repo}")

    return info


@app.get("/standards/updates")
async def standards_updates(since_hours: int = 24):
    """Get updates since a given time period.

    Args:
        since_hours: Look back this many hours (default: 24)
    """
    if not standards_monitor:
        raise HTTPException(status_code=503, detail="Standards monitor not enabled")

    import time
    since_timestamp = time.time() - (since_hours * 3600)

    updates = standards_monitor.get_updates_since(since_timestamp)
    return {
        "since_hours": since_hours,
        "updates": updates,
        "count": len(updates),
        "has_breaking": any(u.get("is_breaking") for u in updates),
    }


@app.post("/standards/check")
async def standards_check_now():
    """Trigger immediate check for updates."""
    if not standards_monitor:
        raise HTTPException(status_code=503, detail="Standards monitor not enabled")

    results = await standards_monitor.check_all()
    return {
        "checked": True,
        "checked_at": results.get("checked_at"),
        "releases_count": len(results.get("releases", [])),
        "new_since_last_check": results.get("new_since_last_check", []),
    }


@app.get("/standards/compatibility")
async def standards_compatibility():
    """Get compatibility matrix for skills and context files across CLIs."""
    if not standards_monitor:
        raise HTTPException(status_code=503, detail="Standards monitor not enabled")

    return standards_monitor.get_compatibility_matrix()


# =========================================================================
# Notification (Twilio) Endpoints
# =========================================================================


@app.get("/notify/status")
async def notify_status():
    """Get notification configuration status."""
    if not twilio:
        return {"enabled": False, "message": "Twilio not initialized"}

    return {
        "enabled": twilio.enabled,
        "configured": twilio.is_configured(),
        "from_number": twilio.from_number[:6] + "****" if twilio.from_number else None,
        "to_number": twilio.to_number[:6] + "****" if twilio.to_number else None,
    }


class SMSRequest(BaseModel):
    """Request to send an SMS."""

    message: str
    to: str | None = None


@app.post("/notify/sms")
async def notify_send_sms(request: SMSRequest):
    """Send an SMS message."""
    if not twilio or not twilio.enabled:
        raise HTTPException(status_code=503, detail="Twilio not configured")

    result = await twilio.send_sms(request.message, request.to)
    return result


class MFARequest(BaseModel):
    """Request to send an MFA code."""

    code: str
    service: str
    to: str | None = None


@app.post("/notify/mfa")
async def notify_send_mfa(request: MFARequest):
    """Send an MFA code via SMS."""
    if not twilio or not twilio.enabled:
        raise HTTPException(status_code=503, detail="Twilio not configured")

    result = await twilio.send_mfa_code(request.code, request.service, request.to)
    return result


class ApprovalRequest(BaseModel):
    """Request for approval via SMS."""

    request_id: str
    action: str
    details: str
    to: str | None = None


@app.post("/notify/approval")
async def notify_request_approval(request: ApprovalRequest):
    """Send an approval request and wait for response."""
    if not twilio or not twilio.enabled:
        raise HTTPException(status_code=503, detail="Twilio not configured")

    result = await twilio.send_approval_request(
        request.request_id,
        request.action,
        request.details,
        request.to,
    )

    return {
        "request_id": request.request_id,
        "result": result,
        "approved": result == "approved",
        "denied": result == "denied",
        "timed_out": result is None,
    }


# Twilio webhook for incoming SMS
@app.post("/notify/webhook")
async def notify_webhook(From: str = Form(...), Body: str = Form(...)):
    """Handle incoming Twilio SMS webhook."""
    if not twilio:
        raise HTTPException(status_code=503, detail="Twilio not configured")

    twilio.handle_incoming_sms(From, Body)

    return Response(
        content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type="application/xml",
    )


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
