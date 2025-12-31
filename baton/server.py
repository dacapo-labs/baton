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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global config, auth, logger, judge, fanout, zones

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

    yield


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
