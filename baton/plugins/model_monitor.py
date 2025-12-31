"""Baton Model Monitor - Track new model availability across providers."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .model_checker import (
    BEDROCK_MODELS,
    VERTEX_MODELS,
    BedrockModelChecker,
    ModelInfo,
    VertexModelChecker,
)

log = logging.getLogger(__name__)


@dataclass
class ModelChange:
    """Represents a change in model availability."""

    change_type: str  # "new", "removed", "access_changed"
    platform: str  # "bedrock", "vertex"
    provider: str  # "anthropic", "google", etc.
    model_id: str
    previous_state: dict[str, Any] | None = None
    current_state: dict[str, Any] | None = None
    detected_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_type": self.change_type,
            "platform": self.platform,
            "provider": self.provider,
            "model_id": self.model_id,
            "previous_state": self.previous_state,
            "current_state": self.current_state,
            "detected_at": self.detected_at,
        }


@dataclass
class MonitorConfig:
    """Configuration for model monitor."""

    check_interval: int = 3600  # Check every hour
    state_file: str | None = None  # Path to persist state
    notify_callback: Callable[[ModelChange], None] | None = None
    check_bedrock: bool = True
    check_vertex: bool = True
    bedrock_profile: str | None = None
    bedrock_region: str = "us-east-1"
    vertex_project: str | None = None
    vertex_region: str = "us-central1"


class ModelMonitor:
    """Monitor for new model availability across providers."""

    def __init__(self, config: MonitorConfig | None = None):
        self.config = config or MonitorConfig()

        # Initialize checkers
        self._bedrock: BedrockModelChecker | None = None
        self._vertex: VertexModelChecker | None = None

        if self.config.check_bedrock:
            self._bedrock = BedrockModelChecker(
                profile=self.config.bedrock_profile,
                region=self.config.bedrock_region,
            )

        if self.config.check_vertex:
            self._vertex = VertexModelChecker(
                project=self.config.vertex_project,
                region=self.config.vertex_region,
            )

        # State tracking
        self._known_models: dict[str, dict[str, set[str]]] = {
            "bedrock": {},
            "vertex": {},
        }
        self._model_states: dict[str, dict[str, ModelInfo]] = {}
        self._changes: list[ModelChange] = []
        self._last_check: float = 0
        self._running: bool = False
        self._task: asyncio.Task | None = None

        # Initialize known models from static lists
        self._init_known_models()

        # Load persisted state if available
        if self.config.state_file:
            self._load_state()

    def _init_known_models(self) -> None:
        """Initialize known models from static lists."""
        for provider, models in BEDROCK_MODELS.items():
            self._known_models["bedrock"][provider] = set(models)

        for provider, models in VERTEX_MODELS.items():
            self._known_models["vertex"][provider] = set(models)

    def _load_state(self) -> None:
        """Load persisted state from file."""
        if not self.config.state_file:
            return

        try:
            state_path = Path(self.config.state_file)
            if state_path.exists():
                with open(state_path) as f:
                    data = json.load(f)

                # Restore known models
                for platform in ["bedrock", "vertex"]:
                    if platform in data.get("known_models", {}):
                        self._known_models[platform] = {
                            provider: set(models)
                            for provider, models in data["known_models"][platform].items()
                        }

                self._last_check = data.get("last_check", 0)
                log.info(f"Loaded model monitor state from {state_path}")

        except Exception as e:
            log.warning(f"Failed to load model monitor state: {e}")

    def _save_state(self) -> None:
        """Save state to file."""
        if not self.config.state_file:
            return

        try:
            state_path = Path(self.config.state_file)
            state_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "known_models": {
                    platform: {
                        provider: list(models)
                        for provider, models in providers.items()
                    }
                    for platform, providers in self._known_models.items()
                },
                "last_check": self._last_check,
            }

            with open(state_path, "w") as f:
                json.dump(data, f, indent=2)

            log.debug(f"Saved model monitor state to {state_path}")

        except Exception as e:
            log.warning(f"Failed to save model monitor state: {e}")

    async def check_for_new_models(self) -> list[ModelChange]:
        """Check all platforms for new models."""
        changes: list[ModelChange] = []

        # Check Bedrock
        if self._bedrock:
            bedrock_changes = await self._check_bedrock_models()
            changes.extend(bedrock_changes)

        # Check Vertex
        if self._vertex:
            vertex_changes = await self._check_vertex_models()
            changes.extend(vertex_changes)

        self._last_check = time.time()
        self._changes.extend(changes)
        self._save_state()

        # Notify about changes
        if self.config.notify_callback:
            for change in changes:
                try:
                    self.config.notify_callback(change)
                except Exception as e:
                    log.error(f"Notification callback failed: {e}")

        return changes

    async def _check_bedrock_models(self) -> list[ModelChange]:
        """Check Bedrock for new models."""
        changes: list[ModelChange] = []

        try:
            # Get all available models from Bedrock API
            available_models = await self._bedrock.list_all_models()

            # Group by provider
            discovered: dict[str, set[str]] = {}
            for model in available_models:
                # Extract provider from model_id (e.g., "anthropic.claude-..." -> "anthropic")
                provider = model.model_id.split(".")[0] if "." in model.model_id else "unknown"
                if provider not in discovered:
                    discovered[provider] = set()
                discovered[provider].add(model.model_id)

            # Compare with known models
            for provider, current_models in discovered.items():
                known = self._known_models["bedrock"].get(provider, set())

                # Find new models
                new_models = current_models - known
                for model_id in new_models:
                    change = ModelChange(
                        change_type="new",
                        platform="bedrock",
                        provider=provider,
                        model_id=model_id,
                        current_state={"discovered": True},
                    )
                    changes.append(change)
                    log.info(f"New Bedrock model discovered: {model_id}")

                # Find removed models (optional - models might just be unavailable)
                removed_models = known - current_models
                for model_id in removed_models:
                    change = ModelChange(
                        change_type="removed",
                        platform="bedrock",
                        provider=provider,
                        model_id=model_id,
                        previous_state={"existed": True},
                    )
                    changes.append(change)
                    log.info(f"Bedrock model no longer available: {model_id}")

                # Update known models
                self._known_models["bedrock"][provider] = current_models

        except Exception as e:
            log.error(f"Failed to check Bedrock models: {e}")

        return changes

    async def _check_vertex_models(self) -> list[ModelChange]:
        """Check Vertex AI for new models."""
        changes: list[ModelChange] = []

        try:
            # Get models via gcloud
            available_models = await self._vertex.list_models_via_gcloud()

            # Track discovered models
            discovered: dict[str, set[str]] = {"google": set()}
            for model in available_models:
                discovered["google"].add(model.model_id)

            # Compare with known models (only for google provider from gcloud)
            known_google = self._known_models["vertex"].get("google", set())

            # Find new models
            new_models = discovered["google"] - known_google
            for model_id in new_models:
                change = ModelChange(
                    change_type="new",
                    platform="vertex",
                    provider="google",
                    model_id=model_id,
                    current_state={"discovered": True},
                )
                changes.append(change)
                log.info(f"New Vertex model discovered: {model_id}")

            # Update known models if any discovered
            if discovered["google"]:
                self._known_models["vertex"]["google"] = discovered["google"]

        except Exception as e:
            log.error(f"Failed to check Vertex models: {e}")

        return changes

    async def check_access_changes(self) -> list[ModelChange]:
        """Check if access to known models has changed."""
        changes: list[ModelChange] = []

        # Check Bedrock model access
        if self._bedrock:
            for provider, model_ids in self._known_models["bedrock"].items():
                for model_id in model_ids:
                    prev_state = self._model_states.get(f"bedrock/{model_id}")
                    current = await self._bedrock.check_model_access(model_id)

                    # Check if access changed
                    if prev_state and prev_state.accessible != current.accessible:
                        change = ModelChange(
                            change_type="access_changed",
                            platform="bedrock",
                            provider=provider,
                            model_id=model_id,
                            previous_state={"accessible": prev_state.accessible},
                            current_state={"accessible": current.accessible},
                        )
                        changes.append(change)
                        log.info(
                            f"Bedrock model access changed: {model_id} "
                            f"{prev_state.accessible} -> {current.accessible}"
                        )

                    self._model_states[f"bedrock/{model_id}"] = current

        return changes

    def get_recent_changes(self, hours: int = 24) -> list[ModelChange]:
        """Get changes from the last N hours."""
        cutoff = time.time() - (hours * 3600)
        return [c for c in self._changes if c.detected_at > cutoff]

    def get_all_known_models(self) -> dict[str, dict[str, list[str]]]:
        """Get all known models by platform and provider."""
        return {
            platform: {
                provider: sorted(models)
                for provider, models in providers.items()
            }
            for platform, providers in self._known_models.items()
        }

    def add_known_model(self, platform: str, provider: str, model_id: str) -> None:
        """Manually add a model to the known list."""
        if platform not in self._known_models:
            self._known_models[platform] = {}
        if provider not in self._known_models[platform]:
            self._known_models[platform][provider] = set()

        self._known_models[platform][provider].add(model_id)
        self._save_state()

    def get_status(self) -> dict[str, Any]:
        """Get monitor status."""
        return {
            "running": self._running,
            "last_check": self._last_check,
            "check_interval": self.config.check_interval,
            "known_model_count": {
                platform: sum(len(models) for models in providers.values())
                for platform, providers in self._known_models.items()
            },
            "recent_changes": len(self.get_recent_changes(24)),
        }

    def start(self) -> None:
        """Start the monitoring loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        log.info("Model monitor started")

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        log.info("Model monitor stopped")

    async def _run_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self.check_for_new_models()
            except Exception as e:
                log.error(f"Model monitor check failed: {e}")

            # Wait for next check interval
            await asyncio.sleep(self.config.check_interval)


# Provider-specific release tracking
class ModelReleaseTracker:
    """Track model releases from provider announcements."""

    # URLs to check for new model announcements
    RELEASE_SOURCES = {
        "anthropic": [
            "https://docs.anthropic.com/en/api/models",
            "https://www.anthropic.com/api",
        ],
        "openai": [
            "https://platform.openai.com/docs/models",
        ],
        "google": [
            "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models",
            "https://ai.google.dev/gemini-api/docs/models/gemini",
        ],
        "meta": [
            "https://ai.meta.com/llama/",
        ],
        "mistral": [
            "https://docs.mistral.ai/getting-started/models/",
        ],
    }

    def __init__(self):
        self._last_known_models: dict[str, set[str]] = {}

    async def check_provider_releases(
        self, provider: str
    ) -> list[dict[str, str]]:
        """Check a provider's documentation for new model releases.

        Note: This is a placeholder. Full implementation would require
        web scraping or RSS feed parsing.
        """
        # This would need to be implemented with actual web fetching
        # For now, return empty list
        return []


# Singleton instance
_monitor: ModelMonitor | None = None


def get_model_monitor() -> ModelMonitor | None:
    """Get the global model monitor instance."""
    return _monitor


def init_model_monitor(config: MonitorConfig | dict[str, Any] | None = None) -> ModelMonitor:
    """Initialize the global model monitor."""
    global _monitor

    if isinstance(config, dict):
        config = MonitorConfig(**config)

    _monitor = ModelMonitor(config)
    return _monitor


def start_model_monitor() -> None:
    """Start the global model monitor."""
    if _monitor:
        _monitor.start()
    else:
        raise RuntimeError("Model monitor not initialized")


def stop_model_monitor() -> None:
    """Stop the global model monitor."""
    if _monitor:
        _monitor.stop()
