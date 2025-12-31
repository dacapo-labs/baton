"""Baton Zones Plugin - Zone-aware routing and defaults."""

from __future__ import annotations

import os
from typing import Any


class BatonZones:
    """Zone-aware configuration and routing.

    Integrates with LifeMaestro zones to provide:
    - Per-zone model defaults
    - Per-zone allowed/blocked providers
    - Per-zone feature flags
    - Zone detection from environment
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.zones = config.get("zones", {})

    def get_current_zone(self) -> str | None:
        """Get current zone from environment."""
        return os.environ.get("MAESTRO_ZONE")

    def get_current_session(self) -> str | None:
        """Get current session from environment."""
        return os.environ.get("MAESTRO_SESSION")

    def get_zone_config(self, zone: str | None = None) -> dict[str, Any]:
        """Get configuration for a specific zone."""
        zone = zone or self.get_current_zone()
        if not zone or zone not in self.zones:
            return {}
        return self.zones[zone]

    def get_default_model(self, zone: str | None = None) -> str | None:
        """Get default model for a zone."""
        zone_config = self.get_zone_config(zone)
        return zone_config.get("default_model")

    def get_default_alias(self, zone: str | None = None) -> str | None:
        """Get default model alias for a zone."""
        zone_config = self.get_zone_config(zone)
        return zone_config.get("default_alias", "smart")

    def get_allowed_providers(self, zone: str | None = None) -> list[str] | None:
        """Get list of allowed providers for a zone. None means all allowed."""
        zone_config = self.get_zone_config(zone)
        return zone_config.get("allowed_providers")

    def get_blocked_providers(self, zone: str | None = None) -> list[str]:
        """Get list of blocked providers for a zone."""
        zone_config = self.get_zone_config(zone)
        return zone_config.get("blocked_providers", [])

    def is_provider_allowed(self, provider: str, zone: str | None = None) -> bool:
        """Check if a provider is allowed for the zone."""
        blocked = self.get_blocked_providers(zone)
        if provider in blocked:
            return False

        allowed = self.get_allowed_providers(zone)
        if allowed is None:
            return True

        return provider in allowed

    def filter_models(self, models: list[str], zone: str | None = None) -> list[str]:
        """Filter models to only those allowed in the zone."""
        return [
            m for m in models
            if self.is_provider_allowed(self._get_provider(m), zone)
        ]

    def _get_provider(self, model: str) -> str:
        """Extract provider from model name."""
        if "/" in model:
            return model.split("/")[0]

        provider_prefixes = {
            "claude": "anthropic",
            "gpt": "openai",
            "o1": "openai",
            "gemini": "google",
            "palm": "google",
            "llama": "meta",
            "mistral": "mistral",
            "deepseek": "deepseek",
            "command": "cohere",
        }

        model_lower = model.lower()
        for prefix, provider in provider_prefixes.items():
            if model_lower.startswith(prefix):
                return provider

        return "unknown"

    def get_zone_headers(self, zone: str | None = None) -> dict[str, str]:
        """Get HTTP headers to add for zone context."""
        zone = zone or self.get_current_zone()
        session = self.get_current_session()

        headers = {}
        if zone:
            headers["X-Maestro-Zone"] = zone
        if session:
            headers["X-Maestro-Session"] = session

        return headers

    def apply_zone_defaults(
        self,
        model: str | None,
        params: dict[str, Any],
        zone: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Apply zone-specific defaults to a request."""
        zone = zone or self.get_current_zone()
        zone_config = self.get_zone_config(zone)

        if not model:
            model = zone_config.get("default_model") or "claude-sonnet-4-20250514"

        zone_params = zone_config.get("default_params", {})
        merged_params = {**zone_params, **params}

        if "temperature" not in merged_params:
            merged_params["temperature"] = zone_config.get("default_temperature", 0.7)

        return model, merged_params

    def get_cost_limit(self, zone: str | None = None) -> float | None:
        """Get cost limit for zone."""
        zone_config = self.get_zone_config(zone)
        return zone_config.get("cost_limit")

    def get_rate_limit(self, zone: str | None = None) -> int | None:
        """Get rate limit (requests per minute) for zone."""
        zone_config = self.get_zone_config(zone)
        return zone_config.get("rate_limit_rpm")
