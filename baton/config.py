"""Baton configuration management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomli

# Config search paths (in priority order)
CONFIG_PATHS = [
    Path(os.environ.get("BATON_CONFIG", "")),
    Path.home() / ".config" / "lifemaestro" / "baton.toml",
    Path.home() / ".config" / "baton" / "config.toml",
    Path("/etc/baton/config.toml"),
]


def find_config() -> Path | None:
    """Find the first existing config file."""
    for path in CONFIG_PATHS:
        if path and path.exists():
            return path
    return None


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load configuration from TOML file."""
    if path is None:
        path = find_config()

    if path is None or not path.exists():
        return get_default_config()

    with open(path, "rb") as f:
        config = tomli.load(f)

    # Merge with defaults
    defaults = get_default_config()
    return deep_merge(defaults, config)


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_default_config() -> dict[str, Any]:
    """Return default configuration."""
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 4000,
            "workers": 1,
        },
        "logging": {
            "dir": "/data/baton/logs",
            "format": "jsonl",
            "rotate_mb": 100,
            "retention_days": 365,
        },
        "auth": {
            "bitwarden": {
                "enabled": True,
                "session_file": "~/.config/bitwarden/session",
                "cache_ttl": 3600,
            },
        },
        "aliases": {
            "fast": ["claude-3-5-haiku-latest", "gpt-4o-mini", "gemini-2.0-flash"],
            "smart": ["claude-sonnet-4-20250514", "gpt-4o", "gemini-2.0-flash-thinking"],
            "code": ["claude-sonnet-4-20250514", "gpt-4o", "deepseek-coder"],
            "local": ["ollama/llama3.2", "ollama/codellama"],
            "cheap": ["gpt-4o-mini", "claude-3-5-haiku-latest"],
        },
        "providers": {},
        "zones": {},
        "fanout": {
            "default_mode": "first",
            "judge_model": "claude-3-5-haiku-latest",
            "timeout_seconds": 60,
        },
        "guardrails": {
            "enabled": True,
            "rate_limit_rpm": 100,
            "require_approval": [],
            "blocked_patterns": [],
        },
    }


def get_zone_config(config: dict[str, Any], zone: str | None) -> dict[str, Any]:
    """Get merged config for a specific zone."""
    if not zone or zone not in config.get("zones", {}):
        return config

    zone_config = config["zones"][zone]
    return deep_merge(config, zone_config)
