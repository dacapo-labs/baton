"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def mock_config():
    """Basic mock configuration."""
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 4000,
        },
        "providers": {
            "anthropic": {
                "bitwarden_item": "Anthropic API",
            },
            "openai": {
                "bitwarden_item": "OpenAI API",
            },
            "google": {
                "bitwarden_item": "Google AI API",
            },
        },
        "rate_limits": {
            "anthropic/api": {"rpm": 60, "rpd": 10000},
            "openai/api": {"rpm": 60, "rpd": 10000},
        },
        "keepalive": {
            "interval": 60,
            "providers": ["aws", "gcp"],
        },
    }


@pytest.fixture
def sample_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
