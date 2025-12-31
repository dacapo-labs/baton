"""Tests for model monitor."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baton.plugins.model_monitor import (
    ModelChange,
    ModelMonitor,
    MonitorConfig,
    get_model_monitor,
    init_model_monitor,
    start_model_monitor,
    stop_model_monitor,
)
from baton.plugins.model_checker import ModelInfo


class TestModelChange:
    """Tests for ModelChange dataclass."""

    def test_new_model_change(self):
        """Test creating a new model change."""
        change = ModelChange(
            change_type="new",
            platform="bedrock",
            provider="anthropic",
            model_id="anthropic.claude-4-opus-v1:0",
            current_state={"discovered": True},
        )

        assert change.change_type == "new"
        assert change.platform == "bedrock"
        assert change.provider == "anthropic"
        assert change.model_id == "anthropic.claude-4-opus-v1:0"
        assert change.detected_at > 0

    def test_removed_model_change(self):
        """Test creating a removed model change."""
        change = ModelChange(
            change_type="removed",
            platform="vertex",
            provider="google",
            model_id="gemini-old",
            previous_state={"existed": True},
        )

        assert change.change_type == "removed"
        assert change.previous_state == {"existed": True}

    def test_access_changed(self):
        """Test creating an access changed event."""
        change = ModelChange(
            change_type="access_changed",
            platform="bedrock",
            provider="anthropic",
            model_id="anthropic.claude-3-opus-v1:0",
            previous_state={"accessible": False},
            current_state={"accessible": True},
        )

        assert change.change_type == "access_changed"
        assert change.previous_state["accessible"] is False
        assert change.current_state["accessible"] is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        change = ModelChange(
            change_type="new",
            platform="bedrock",
            provider="anthropic",
            model_id="test-model",
            current_state={"test": True},
        )

        result = change.to_dict()

        assert result["change_type"] == "new"
        assert result["platform"] == "bedrock"
        assert result["provider"] == "anthropic"
        assert result["model_id"] == "test-model"
        assert result["current_state"] == {"test": True}
        assert "detected_at" in result


class TestMonitorConfig:
    """Tests for MonitorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MonitorConfig()

        assert config.check_interval == 3600
        assert config.state_file is None
        assert config.check_bedrock is True
        assert config.check_vertex is True
        assert config.bedrock_region == "us-east-1"
        assert config.vertex_region == "us-central1"

    def test_custom_config(self):
        """Test custom configuration."""
        config = MonitorConfig(
            check_interval=1800,
            state_file="/tmp/model_state.json",
            check_bedrock=True,
            check_vertex=False,
            bedrock_profile="production",
            bedrock_region="us-west-2",
        )

        assert config.check_interval == 1800
        assert config.state_file == "/tmp/model_state.json"
        assert config.check_bedrock is True
        assert config.check_vertex is False
        assert config.bedrock_profile == "production"
        assert config.bedrock_region == "us-west-2"


class TestModelMonitor:
    """Tests for ModelMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create monitor instance without checkers."""
        config = MonitorConfig(
            check_bedrock=False,
            check_vertex=False,
        )
        return ModelMonitor(config)

    @pytest.fixture
    def monitor_with_bedrock(self):
        """Create monitor with Bedrock enabled."""
        config = MonitorConfig(
            check_bedrock=True,
            check_vertex=False,
            bedrock_profile="test",
        )
        return ModelMonitor(config)

    @pytest.fixture
    def monitor_with_state_file(self):
        """Create monitor with state persistence."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            config = MonitorConfig(
                check_bedrock=False,
                check_vertex=False,
                state_file=f.name,
            )
            monitor = ModelMonitor(config)
            yield monitor
            # Cleanup
            Path(f.name).unlink(missing_ok=True)

    def test_init_known_models(self, monitor):
        """Test that known models are initialized from static lists."""
        known = monitor.get_all_known_models()

        assert "bedrock" in known
        assert "vertex" in known
        assert "anthropic" in known["bedrock"]
        assert "google" in known["vertex"]

    def test_get_status(self, monitor):
        """Test getting monitor status."""
        status = monitor.get_status()

        assert "running" in status
        assert "last_check" in status
        assert "check_interval" in status
        assert "known_model_count" in status
        assert "recent_changes" in status
        assert status["running"] is False

    def test_add_known_model(self, monitor):
        """Test adding a model to known list."""
        monitor.add_known_model("bedrock", "test-provider", "test-model-id")

        known = monitor.get_all_known_models()
        assert "test-provider" in known["bedrock"]
        assert "test-model-id" in known["bedrock"]["test-provider"]

    def test_get_recent_changes_empty(self, monitor):
        """Test getting recent changes when none exist."""
        changes = monitor.get_recent_changes(24)
        assert changes == []

    def test_get_recent_changes_with_data(self, monitor):
        """Test getting recent changes with data."""
        # Add some changes
        change = ModelChange(
            change_type="new",
            platform="bedrock",
            provider="anthropic",
            model_id="test-model",
        )
        monitor._changes.append(change)

        changes = monitor.get_recent_changes(24)
        assert len(changes) == 1
        assert changes[0].model_id == "test-model"

    def test_get_recent_changes_filters_old(self, monitor):
        """Test that old changes are filtered out."""
        # Add an old change
        old_change = ModelChange(
            change_type="new",
            platform="bedrock",
            provider="anthropic",
            model_id="old-model",
        )
        old_change.detected_at = time.time() - (48 * 3600)  # 48 hours ago
        monitor._changes.append(old_change)

        # Add a recent change
        recent_change = ModelChange(
            change_type="new",
            platform="bedrock",
            provider="anthropic",
            model_id="recent-model",
        )
        monitor._changes.append(recent_change)

        changes = monitor.get_recent_changes(24)
        assert len(changes) == 1
        assert changes[0].model_id == "recent-model"

    def test_state_persistence_save(self, monitor_with_state_file):
        """Test saving state to file."""
        monitor = monitor_with_state_file
        monitor.add_known_model("bedrock", "test", "test-model")

        # State should be saved automatically
        state_path = Path(monitor.config.state_file)
        assert state_path.exists()

        with open(state_path) as f:
            data = json.load(f)

        assert "known_models" in data
        assert "bedrock" in data["known_models"]

    def test_state_persistence_load(self):
        """Test loading state from file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            state = {
                "known_models": {
                    "bedrock": {"custom": ["custom-model"]},
                    "vertex": {},
                },
                "last_check": 12345.0,
            }
            json.dump(state, f)
            state_file = f.name

        try:
            config = MonitorConfig(
                check_bedrock=False,
                check_vertex=False,
                state_file=state_file,
            )
            monitor = ModelMonitor(config)

            # Should have loaded custom model
            known = monitor.get_all_known_models()
            assert "custom" in known["bedrock"]
            assert "custom-model" in known["bedrock"]["custom"]
            assert monitor._last_check == 12345.0

        finally:
            Path(state_file).unlink()

    @pytest.mark.asyncio
    async def test_check_for_new_models_empty(self, monitor):
        """Test checking for new models with no checkers."""
        changes = await monitor.check_for_new_models()
        assert changes == []
        assert monitor._last_check > 0

    @pytest.mark.asyncio
    async def test_check_bedrock_models_new_discovery(self, monitor_with_bedrock):
        """Test discovering new Bedrock models."""
        monitor = monitor_with_bedrock

        # Mock the bedrock checker to return a new model
        new_model = ModelInfo(
            model_id="anthropic.claude-4-v1:0",  # New model not in known list
            provider="bedrock/anthropic",
        )

        with patch.object(
            monitor._bedrock,
            "list_all_models",
            new_callable=AsyncMock,
            return_value=[new_model],
        ):
            changes = await monitor._check_bedrock_models()

            # Should detect the new model
            new_changes = [c for c in changes if c.change_type == "new"]
            assert any(c.model_id == "anthropic.claude-4-v1:0" for c in new_changes)

    @pytest.mark.asyncio
    async def test_check_bedrock_models_error_handling(self, monitor_with_bedrock):
        """Test error handling during Bedrock check."""
        monitor = monitor_with_bedrock

        with patch.object(
            monitor._bedrock,
            "list_all_models",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            changes = await monitor._check_bedrock_models()

            # Should return empty list on error
            assert changes == []

    @pytest.mark.asyncio
    async def test_notify_callback(self, monitor):
        """Test notification callback is called."""
        callback_called = []

        def notify(change: ModelChange):
            callback_called.append(change)

        monitor.config.notify_callback = notify

        # Simulate a change
        monitor._changes.append(
            ModelChange(
                change_type="new",
                platform="bedrock",
                provider="test",
                model_id="test-model",
            )
        )

        # Trigger notification via check (even with no checkers)
        with patch.object(monitor, "_check_bedrock_models", new_callable=AsyncMock) as mock_check:
            # Return a change from the mock
            mock_check.return_value = [
                ModelChange(
                    change_type="new",
                    platform="bedrock",
                    provider="test",
                    model_id="new-test-model",
                )
            ]
            monitor._bedrock = MagicMock()  # Enable bedrock checking

            await monitor.check_for_new_models()

            assert len(callback_called) == 1
            assert callback_called[0].model_id == "new-test-model"

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """Test starting and stopping the monitor."""
        assert monitor._running is False

        monitor.start()
        assert monitor._running is True
        assert monitor._task is not None

        await asyncio.sleep(0.1)

        monitor.stop()
        assert monitor._running is False

    @pytest.mark.asyncio
    async def test_check_access_changes(self, monitor_with_bedrock):
        """Test checking for access changes."""
        monitor = monitor_with_bedrock

        # Set up initial state
        initial_model = ModelInfo(
            model_id="anthropic.claude-3-opus-v1:0",
            provider="bedrock/anthropic",
            accessible=False,
        )
        monitor._model_states["bedrock/anthropic.claude-3-opus-v1:0"] = initial_model
        monitor._known_models["bedrock"]["anthropic"] = {"anthropic.claude-3-opus-v1:0"}

        # Mock check_model_access to return changed access
        current_model = ModelInfo(
            model_id="anthropic.claude-3-opus-v1:0",
            provider="bedrock/anthropic",
            accessible=True,  # Changed!
        )

        with patch.object(
            monitor._bedrock,
            "check_model_access",
            new_callable=AsyncMock,
            return_value=current_model,
        ):
            changes = await monitor.check_access_changes()

            assert len(changes) == 1
            assert changes[0].change_type == "access_changed"
            assert changes[0].previous_state["accessible"] is False
            assert changes[0].current_state["accessible"] is True


class TestGlobalModelMonitor:
    """Tests for global model monitor functions."""

    def test_init_model_monitor_with_config(self):
        """Test initializing global monitor with config."""
        config = MonitorConfig(
            check_interval=1800,
            check_bedrock=False,
            check_vertex=False,
        )
        monitor = init_model_monitor(config)

        assert monitor is not None
        assert monitor.config.check_interval == 1800
        assert get_model_monitor() is monitor

    def test_init_model_monitor_with_dict(self):
        """Test initializing global monitor with dict config."""
        monitor = init_model_monitor({
            "check_interval": 900,
            "check_bedrock": False,
            "check_vertex": False,
        })

        assert monitor is not None
        assert monitor.config.check_interval == 900

    def test_get_model_monitor_none(self):
        """Test get_model_monitor when not initialized."""
        import baton.plugins.model_monitor as mm_module

        mm_module._monitor = None
        assert get_model_monitor() is None

    @pytest.mark.asyncio
    async def test_start_stop_model_monitor(self):
        """Test start and stop functions."""
        monitor = init_model_monitor({
            "check_bedrock": False,
            "check_vertex": False,
        })

        start_model_monitor()
        assert monitor._running is True

        await asyncio.sleep(0.1)

        stop_model_monitor()
        assert monitor._running is False

    def test_start_model_monitor_not_initialized(self):
        """Test starting monitor when not initialized."""
        import baton.plugins.model_monitor as mm_module

        mm_module._monitor = None

        with pytest.raises(RuntimeError, match="not initialized"):
            start_model_monitor()
