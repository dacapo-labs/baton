"""Tests for credential keepalive daemon."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from baton.plugins.keepalive import (
    AWSCredentialChecker,
    AzureCredentialChecker,
    BitwradenChecker,
    CredentialStatus,
    GCPCredentialChecker,
    KeepaliveConfig,
    KeepaliveDaemon,
    init_keepalive_daemon,
    get_keepalive_daemon,
)


class TestCredentialStatus:
    """Tests for CredentialStatus dataclass."""

    def test_valid_status(self):
        """Test valid credential status."""
        status = CredentialStatus(
            provider="aws",
            auth_method="sso",
            status="valid",
            ttl_seconds=3600,
            user="arn:aws:iam::123456789:user/test",
        )

        assert status.is_healthy is True
        assert status.needs_refresh is False

    def test_expiring_status(self):
        """Test expiring credential status."""
        status = CredentialStatus(
            provider="aws",
            auth_method="sso",
            status="expiring",
            ttl_seconds=600,
        )

        assert status.is_healthy is True
        assert status.needs_refresh is True

    def test_expired_status(self):
        """Test expired credential status."""
        status = CredentialStatus(
            provider="aws",
            auth_method="sso",
            status="expired",
        )

        assert status.is_healthy is False
        assert status.needs_refresh is True

    def test_error_status(self):
        """Test error credential status."""
        status = CredentialStatus(
            provider="aws",
            auth_method="sso",
            status="error",
            error="CLI not installed",
        )

        assert status.is_healthy is False
        assert status.needs_refresh is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        status = CredentialStatus(
            provider="aws",
            auth_method="sso",
            status="valid",
            ttl_seconds=3600,
            user="test-user",
        )

        result = status.to_dict()

        assert result["provider"] == "aws"
        assert result["auth_method"] == "sso"
        assert result["status"] == "valid"
        assert result["ttl_seconds"] == 3600
        assert result["user"] == "test-user"


class TestAWSCredentialChecker:
    """Tests for AWS credential checking."""

    @pytest.fixture
    def temp_aws_dir(self):
        """Create temporary AWS config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sso_cache = Path(tmpdir) / ".aws" / "sso" / "cache"
            sso_cache.mkdir(parents=True)
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_check_sso_no_cache_dir(self):
        """Test SSO check with no cache directory."""
        with patch.object(Path, "home", return_value=Path("/nonexistent")):
            status = await AWSCredentialChecker.check_sso_session()

            assert status.status == "error"
            assert "No SSO cache" in status.error

    @pytest.mark.asyncio
    async def test_check_sso_expired(self, temp_aws_dir):
        """Test SSO check with expired session."""
        cache_dir = temp_aws_dir / ".aws" / "sso" / "cache"
        cache_file = cache_dir / "test-cache.json"

        # Create expired cache file
        expired_time = "2020-01-01T00:00:00Z"
        cache_file.write_text(json.dumps({"expiresAt": expired_time}))

        with patch.object(Path, "home", return_value=temp_aws_dir):
            status = await AWSCredentialChecker.check_sso_session()

            assert status.status == "expired"

    @pytest.mark.asyncio
    async def test_check_sso_valid(self, temp_aws_dir):
        """Test SSO check with valid session."""
        cache_dir = temp_aws_dir / ".aws" / "sso" / "cache"
        cache_file = cache_dir / "test-cache.json"

        # Create valid cache file (expires in future)
        future_time = "2030-01-01T00:00:00Z"
        cache_file.write_text(json.dumps({"expiresAt": future_time}))

        with patch.object(Path, "home", return_value=temp_aws_dir):
            status = await AWSCredentialChecker.check_sso_session()

            assert status.status == "valid"
            assert status.ttl_seconds > 0

    @pytest.mark.asyncio
    async def test_check_profile_success(self):
        """Test profile check success."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "UserId": "AIDAXXXXXXXX",
            "Account": "123456789012",
            "Arn": "arn:aws:iam::123456789012:user/test",
        })

        with patch("subprocess.run", return_value=mock_result):
            status = await AWSCredentialChecker.check_profile("default")

            assert status.status == "valid"
            assert "arn:aws:iam" in status.user

    @pytest.mark.asyncio
    async def test_check_profile_expired(self):
        """Test profile check with expired credentials."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "The security token included in the request is expired"

        with patch("subprocess.run", return_value=mock_result):
            status = await AWSCredentialChecker.check_profile("default")

            assert status.status == "expired"

    @pytest.mark.asyncio
    async def test_check_profile_cli_not_installed(self):
        """Test profile check when AWS CLI not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            status = await AWSCredentialChecker.check_profile("default")

            assert status.status == "error"
            assert "not installed" in status.error


class TestGCPCredentialChecker:
    """Tests for GCP credential checking."""

    @pytest.mark.asyncio
    async def test_check_adc_no_file(self):
        """Test ADC check with no credentials file."""
        with patch.dict("os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": "/nonexistent"}):
            with patch.object(Path, "exists", return_value=False):
                status = await GCPCredentialChecker.check_adc()

                assert status.status == "error"
                assert "No ADC file" in status.error

    @pytest.mark.asyncio
    async def test_check_adc_cli_not_installed(self):
        """Test ADC check when gcloud CLI not installed."""
        with patch.object(Path, "exists", return_value=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                status = await GCPCredentialChecker.check_adc()

                assert status.status == "error"
                assert "not installed" in status.error


class TestAzureCredentialChecker:
    """Tests for Azure credential checking."""

    @pytest.fixture
    def temp_azure_dir(self):
        """Create temporary Azure config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            azure_dir = Path(tmpdir) / ".azure"
            azure_dir.mkdir(parents=True)
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_check_token_no_file(self):
        """Test token check with no cache file."""
        with patch.object(Path, "home", return_value=Path("/nonexistent")):
            status = await AzureCredentialChecker.check_token()

            assert status.status == "error"
            assert "No token cache" in status.error

    @pytest.mark.asyncio
    async def test_check_token_expired(self, temp_azure_dir):
        """Test token check with expired token."""
        token_file = temp_azure_dir / ".azure" / "msal_token_cache.json"

        # Create expired token cache
        expired_time = int(time.time()) - 3600
        token_file.write_text(json.dumps({
            "AccessToken": {
                "token1": {"expires_on": str(expired_time)}
            }
        }))

        with patch.object(Path, "home", return_value=temp_azure_dir):
            status = await AzureCredentialChecker.check_token()

            assert status.status == "expired"

    @pytest.mark.asyncio
    async def test_check_token_valid(self, temp_azure_dir):
        """Test token check with valid token."""
        token_file = temp_azure_dir / ".azure" / "msal_token_cache.json"

        # Create valid token cache (expires in future)
        future_time = int(time.time()) + 7200
        token_file.write_text(json.dumps({
            "AccessToken": {
                "token1": {"expires_on": str(future_time)}
            }
        }))

        with patch.object(Path, "home", return_value=temp_azure_dir):
            status = await AzureCredentialChecker.check_token()

            assert status.status == "valid"
            assert status.ttl_seconds > 0


class TestBitwardenChecker:
    """Tests for Bitwarden vault checking."""

    @pytest.mark.asyncio
    async def test_check_status_unlocked(self):
        """Test vault status when unlocked."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({
            "status": "unlocked",
            "userEmail": "test@example.com",
        })

        with patch("subprocess.run", return_value=mock_result):
            status = await BitwradenChecker.check_status()

            assert status.status == "valid"
            assert status.user == "test@example.com"

    @pytest.mark.asyncio
    async def test_check_status_locked(self):
        """Test vault status when locked."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"status": "locked"})

        with patch("subprocess.run", return_value=mock_result):
            status = await BitwradenChecker.check_status()

            assert status.status == "expired"

    @pytest.mark.asyncio
    async def test_check_status_not_installed(self):
        """Test vault status when Bitwarden CLI not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            status = await BitwradenChecker.check_status()

            assert status.status == "error"
            assert "not installed" in status.error


class TestKeepaliveConfig:
    """Tests for keepalive configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KeepaliveConfig({})

        assert config.interval == 300
        assert "aws_sso" in config.thresholds
        assert "gcp_adc" in config.thresholds

    def test_custom_config(self):
        """Test custom configuration."""
        config = KeepaliveConfig({
            "interval": 60,
            "thresholds": {"aws_sso": 7200},
            "providers": ["aws"],
            "aws_profiles": ["work", "personal"],
        })

        assert config.interval == 60
        assert config.thresholds["aws_sso"] == 7200
        assert config.enabled_providers == ["aws"]
        assert config.aws_profiles == ["work", "personal"]


class TestKeepaliveDaemon:
    """Tests for keepalive daemon."""

    @pytest.fixture
    def daemon(self):
        """Create daemon instance."""
        config = KeepaliveConfig({
            "interval": 1,
            "providers": [],  # Disable all providers for testing
        })
        return KeepaliveDaemon(config)

    @pytest.mark.asyncio
    async def test_check_all_empty(self, daemon):
        """Test check_all with no providers enabled."""
        results = await daemon.check_all()

        assert results == {}

    @pytest.mark.asyncio
    async def test_check_all_with_aws(self):
        """Test check_all with AWS enabled."""
        config = KeepaliveConfig({
            "providers": ["aws"],
            "aws_profiles": ["default"],
        })
        daemon = KeepaliveDaemon(config)

        with patch.object(
            AWSCredentialChecker,
            "check_sso_session",
            new_callable=AsyncMock,
            return_value=CredentialStatus(
                provider="aws",
                auth_method="sso",
                status="valid",
            ),
        ):
            with patch.object(
                AWSCredentialChecker,
                "check_profile",
                new_callable=AsyncMock,
                return_value=CredentialStatus(
                    provider="aws",
                    auth_method="profile",
                    status="valid",
                ),
            ):
                results = await daemon.check_all()

                assert "aws/sso" in results
                assert "aws/profile/default" in results

    def test_get_status(self, daemon):
        """Test getting daemon status."""
        status = daemon.get_status()

        assert "running" in status
        assert "last_check" in status
        assert "credentials" in status

    def test_get_health(self, daemon):
        """Test getting health summary."""
        health = daemon.get_health()

        assert "healthy" in health
        assert "total" in health
        assert "healthy_count" in health
        assert "unhealthy_count" in health

    @pytest.mark.asyncio
    async def test_start_stop(self, daemon):
        """Test starting and stopping daemon."""
        import asyncio

        # Initially not running
        assert daemon._running is False

        # Start
        daemon.start()
        assert daemon._running is True
        assert daemon._task is not None

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop
        daemon.stop()
        assert daemon._running is False


class TestGlobalKeepaliveDaemon:
    """Tests for global keepalive daemon functions."""

    def test_init_keepalive_daemon(self):
        """Test initializing global daemon."""
        daemon = init_keepalive_daemon({"interval": 60})

        assert daemon is not None
        assert get_keepalive_daemon() is daemon

    def test_get_keepalive_daemon_none(self):
        """Test get_keepalive_daemon when not initialized."""
        import baton.plugins.keepalive as keepalive_module
        keepalive_module._daemon = None

        daemon = get_keepalive_daemon()
        assert daemon is None
