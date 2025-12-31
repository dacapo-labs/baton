"""Baton Keepalive - Credential monitoring and refresh daemon."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class CredentialStatus:
    """Status of a credential."""

    provider: str
    auth_method: str
    status: str  # "valid", "expiring", "expired", "error", "unknown"
    ttl_seconds: int | None = None
    expires_at: float | None = None
    user: str | None = None
    error: str | None = None
    last_checked: float = field(default_factory=time.time)

    @property
    def is_healthy(self) -> bool:
        return self.status in ("valid", "expiring")

    @property
    def needs_refresh(self) -> bool:
        return self.status in ("expiring", "expired")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "auth_method": self.auth_method,
            "status": self.status,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at,
            "user": self.user,
            "error": self.error,
            "last_checked": self.last_checked,
        }


# Refresh thresholds (seconds before expiry to trigger refresh)
DEFAULT_THRESHOLDS = {
    "aws_sso": 3600,  # 1 hour before expiry
    "gcp_adc": 600,  # 10 minutes before expiry
    "azure_ad": 3000,  # 50 minutes before expiry
    "oauth": 3600,  # 1 hour before expiry
    "api_key": None,  # API keys don't expire (but can be invalidated)
}


class AWSCredentialChecker:
    """Check and manage AWS credentials (SSO, profiles)."""

    @staticmethod
    async def check_sso_session(profile: str | None = None) -> CredentialStatus:
        """Check AWS SSO session status."""
        cache_dir = Path.home() / ".aws" / "sso" / "cache"

        if not cache_dir.exists():
            return CredentialStatus(
                provider="aws",
                auth_method="sso",
                status="error",
                error="No SSO cache directory",
            )

        # Find most recent cache file
        cache_files = sorted(cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not cache_files:
            return CredentialStatus(
                provider="aws",
                auth_method="sso",
                status="expired",
                error="No SSO cache files",
            )

        cache_file = cache_files[-1]

        try:
            data = json.loads(cache_file.read_text())
            expires_at_str = data.get("expiresAt")

            if not expires_at_str:
                return CredentialStatus(
                    provider="aws",
                    auth_method="sso",
                    status="unknown",
                    error="No expiry in cache",
                )

            # Parse ISO date
            expires_at = datetime.fromisoformat(
                expires_at_str.replace("Z", "+00:00")
            ).timestamp()
            ttl = int(expires_at - time.time())

            if ttl <= 0:
                status = "expired"
            elif ttl < DEFAULT_THRESHOLDS["aws_sso"]:
                status = "expiring"
            else:
                status = "valid"

            return CredentialStatus(
                provider="aws",
                auth_method="sso",
                status=status,
                ttl_seconds=max(0, ttl),
                expires_at=expires_at,
            )

        except (json.JSONDecodeError, ValueError) as e:
            return CredentialStatus(
                provider="aws",
                auth_method="sso",
                status="error",
                error=str(e),
            )

    @staticmethod
    async def check_profile(profile: str) -> CredentialStatus:
        """Check if AWS profile credentials are valid."""
        try:
            result = subprocess.run(
                ["aws", "sts", "get-caller-identity", "--profile", profile],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return CredentialStatus(
                    provider="aws",
                    auth_method="profile",
                    status="valid",
                    user=data.get("Arn"),
                )
            else:
                return CredentialStatus(
                    provider="aws",
                    auth_method="profile",
                    status="expired",
                    error=result.stderr,
                )

        except subprocess.TimeoutExpired:
            return CredentialStatus(
                provider="aws",
                auth_method="profile",
                status="error",
                error="Command timed out",
            )
        except FileNotFoundError:
            return CredentialStatus(
                provider="aws",
                auth_method="profile",
                status="error",
                error="AWS CLI not installed",
            )

    @staticmethod
    async def refresh_sso(profile: str | None = None) -> bool:
        """Attempt to refresh SSO session."""
        args = ["aws", "sso", "login"]
        if profile:
            args.extend(["--profile", profile])
        args.append("--no-browser")

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False


class GCPCredentialChecker:
    """Check and manage GCP credentials (ADC, service accounts)."""

    @staticmethod
    async def check_adc() -> CredentialStatus:
        """Check GCP Application Default Credentials."""
        adc_path = Path(
            os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS",
                Path.home() / ".config" / "gcloud" / "application_default_credentials.json",
            )
        )

        if not adc_path.exists():
            return CredentialStatus(
                provider="gcp",
                auth_method="adc",
                status="error",
                error="No ADC file found",
            )

        # Try to get a token to check validity
        try:
            result = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return CredentialStatus(
                    provider="gcp",
                    auth_method="adc",
                    status="expired",
                    error=result.stderr,
                )

            token = result.stdout.strip()

            # Check token expiry via tokeninfo
            import urllib.request

            req = urllib.request.Request(
                f"https://oauth2.googleapis.com/tokeninfo?access_token={token}"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                expires_in = int(data.get("expires_in", 0))

                if expires_in <= 0:
                    status = "expired"
                elif expires_in < DEFAULT_THRESHOLDS["gcp_adc"]:
                    status = "expiring"
                else:
                    status = "valid"

                return CredentialStatus(
                    provider="gcp",
                    auth_method="adc",
                    status=status,
                    ttl_seconds=expires_in,
                    expires_at=time.time() + expires_in,
                    user=data.get("email"),
                )

        except subprocess.TimeoutExpired:
            return CredentialStatus(
                provider="gcp",
                auth_method="adc",
                status="error",
                error="Command timed out",
            )
        except FileNotFoundError:
            return CredentialStatus(
                provider="gcp",
                auth_method="adc",
                status="error",
                error="gcloud CLI not installed",
            )
        except Exception as e:
            return CredentialStatus(
                provider="gcp",
                auth_method="adc",
                status="error",
                error=str(e),
            )

    @staticmethod
    async def refresh_adc() -> bool:
        """Refresh ADC credentials."""
        try:
            # Try silent refresh first
            result = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False


class AzureCredentialChecker:
    """Check and manage Azure AD credentials."""

    @staticmethod
    async def check_token() -> CredentialStatus:
        """Check Azure AD token status."""
        token_file = Path.home() / ".azure" / "msal_token_cache.json"

        if not token_file.exists():
            return CredentialStatus(
                provider="azure",
                auth_method="ad",
                status="error",
                error="No token cache file",
            )

        try:
            data = json.loads(token_file.read_text())
            access_tokens = data.get("AccessToken", {})

            if not access_tokens:
                return CredentialStatus(
                    provider="azure",
                    auth_method="ad",
                    status="expired",
                    error="No access tokens in cache",
                )

            # Get first token's expiry
            first_token = next(iter(access_tokens.values()), {})
            expires_on = first_token.get("expires_on", 0)

            if isinstance(expires_on, str):
                expires_on = int(expires_on)

            ttl = expires_on - int(time.time())

            if ttl <= 0:
                status = "expired"
            elif ttl < DEFAULT_THRESHOLDS["azure_ad"]:
                status = "expiring"
            else:
                status = "valid"

            return CredentialStatus(
                provider="azure",
                auth_method="ad",
                status=status,
                ttl_seconds=max(0, ttl),
                expires_at=expires_on,
            )

        except (json.JSONDecodeError, ValueError) as e:
            return CredentialStatus(
                provider="azure",
                auth_method="ad",
                status="error",
                error=str(e),
            )

    @staticmethod
    async def refresh_token() -> bool:
        """Attempt silent token refresh."""
        try:
            result = subprocess.run(
                ["az", "account", "get-access-token"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False


class BitwradenChecker:
    """Check Bitwarden vault status."""

    @staticmethod
    async def check_status() -> CredentialStatus:
        """Check Bitwarden vault status."""
        try:
            result = subprocess.run(
                ["bw", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return CredentialStatus(
                    provider="bitwarden",
                    auth_method="vault",
                    status="error",
                    error=result.stderr,
                )

            data = json.loads(result.stdout)
            vault_status = data.get("status", "unknown")

            if vault_status == "unlocked":
                status = "valid"
            elif vault_status == "locked":
                status = "expired"
            else:
                status = "error"

            return CredentialStatus(
                provider="bitwarden",
                auth_method="vault",
                status=status,
                user=data.get("userEmail"),
            )

        except FileNotFoundError:
            return CredentialStatus(
                provider="bitwarden",
                auth_method="vault",
                status="error",
                error="Bitwarden CLI not installed",
            )
        except Exception as e:
            return CredentialStatus(
                provider="bitwarden",
                auth_method="vault",
                status="error",
                error=str(e),
            )


class KeepaliveConfig:
    """Configuration for keepalive daemon."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.interval = config.get("interval", 300)  # Check every 5 minutes
        self.thresholds = {**DEFAULT_THRESHOLDS, **config.get("thresholds", {})}
        self.enabled_providers = config.get(
            "providers",
            ["aws", "gcp", "azure", "bitwarden"],
        )
        self.aws_profiles = config.get("aws_profiles", [])
        self.notify_callback: Callable[[str, str], None] | None = None


class KeepaliveDaemon:
    """Background daemon for credential monitoring and refresh."""

    def __init__(
        self,
        config: KeepaliveConfig,
        playwright_auth: Any | None = None,
        cli_auth: Any | None = None,
    ):
        self.config = config
        self.playwright_auth = playwright_auth
        self.cli_auth = cli_auth
        self._running = False
        self._task: asyncio.Task | None = None
        self._statuses: dict[str, CredentialStatus] = {}
        self._last_check: float = 0

    async def check_all(self) -> dict[str, CredentialStatus]:
        """Check all configured credentials."""
        results = {}

        # AWS
        if "aws" in self.config.enabled_providers:
            # Check SSO
            results["aws/sso"] = await AWSCredentialChecker.check_sso_session()

            # Check configured profiles
            for profile in self.config.aws_profiles:
                results[f"aws/profile/{profile}"] = await AWSCredentialChecker.check_profile(
                    profile
                )

        # GCP
        if "gcp" in self.config.enabled_providers:
            results["gcp/adc"] = await GCPCredentialChecker.check_adc()

        # Azure
        if "azure" in self.config.enabled_providers:
            results["azure/ad"] = await AzureCredentialChecker.check_token()

        # Bitwarden
        if "bitwarden" in self.config.enabled_providers:
            results["bitwarden/vault"] = await BitwradenChecker.check_status()

        # CLI OAuth (Claude, Codex, Gemini)
        if self.cli_auth:
            try:
                cli_statuses = await self.cli_auth.check_all_auth()
                for name, status in cli_statuses.items():
                    results[f"{name}/oauth"] = CredentialStatus(
                        provider=name,
                        auth_method="oauth",
                        status="valid" if status.authenticated else "expired",
                        ttl_seconds=status.ttl_seconds,
                        expires_at=status.expires_at,
                        user=status.user,
                        error=status.error,
                    )
            except Exception as e:
                log.error(f"Failed to check CLI auth: {e}")

        # Playwright OAuth sessions
        if self.playwright_auth:
            try:
                pw_statuses = await self.playwright_auth.get_status()
                for name, status in pw_statuses.items():
                    if status.get("authenticated"):
                        results[f"{name}/playwright"] = CredentialStatus(
                            provider=name,
                            auth_method="playwright",
                            status="valid" if status["ttl_seconds"] and status["ttl_seconds"] > 3600 else "expiring",
                            ttl_seconds=status.get("ttl_seconds"),
                            expires_at=status.get("expires_at"),
                            user=status.get("user"),
                        )
            except Exception as e:
                log.error(f"Failed to check Playwright auth: {e}")

        self._statuses = results
        self._last_check = time.time()
        return results

    async def refresh_expiring(self) -> dict[str, bool]:
        """Refresh credentials that are expiring."""
        results = {}

        for key, status in self._statuses.items():
            if not status.needs_refresh:
                continue

            log.info(f"Attempting to refresh {key} (status: {status.status})")

            provider, auth_method = key.split("/", 1)
            success = False

            try:
                if provider == "aws" and auth_method == "sso":
                    success = await AWSCredentialChecker.refresh_sso()

                elif provider == "gcp" and auth_method == "adc":
                    success = await GCPCredentialChecker.refresh_adc()

                elif provider == "azure" and auth_method == "ad":
                    success = await AzureCredentialChecker.refresh_token()

                elif auth_method == "playwright" and self.playwright_auth:
                    session = await self.playwright_auth.refresh(provider)
                    success = session.authenticated

                elif auth_method == "oauth" and self.cli_auth:
                    # CLI OAuth typically requires user interaction
                    # Just notify instead of trying to refresh
                    self._notify(
                        f"{provider} OAuth Expiring",
                        f"Run '{self.cli_auth.get_provider(provider).cli_name} login' to refresh",
                    )

                results[key] = success

                if success:
                    log.info(f"Successfully refreshed {key}")
                else:
                    log.warning(f"Failed to refresh {key}")
                    self._notify(f"{key} Refresh Failed", f"Manual intervention required")

            except Exception as e:
                log.error(f"Error refreshing {key}: {e}")
                results[key] = False

        return results

    def _notify(self, title: str, message: str) -> None:
        """Send notification."""
        if self.config.notify_callback:
            self.config.notify_callback(title, message)
        else:
            log.warning(f"NOTIFICATION: {title} - {message}")

    async def _run_loop(self) -> None:
        """Main daemon loop."""
        while self._running:
            try:
                log.debug("Running keepalive check...")
                await self.check_all()
                await self.refresh_expiring()

            except Exception as e:
                log.error(f"Keepalive check failed: {e}")

            await asyncio.sleep(self.config.interval)

    def start(self) -> None:
        """Start the keepalive daemon."""
        if self._running:
            log.warning("Keepalive daemon already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        log.info(f"Keepalive daemon started (interval: {self.config.interval}s)")

    def stop(self) -> None:
        """Stop the keepalive daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        log.info("Keepalive daemon stopped")

    def get_status(self) -> dict[str, Any]:
        """Get current status of all credentials."""
        return {
            "running": self._running,
            "last_check": self._last_check,
            "credentials": {k: v.to_dict() for k, v in self._statuses.items()},
        }

    def get_health(self) -> dict[str, Any]:
        """Get health summary."""
        healthy = sum(1 for s in self._statuses.values() if s.is_healthy)
        unhealthy = len(self._statuses) - healthy

        return {
            "healthy": unhealthy == 0,
            "total": len(self._statuses),
            "healthy_count": healthy,
            "unhealthy_count": unhealthy,
            "unhealthy": [
                k for k, v in self._statuses.items() if not v.is_healthy
            ],
        }


# Singleton instance
_daemon: KeepaliveDaemon | None = None


def get_keepalive_daemon() -> KeepaliveDaemon | None:
    """Get the global keepalive daemon instance."""
    return _daemon


def init_keepalive_daemon(
    config: dict[str, Any],
    playwright_auth: Any | None = None,
    cli_auth: Any | None = None,
) -> KeepaliveDaemon:
    """Initialize and return the global keepalive daemon."""
    global _daemon
    _daemon = KeepaliveDaemon(
        KeepaliveConfig(config),
        playwright_auth=playwright_auth,
        cli_auth=cli_auth,
    )
    return _daemon
