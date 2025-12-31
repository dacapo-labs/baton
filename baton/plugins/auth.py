"""Baton Auth Plugin - Bitwarden credential fetching with caching."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CachedCredential:
    """Cached credential with TTL."""

    value: str
    expires_at: float


class BatonAuth:
    """Bitwarden-based authentication for AI providers.

    Fetches API keys and OAuth tokens from Bitwarden vault.
    Supports auto-TOTP for services requiring 2FA.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.bw_config = config.get("auth", {}).get("bitwarden", {})
        self.enabled = self.bw_config.get("enabled", True)
        self.session_file = Path(
            os.path.expanduser(self.bw_config.get("session_file", "~/.config/bitwarden/session"))
        )
        self.cache_ttl = self.bw_config.get("cache_ttl", 3600)
        self._cache: dict[str, CachedCredential] = {}
        self._bw_session: str | None = None

    def _get_bw_session(self) -> str | None:
        """Get Bitwarden session from environment or session file."""
        session = os.environ.get("BW_SESSION")
        if session:
            return session

        if self.session_file.exists():
            session = self.session_file.read_text().strip()
            if session:
                result = subprocess.run(
                    ["bw", "status", "--session", session],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    status = json.loads(result.stdout)
                    if status.get("status") == "unlocked":
                        return session

        return None

    def _bw_get(self, item_name: str, field: str = "password") -> str | None:
        """Get a field from a Bitwarden item."""
        session = self._get_bw_session()
        if not session:
            return None

        try:
            result = subprocess.run(
                ["bw", "get", "item", item_name, "--session", session],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return None

            item = json.loads(result.stdout)

            if field == "password":
                return item.get("login", {}).get("password")
            elif field == "username":
                return item.get("login", {}).get("username")
            elif field == "totp":
                totp_result = subprocess.run(
                    ["bw", "get", "totp", item_name, "--session", session],
                    capture_output=True,
                    text=True,
                )
                if totp_result.returncode == 0:
                    return totp_result.stdout.strip()
            else:
                fields = item.get("fields", [])
                for f in fields:
                    if f.get("name") == field:
                        return f.get("value")

            return None

        except (json.JSONDecodeError, subprocess.SubprocessError):
            return None

    def get_api_key(self, provider: str, force_refresh: bool = False) -> str | None:
        """Get API key for a provider from Bitwarden."""
        if not self.enabled:
            return None

        cache_key = f"api_key:{provider}"

        if not force_refresh and cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() < cached.expires_at:
                return cached.value

        bw_items = {
            "anthropic": ["Anthropic API", "anthropic-api", "Claude API"],
            "openai": ["OpenAI API", "openai-api", "ChatGPT API"],
            "google": ["Google AI API", "google-ai-api", "Gemini API"],
            "deepseek": ["DeepSeek API", "deepseek-api"],
            "mistral": ["Mistral API", "mistral-api"],
            "groq": ["Groq API", "groq-api"],
            "together": ["Together API", "together-api"],
            "openrouter": ["OpenRouter API", "openrouter-api"],
        }

        item_names = bw_items.get(provider.lower(), [f"{provider} API"])

        for item_name in item_names:
            api_key = self._bw_get(item_name, "password")
            if api_key:
                self._cache[cache_key] = CachedCredential(
                    value=api_key,
                    expires_at=time.time() + self.cache_ttl,
                )
                return api_key

        return None

    def get_totp(self, service: str) -> str | None:
        """Get current TOTP code for a service."""
        if not self.enabled:
            return None
        return self._bw_get(service, "totp")

    async def refresh_all_keys(self) -> dict[str, bool]:
        """Refresh all provider API keys from Bitwarden."""
        providers = ["anthropic", "openai", "google", "deepseek", "mistral", "groq"]
        results = {}

        for provider in providers:
            key = self.get_api_key(provider, force_refresh=True)
            results[provider] = key is not None

        return results

    def export_env_vars(self) -> dict[str, str]:
        """Export API keys as environment variables."""
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        env_vars = {}
        for provider, env_var in env_map.items():
            key = self.get_api_key(provider)
            if key:
                env_vars[env_var] = key

        return env_vars

    def save_session(self, path: Path | None = None) -> bool:
        """Save current BW session for later restore."""
        path = path or Path("/data/baton/auth/bw_session")
        session = self._get_bw_session()
        if not session:
            return False

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(session)
        path.chmod(0o600)
        return True

    def restore_session(self, path: Path | None = None) -> bool:
        """Restore BW session from saved file."""
        path = path or Path("/data/baton/auth/bw_session")
        if not path.exists():
            return False

        session = path.read_text().strip()
        if not session:
            return False

        result = subprocess.run(
            ["bw", "status", "--session", session],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False

        try:
            status = json.loads(result.stdout)
            if status.get("status") != "unlocked":
                return False
        except json.JSONDecodeError:
            return False

        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_file.write_text(session)
        self.session_file.chmod(0o600)
        os.environ["BW_SESSION"] = session
        return True

    # =========================================================================
    # Multi-Auth Support (OAuth, API Key, GCP ADC, AWS)
    # =========================================================================

    def get_oauth_token(self, service: str, force_refresh: bool = False) -> dict[str, str] | None:
        """Get OAuth credentials for a service from Bitwarden."""
        if not self.enabled:
            return None

        cache_key = f"oauth:{service}"

        if not force_refresh and cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() < cached.expires_at:
                return json.loads(cached.value)

        session = self._get_bw_session()
        if not session:
            return None

        oauth_items = [f"{service} OAuth", f"{service}-oauth", service]

        for item_name in oauth_items:
            try:
                result = subprocess.run(
                    ["bw", "get", "item", item_name, "--session", session],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    continue

                item = json.loads(result.stdout)
                fields = {f.get("name"): f.get("value") for f in item.get("fields", [])}

                oauth_data = {}
                if "client_id" in fields:
                    oauth_data["client_id"] = fields["client_id"]
                if "client_secret" in fields:
                    oauth_data["client_secret"] = fields["client_secret"]
                if "refresh_token" in fields:
                    oauth_data["refresh_token"] = fields["refresh_token"]

                login = item.get("login", {})
                if not oauth_data.get("client_id") and login.get("username"):
                    oauth_data["client_id"] = login["username"]
                if not oauth_data.get("client_secret") and login.get("password"):
                    oauth_data["client_secret"] = login["password"]

                if oauth_data.get("client_id"):
                    self._cache[cache_key] = CachedCredential(
                        value=json.dumps(oauth_data),
                        expires_at=time.time() + self.cache_ttl,
                    )
                    return oauth_data

            except (json.JSONDecodeError, subprocess.SubprocessError):
                continue

        return None

    def get_gcp_credentials(self) -> dict[str, Any] | None:
        """Get GCP credentials - supports ADC, service account, or API key."""
        adc_path = Path(os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS",
            os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
        ))

        if adc_path.exists():
            try:
                with open(adc_path) as f:
                    creds = json.load(f)
                return {"type": "adc", "credentials": creds}
            except (json.JSONDecodeError, IOError):
                pass

        service_account = self._bw_get("GCP Service Account", "password")
        if service_account:
            try:
                creds = json.loads(service_account)
                return {"type": "service_account", "credentials": creds}
            except json.JSONDecodeError:
                pass

        api_key = self.get_api_key("google")
        if api_key:
            return {"type": "api_key", "key": api_key}

        return None

    def get_aws_credentials(self, profile: str | None = None) -> dict[str, str] | None:
        """Get AWS credentials - supports profiles, env vars, or Bitwarden."""
        profile = profile or os.environ.get("AWS_PROFILE")

        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            return {
                "access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
                "secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
                "session_token": os.environ.get("AWS_SESSION_TOKEN"),
            }

        creds_file = Path(os.path.expanduser("~/.aws/credentials"))
        if creds_file.exists() and profile:
            try:
                import configparser
                config = configparser.ConfigParser()
                config.read(creds_file)

                if profile in config:
                    return {
                        "access_key_id": config[profile].get("aws_access_key_id"),
                        "secret_access_key": config[profile].get("aws_secret_access_key"),
                        "session_token": config[profile].get("aws_session_token"),
                    }
            except Exception:
                pass

        if self.enabled:
            item_names = [f"AWS {profile}" if profile else "AWS", "AWS Credentials"]
            for item_name in item_names:
                session = self._get_bw_session()
                if not session:
                    break

                try:
                    result = subprocess.run(
                        ["bw", "get", "item", item_name, "--session", session],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        continue

                    item = json.loads(result.stdout)
                    fields = {f.get("name"): f.get("value") for f in item.get("fields", [])}

                    if "access_key_id" in fields:
                        return {
                            "access_key_id": fields.get("access_key_id"),
                            "secret_access_key": fields.get("secret_access_key"),
                            "session_token": fields.get("session_token"),
                        }
                except (json.JSONDecodeError, subprocess.SubprocessError):
                    continue

        return None

    async def auto_totp_auth(
        self,
        service: str,
        login_callback,
        max_attempts: int = 3,
    ) -> bool:
        """Perform automatic TOTP authentication."""
        username = self._bw_get(service, "username")
        password = self._bw_get(service, "password")

        if not username or not password:
            return False

        for attempt in range(max_attempts):
            totp_code = self.get_totp(service)
            if not totp_code:
                return False

            try:
                success = await login_callback(username, password, totp_code)
                if success:
                    return True
                await asyncio.sleep(2)
            except Exception:
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(1)

        return False

    def get_all_credentials(self, provider: str) -> dict[str, Any]:
        """Get all available credentials for a provider."""
        creds: dict[str, Any] = {}

        api_key = self.get_api_key(provider)
        if api_key:
            creds["api_key"] = api_key

        oauth = self.get_oauth_token(provider)
        if oauth:
            creds["oauth"] = oauth

        if provider.lower() in ("google", "vertex", "gemini"):
            gcp = self.get_gcp_credentials()
            if gcp:
                creds["gcp"] = gcp

        if provider.lower() in ("aws", "bedrock", "sagemaker"):
            aws = self.get_aws_credentials()
            if aws:
                creds["aws"] = aws

        return creds
