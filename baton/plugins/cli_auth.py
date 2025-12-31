"""Baton CLI Auth - Wrapper for OAuth CLIs (claude, codex, gemini)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AuthStatus:
    """Authentication status for a CLI."""

    authenticated: bool
    user: str | None = None
    plan: str | None = None
    ttl_seconds: int | None = None
    expires_at: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "authenticated": self.authenticated,
            "user": self.user,
            "plan": self.plan,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at,
            "error": self.error,
        }


@dataclass
class CompletionResult:
    """Result from a CLI completion call."""

    success: bool
    content: str | None = None
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None
    raw_response: dict | None = None


class CLIAuthProvider(ABC):
    """Abstract base class for CLI-based OAuth providers."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._cached_status: AuthStatus | None = None
        self._status_cache_time: float = 0
        self._status_cache_ttl: int = 60  # Cache status for 60 seconds

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (anthropic, openai, google)."""
        pass

    @property
    @abstractmethod
    def cli_name(self) -> str:
        """Name of the CLI executable."""
        pass

    @abstractmethod
    def get_auth_file_paths(self) -> list[Path]:
        """Return possible paths to auth/session files."""
        pass

    @abstractmethod
    async def check_auth(self) -> AuthStatus:
        """Check current authentication status."""
        pass

    @abstractmethod
    async def login(self, headless: bool = False) -> AuthStatus:
        """Initiate login flow."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Execute a completion request via CLI."""
        pass

    def is_installed(self) -> bool:
        """Check if the CLI is installed."""
        return shutil.which(self.cli_name) is not None

    def _run_command(
        self,
        args: list[str],
        timeout: int = 30,
        input_data: str | None = None,
    ) -> tuple[int, str, str]:
        """Run a CLI command and return (returncode, stdout, stderr)."""
        try:
            result = subprocess.run(
                [self.cli_name] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                input=input_data,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", f"{self.cli_name} not found"
        except Exception as e:
            return -1, "", str(e)

    async def _run_command_async(
        self,
        args: list[str],
        timeout: int = 30,
        input_data: str | None = None,
    ) -> tuple[int, str, str]:
        """Run a CLI command asynchronously."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self.cli_name,
                *args,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input_data.encode() if input_data else None),
                timeout=timeout,
            )
            return (
                proc.returncode or 0,
                stdout.decode() if stdout else "",
                stderr.decode() if stderr else "",
            )
        except asyncio.TimeoutError:
            proc.kill()
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", f"{self.cli_name} not found"
        except Exception as e:
            return -1, "", str(e)

    def _read_json_file(self, path: Path) -> dict | None:
        """Read and parse a JSON file."""
        try:
            if path.exists():
                return json.loads(path.read_text())
        except (json.JSONDecodeError, IOError):
            pass
        return None


class ClaudeCliAuth(CLIAuthProvider):
    """Claude Code CLI authentication wrapper."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def cli_name(self) -> str:
        return "claude"

    def get_auth_file_paths(self) -> list[Path]:
        return [
            Path.home() / ".claude" / "auth.json",
            Path.home() / ".config" / "claude" / "auth.json",
            Path.home() / ".claude.json",
        ]

    def _get_auth_data(self) -> dict | None:
        """Read auth data from file."""
        for path in self.get_auth_file_paths():
            data = self._read_json_file(path)
            if data:
                return data
        return None

    async def check_auth(self) -> AuthStatus:
        """Check Claude CLI authentication status."""
        # Check cache first
        now = time.time()
        if (
            self._cached_status
            and now - self._status_cache_time < self._status_cache_ttl
        ):
            return self._cached_status

        if not self.is_installed():
            return AuthStatus(
                authenticated=False,
                error="Claude CLI not installed",
            )

        # Try running claude with minimal command to check auth
        returncode, stdout, stderr = await self._run_command_async(
            ["--version"],
            timeout=10,
        )

        if returncode != 0:
            return AuthStatus(
                authenticated=False,
                error=stderr or "Failed to run claude CLI",
            )

        # Check auth file for more details
        auth_data = self._get_auth_data()
        if not auth_data:
            return AuthStatus(
                authenticated=False,
                error="No auth file found",
            )

        # Parse expiry
        expires_at = None
        ttl_seconds = None
        for key in ["expiresAt", "expires_at", "expiry"]:
            if key in auth_data:
                expires_str = auth_data[key]
                try:
                    if isinstance(expires_str, (int, float)):
                        expires_at = float(expires_str)
                    else:
                        from datetime import datetime

                        expires_at = datetime.fromisoformat(
                            expires_str.replace("Z", "+00:00")
                        ).timestamp()
                    ttl_seconds = int(expires_at - now)
                except (ValueError, TypeError):
                    pass
                break

        # Determine plan from auth data
        plan = auth_data.get("plan") or auth_data.get("subscription_type")

        status = AuthStatus(
            authenticated=True,
            user=auth_data.get("email") or auth_data.get("user"),
            plan=plan,
            ttl_seconds=ttl_seconds if ttl_seconds and ttl_seconds > 0 else None,
            expires_at=expires_at,
        )

        self._cached_status = status
        self._status_cache_time = now
        return status

    async def login(self, headless: bool = False) -> AuthStatus:
        """Initiate Claude login flow."""
        if headless:
            return AuthStatus(
                authenticated=False,
                error="Headless login not yet supported for Claude CLI",
            )

        # This will open a browser
        returncode, stdout, stderr = await self._run_command_async(
            ["login"],
            timeout=300,  # 5 minutes for manual login
        )

        if returncode != 0:
            return AuthStatus(
                authenticated=False,
                error=stderr or "Login failed",
            )

        return await self.check_auth()

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Execute completion via Claude CLI."""
        # Build the prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"Human: {content}")

        prompt = "\n\n".join(prompt_parts)

        # Use claude CLI in non-interactive mode
        # Note: This is a simplified version - actual CLI usage may vary
        args = ["--print", "--model", model]

        returncode, stdout, stderr = await self._run_command_async(
            args,
            timeout=kwargs.get("timeout", 120),
            input_data=prompt,
        )

        if returncode != 0:
            return CompletionResult(
                success=False,
                error=stderr or "Completion failed",
            )

        return CompletionResult(
            success=True,
            content=stdout.strip(),
            model=model,
        )


class CodexCliAuth(CLIAuthProvider):
    """OpenAI Codex CLI authentication wrapper."""

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def cli_name(self) -> str:
        return "codex"

    def get_auth_file_paths(self) -> list[Path]:
        # Codex stores auth in system credential store, but may have config
        return [
            Path.home() / ".codex" / "auth.json",
            Path.home() / ".config" / "codex" / "config.json",
        ]

    async def check_auth(self) -> AuthStatus:
        """Check Codex CLI authentication status."""
        now = time.time()
        if (
            self._cached_status
            and now - self._status_cache_time < self._status_cache_ttl
        ):
            return self._cached_status

        if not self.is_installed():
            return AuthStatus(
                authenticated=False,
                error="Codex CLI not installed",
            )

        # Check auth status
        returncode, stdout, stderr = await self._run_command_async(
            ["auth", "status"],
            timeout=10,
        )

        if returncode != 0:
            # Try whoami as fallback
            returncode, stdout, stderr = await self._run_command_async(
                ["whoami"],
                timeout=10,
            )

        if returncode != 0:
            return AuthStatus(
                authenticated=False,
                error=stderr or "Not authenticated",
            )

        # Parse output for user info
        user = None
        plan = None
        for line in stdout.split("\n"):
            line = line.strip().lower()
            if "email:" in line or "user:" in line:
                user = line.split(":", 1)[-1].strip()
            if "plan:" in line or "subscription:" in line:
                plan = line.split(":", 1)[-1].strip()

        status = AuthStatus(
            authenticated=True,
            user=user,
            plan=plan,
        )

        self._cached_status = status
        self._status_cache_time = now
        return status

    async def login(self, headless: bool = False) -> AuthStatus:
        """Initiate Codex login flow."""
        if headless:
            return AuthStatus(
                authenticated=False,
                error="Headless login not yet supported for Codex CLI",
            )

        returncode, stdout, stderr = await self._run_command_async(
            ["login"],
            timeout=300,
        )

        if returncode != 0:
            return AuthStatus(
                authenticated=False,
                error=stderr or "Login failed",
            )

        return await self.check_auth()

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Execute completion via Codex CLI."""
        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(content)

        prompt = "\n\n".join(prompt_parts)

        # Codex CLI usage - adjust based on actual CLI interface
        args = ["--model", model, "--quiet"]

        returncode, stdout, stderr = await self._run_command_async(
            args,
            timeout=kwargs.get("timeout", 120),
            input_data=prompt,
        )

        if returncode != 0:
            return CompletionResult(
                success=False,
                error=stderr or "Completion failed",
            )

        return CompletionResult(
            success=True,
            content=stdout.strip(),
            model=model,
        )


class GeminiCliAuth(CLIAuthProvider):
    """Google Gemini CLI authentication wrapper."""

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def cli_name(self) -> str:
        return "gemini"

    def get_auth_file_paths(self) -> list[Path]:
        return [
            Path.home() / ".gemini" / "auth.json",
            Path.home() / ".config" / "gemini" / "credentials.json",
        ]

    async def check_auth(self) -> AuthStatus:
        """Check Gemini CLI authentication status."""
        now = time.time()
        if (
            self._cached_status
            and now - self._status_cache_time < self._status_cache_ttl
        ):
            return self._cached_status

        if not self.is_installed():
            return AuthStatus(
                authenticated=False,
                error="Gemini CLI not installed",
            )

        # Check auth status
        returncode, stdout, stderr = await self._run_command_async(
            ["auth", "status"],
            timeout=10,
        )

        if returncode != 0:
            return AuthStatus(
                authenticated=False,
                error=stderr or "Not authenticated",
            )

        # Parse output
        user = None
        plan = None
        for line in stdout.split("\n"):
            line_lower = line.strip().lower()
            if "email:" in line_lower or "account:" in line_lower:
                user = line.split(":", 1)[-1].strip()
            if "plan:" in line_lower or "tier:" in line_lower:
                plan = line.split(":", 1)[-1].strip()

        status = AuthStatus(
            authenticated=True,
            user=user,
            plan=plan,
        )

        self._cached_status = status
        self._status_cache_time = now
        return status

    async def login(self, headless: bool = False) -> AuthStatus:
        """Initiate Gemini login flow."""
        if headless:
            return AuthStatus(
                authenticated=False,
                error="Headless login not yet supported for Gemini CLI",
            )

        returncode, stdout, stderr = await self._run_command_async(
            ["auth", "login"],
            timeout=300,
        )

        if returncode != 0:
            return AuthStatus(
                authenticated=False,
                error=stderr or "Login failed",
            )

        return await self.check_auth()

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> CompletionResult:
        """Execute completion via Gemini CLI."""
        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            content = msg.get("content", "")
            prompt_parts.append(content)

        prompt = "\n\n".join(prompt_parts)

        # Gemini CLI usage - adjust based on actual CLI interface
        args = ["--model", model]

        returncode, stdout, stderr = await self._run_command_async(
            args,
            timeout=kwargs.get("timeout", 120),
            input_data=prompt,
        )

        if returncode != 0:
            return CompletionResult(
                success=False,
                error=stderr or "Completion failed",
            )

        return CompletionResult(
            success=True,
            content=stdout.strip(),
            model=model,
        )


class CLIAuthManager:
    """Manager for all CLI-based OAuth providers."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._providers: dict[str, CLIAuthProvider] = {}
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize all CLI providers."""
        provider_classes = {
            "anthropic": ClaudeCliAuth,
            "openai": CodexCliAuth,
            "google": GeminiCliAuth,
        }

        for name, cls in provider_classes.items():
            provider_config = self.config.get("providers", {}).get(name, {})
            self._providers[name] = cls(provider_config)

    def get_provider(self, name: str) -> CLIAuthProvider | None:
        """Get a provider by name."""
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """List all available providers."""
        return list(self._providers.keys())

    async def check_all_auth(self) -> dict[str, AuthStatus]:
        """Check auth status for all providers."""
        results = {}
        for name, provider in self._providers.items():
            if provider.is_installed():
                results[name] = await provider.check_auth()
            else:
                results[name] = AuthStatus(
                    authenticated=False,
                    error=f"{provider.cli_name} not installed",
                )
        return results

    async def get_status(self) -> dict[str, dict[str, Any]]:
        """Get status for all providers as dicts."""
        statuses = await self.check_all_auth()
        return {name: status.to_dict() for name, status in statuses.items()}
