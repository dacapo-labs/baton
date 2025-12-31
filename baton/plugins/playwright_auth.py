"""Baton Playwright Auth - Headless OAuth automation using Playwright."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Optional import - Playwright may not be installed
try:
    from playwright.async_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
        async_playwright,
    )

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = Any
    BrowserContext = Any
    Page = Any
    Playwright = Any


@dataclass
class OAuthSession:
    """Stored OAuth session data."""

    provider: str
    authenticated: bool
    user: str | None = None
    storage_state: dict | None = None
    cookies: list[dict] | None = None
    authenticated_at: float | None = None
    expires_at: float | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ttl_seconds(self) -> int | None:
        """Time to live in seconds."""
        if self.expires_at:
            return max(0, int(self.expires_at - time.time()))
        return None

    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        if self.expires_at:
            return time.time() >= self.expires_at
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "authenticated": self.authenticated,
            "user": self.user,
            "authenticated_at": self.authenticated_at,
            "expires_at": self.expires_at,
            "ttl_seconds": self.ttl_seconds,
            "error": self.error,
        }

    def to_storage(self) -> dict[str, Any]:
        """Full data for storage (includes sensitive data)."""
        return {
            "provider": self.provider,
            "authenticated": self.authenticated,
            "user": self.user,
            "storage_state": self.storage_state,
            "cookies": self.cookies,
            "authenticated_at": self.authenticated_at,
            "expires_at": self.expires_at,
            "extra": self.extra,
        }

    @classmethod
    def from_storage(cls, data: dict[str, Any]) -> "OAuthSession":
        return cls(
            provider=data.get("provider", "unknown"),
            authenticated=data.get("authenticated", False),
            user=data.get("user"),
            storage_state=data.get("storage_state"),
            cookies=data.get("cookies"),
            authenticated_at=data.get("authenticated_at"),
            expires_at=data.get("expires_at"),
            extra=data.get("extra", {}),
        )


class PlaywrightOAuthProvider(ABC):
    """Abstract base class for Playwright-based OAuth providers."""

    def __init__(
        self,
        config: dict[str, Any],
        bitwarden_getter: Any | None = None,
    ):
        self.config = config
        self.bw = bitwarden_getter
        self._session: OAuthSession | None = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier."""
        pass

    @property
    @abstractmethod
    def login_url(self) -> str:
        """URL to start OAuth flow."""
        pass

    @property
    @abstractmethod
    def success_url_pattern(self) -> str:
        """URL pattern indicating successful login."""
        pass

    @property
    def bitwarden_item_name(self) -> str:
        """Bitwarden item name for credentials."""
        return self.config.get("bitwarden_item", f"{self.provider_name} OAuth")

    @abstractmethod
    async def fill_login_form(self, page: Page, username: str, password: str) -> None:
        """Fill in the login form."""
        pass

    @abstractmethod
    async def handle_2fa(self, page: Page, totp_code: str) -> None:
        """Handle 2FA if present."""
        pass

    @abstractmethod
    async def extract_session_info(
        self, page: Page, context: BrowserContext
    ) -> OAuthSession:
        """Extract session info after successful login."""
        pass

    async def authenticate(
        self,
        headless: bool = True,
        timeout: int = 300,
    ) -> OAuthSession:
        """Run the full OAuth flow."""
        if not PLAYWRIGHT_AVAILABLE:
            return OAuthSession(
                provider=self.provider_name,
                authenticated=False,
                error="Playwright not installed. Run: pip install playwright && playwright install chromium",
            )

        # Get credentials from Bitwarden if available
        username = None
        password = None
        totp_secret = None

        if self.bw:
            try:
                creds = self.bw.get_oauth_token(self.bitwarden_item_name)
                if creds:
                    username = creds.get("username") or creds.get("client_id")
                    password = creds.get("password") or creds.get("client_secret")

                # Get TOTP if available
                totp_code = self.bw.get_totp(self.bitwarden_item_name)
            except Exception as e:
                log.warning(f"Failed to get credentials from Bitwarden: {e}")

        if not username or not password:
            return OAuthSession(
                provider=self.provider_name,
                authenticated=False,
                error="No credentials available. Configure Bitwarden or provide credentials.",
            )

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                ],
            )

            try:
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                )

                # Load existing session if available
                if self._session and self._session.storage_state:
                    await context.add_cookies(self._session.cookies or [])

                page = await context.new_page()

                # Navigate to login
                await page.goto(self.login_url)
                await page.wait_for_load_state("networkidle")

                # Check if already logged in
                if await self._check_already_logged_in(page):
                    session = await self.extract_session_info(page, context)
                    self._session = session
                    return session

                # Fill login form
                await self.fill_login_form(page, username, password)

                # Wait for navigation
                try:
                    await page.wait_for_url(
                        f"**{self.success_url_pattern}**",
                        timeout=30000,
                    )
                except Exception:
                    # Check for 2FA
                    if await self._check_2fa_required(page):
                        if totp_code:
                            await self.handle_2fa(page, totp_code)
                            await page.wait_for_url(
                                f"**{self.success_url_pattern}**",
                                timeout=30000,
                            )
                        else:
                            return OAuthSession(
                                provider=self.provider_name,
                                authenticated=False,
                                error="2FA required but no TOTP available",
                            )
                    else:
                        # Check for error messages
                        error = await self._check_login_error(page)
                        return OAuthSession(
                            provider=self.provider_name,
                            authenticated=False,
                            error=error or "Login failed",
                        )

                # Extract session
                session = await self.extract_session_info(page, context)
                self._session = session
                return session

            except Exception as e:
                log.error(f"OAuth flow failed: {e}")
                return OAuthSession(
                    provider=self.provider_name,
                    authenticated=False,
                    error=str(e),
                )
            finally:
                await browser.close()

    async def refresh_session(self, headless: bool = True) -> OAuthSession:
        """Refresh an existing session by visiting authenticated page."""
        if not self._session or not self._session.storage_state:
            return await self.authenticate(headless=headless)

        if not PLAYWRIGHT_AVAILABLE:
            return OAuthSession(
                provider=self.provider_name,
                authenticated=False,
                error="Playwright not installed",
            )

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)

            try:
                context = await browser.new_context(
                    storage_state=self._session.storage_state,
                )
                page = await context.new_page()

                # Visit a page that requires auth
                await page.goto(self.login_url.replace("/login", "/dashboard"))
                await page.wait_for_load_state("networkidle")

                # Check if still authenticated
                if await self._check_already_logged_in(page):
                    session = await self.extract_session_info(page, context)
                    self._session = session
                    return session
                else:
                    # Need full re-auth
                    return await self.authenticate(headless=headless)

            finally:
                await browser.close()

    async def _check_already_logged_in(self, page: Page) -> bool:
        """Check if already logged in."""
        current_url = page.url
        return self.success_url_pattern in current_url

    async def _check_2fa_required(self, page: Page) -> bool:
        """Check if 2FA is required."""
        selectors = [
            'input[name="totp"]',
            'input[name="code"]',
            'input[name="otp"]',
            'input[placeholder*="code"]',
            'input[placeholder*="2FA"]',
            "[data-testid*='2fa']",
            "[data-testid*='totp']",
        ]
        for selector in selectors:
            try:
                if await page.locator(selector).is_visible(timeout=1000):
                    return True
            except Exception:
                pass
        return False

    async def _check_login_error(self, page: Page) -> str | None:
        """Check for login error messages."""
        error_selectors = [
            "[class*='error']",
            "[class*='alert']",
            "[role='alert']",
            "[data-testid*='error']",
        ]
        for selector in error_selectors:
            try:
                element = page.locator(selector).first
                if await element.is_visible(timeout=1000):
                    return await element.text_content()
            except Exception:
                pass
        return None

    def get_session(self) -> OAuthSession | None:
        """Get current session."""
        return self._session

    def set_session(self, session: OAuthSession) -> None:
        """Set session from storage."""
        self._session = session


class AnthropicOAuthProvider(PlaywrightOAuthProvider):
    """Anthropic Console OAuth provider."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def login_url(self) -> str:
        return "https://console.anthropic.com/login"

    @property
    def success_url_pattern(self) -> str:
        return "/dashboard"

    async def fill_login_form(self, page: Page, username: str, password: str) -> None:
        """Fill Anthropic login form."""
        # Wait for form to load
        await page.wait_for_selector('input[type="email"], input[name="email"]')

        # Fill email
        email_input = page.locator('input[type="email"], input[name="email"]').first
        await email_input.fill(username)

        # Click continue/next if present (multi-step login)
        continue_btn = page.locator('button:has-text("Continue"), button:has-text("Next")')
        if await continue_btn.is_visible(timeout=2000):
            await continue_btn.click()
            await page.wait_for_load_state("networkidle")

        # Fill password
        password_input = page.locator('input[type="password"]').first
        await password_input.fill(password)

        # Submit
        submit_btn = page.locator(
            'button[type="submit"], button:has-text("Sign in"), button:has-text("Log in")'
        ).first
        await submit_btn.click()

    async def handle_2fa(self, page: Page, totp_code: str) -> None:
        """Handle Anthropic 2FA."""
        totp_input = page.locator(
            'input[name="totp"], input[name="code"], input[placeholder*="code"]'
        ).first
        await totp_input.fill(totp_code)

        submit_btn = page.locator(
            'button[type="submit"], button:has-text("Verify")'
        ).first
        await submit_btn.click()

    async def extract_session_info(
        self, page: Page, context: BrowserContext
    ) -> OAuthSession:
        """Extract session info from Anthropic."""
        storage_state = await context.storage_state()
        cookies = await context.cookies()

        # Try to get user info from page
        user = None
        try:
            # Look for user email in page
            user_element = page.locator("[data-testid='user-email'], .user-email")
            if await user_element.is_visible(timeout=2000):
                user = await user_element.text_content()
        except Exception:
            pass

        # Default session expiry (8 hours for Anthropic)
        expires_at = time.time() + (8 * 3600)

        return OAuthSession(
            provider=self.provider_name,
            authenticated=True,
            user=user,
            storage_state=storage_state,
            cookies=cookies,
            authenticated_at=time.time(),
            expires_at=expires_at,
        )


class OpenAIOAuthProvider(PlaywrightOAuthProvider):
    """OpenAI/ChatGPT OAuth provider."""

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def login_url(self) -> str:
        return "https://platform.openai.com/login"

    @property
    def success_url_pattern(self) -> str:
        return "/account"

    async def fill_login_form(self, page: Page, username: str, password: str) -> None:
        """Fill OpenAI login form."""
        # OpenAI uses Auth0-style login
        await page.wait_for_selector('input[type="email"], input[name="email"]')

        email_input = page.locator('input[type="email"], input[name="email"]').first
        await email_input.fill(username)

        # Click continue
        continue_btn = page.locator('button:has-text("Continue")').first
        await continue_btn.click()
        await page.wait_for_load_state("networkidle")

        # Fill password
        password_input = page.locator('input[type="password"]').first
        await password_input.fill(password)

        # Submit
        submit_btn = page.locator('button:has-text("Continue")').first
        await submit_btn.click()

    async def handle_2fa(self, page: Page, totp_code: str) -> None:
        """Handle OpenAI 2FA."""
        totp_input = page.locator('input[name="code"], input[inputmode="numeric"]').first
        await totp_input.fill(totp_code)

        submit_btn = page.locator('button:has-text("Continue")').first
        await submit_btn.click()

    async def extract_session_info(
        self, page: Page, context: BrowserContext
    ) -> OAuthSession:
        """Extract session info from OpenAI."""
        storage_state = await context.storage_state()
        cookies = await context.cookies()

        user = None
        # OpenAI session typically lasts 30 days
        expires_at = time.time() + (30 * 24 * 3600)

        return OAuthSession(
            provider=self.provider_name,
            authenticated=True,
            user=user,
            storage_state=storage_state,
            cookies=cookies,
            authenticated_at=time.time(),
            expires_at=expires_at,
        )


class GoogleOAuthProvider(PlaywrightOAuthProvider):
    """Google AI Studio OAuth provider."""

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def login_url(self) -> str:
        return "https://aistudio.google.com/"

    @property
    def success_url_pattern(self) -> str:
        return "/app"

    async def fill_login_form(self, page: Page, username: str, password: str) -> None:
        """Fill Google login form."""
        # Google login is multi-step
        await page.wait_for_selector('input[type="email"]')

        email_input = page.locator('input[type="email"]').first
        await email_input.fill(username)

        # Click next
        next_btn = page.locator('button:has-text("Next"), #identifierNext').first
        await next_btn.click()
        await page.wait_for_load_state("networkidle")

        # Wait for password field
        await page.wait_for_selector('input[type="password"]', state="visible")

        password_input = page.locator('input[type="password"]').first
        await password_input.fill(password)

        # Submit
        submit_btn = page.locator('button:has-text("Next"), #passwordNext').first
        await submit_btn.click()

    async def handle_2fa(self, page: Page, totp_code: str) -> None:
        """Handle Google 2FA."""
        # Google 2FA can be TOTP or other methods
        totp_input = page.locator('input[name="totpPin"], input[type="tel"]').first
        await totp_input.fill(totp_code)

        submit_btn = page.locator('button:has-text("Next")').first
        await submit_btn.click()

    async def extract_session_info(
        self, page: Page, context: BrowserContext
    ) -> OAuthSession:
        """Extract session info from Google."""
        storage_state = await context.storage_state()
        cookies = await context.cookies()

        user = None
        # Google sessions typically last 24 hours for OAuth tokens
        expires_at = time.time() + (24 * 3600)

        return OAuthSession(
            provider=self.provider_name,
            authenticated=True,
            user=user,
            storage_state=storage_state,
            cookies=cookies,
            authenticated_at=time.time(),
            expires_at=expires_at,
        )


class SessionStore:
    """Encrypted storage for OAuth sessions."""

    def __init__(self, storage_dir: Path | str, encryption_key: bytes | None = None):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._key = encryption_key

    def _get_path(self, provider: str) -> Path:
        return self.storage_dir / f"{provider}.json"

    def save(self, session: OAuthSession) -> bool:
        """Save session to disk."""
        try:
            path = self._get_path(session.provider)
            data = session.to_storage()

            # TODO: Encrypt data if key is available
            if self._key:
                # Implement encryption
                pass

            path.write_text(json.dumps(data, indent=2))
            path.chmod(0o600)
            return True
        except Exception as e:
            log.error(f"Failed to save session: {e}")
            return False

    def load(self, provider: str) -> OAuthSession | None:
        """Load session from disk."""
        try:
            path = self._get_path(provider)
            if not path.exists():
                return None

            data = json.loads(path.read_text())

            # TODO: Decrypt if encrypted
            if self._key:
                pass

            return OAuthSession.from_storage(data)
        except Exception as e:
            log.error(f"Failed to load session: {e}")
            return None

    def delete(self, provider: str) -> bool:
        """Delete a stored session."""
        try:
            path = self._get_path(provider)
            if path.exists():
                path.unlink()
            return True
        except Exception as e:
            log.error(f"Failed to delete session: {e}")
            return False

    def list_sessions(self) -> list[str]:
        """List all stored session providers."""
        return [p.stem for p in self.storage_dir.glob("*.json")]


class PlaywrightAuthManager:
    """Manager for Playwright-based OAuth providers."""

    def __init__(
        self,
        config: dict[str, Any],
        bitwarden_getter: Any | None = None,
        storage_dir: Path | str | None = None,
    ):
        self.config = config
        self.bw = bitwarden_getter

        self._storage = SessionStore(
            storage_dir or Path.home() / ".config" / "baton" / "sessions"
        )

        self._providers: dict[str, PlaywrightOAuthProvider] = {}
        self._init_providers()

    def _init_providers(self) -> None:
        """Initialize OAuth providers."""
        provider_classes = {
            "anthropic": AnthropicOAuthProvider,
            "openai": OpenAIOAuthProvider,
            "google": GoogleOAuthProvider,
        }

        for name, cls in provider_classes.items():
            provider_config = self.config.get("providers", {}).get(name, {})
            provider = cls(provider_config, self.bw)

            # Load existing session
            session = self._storage.load(name)
            if session and not session.is_expired:
                provider.set_session(session)

            self._providers[name] = provider

    def get_provider(self, name: str) -> PlaywrightOAuthProvider | None:
        """Get a provider by name."""
        return self._providers.get(name)

    async def authenticate(
        self,
        provider_name: str,
        headless: bool = True,
    ) -> OAuthSession:
        """Authenticate with a provider."""
        provider = self._providers.get(provider_name)
        if not provider:
            return OAuthSession(
                provider=provider_name,
                authenticated=False,
                error=f"Unknown provider: {provider_name}",
            )

        session = await provider.authenticate(headless=headless)

        if session.authenticated:
            self._storage.save(session)

        return session

    async def refresh(
        self,
        provider_name: str,
        headless: bool = True,
    ) -> OAuthSession:
        """Refresh a provider's session."""
        provider = self._providers.get(provider_name)
        if not provider:
            return OAuthSession(
                provider=provider_name,
                authenticated=False,
                error=f"Unknown provider: {provider_name}",
            )

        session = await provider.refresh_session(headless=headless)

        if session.authenticated:
            self._storage.save(session)

        return session

    async def get_status(self) -> dict[str, dict[str, Any]]:
        """Get status for all providers."""
        results = {}
        for name, provider in self._providers.items():
            session = provider.get_session()
            if session:
                results[name] = session.to_dict()
            else:
                results[name] = {
                    "provider": name,
                    "authenticated": False,
                    "error": "No session",
                }
        return results

    async def check_and_refresh_expiring(
        self,
        threshold_seconds: int = 3600,
        headless: bool = True,
    ) -> dict[str, OAuthSession]:
        """Check and refresh sessions expiring within threshold."""
        results = {}
        for name, provider in self._providers.items():
            session = provider.get_session()
            if session and session.ttl_seconds is not None:
                if session.ttl_seconds < threshold_seconds:
                    log.info(f"Refreshing {name} session (TTL: {session.ttl_seconds}s)")
                    results[name] = await provider.refresh_session(headless=headless)
        return results
