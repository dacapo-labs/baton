"""Baton Twilio Plugin - 2-way SMS for auth flows and approvals."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, Callable

import httpx


class BatonTwilio:
    """Twilio integration for SMS-based interactions."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        twilio_config = config.get("twilio", {})

        self.account_sid = twilio_config.get("account_sid") or os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = twilio_config.get("auth_token") or os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_number = twilio_config.get("from_number") or os.environ.get("TWILIO_FROM_NUMBER")
        self.to_number = twilio_config.get("to_number") or os.environ.get("TWILIO_TO_NUMBER")

        self.enabled = bool(self.account_sid and self.auth_token and self.from_number)

        self._pending_responses: dict[str, asyncio.Future] = {}
        self._response_handlers: list[Callable[[str, str], None]] = []

    def is_configured(self) -> bool:
        """Check if Twilio is properly configured."""
        return self.enabled and bool(self.to_number)

    async def send_sms(self, message: str, to: str | None = None) -> dict[str, Any]:
        """Send an SMS message."""
        if not self.enabled:
            return {"error": "Twilio not configured", "sent": False}

        to_number = to or self.to_number
        if not to_number:
            return {"error": "No recipient number", "sent": False}

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                auth=(self.account_sid, self.auth_token),
                data={
                    "From": self.from_number,
                    "To": to_number,
                    "Body": message,
                },
            )

            if response.status_code == 201:
                return {"sent": True, "sid": response.json().get("sid")}
            else:
                return {
                    "sent": False,
                    "error": response.text,
                    "status_code": response.status_code,
                }

    async def send_and_wait(
        self,
        message: str,
        timeout_seconds: int = 300,
        to: str | None = None,
    ) -> str | None:
        """Send SMS and wait for reply."""
        result = await self.send_sms(message, to)
        if not result.get("sent"):
            return None

        future: asyncio.Future[str] = asyncio.Future()
        response_key = (to or self.to_number).replace("+", "")
        self._pending_responses[response_key] = future

        try:
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_responses.pop(response_key, None)

    def handle_incoming_sms(self, from_number: str, body: str) -> None:
        """Handle incoming SMS (called from webhook)."""
        key = from_number.replace("+", "").replace("-", "").replace(" ", "")

        if key in self._pending_responses:
            future = self._pending_responses[key]
            if not future.done():
                future.set_result(body)

        for handler in self._response_handlers:
            try:
                handler(from_number, body)
            except Exception:
                pass

    def register_handler(self, handler: Callable[[str, str], None]) -> None:
        """Register a handler for incoming SMS."""
        self._response_handlers.append(handler)

    async def send_mfa_code(self, code: str, service: str, to: str | None = None) -> dict[str, Any]:
        """Send an MFA code via SMS."""
        message = f"Baton MFA code for {service}: {code}\n\nThis code expires in 30 seconds."
        return await self.send_sms(message, to)

    async def send_approval_request(
        self,
        request_id: str,
        action: str,
        details: str,
        to: str | None = None,
    ) -> str | None:
        """Send approval request and wait for YES/NO response."""
        message = (
            f"Baton Approval [{request_id[:8]}]\n"
            f"Action: {action}\n"
            f"{details}\n\n"
            f"Reply YES to approve, NO to deny."
        )

        response = await self.send_and_wait(message, timeout_seconds=300, to=to)

        if response is None:
            return None

        response_upper = response.strip().upper()
        if response_upper in ("YES", "Y", "APPROVE", "OK"):
            return "approved"
        elif response_upper in ("NO", "N", "DENY", "REJECT"):
            return "denied"
        else:
            await self.send_sms("Please reply YES or NO.", to)
            return None

    def get_webhook_handler(self):
        """Get FastAPI route handler for Twilio webhooks."""
        from fastapi import Form, Response

        async def twilio_webhook(From: str = Form(...), Body: str = Form(...)):
            """Handle incoming Twilio SMS webhook."""
            self.handle_incoming_sms(From, Body)
            return Response(
                content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
                media_type="application/xml",
            )

        return twilio_webhook
