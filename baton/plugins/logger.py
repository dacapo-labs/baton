"""Baton Logger Plugin - JSONL logging to persistent storage."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles


class BatonLogger:
    """Log all AI interactions to JSONL files.

    Logs are stored on LUKS-encrypted EBS volume for persistence.
    Supports log rotation and retention policies.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        log_config = config.get("logging", {})
        self.log_dir = Path(os.path.expanduser(log_config.get("dir", "/data/baton/logs")))
        self.rotate_mb = log_config.get("rotate_mb", 100)
        self.retention_days = log_config.get("retention_days", 365)
        self._current_file: Path | None = None
        self._lock = asyncio.Lock()

    def _get_log_path(self) -> Path:
        """Get current log file path (daily rotation)."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"baton-{today}.jsonl"

    async def _ensure_log_dir(self) -> None:
        """Ensure log directory exists."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    async def log_request(
        self,
        request_id: str,
        model: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        zone: str | None = None,
        session: str | None = None,
        fanout_mode: str | None = None,
    ) -> None:
        """Log an incoming request."""
        await self._write_log(
            {
                "type": "request",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": model,
                "messages": messages,
                "params": params,
                "zone": zone,
                "session": session,
                "fanout_mode": fanout_mode,
            }
        )

    async def log_response(
        self,
        request_id: str,
        model: str,
        response: dict[str, Any],
        latency_ms: int,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
        cost: float | None = None,
    ) -> None:
        """Log a response from a model."""
        await self._write_log(
            {
                "type": "response",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model": model,
                "response": response,
                "latency_ms": latency_ms,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost": cost,
            }
        )

    async def log_judge_decision(
        self,
        request_id: str,
        judge_model: str,
        candidates: list[dict[str, Any]],
        winner: str,
        scores: dict[str, float],
        reasoning: str,
        query_type: str | None = None,
    ) -> None:
        """Log a judge decision for model selection."""
        await self._write_log(
            {
                "type": "judge_decision",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "judge_model": judge_model,
                "candidates": candidates,
                "winner": winner,
                "scores": scores,
                "reasoning": reasoning,
                "query_type": query_type,
            }
        )

    async def log_error(
        self,
        request_id: str,
        error_type: str,
        error_message: str,
        model: str | None = None,
        traceback: str | None = None,
    ) -> None:
        """Log an error."""
        await self._write_log(
            {
                "type": "error",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error_type": error_type,
                "error_message": error_message,
                "model": model,
                "traceback": traceback,
            }
        )

    async def log_fanout(
        self,
        request_id: str,
        mode: str,
        models: list[str],
        results: list[dict[str, Any]],
        selected_model: str | None = None,
    ) -> None:
        """Log a fan-out operation."""
        await self._write_log(
            {
                "type": "fanout",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "mode": mode,
                "models": models,
                "results": results,
                "selected_model": selected_model,
            }
        )

    async def _write_log(self, entry: dict[str, Any]) -> None:
        """Write a log entry to the current log file."""
        await self._ensure_log_dir()

        async with self._lock:
            log_path = self._get_log_path()

            if log_path.exists() and log_path.stat().st_size > self.rotate_mb * 1024 * 1024:
                timestamp = datetime.now().strftime("%H%M%S")
                rotated = log_path.with_suffix(f".{timestamp}.jsonl")
                log_path.rename(rotated)

            async with aiofiles.open(log_path, "a") as f:
                await f.write(json.dumps(entry, default=str) + "\n")

    async def cleanup_old_logs(self) -> int:
        """Remove logs older than retention_days. Returns count of deleted files."""
        if not self.log_dir.exists():
            return 0

        cutoff = time.time() - (self.retention_days * 86400)
        deleted = 0

        for log_file in self.log_dir.glob("baton-*.jsonl"):
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()
                deleted += 1

        return deleted

    def get_log_stats(self) -> dict[str, Any]:
        """Get statistics about logs."""
        if not self.log_dir.exists():
            return {"total_files": 0, "total_size_mb": 0, "oldest_log": None}

        files = list(self.log_dir.glob("baton-*.jsonl"))
        total_size = sum(f.stat().st_size for f in files)
        oldest = min(files, key=lambda f: f.stat().st_mtime) if files else None

        return {
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_log": oldest.name if oldest else None,
            "log_dir": str(self.log_dir),
        }
