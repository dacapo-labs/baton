"""Baton Router Plugin - Adaptive routing from learned judge patterns."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class BatonRouter:
    """Adaptive routing based on learned judge decisions.

    Analyzes historical judge decisions to automatically route queries
    to the model most likely to win for a given query type.
    """

    def __init__(self, config: dict[str, Any], log_dir: Path | None = None):
        self.config = config
        self.log_dir = log_dir or Path(config.get("logging", {}).get("dir", "/data/baton/logs"))
        self._win_rates: dict[str, dict[str, float]] = {}
        self._last_refresh: datetime | None = None
        self.refresh_interval = timedelta(hours=1)
        self.min_samples = 10

    def _load_judge_decisions(self, days: int = 30) -> list[dict[str, Any]]:
        """Load judge decisions from logs."""
        decisions = []
        cutoff = datetime.now() - timedelta(days=days)

        if not self.log_dir.exists():
            return decisions

        for log_file in self.log_dir.glob("baton-*.jsonl"):
            try:
                date_str = log_file.stem.replace("baton-", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    continue
            except ValueError:
                continue

            with open(log_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("type") == "judge_decision":
                            decisions.append(entry)
                    except json.JSONDecodeError:
                        continue

        return decisions

    def _calculate_win_rates(self, decisions: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        """Calculate win rates by query type and model."""
        appearances: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        wins: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for d in decisions:
            query_type = d.get("query_type") or "general"
            winner = d.get("winner")
            candidates = d.get("candidates", [])

            for c in candidates:
                model = c.get("model")
                if model:
                    appearances[query_type][model] += 1
                    if model == winner:
                        wins[query_type][model] += 1

        win_rates: dict[str, dict[str, float]] = {}
        for query_type, models in appearances.items():
            win_rates[query_type] = {}
            for model, app_count in models.items():
                if app_count >= self.min_samples:
                    win_rates[query_type][model] = wins[query_type][model] / app_count

        return win_rates

    def refresh_win_rates(self, force: bool = False) -> None:
        """Refresh win rates from logs."""
        now = datetime.now()
        if not force and self._last_refresh:
            if now - self._last_refresh < self.refresh_interval:
                return

        decisions = self._load_judge_decisions()
        self._win_rates = self._calculate_win_rates(decisions)
        self._last_refresh = now

    def get_best_model(
        self,
        query_type: str,
        available_models: list[str],
        fallback: str | None = None,
    ) -> str | None:
        """Get the best model for a query type based on learned patterns."""
        self.refresh_win_rates()

        if query_type not in self._win_rates:
            return fallback

        type_rates = self._win_rates[query_type]

        best_model = None
        best_rate = -1.0

        for model in available_models:
            if model in type_rates and type_rates[model] > best_rate:
                best_rate = type_rates[model]
                best_model = model

        return best_model or fallback

    def should_use_adaptive(self, query_type: str) -> bool:
        """Check if we have enough data to use adaptive routing."""
        self.refresh_win_rates()
        return query_type in self._win_rates and len(self._win_rates[query_type]) > 0

    def get_routing_stats(self) -> dict[str, Any]:
        """Get current routing statistics."""
        self.refresh_win_rates()

        stats = {
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "query_types": len(self._win_rates),
            "win_rates": {},
        }

        for query_type, models in self._win_rates.items():
            stats["win_rates"][query_type] = {
                model: f"{rate:.1%}" for model, rate in sorted(models.items(), key=lambda x: -x[1])
            }

        return stats

    def suggest_fanout_models(
        self,
        query_type: str,
        available_models: list[str],
        top_n: int = 3,
    ) -> list[str]:
        """Suggest top N models for fan-out based on win rates."""
        self.refresh_win_rates()

        if query_type not in self._win_rates:
            return available_models[:top_n]

        type_rates = self._win_rates[query_type]

        rated_models = [
            (model, type_rates.get(model, 0.0))
            for model in available_models
        ]
        rated_models.sort(key=lambda x: -x[1])

        return [model for model, _ in rated_models[:top_n]]
