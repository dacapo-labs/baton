"""Baton Fan-out Plugin - Multi-model parallel queries with aggregation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any
from uuid import uuid4

import litellm


class FanoutMode(Enum):
    """Fan-out aggregation modes."""

    FIRST = "first"  # Return first successful response
    ALL = "all"  # Return all responses
    RACE = "race"  # Return fastest response
    VOTE = "vote"  # Majority vote (for classification tasks)
    JUDGE = "judge"  # Use a judge model to pick best response


@dataclass
class FanoutResult:
    """Result from a single model in a fan-out."""

    model: str
    response: dict[str, Any] | None
    error: str | None
    latency_ms: int
    tokens_in: int | None
    tokens_out: int | None


class BatonFanout:
    """Execute queries across multiple models in parallel with aggregation."""

    def __init__(self, config: dict[str, Any], judge: Any = None):
        self.config = config
        self.judge = judge
        fanout_config = config.get("fanout", {})
        self.default_mode = FanoutMode(fanout_config.get("default_mode", "first"))
        self.timeout_seconds = fanout_config.get("timeout_seconds", 60)

    def resolve_alias(self, model_or_alias: str) -> list[str]:
        """Resolve a model alias to a list of models."""
        aliases = self.config.get("aliases", {})
        if model_or_alias in aliases:
            return aliases[model_or_alias]
        return [model_or_alias]

    async def _call_model(
        self,
        model: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> FanoutResult:
        """Call a single model and return result."""
        import time

        start = time.perf_counter()

        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                **params,
            )

            latency_ms = int((time.perf_counter() - start) * 1000)

            return FanoutResult(
                model=model,
                response=response.model_dump() if hasattr(response, "model_dump") else dict(response),
                error=None,
                latency_ms=latency_ms,
                tokens_in=response.usage.prompt_tokens if response.usage else None,
                tokens_out=response.usage.completion_tokens if response.usage else None,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return FanoutResult(
                model=model,
                response=None,
                error=str(e),
                latency_ms=latency_ms,
                tokens_in=None,
                tokens_out=None,
            )

    async def execute(
        self,
        models: list[str],
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        mode: FanoutMode | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute fan-out query across multiple models."""
        mode = mode or self.default_mode
        request_id = request_id or str(uuid4())

        all_models = []
        for m in models:
            all_models.extend(self.resolve_alias(m))

        tasks = [
            asyncio.wait_for(
                self._call_model(model, messages, params),
                timeout=self.timeout_seconds,
            )
            for model in all_models
        ]

        if mode == FanoutMode.RACE:
            done, pending = await asyncio.wait(
                [asyncio.create_task(t) for t in tasks],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            for task in done:
                result = task.result()
                if result.response:
                    return {
                        "mode": "race",
                        "selected_model": result.model,
                        "response": result.response,
                        "latency_ms": result.latency_ms,
                        "all_results": [result],
                    }

            return {"mode": "race", "error": "All models failed", "all_results": []}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    FanoutResult(
                        model=all_models[i],
                        response=None,
                        error=str(result),
                        latency_ms=0,
                        tokens_in=None,
                        tokens_out=None,
                    )
                )
            else:
                processed_results.append(result)

        successful = [r for r in processed_results if r.response]

        if not successful:
            return {
                "mode": mode.value,
                "error": "All models failed",
                "all_results": [{"model": r.model, "error": r.error} for r in processed_results],
            }

        if mode == FanoutMode.FIRST:
            selected = successful[0]
            return {
                "mode": "first",
                "selected_model": selected.model,
                "response": selected.response,
                "latency_ms": selected.latency_ms,
            }

        elif mode == FanoutMode.ALL:
            return {
                "mode": "all",
                "responses": [
                    {
                        "model": r.model,
                        "response": r.response,
                        "latency_ms": r.latency_ms,
                    }
                    for r in successful
                ],
            }

        elif mode == FanoutMode.VOTE:
            responses = {}
            for r in successful:
                text = self._extract_text(r.response)
                normalized = text.strip().lower()
                if normalized not in responses:
                    responses[normalized] = {"text": text, "votes": 0, "models": []}
                responses[normalized]["votes"] += 1
                responses[normalized]["models"].append(r.model)

            winner = max(responses.values(), key=lambda x: x["votes"])
            return {
                "mode": "vote",
                "selected_text": winner["text"],
                "votes": winner["votes"],
                "total_models": len(successful),
                "voting_models": winner["models"],
                "all_votes": responses,
            }

        elif mode == FanoutMode.JUDGE:
            if not self.judge:
                selected = successful[0]
                return {
                    "mode": "judge",
                    "error": "Judge not configured, falling back to first",
                    "selected_model": selected.model,
                    "response": selected.response,
                }

            judge_result = await self.judge.select_best(
                messages=messages,
                candidates=[
                    {"model": r.model, "response": self._extract_text(r.response)}
                    for r in successful
                ],
                request_id=request_id,
            )

            winner_model = judge_result.get("winner")
            winner_result = next(
                (r for r in successful if r.model == winner_model),
                successful[0],
            )

            return {
                "mode": "judge",
                "selected_model": winner_result.model,
                "response": winner_result.response,
                "latency_ms": winner_result.latency_ms,
                "judge_scores": judge_result.get("scores", {}),
                "judge_reasoning": judge_result.get("reasoning", ""),
            }

        return {"error": f"Unknown mode: {mode}"}

    def _extract_text(self, response: dict[str, Any]) -> str:
        """Extract text content from a response."""
        try:
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return message.get("content", "")
        except (KeyError, IndexError, TypeError):
            pass
        return ""
