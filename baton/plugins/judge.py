"""Baton Judge Plugin - PAI rate_content pattern for response selection."""

from __future__ import annotations

import json
import re
from typing import Any

import litellm


JUDGE_PROMPT = """You are an expert evaluator tasked with selecting the best AI response.

## Task

Evaluate the candidate responses to the user's query and select the best one.

## User Query

{query}

## Candidate Responses

{candidates}

## Evaluation Criteria

Rate each response on these dimensions (1-10 scale):
- **Accuracy**: Is the information correct and factual?
- **Completeness**: Does it fully address the query?
- **Clarity**: Is it well-organized and easy to understand?
- **Helpfulness**: Does it provide actionable, useful information?
- **Conciseness**: Is it appropriately brief without sacrificing quality?

## Output Format

Respond with JSON only:
```json
{{
  "scores": {{
    "model_name": {{
      "accuracy": N,
      "completeness": N,
      "clarity": N,
      "helpfulness": N,
      "conciseness": N,
      "total": N
    }}
  }},
  "winner": "model_name",
  "reasoning": "Brief explanation of why this response is best",
  "query_type": "classification of query type (code, explanation, creative, analysis, etc.)"
}}
```
"""


class BatonJudge:
    """Judge model for selecting best response from multiple candidates."""

    def __init__(self, config: dict[str, Any], logger: Any = None):
        self.config = config
        self.logger = logger
        fanout_config = config.get("fanout", {})
        self.judge_model = fanout_config.get("judge_model", "claude-3-5-haiku-latest")

    async def select_best(
        self,
        messages: list[dict[str, Any]],
        candidates: list[dict[str, str]],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Select the best response from candidates."""
        query = self._extract_query(messages)

        candidates_text = "\n\n".join(
            f"### Response from {c['model']}\n{c['response']}"
            for c in candidates
        )

        prompt = JUDGE_PROMPT.format(
            query=query,
            candidates=candidates_text,
        )

        try:
            response = await litellm.acompletion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
            )

            content = response.choices[0].message.content
            result = self._parse_judge_response(content, candidates)

            if self.logger and request_id:
                await self.logger.log_judge_decision(
                    request_id=request_id,
                    judge_model=self.judge_model,
                    candidates=candidates,
                    winner=result["winner"],
                    scores=result["scores"],
                    reasoning=result["reasoning"],
                    query_type=result.get("query_type"),
                )

            return result

        except Exception as e:
            return {
                "winner": candidates[0]["model"] if candidates else None,
                "scores": {},
                "reasoning": f"Judge error: {str(e)}",
                "error": str(e),
            }

    def _extract_query(self, messages: list[dict[str, Any]]) -> str:
        """Extract the user's query from messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    text_parts = [
                        p.get("text", "") for p in content if p.get("type") == "text"
                    ]
                    return " ".join(text_parts)
        return ""

    def _parse_judge_response(
        self,
        content: str,
        candidates: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Parse the judge's JSON response."""
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {
                    "winner": candidates[0]["model"] if candidates else None,
                    "scores": {},
                    "reasoning": "Could not parse judge response",
                    "raw_response": content,
                }

        try:
            result = json.loads(json_str)

            winner = result.get("winner")
            candidate_models = [c["model"] for c in candidates]
            if winner not in candidate_models:
                for model in candidate_models:
                    if winner and (winner in model or model in winner):
                        winner = model
                        break
                else:
                    winner = candidates[0]["model"] if candidates else None

            return {
                "winner": winner,
                "scores": result.get("scores", {}),
                "reasoning": result.get("reasoning", ""),
                "query_type": result.get("query_type"),
            }

        except json.JSONDecodeError:
            return {
                "winner": candidates[0]["model"] if candidates else None,
                "scores": {},
                "reasoning": "JSON parse error",
                "raw_response": content,
            }

    async def classify_query(self, messages: list[dict[str, Any]]) -> str:
        """Classify the query type for routing decisions."""
        query = self._extract_query(messages)

        classify_prompt = f"""Classify this query into exactly one category:
- code: Programming, debugging, code review
- explanation: How/why questions, concepts
- creative: Writing, brainstorming, ideas
- analysis: Data analysis, comparison, evaluation
- factual: Facts, definitions, lookups
- conversation: Casual chat, small talk

Query: {query}

Respond with just the category name."""

        try:
            response = await litellm.acompletion(
                model=self.judge_model,
                messages=[{"role": "user", "content": classify_prompt}],
                temperature=0,
                max_tokens=20,
            )
            category = response.choices[0].message.content.strip().lower()

            valid_categories = ["code", "explanation", "creative", "analysis", "factual", "conversation"]
            if category in valid_categories:
                return category

        except Exception:
            pass

        return "conversation"
