"""Baton Model Checker - Check model availability across providers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model."""

    model_id: str
    provider: str
    name: str | None = None
    accessible: bool = False
    requires_agreement: bool = False
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "name": self.name,
            "accessible": self.accessible,
            "requires_agreement": self.requires_agreement,
            "input_modalities": self.input_modalities,
            "output_modalities": self.output_modalities,
            "regions": self.regions,
            "error": self.error,
        }


# Known Bedrock model IDs by provider
BEDROCK_MODELS = {
    "anthropic": [
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-3-5-haiku-20241022-v1:0",
        "anthropic.claude-3-opus-20240229-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-v2:1",
        "anthropic.claude-v2",
        "anthropic.claude-instant-v1",
    ],
    "amazon": [
        "amazon.titan-text-premier-v1:0",
        "amazon.titan-text-express-v1",
        "amazon.titan-text-lite-v1",
        "amazon.titan-embed-text-v2:0",
        "amazon.titan-embed-text-v1",
        "amazon.titan-embed-image-v1",
        "amazon.nova-pro-v1:0",
        "amazon.nova-lite-v1:0",
        "amazon.nova-micro-v1:0",
    ],
    "meta": [
        "meta.llama3-2-90b-instruct-v1:0",
        "meta.llama3-2-11b-instruct-v1:0",
        "meta.llama3-2-3b-instruct-v1:0",
        "meta.llama3-2-1b-instruct-v1:0",
        "meta.llama3-1-405b-instruct-v1:0",
        "meta.llama3-1-70b-instruct-v1:0",
        "meta.llama3-1-8b-instruct-v1:0",
        "meta.llama3-70b-instruct-v1:0",
        "meta.llama3-8b-instruct-v1:0",
    ],
    "mistral": [
        "mistral.mistral-large-2407-v1:0",
        "mistral.mistral-large-2402-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral.mistral-7b-instruct-v0:2",
    ],
    "cohere": [
        "cohere.command-r-plus-v1:0",
        "cohere.command-r-v1:0",
        "cohere.command-text-v14",
        "cohere.command-light-text-v14",
        "cohere.embed-english-v3",
        "cohere.embed-multilingual-v3",
    ],
    "ai21": [
        "ai21.jamba-1-5-large-v1:0",
        "ai21.jamba-1-5-mini-v1:0",
        "ai21.j2-ultra-v1",
        "ai21.j2-mid-v1",
    ],
}

# Known Vertex AI model IDs
VERTEX_MODELS = {
    "google": [
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.0-pro",
        "gemini-1.0-pro-vision",
        "text-bison",
        "text-bison-32k",
        "text-unicorn",
        "chat-bison",
        "chat-bison-32k",
        "codechat-bison",
        "codechat-bison-32k",
        "code-bison",
        "code-bison-32k",
        "code-gecko",
        "textembedding-gecko",
        "textembedding-gecko-multilingual",
    ],
    "anthropic": [
        # Claude models available via Vertex AI
        "claude-3-5-sonnet-v2@20241022",
        "claude-3-5-haiku@20241022",
        "claude-3-opus@20240229",
        "claude-3-sonnet@20240229",
        "claude-3-haiku@20240307",
    ],
    "meta": [
        "llama-3.2-90b-vision-instruct-maas",
        "llama-3.2-11b-vision-instruct-maas",
        "llama-3.1-405b-instruct-maas",
        "llama-3.1-70b-instruct-maas",
        "llama-3.1-8b-instruct-maas",
    ],
    "mistral": [
        "mistral-large@2407",
        "mistral-nemo@2407",
        "codestral@2405",
    ],
}


class BedrockModelChecker:
    """Check model availability on AWS Bedrock."""

    def __init__(self, profile: str | None = None, region: str = "us-east-1"):
        self.profile = profile
        self.region = region
        self._client = None
        self._runtime_client = None

    def _get_client(self):
        """Get or create Bedrock client."""
        if self._client is None:
            try:
                import boto3

                session_kwargs = {}
                if self.profile:
                    session_kwargs["profile_name"] = self.profile

                session = boto3.Session(**session_kwargs)
                self._client = session.client("bedrock", region_name=self.region)
            except ImportError:
                raise ImportError("boto3 is required for Bedrock model checking")
        return self._client

    def _get_runtime_client(self):
        """Get or create Bedrock Runtime client."""
        if self._runtime_client is None:
            try:
                import boto3

                session_kwargs = {}
                if self.profile:
                    session_kwargs["profile_name"] = self.profile

                session = boto3.Session(**session_kwargs)
                self._runtime_client = session.client(
                    "bedrock-runtime", region_name=self.region
                )
            except ImportError:
                raise ImportError("boto3 is required for Bedrock model checking")
        return self._runtime_client

    async def list_all_models(self) -> list[ModelInfo]:
        """List all foundation models available in Bedrock."""
        try:
            client = self._get_client()
            response = client.list_foundation_models()

            models = []
            for model in response.get("modelSummaries", []):
                model_id = model.get("modelId", "")
                provider = model.get("providerName", "unknown").lower()

                models.append(
                    ModelInfo(
                        model_id=model_id,
                        provider=f"bedrock/{provider}",
                        name=model.get("modelName"),
                        input_modalities=model.get("inputModalities", []),
                        output_modalities=model.get("outputModalities", []),
                        # Note: This doesn't tell us if we have access
                        accessible=True,  # Assume accessible, verify separately
                    )
                )

            return models

        except Exception as e:
            log.error(f"Failed to list Bedrock models: {e}")
            return []

    async def check_model_access(self, model_id: str) -> ModelInfo:
        """Check if a specific model is accessible."""
        provider = model_id.split(".")[0] if "." in model_id else "unknown"

        try:
            runtime = self._get_runtime_client()

            # Minimal request to check access
            # Different models need different request formats
            if "anthropic" in model_id:
                body = json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                )
            elif "titan" in model_id:
                body = json.dumps(
                    {
                        "inputText": "hi",
                        "textGenerationConfig": {"maxTokenCount": 1},
                    }
                )
            elif "llama" in model_id or "meta" in model_id:
                body = json.dumps({"prompt": "hi", "max_gen_len": 1})
            elif "mistral" in model_id:
                body = json.dumps({"prompt": "hi", "max_tokens": 1})
            elif "cohere" in model_id:
                body = json.dumps({"prompt": "hi", "max_tokens": 1})
            else:
                body = json.dumps({"prompt": "hi", "max_tokens": 1})

            # Try to invoke the model
            response = runtime.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )

            return ModelInfo(
                model_id=model_id,
                provider=f"bedrock/{provider}",
                accessible=True,
            )

        except Exception as e:
            error_str = str(e)

            # Check for specific error types
            if "AccessDeniedException" in error_str:
                return ModelInfo(
                    model_id=model_id,
                    provider=f"bedrock/{provider}",
                    accessible=False,
                    error="Access denied - model not enabled",
                )
            elif "ValidationException" in error_str:
                # Model exists but request format was wrong - still accessible
                return ModelInfo(
                    model_id=model_id,
                    provider=f"bedrock/{provider}",
                    accessible=True,
                )
            elif "ResourceNotFoundException" in error_str:
                return ModelInfo(
                    model_id=model_id,
                    provider=f"bedrock/{provider}",
                    accessible=False,
                    error="Model not found in region",
                )
            else:
                return ModelInfo(
                    model_id=model_id,
                    provider=f"bedrock/{provider}",
                    accessible=False,
                    error=error_str[:200],
                )

    async def check_all_known_models(
        self, check_access: bool = True
    ) -> dict[str, list[ModelInfo]]:
        """Check accessibility of all known Bedrock models."""
        results: dict[str, list[ModelInfo]] = {}

        for provider, model_ids in BEDROCK_MODELS.items():
            results[provider] = []

            for model_id in model_ids:
                if check_access:
                    model_info = await self.check_model_access(model_id)
                else:
                    model_info = ModelInfo(
                        model_id=model_id,
                        provider=f"bedrock/{provider}",
                        accessible=True,  # Assume accessible
                    )
                results[provider].append(model_info)

        return results

    async def get_accessible_models(self) -> list[str]:
        """Get list of accessible model IDs."""
        all_models = await self.check_all_known_models(check_access=True)

        accessible = []
        for provider_models in all_models.values():
            for model in provider_models:
                if model.accessible:
                    accessible.append(model.model_id)

        return accessible


class VertexModelChecker:
    """Check model availability on Google Vertex AI."""

    def __init__(self, project: str | None = None, region: str = "us-central1"):
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.region = region

    async def list_models_via_gcloud(self) -> list[ModelInfo]:
        """List models using gcloud CLI."""
        try:
            cmd = [
                "gcloud",
                "ai",
                "models",
                "list",
                "--region",
                self.region,
                "--format",
                "json",
            ]
            if self.project:
                cmd.extend(["--project", self.project])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                log.error(f"gcloud command failed: {result.stderr}")
                return []

            models_data = json.loads(result.stdout) if result.stdout else []

            models = []
            for model in models_data:
                models.append(
                    ModelInfo(
                        model_id=model.get("name", ""),
                        provider="vertex",
                        name=model.get("displayName"),
                        accessible=True,
                    )
                )

            return models

        except FileNotFoundError:
            log.error("gcloud CLI not found")
            return []
        except subprocess.TimeoutExpired:
            log.error("gcloud command timed out")
            return []
        except Exception as e:
            log.error(f"Failed to list Vertex models: {e}")
            return []

    async def check_model_access(self, model_id: str) -> ModelInfo:
        """Check if a specific model is accessible via API call."""
        provider = "google"
        for p, models in VERTEX_MODELS.items():
            if model_id in models:
                provider = p
                break

        try:
            # Try using the Google Gen AI SDK
            from google import genai

            client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.region,
            )

            # Try to get model info
            # This varies by model type
            model = client.models.get(model_id)

            return ModelInfo(
                model_id=model_id,
                provider=f"vertex/{provider}",
                name=getattr(model, "display_name", model_id),
                accessible=True,
            )

        except ImportError:
            # Fall back to REST API check
            return await self._check_model_rest(model_id, provider)
        except Exception as e:
            error_str = str(e)

            if "403" in error_str or "PERMISSION_DENIED" in error_str:
                return ModelInfo(
                    model_id=model_id,
                    provider=f"vertex/{provider}",
                    accessible=False,
                    error="Permission denied",
                )
            elif "404" in error_str or "NOT_FOUND" in error_str:
                return ModelInfo(
                    model_id=model_id,
                    provider=f"vertex/{provider}",
                    accessible=False,
                    error="Model not found",
                )
            else:
                return ModelInfo(
                    model_id=model_id,
                    provider=f"vertex/{provider}",
                    accessible=False,
                    error=error_str[:200],
                )

    async def _check_model_rest(self, model_id: str, provider: str) -> ModelInfo:
        """Check model access via REST API."""
        try:
            import subprocess

            # Get access token
            token_result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if token_result.returncode != 0:
                return ModelInfo(
                    model_id=model_id,
                    provider=f"vertex/{provider}",
                    accessible=False,
                    error="Failed to get access token",
                )

            token = token_result.stdout.strip()

            # Try to access the model endpoint
            import urllib.request

            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project}/locations/{self.region}/publishers/google/models/{model_id}"

            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {token}")

            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return ModelInfo(
                        model_id=model_id,
                        provider=f"vertex/{provider}",
                        accessible=True,
                    )

        except urllib.error.HTTPError as e:
            if e.code == 403:
                return ModelInfo(
                    model_id=model_id,
                    provider=f"vertex/{provider}",
                    accessible=False,
                    error="Permission denied",
                )
            elif e.code == 404:
                return ModelInfo(
                    model_id=model_id,
                    provider=f"vertex/{provider}",
                    accessible=False,
                    error="Model not found",
                )

        except Exception as e:
            return ModelInfo(
                model_id=model_id,
                provider=f"vertex/{provider}",
                accessible=False,
                error=str(e)[:200],
            )

        return ModelInfo(
            model_id=model_id,
            provider=f"vertex/{provider}",
            accessible=False,
            error="Unknown error",
        )

    async def check_all_known_models(
        self, check_access: bool = False
    ) -> dict[str, list[ModelInfo]]:
        """Check accessibility of all known Vertex models."""
        results: dict[str, list[ModelInfo]] = {}

        for provider, model_ids in VERTEX_MODELS.items():
            results[provider] = []

            for model_id in model_ids:
                if check_access:
                    model_info = await self.check_model_access(model_id)
                else:
                    model_info = ModelInfo(
                        model_id=model_id,
                        provider=f"vertex/{provider}",
                        accessible=True,  # Assume accessible
                    )
                results[provider].append(model_info)

        return results

    async def get_accessible_models(self) -> list[str]:
        """Get list of accessible model IDs."""
        all_models = await self.check_all_known_models(check_access=True)

        accessible = []
        for provider_models in all_models.values():
            for model in provider_models:
                if model.accessible:
                    accessible.append(model.model_id)

        return accessible


class ModelAvailabilityChecker:
    """Unified model availability checker across providers."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Initialize provider checkers
        bedrock_config = self.config.get("bedrock", {})
        self.bedrock = BedrockModelChecker(
            profile=bedrock_config.get("profile"),
            region=bedrock_config.get("region", "us-east-1"),
        )

        vertex_config = self.config.get("vertex", {})
        self.vertex = VertexModelChecker(
            project=vertex_config.get("project"),
            region=vertex_config.get("region", "us-central1"),
        )

        self._cache: dict[str, dict[str, list[ModelInfo]]] = {}
        self._cache_time: float = 0

    async def check_all(
        self,
        check_access: bool = True,
        use_cache: bool = True,
        cache_ttl: int = 3600,
    ) -> dict[str, dict[str, list[ModelInfo]]]:
        """Check model availability across all providers."""
        import time

        now = time.time()

        # Return cached results if valid
        if use_cache and self._cache and (now - self._cache_time) < cache_ttl:
            return self._cache

        results = {}

        # Check Bedrock
        try:
            results["bedrock"] = await self.bedrock.check_all_known_models(
                check_access=check_access
            )
        except Exception as e:
            log.error(f"Failed to check Bedrock models: {e}")
            results["bedrock"] = {}

        # Check Vertex
        try:
            results["vertex"] = await self.vertex.check_all_known_models(
                check_access=check_access
            )
        except Exception as e:
            log.error(f"Failed to check Vertex models: {e}")
            results["vertex"] = {}

        # Cache results
        self._cache = results
        self._cache_time = now

        return results

    async def get_accessible_models(self) -> dict[str, list[str]]:
        """Get list of all accessible models by provider."""
        results = await self.check_all(check_access=True)

        accessible = {}

        for platform, providers in results.items():
            for provider, models in providers.items():
                key = f"{platform}/{provider}"
                accessible[key] = [m.model_id for m in models if m.accessible]

        return accessible

    async def get_summary(self) -> dict[str, Any]:
        """Get summary of model availability."""
        results = await self.check_all(check_access=True)

        summary = {
            "bedrock": {"total": 0, "accessible": 0, "by_provider": {}},
            "vertex": {"total": 0, "accessible": 0, "by_provider": {}},
        }

        for platform in ["bedrock", "vertex"]:
            if platform in results:
                for provider, models in results[platform].items():
                    total = len(models)
                    accessible = sum(1 for m in models if m.accessible)

                    summary[platform]["total"] += total
                    summary[platform]["accessible"] += accessible
                    summary[platform]["by_provider"][provider] = {
                        "total": total,
                        "accessible": accessible,
                        "models": [m.model_id for m in models if m.accessible],
                    }

        return summary

    def get_litellm_model_list(self, accessible_only: bool = True) -> list[str]:
        """Get models in LiteLLM format."""
        models = []

        # Bedrock format: bedrock/<model_id>
        for provider, model_ids in BEDROCK_MODELS.items():
            for model_id in model_ids:
                models.append(f"bedrock/{model_id}")

        # Vertex format: vertex_ai/<model_id>
        for provider, model_ids in VERTEX_MODELS.items():
            for model_id in model_ids:
                models.append(f"vertex_ai/{model_id}")

        return models


# Singleton instance
_checker: ModelAvailabilityChecker | None = None


def get_model_checker() -> ModelAvailabilityChecker:
    """Get or create the global model checker instance."""
    global _checker
    if _checker is None:
        _checker = ModelAvailabilityChecker()
    return _checker


def init_model_checker(config: dict[str, Any]) -> ModelAvailabilityChecker:
    """Initialize the global model checker with config."""
    global _checker
    _checker = ModelAvailabilityChecker(config)
    return _checker
