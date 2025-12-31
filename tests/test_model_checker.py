"""Tests for model availability checker."""

import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from baton.plugins.model_checker import (
    ModelInfo,
    BedrockModelChecker,
    VertexModelChecker,
    ModelAvailabilityChecker,
    BEDROCK_MODELS,
    VERTEX_MODELS,
    get_model_checker,
    init_model_checker,
)


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_basic_model_info(self):
        """Test basic model info creation."""
        info = ModelInfo(
            model_id="anthropic.claude-3-opus-20240229-v1:0",
            provider="bedrock/anthropic",
            name="Claude 3 Opus",
            accessible=True,
        )

        assert info.model_id == "anthropic.claude-3-opus-20240229-v1:0"
        assert info.provider == "bedrock/anthropic"
        assert info.name == "Claude 3 Opus"
        assert info.accessible is True
        assert info.error is None

    def test_model_info_with_error(self):
        """Test model info with error."""
        info = ModelInfo(
            model_id="test-model",
            provider="bedrock/test",
            accessible=False,
            error="Access denied",
        )

        assert info.accessible is False
        assert info.error == "Access denied"

    def test_model_info_with_modalities(self):
        """Test model info with modalities."""
        info = ModelInfo(
            model_id="test-model",
            provider="bedrock/test",
            input_modalities=["TEXT", "IMAGE"],
            output_modalities=["TEXT"],
        )

        assert info.input_modalities == ["TEXT", "IMAGE"]
        assert info.output_modalities == ["TEXT"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = ModelInfo(
            model_id="test-model",
            provider="bedrock/test",
            name="Test Model",
            accessible=True,
            requires_agreement=True,
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
        )

        result = info.to_dict()

        assert result["model_id"] == "test-model"
        assert result["provider"] == "bedrock/test"
        assert result["name"] == "Test Model"
        assert result["accessible"] is True
        assert result["requires_agreement"] is True
        assert result["input_modalities"] == ["TEXT"]
        assert result["output_modalities"] == ["TEXT"]
        assert result["regions"] == ["us-east-1"]


class TestBedrockModelChecker:
    """Tests for Bedrock model checker."""

    @pytest.fixture
    def checker(self):
        """Create Bedrock checker instance."""
        return BedrockModelChecker(profile="test", region="us-east-1")

    def test_init_with_profile(self):
        """Test initialization with profile."""
        checker = BedrockModelChecker(profile="work", region="us-west-2")

        assert checker.profile == "work"
        assert checker.region == "us-west-2"

    def test_init_default(self):
        """Test default initialization."""
        checker = BedrockModelChecker()

        assert checker.profile is None
        assert checker.region == "us-east-1"

    @pytest.mark.asyncio
    async def test_list_all_models_success(self, checker):
        """Test listing all models successfully."""
        mock_client = MagicMock()
        mock_client.list_foundation_models.return_value = {
            "modelSummaries": [
                {
                    "modelId": "anthropic.claude-3-opus-20240229-v1:0",
                    "providerName": "Anthropic",
                    "modelName": "Claude 3 Opus",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                },
                {
                    "modelId": "amazon.titan-text-express-v1",
                    "providerName": "Amazon",
                    "modelName": "Titan Text Express",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                },
            ]
        }

        with patch.object(checker, "_get_client", return_value=mock_client):
            models = await checker.list_all_models()

            assert len(models) == 2
            assert models[0].model_id == "anthropic.claude-3-opus-20240229-v1:0"
            assert models[0].provider == "bedrock/anthropic"
            assert models[1].provider == "bedrock/amazon"

    @pytest.mark.asyncio
    async def test_list_all_models_error(self, checker):
        """Test listing models with error."""
        mock_client = MagicMock()
        mock_client.list_foundation_models.side_effect = Exception("API error")

        with patch.object(checker, "_get_client", return_value=mock_client):
            models = await checker.list_all_models()

            assert models == []

    @pytest.mark.asyncio
    async def test_check_model_access_success(self, checker):
        """Test successful model access check."""
        mock_runtime = MagicMock()
        mock_runtime.invoke_model.return_value = {
            "body": MagicMock(read=lambda: b'{"content": "test"}')
        }

        with patch.object(checker, "_get_runtime_client", return_value=mock_runtime):
            result = await checker.check_model_access(
                "anthropic.claude-3-opus-20240229-v1:0"
            )

            assert result.accessible is True
            assert result.provider == "bedrock/anthropic"

    @pytest.mark.asyncio
    async def test_check_model_access_denied(self, checker):
        """Test model access denied."""
        mock_runtime = MagicMock()
        mock_runtime.invoke_model.side_effect = Exception(
            "AccessDeniedException: You don't have access"
        )

        with patch.object(checker, "_get_runtime_client", return_value=mock_runtime):
            result = await checker.check_model_access(
                "anthropic.claude-3-opus-20240229-v1:0"
            )

            assert result.accessible is False
            assert "Access denied" in result.error

    @pytest.mark.asyncio
    async def test_check_model_access_not_found(self, checker):
        """Test model not found."""
        mock_runtime = MagicMock()
        mock_runtime.invoke_model.side_effect = Exception(
            "ResourceNotFoundException: Model not found"
        )

        with patch.object(checker, "_get_runtime_client", return_value=mock_runtime):
            result = await checker.check_model_access("nonexistent-model")

            assert result.accessible is False
            assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_check_model_access_validation_error(self, checker):
        """Test validation error (model exists but wrong format)."""
        mock_runtime = MagicMock()
        mock_runtime.invoke_model.side_effect = Exception(
            "ValidationException: Invalid request"
        )

        with patch.object(checker, "_get_runtime_client", return_value=mock_runtime):
            result = await checker.check_model_access(
                "anthropic.claude-3-opus-20240229-v1:0"
            )

            # Validation error means model exists, just wrong request format
            assert result.accessible is True

    @pytest.mark.asyncio
    async def test_check_all_known_models_no_access_check(self, checker):
        """Test checking all known models without access verification."""
        results = await checker.check_all_known_models(check_access=False)

        # Should have all providers
        assert "anthropic" in results
        assert "amazon" in results
        assert "meta" in results

        # All should be marked accessible (assumed)
        for provider_models in results.values():
            for model in provider_models:
                assert model.accessible is True

    @pytest.mark.asyncio
    async def test_check_all_known_models_with_access_check(self, checker):
        """Test checking all known models with access verification."""
        mock_runtime = MagicMock()
        # First call succeeds, rest fail
        mock_runtime.invoke_model.side_effect = [
            {"body": MagicMock(read=lambda: b'{}')},
            Exception("AccessDeniedException"),
        ] * 50  # Enough for all models

        with patch.object(checker, "_get_runtime_client", return_value=mock_runtime):
            results = await checker.check_all_known_models(check_access=True)

            # Should have results for all providers
            assert len(results) > 0


class TestVertexModelChecker:
    """Tests for Vertex AI model checker."""

    @pytest.fixture
    def checker(self):
        """Create Vertex checker instance."""
        return VertexModelChecker(project="test-project", region="us-central1")

    def test_init_with_config(self):
        """Test initialization with config."""
        checker = VertexModelChecker(project="my-project", region="europe-west1")

        assert checker.project == "my-project"
        assert checker.region == "europe-west1"

    def test_init_from_env(self):
        """Test initialization from environment."""
        with patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "env-project"}):
            checker = VertexModelChecker()

            assert checker.project == "env-project"

    @pytest.mark.asyncio
    async def test_list_models_via_gcloud_success(self, checker):
        """Test listing models via gcloud CLI."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            [
                {"name": "gemini-1.5-pro", "displayName": "Gemini 1.5 Pro"},
                {"name": "gemini-1.5-flash", "displayName": "Gemini 1.5 Flash"},
            ]
        )

        with patch("subprocess.run", return_value=mock_result):
            models = await checker.list_models_via_gcloud()

            assert len(models) == 2
            assert models[0].model_id == "gemini-1.5-pro"
            assert models[0].name == "Gemini 1.5 Pro"

    @pytest.mark.asyncio
    async def test_list_models_gcloud_not_found(self, checker):
        """Test when gcloud CLI is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            models = await checker.list_models_via_gcloud()

            assert models == []

    @pytest.mark.asyncio
    async def test_list_models_gcloud_timeout(self, checker):
        """Test gcloud command timeout."""
        import subprocess

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("gcloud", 30)):
            models = await checker.list_models_via_gcloud()

            assert models == []

    @pytest.mark.asyncio
    async def test_check_model_access_permission_denied(self, checker):
        """Test model access with permission denied."""
        with patch(
            "baton.plugins.model_checker.VertexModelChecker._check_model_rest"
        ) as mock_rest:
            mock_rest.return_value = ModelInfo(
                model_id="gemini-1.5-pro",
                provider="vertex/google",
                accessible=False,
                error="Permission denied",
            )

            # Mock the import to raise ImportError so it falls back to REST
            with patch.dict("sys.modules", {"google.genai": None}):
                with patch(
                    "builtins.__import__",
                    side_effect=ImportError("No module named 'google'"),
                ):
                    result = await checker.check_model_access("gemini-1.5-pro")

                    assert result.accessible is False

    @pytest.mark.asyncio
    async def test_check_all_known_models_no_access_check(self, checker):
        """Test checking all known models without access verification."""
        results = await checker.check_all_known_models(check_access=False)

        # Should have all providers
        assert "google" in results
        assert "anthropic" in results

        # All should be marked accessible (assumed)
        for provider_models in results.values():
            for model in provider_models:
                assert model.accessible is True


class TestModelAvailabilityChecker:
    """Tests for unified model availability checker."""

    @pytest.fixture
    def checker(self):
        """Create unified checker instance."""
        return ModelAvailabilityChecker(
            {
                "bedrock": {"profile": "test", "region": "us-east-1"},
                "vertex": {"project": "test-project", "region": "us-central1"},
            }
        )

    def test_init_with_config(self, checker):
        """Test initialization with config."""
        assert checker.bedrock.profile == "test"
        assert checker.bedrock.region == "us-east-1"
        assert checker.vertex.project == "test-project"
        assert checker.vertex.region == "us-central1"

    def test_init_default(self):
        """Test default initialization."""
        checker = ModelAvailabilityChecker()

        assert checker.bedrock.region == "us-east-1"
        assert checker.vertex.region == "us-central1"

    @pytest.mark.asyncio
    async def test_check_all(self, checker):
        """Test checking all providers."""
        mock_bedrock_results = {
            "anthropic": [
                ModelInfo(
                    model_id="anthropic.claude-3-opus",
                    provider="bedrock/anthropic",
                    accessible=True,
                )
            ]
        }
        mock_vertex_results = {
            "google": [
                ModelInfo(
                    model_id="gemini-1.5-pro",
                    provider="vertex/google",
                    accessible=True,
                )
            ]
        }

        with patch.object(
            checker.bedrock,
            "check_all_known_models",
            new_callable=AsyncMock,
            return_value=mock_bedrock_results,
        ):
            with patch.object(
                checker.vertex,
                "check_all_known_models",
                new_callable=AsyncMock,
                return_value=mock_vertex_results,
            ):
                results = await checker.check_all(check_access=True, use_cache=False)

                assert "bedrock" in results
                assert "vertex" in results
                assert "anthropic" in results["bedrock"]
                assert "google" in results["vertex"]

    @pytest.mark.asyncio
    async def test_check_all_with_cache(self, checker):
        """Test caching behavior."""
        mock_bedrock_results = {"anthropic": []}
        mock_vertex_results = {"google": []}

        with patch.object(
            checker.bedrock,
            "check_all_known_models",
            new_callable=AsyncMock,
            return_value=mock_bedrock_results,
        ) as mock_bedrock:
            with patch.object(
                checker.vertex,
                "check_all_known_models",
                new_callable=AsyncMock,
                return_value=mock_vertex_results,
            ) as mock_vertex:
                # First call
                await checker.check_all(use_cache=True)

                # Second call should use cache
                await checker.check_all(use_cache=True)

                # Should only be called once due to caching
                assert mock_bedrock.call_count == 1
                assert mock_vertex.call_count == 1

    @pytest.mark.asyncio
    async def test_check_all_error_handling(self, checker):
        """Test error handling during check."""
        with patch.object(
            checker.bedrock,
            "check_all_known_models",
            new_callable=AsyncMock,
            side_effect=Exception("Bedrock error"),
        ):
            with patch.object(
                checker.vertex,
                "check_all_known_models",
                new_callable=AsyncMock,
                return_value={"google": []},
            ):
                results = await checker.check_all(use_cache=False)

                # Should have empty bedrock but vertex should work
                assert results["bedrock"] == {}
                assert "google" in results["vertex"]

    @pytest.mark.asyncio
    async def test_get_accessible_models(self, checker):
        """Test getting accessible models."""
        mock_results = {
            "bedrock": {
                "anthropic": [
                    ModelInfo(
                        model_id="claude-3-opus",
                        provider="bedrock/anthropic",
                        accessible=True,
                    ),
                    ModelInfo(
                        model_id="claude-3-sonnet",
                        provider="bedrock/anthropic",
                        accessible=False,
                    ),
                ]
            },
            "vertex": {
                "google": [
                    ModelInfo(
                        model_id="gemini-1.5-pro",
                        provider="vertex/google",
                        accessible=True,
                    ),
                ]
            },
        }

        with patch.object(
            checker, "check_all", new_callable=AsyncMock, return_value=mock_results
        ):
            accessible = await checker.get_accessible_models()

            assert "bedrock/anthropic" in accessible
            assert "vertex/google" in accessible
            assert "claude-3-opus" in accessible["bedrock/anthropic"]
            assert "claude-3-sonnet" not in accessible["bedrock/anthropic"]

    @pytest.mark.asyncio
    async def test_get_summary(self, checker):
        """Test getting availability summary."""
        mock_results = {
            "bedrock": {
                "anthropic": [
                    ModelInfo(
                        model_id="claude-3-opus",
                        provider="bedrock/anthropic",
                        accessible=True,
                    ),
                    ModelInfo(
                        model_id="claude-3-sonnet",
                        provider="bedrock/anthropic",
                        accessible=False,
                    ),
                ]
            },
            "vertex": {
                "google": [
                    ModelInfo(
                        model_id="gemini-1.5-pro",
                        provider="vertex/google",
                        accessible=True,
                    ),
                ]
            },
        }

        with patch.object(
            checker, "check_all", new_callable=AsyncMock, return_value=mock_results
        ):
            summary = await checker.get_summary()

            assert summary["bedrock"]["total"] == 2
            assert summary["bedrock"]["accessible"] == 1
            assert summary["vertex"]["total"] == 1
            assert summary["vertex"]["accessible"] == 1

    def test_get_litellm_model_list(self, checker):
        """Test getting LiteLLM format model list."""
        models = checker.get_litellm_model_list()

        # Should include bedrock models
        assert any("bedrock/" in m for m in models)

        # Should include vertex models
        assert any("vertex_ai/" in m for m in models)


class TestKnownModels:
    """Tests for known model lists."""

    def test_bedrock_models_structure(self):
        """Test BEDROCK_MODELS structure."""
        assert "anthropic" in BEDROCK_MODELS
        assert "amazon" in BEDROCK_MODELS
        assert "meta" in BEDROCK_MODELS
        assert "mistral" in BEDROCK_MODELS
        assert "cohere" in BEDROCK_MODELS
        assert "ai21" in BEDROCK_MODELS

        # Check that all values are lists
        for provider, models in BEDROCK_MODELS.items():
            assert isinstance(models, list)
            assert len(models) > 0

    def test_vertex_models_structure(self):
        """Test VERTEX_MODELS structure."""
        assert "google" in VERTEX_MODELS
        assert "anthropic" in VERTEX_MODELS
        assert "meta" in VERTEX_MODELS
        assert "mistral" in VERTEX_MODELS

        # Check that all values are lists
        for provider, models in VERTEX_MODELS.items():
            assert isinstance(models, list)
            assert len(models) > 0

    def test_bedrock_anthropic_models(self):
        """Test Anthropic models in Bedrock."""
        anthropic_models = BEDROCK_MODELS["anthropic"]

        # Should have Claude 3 models
        assert any("claude-3-5-sonnet" in m for m in anthropic_models)
        assert any("claude-3-opus" in m for m in anthropic_models)
        assert any("claude-3-haiku" in m for m in anthropic_models)

    def test_vertex_google_models(self):
        """Test Google models in Vertex."""
        google_models = VERTEX_MODELS["google"]

        # Should have Gemini models
        assert any("gemini-2.0" in m for m in google_models)
        assert any("gemini-1.5-pro" in m for m in google_models)
        assert any("gemini-1.5-flash" in m for m in google_models)


class TestGlobalModelChecker:
    """Tests for global model checker functions."""

    def test_init_model_checker(self):
        """Test initializing global model checker."""
        checker = init_model_checker(
            {"bedrock": {"region": "us-west-2"}}
        )

        assert checker is not None
        assert checker.bedrock.region == "us-west-2"
        assert get_model_checker() is checker

    def test_get_model_checker_creates_default(self):
        """Test get_model_checker creates default if not initialized."""
        import baton.plugins.model_checker as mc_module

        mc_module._checker = None

        checker = get_model_checker()

        assert checker is not None
        assert checker.bedrock.region == "us-east-1"  # Default region
