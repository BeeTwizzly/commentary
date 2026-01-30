"""Tests for the LLM client module."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date

import httpx

from src.config import LLMConfig, AppConfig, load_config_from_env
from src.generation.llm_client import (
    LLMClient,
    GenerationResult,
    TokenUsage,
    BatchProgress,
    TOKEN_COSTS,
)
from src.generation.prompt_builder import AssembledPrompt, PromptContext
from src.models import HoldingData, ThesisLookupResult, ExemplarSelection


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create test LLM configuration."""
    return LLMConfig(
        api_key="test-api-key",
        model="gpt-4o",
        temperature=0.8,
        max_tokens=1500,
        max_retries=2,
    )


@pytest.fixture
def sample_prompt() -> AssembledPrompt:
    """Create sample assembled prompt for testing."""
    holding = HoldingData(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        strategy="Test Strategy",
        avg_weight=5.0,
        begin_weight=4.5,
        end_weight=5.5,
        benchmark_weight=3.0,
        benchmark_return=12.5,
        total_attribution=0.85,
        selection_effect=0.65,
        allocation_effect=0.20,
        rank=1,
        is_contributor=True,
    )
    thesis = ThesisLookupResult(
        ticker="NVDA",
        found=True,
        entry=None,
        placeholder_text="",
    )
    exemplars = ExemplarSelection(
        target_ticker="NVDA",
        target_is_contributor=True,
        same_ticker_exemplar=None,
        similar_exemplars=[],
    )
    context = PromptContext(
        holding=holding,
        thesis=thesis,
        exemplars=exemplars,
        quarter="Q4 2025",
        strategy_name="Test Strategy",
        num_variations=3,
    )
    return AssembledPrompt(
        system_prompt="You are a test assistant.",
        user_prompt="Generate test commentary for NVDA.",
        context=context,
        metadata={"ticker": "NVDA", "strategy": "Test"},
    )


@pytest.fixture
def mock_api_response():
    """Create mock API response."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": """[A] NVIDIA contributed strongly as datacenter revenue exceeded expectations. The AI infrastructure thesis remains intact with demand visibility extending into next year. We maintain our overweight position.

[B] Our NVIDIA position drove relative performance this quarter. Datacenter growth of 150% year-over-year validated our AI demand thesis. The position remains a high-conviction holding.

[C] NVIDIA was the primary contributor to returns. Management's commentary reinforced the durability of AI infrastructure spending. We continue to view the risk-reward favorably."""
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 500,
            "completion_tokens": 150,
            "total_tokens": 650
        }
    }


class TestLLMConfig:
    """Tests for LLMConfig validation."""

    def test_valid_config(self):
        """Should accept valid configuration."""
        config = LLMConfig(
            api_key="test-key",
            model="gpt-4o",
            temperature=0.7,
        )
        assert config.api_key == "test-key"
        assert config.model == "gpt-4o"

    def test_rejects_empty_api_key(self):
        """Should detect unconfigured state with empty API key."""
        config = LLMConfig(api_key="", model="gpt-4o")
        assert config.is_configured is False

    def test_rejects_invalid_temperature(self):
        """Should allow any temperature (validation moved to AppConfig)."""
        config = LLMConfig(api_key="sk-test", temperature=2.5)
        assert config.temperature == 2.5

    def test_rejects_low_max_tokens(self):
        """Should allow any max_tokens (validation moved to AppConfig)."""
        config = LLMConfig(api_key="sk-test", max_tokens=50)
        assert config.max_tokens == 50


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_is_empty(self):
        """Should detect empty usage."""
        empty = TokenUsage()
        assert empty.is_empty is True

        with_tokens = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert with_tokens.is_empty is False


class TestLLMClient:
    """Tests for LLMClient."""

    @pytest.mark.asyncio
    async def test_successful_generation(self, llm_config, sample_prompt, mock_api_response):
        """Should successfully generate and parse variations."""
        client = LLMClient(llm_config)

        # Mock the HTTP response (json() is sync in httpx, so use MagicMock)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_response
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            result = await client.generate(sample_prompt)

        assert result.success is True
        assert result.parsed is not None
        assert result.parsed.variation_count == 3
        assert result.usage.total_tokens == 650
        assert result.cost_usd > 0

        await client.close()

    @pytest.mark.asyncio
    async def test_handles_api_error(self, llm_config, sample_prompt):
        """Should handle API errors gracefully."""
        client = LLMClient(llm_config)

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=mock_response,
            )
            mock_http_client.post.return_value = mock_response
            mock_get_client.return_value = mock_http_client

            result = await client.generate(sample_prompt)

        assert result.success is False
        assert result.error_message != ""
        assert result.parsed is None

        await client.close()

    @pytest.mark.asyncio
    async def test_handles_rate_limiting(self, llm_config, sample_prompt, mock_api_response):
        """Should retry on rate limiting."""
        client = LLMClient(llm_config)

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call returns rate limit
                mock_response = MagicMock()
                mock_response.status_code = 429
                mock_response.headers = {"Retry-After": "1"}
                return mock_response
            else:
                # Second call succeeds (json() is sync in httpx)
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = mock_api_response
                mock_response.raise_for_status = MagicMock()
                return mock_response

        with patch.object(client, '_get_client') as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post.side_effect = mock_post
            mock_get_client.return_value = mock_http_client

            result = await client.generate(sample_prompt)

        assert result.success is True
        assert call_count == 2

        await client.close()

    def test_cost_calculation(self, llm_config):
        """Should calculate cost correctly."""
        client = LLMClient(llm_config)

        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )

        cost = client._calculate_cost(usage)

        # gpt-4o: $2.50/1M input, $10.00/1M output
        expected_input = (1000 / 1_000_000) * 2.50
        expected_output = (500 / 1_000_000) * 10.00
        expected_total = expected_input + expected_output

        assert abs(cost - expected_total) < 0.0001

    def test_sync_wrapper(self, llm_config, sample_prompt, mock_api_response):
        """Should provide sync wrapper for async generate."""
        client = LLMClient(llm_config)

        # This test verifies the sync wrapper exists and has correct signature
        # Full integration test would require mocking at a lower level
        assert hasattr(client, 'generate_sync')
        assert callable(client.generate_sync)


class TestBatchProgress:
    """Tests for BatchProgress tracking."""

    def test_percent_complete(self):
        """Should calculate percent complete correctly."""
        progress = BatchProgress(total=10, completed=3, failed=2)

        assert progress.percent_complete == 50.0

    def test_handles_zero_total(self):
        """Should handle zero total gracefully."""
        progress = BatchProgress(total=0)

        assert progress.percent_complete == 100.0


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_variations_accessor(self, mock_api_response):
        """Should provide convenient variations access."""
        from src.generation.response_parser import parse_llm_response

        content = mock_api_response["choices"][0]["message"]["content"]
        parsed = parse_llm_response(content)

        result = GenerationResult(
            parsed=parsed,
            usage=TokenUsage(),
            cost_usd=0.01,
            latency_seconds=1.0,
            model="gpt-4o",
            success=True,
        )

        assert len(result.variations) == 3

    def test_variations_empty_on_failure(self):
        """Should return empty list on failure."""
        result = GenerationResult(
            parsed=None,
            usage=TokenUsage(),
            cost_usd=0.0,
            latency_seconds=1.0,
            model="gpt-4o",
            success=False,
            error_message="Test error",
        )

        assert result.variations == []


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_from_env(self, monkeypatch):
        """Should load config from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-from-env")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("OPENAI_TEMPERATURE", "0.5")

        config = load_config_from_env()

        assert config.llm.api_key == "test-key-from-env"
        assert config.llm.model == "gpt-4o-mini"
        assert config.llm.temperature == 0.5

    def test_reports_unconfigured_without_api_key(self, monkeypatch):
        """Should detect unconfigured state when API key not set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config = load_config_from_env()
        assert config.llm.is_configured is False
