"""LLM client for generating commentary via OpenAI API."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from src.config import LLMConfig
from src.generation.prompt_builder import AssembledPrompt
from src.generation.response_parser import ParsedResponse, parse_llm_response

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (approximate, update as needed)
# https://openai.com/pricing
TOKEN_COSTS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

# Default fallback pricing
DEFAULT_COST = {"input": 5.00, "output": 15.00}


@dataclass
class TokenUsage:
    """Token usage from a single API call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return self.total_tokens == 0


@dataclass
class GenerationResult:
    """
    Result of a single commentary generation.

    Attributes:
        parsed: Parsed response with variations
        usage: Token usage statistics
        cost_usd: Estimated cost in USD
        latency_seconds: Time taken for API call
        model: Model used for generation
        success: Whether generation succeeded
        error_message: Error details if failed
    """
    parsed: ParsedResponse | None
    usage: TokenUsage
    cost_usd: float
    latency_seconds: float
    model: str
    success: bool
    error_message: str = ""

    @property
    def variations(self):
        """Convenience accessor for parsed variations."""
        if self.parsed:
            return self.parsed.variations
        return []


@dataclass
class BatchProgress:
    """Progress tracking for batch generation."""
    total: int
    completed: int = 0
    failed: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    @property
    def percent_complete(self) -> float:
        if self.total == 0:
            return 100.0
        return (self.completed + self.failed) / self.total * 100


class LLMClient:
    """
    Client for generating commentary via OpenAI-compatible API.

    Handles API calls with retry logic, rate limiting awareness,
    and cost tracking. Supports both sync and async operation.

    Usage:
        config = LLMConfig(api_key="sk-...")
        client = LLMClient(config)

        result = await client.generate(prompt)
        print(result.parsed.variations)
        print(f"Cost: ${result.cost_usd:.4f}")
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize client with configuration.

        Args:
            config: LLM configuration including API key and model settings
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None

        # Determine base URL
        self._base_url = config.base_url or "https://api.openai.com/v1"

        logger.info(
            "LLMClient initialized: model=%s, temperature=%.1f, max_tokens=%d",
            config.model,
            config.temperature,
            config.max_tokens,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        prompt: AssembledPrompt,
        num_variations: int = 3,
    ) -> GenerationResult:
        """
        Generate commentary variations for a single holding.

        Args:
            prompt: Assembled prompt with system and user messages
            num_variations: Expected number of variations

        Returns:
            GenerationResult with parsed variations and metadata
        """
        start_time = time.monotonic()

        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ]

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        # Retry loop with exponential backoff
        last_error = ""
        for attempt in range(self.config.max_retries):
            try:
                client = await self._get_client()

                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning(
                        "Rate limited, waiting %d seconds (attempt %d/%d)",
                        retry_after,
                        attempt + 1,
                        self.config.max_retries,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                data = response.json()

                # Extract response content
                content = data["choices"][0]["message"]["content"]

                # Parse usage
                usage_data = data.get("usage", {})
                usage = TokenUsage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )

                # Calculate cost
                cost = self._calculate_cost(usage)

                # Parse variations
                parsed = parse_llm_response(content, expected_count=num_variations)

                latency = time.monotonic() - start_time

                logger.info(
                    "Generation complete: ticker=%s, variations=%d, tokens=%d, cost=$%.4f, latency=%.1fs",
                    prompt.metadata.get("ticker", "unknown"),
                    parsed.variation_count,
                    usage.total_tokens,
                    cost,
                    latency,
                )

                return GenerationResult(
                    parsed=parsed,
                    usage=usage,
                    cost_usd=cost,
                    latency_seconds=latency,
                    model=self.config.model,
                    success=True,
                )

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                logger.warning(
                    "API error (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    last_error,
                )

            except httpx.RequestError as e:
                last_error = f"Request failed: {e}"
                logger.warning(
                    "Request error (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    last_error,
                )

            except Exception as e:
                last_error = f"Unexpected error: {e}"
                logger.exception("Unexpected error during generation")

            # Exponential backoff
            if attempt < self.config.max_retries - 1:
                wait_time = 2 ** attempt
                logger.debug("Waiting %d seconds before retry", wait_time)
                await asyncio.sleep(wait_time)

        # All retries exhausted
        latency = time.monotonic() - start_time

        return GenerationResult(
            parsed=None,
            usage=TokenUsage(),
            cost_usd=0.0,
            latency_seconds=latency,
            model=self.config.model,
            success=False,
            error_message=last_error,
        )

    async def generate_batch(
        self,
        prompts: list[AssembledPrompt],
        num_variations: int = 3,
        on_progress: Callable[[BatchProgress], None] | None = None,
        max_concurrent: int = 3,
    ) -> list[GenerationResult]:
        """
        Generate commentary for multiple holdings with concurrency control.

        Args:
            prompts: List of assembled prompts
            num_variations: Expected variations per prompt
            on_progress: Optional callback for progress updates
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of GenerationResult in same order as prompts
        """
        progress = BatchProgress(total=len(prompts))
        results: list[GenerationResult | None] = [None] * len(prompts)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_one(index: int, prompt: AssembledPrompt):
            async with semaphore:
                result = await self.generate(prompt, num_variations)
                results[index] = result

                # Update progress
                if result.success:
                    progress.completed += 1
                else:
                    progress.failed += 1

                progress.total_cost_usd += result.cost_usd
                progress.total_tokens += result.usage.total_tokens

                if on_progress:
                    on_progress(progress)

        # Create tasks for all prompts
        tasks = [
            generate_one(i, prompt)
            for i, prompt in enumerate(prompts)
        ]

        # Execute with concurrency limit
        await asyncio.gather(*tasks)

        logger.info(
            "Batch complete: %d/%d succeeded, total_cost=$%.4f, total_tokens=%d",
            progress.completed,
            progress.total,
            progress.total_cost_usd,
            progress.total_tokens,
        )

        return [r for r in results if r is not None]

    def generate_sync(
        self,
        prompt: AssembledPrompt,
        num_variations: int = 3,
    ) -> GenerationResult:
        """
        Synchronous wrapper for generate().

        Useful for Streamlit which has its own event loop management.

        Args:
            prompt: Assembled prompt
            num_variations: Expected number of variations

        Returns:
            GenerationResult with parsed variations
        """
        return asyncio.run(self.generate(prompt, num_variations))

    def _calculate_cost(self, usage: TokenUsage) -> float:
        """Calculate estimated cost in USD."""
        costs = TOKEN_COSTS.get(self.config.model, DEFAULT_COST)

        input_cost = (usage.prompt_tokens / 1_000_000) * costs["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * costs["output"]

        return input_cost + output_cost

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def create_client_from_config() -> LLMClient:
    """
    Create LLM client from application configuration.

    Convenience function that loads config and creates client.

    Returns:
        Configured LLMClient
    """
    from src.config import get_config

    config = get_config()
    return LLMClient(config.llm)
