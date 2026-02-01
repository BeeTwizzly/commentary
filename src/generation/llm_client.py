"""LLM client for generating commentary via OpenAI Responses API."""

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
    "gpt-5-mini": {"input": 0.30, "output": 1.20},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

# Default fallback pricing
DEFAULT_COST = {"input": 5.00, "output": 15.00}

# Response status values
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_IN_PROGRESS = "in_progress"
TERMINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED}


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
    Client for generating commentary via OpenAI Responses API.

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
            "LLMClient initialized: model=%s, max_tokens=%d",
            config.model,
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

    async def create_response(
        self,
        prompt: AssembledPrompt,
    ) -> dict:
        """
        Submit a request to the Responses API.

        Args:
            prompt: Assembled prompt with system and user messages

        Returns:
            Response data dict from API

        Raises:
            httpx.HTTPStatusError: On API errors
        """
        # Build Responses API payload
        # Note: temperature is not supported for GPT-5 and later models
        payload = {
            "model": self.config.model,
            "input": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "max_output_tokens": self.config.max_tokens,
        }

        client = await self._get_client()
        response = await client.post(
            f"{self._base_url}/responses",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def poll_response(self, response_id: str) -> dict:
        """
        Poll for response completion.

        Args:
            response_id: ID of the response to poll

        Returns:
            Completed response data dict
        """
        client = await self._get_client()

        while True:
            response = await client.get(
                f"{self._base_url}/responses/{response_id}",
            )
            response.raise_for_status()
            data = response.json()

            status = data.get("status", STATUS_COMPLETED)
            if status in TERMINAL_STATUSES:
                return data

            logger.debug(
                "Response %s status: %s, polling again in %.1fs",
                response_id,
                status,
                self.config.poll_interval_seconds,
            )
            await asyncio.sleep(self.config.poll_interval_seconds)

    def _extract_output_text(self, response_data: dict) -> str:
        """
        Extract text content from Responses API output.

        The Responses API returns output in the format:
        response_data["output"][*]["content"][*]["text"]

        Args:
            response_data: Response data from API

        Returns:
            Concatenated text from all output items
        """
        outputs = response_data.get("output", [])
        chunks = []

        for item in outputs:
            # Handle message-type outputs
            content_list = item.get("content", [])
            for content in content_list:
                if content.get("type") == "text":
                    text = content.get("text", "")
                    if text:
                        chunks.append(text)
                elif isinstance(content, str):
                    # Some responses may have plain string content
                    chunks.append(content)

            # Also check for direct text field (older format compatibility)
            if "text" in item:
                chunks.append(item["text"])

        return "\n".join(chunks).strip()

    def _parse_usage(self, response_data: dict) -> TokenUsage:
        """
        Parse token usage from Responses API response.

        Args:
            response_data: Response data from API

        Returns:
            TokenUsage with parsed values
        """
        usage_data = response_data.get("usage", {})

        # Responses API uses input_tokens/output_tokens
        input_tokens = usage_data.get("input_tokens", usage_data.get("prompt_tokens", 0))
        output_tokens = usage_data.get("output_tokens", usage_data.get("completion_tokens", 0))
        total_tokens = usage_data.get("total_tokens", input_tokens + output_tokens)

        return TokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    async def generate(
        self,
        prompt: AssembledPrompt,
        num_variations: int = 3,
    ) -> GenerationResult:
        """
        Generate commentary variations for a single holding.

        Uses the OpenAI Responses API with polling for completion.

        Args:
            prompt: Assembled prompt with system and user messages
            num_variations: Expected number of variations

        Returns:
            GenerationResult with parsed variations and metadata
        """
        start_time = time.monotonic()

        # Retry loop with exponential backoff
        last_error = ""
        for attempt in range(self.config.max_retries):
            try:
                # Create the response
                response_data = await self.create_response(prompt)

                # Check if we need to poll (status might be in_progress)
                status = response_data.get("status", STATUS_COMPLETED)
                if status == STATUS_IN_PROGRESS:
                    response_id = response_data.get("id")
                    if response_id:
                        response_data = await self.poll_response(response_id)
                        status = response_data.get("status", STATUS_COMPLETED)

                # Handle non-success statuses
                if status == STATUS_FAILED:
                    error = response_data.get("error", {})
                    last_error = error.get("message", "Response failed")
                    logger.warning("Response failed: %s", last_error)
                    continue

                if status == STATUS_CANCELLED:
                    last_error = "Response was cancelled"
                    logger.warning(last_error)
                    continue

                # Extract content from output
                content = self._extract_output_text(response_data)

                if not content:
                    last_error = "Empty response content"
                    logger.warning("Empty response from API")
                    continue

                # Parse usage
                usage = self._parse_usage(response_data)

                # Calculate cost
                cost = self._calculate_cost(usage)

                # Parse variations from content
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

                # Handle rate limiting
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 5))
                    logger.warning("Rate limited, waiting %d seconds", retry_after)
                    await asyncio.sleep(retry_after)
                    continue

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
        Uses a fresh HTTP client per call to avoid event loop conflicts.

        Args:
            prompt: Assembled prompt
            num_variations: Expected number of variations

        Returns:
            GenerationResult with parsed variations
        """
        async def _generate_with_fresh_client() -> GenerationResult:
            # Use a fresh client for each sync call to avoid event loop issues.
            # When asyncio.run() completes, it closes the event loop, which can
            # leave a cached AsyncClient in an invalid state for subsequent calls.
            old_client = self._client
            self._client = None  # Force creation of new client
            try:
                return await self.generate(prompt, num_variations)
            finally:
                # Clean up the client we created
                if self._client and not self._client.is_closed:
                    await self._client.aclose()
                # Don't restore old_client - it's tied to a dead event loop
                self._client = None

        return asyncio.run(_generate_with_fresh_client())

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
