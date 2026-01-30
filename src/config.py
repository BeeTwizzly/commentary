"""Application configuration management."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """
    Configuration for LLM API calls.

    Attributes:
        api_key: OpenAI API key
        model: Model identifier (e.g., "gpt-4o", "gpt-4o-mini")
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        timeout_seconds: Request timeout
        max_retries: Maximum retry attempts
        base_url: Optional custom API base URL (for enterprise endpoints)
    """
    api_key: str
    model: str = "gpt-4o"
    temperature: float = 0.8
    max_tokens: int = 1500
    timeout_seconds: float = 60.0
    max_retries: int = 3
    base_url: str | None = None

    def __post_init__(self):
        if not self.api_key:
            raise ValueError("API key is required")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"Temperature must be 0.0-2.0, got {self.temperature}")
        if self.max_tokens < 100:
            raise ValueError(f"max_tokens must be >= 100, got {self.max_tokens}")


@dataclass
class AppConfig:
    """
    Complete application configuration.

    Attributes:
        llm: LLM-specific configuration
        data_dir: Base directory for data files
        thesis_registry_path: Path to thesis CSV
        exemplars_path: Path to exemplars JSON
        log_level: Logging level
        num_variations: Number of commentary variations to generate
    """
    llm: LLMConfig
    data_dir: Path = field(default_factory=lambda: Path("data"))
    thesis_registry_path: Path | None = None
    exemplars_path: Path | None = None
    log_level: str = "INFO"
    num_variations: int = 3

    def __post_init__(self):
        # Set default paths if not specified
        if self.thesis_registry_path is None:
            self.thesis_registry_path = self.data_dir / "thesis_registry.csv"
        if self.exemplars_path is None:
            self.exemplars_path = self.data_dir / "exemplars" / "exemplars.json"


def load_config_from_env() -> AppConfig:
    """
    Load configuration from environment variables.

    Environment variables:
        OPENAI_API_KEY: Required API key
        OPENAI_MODEL: Model name (default: gpt-4o)
        OPENAI_BASE_URL: Optional custom base URL
        OPENAI_TEMPERATURE: Sampling temperature (default: 0.8)
        OPENAI_MAX_TOKENS: Max response tokens (default: 1500)
        LOG_LEVEL: Logging level (default: INFO)

    Returns:
        AppConfig with values from environment

    Raises:
        ValueError: If required environment variables are missing
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it in your environment or .streamlit/secrets.toml"
        )

    llm_config = LLMConfig(
        api_key=api_key,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.8")),
        max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "1500")),
    )

    return AppConfig(
        llm=llm_config,
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )


def load_config_from_streamlit() -> AppConfig:
    """
    Load configuration from Streamlit secrets.

    Expects secrets in .streamlit/secrets.toml:
        [openai]
        api_key = "sk-..."
        model = "gpt-4o"  # optional
        base_url = "..."  # optional

    Returns:
        AppConfig with values from Streamlit secrets

    Raises:
        ValueError: If required secrets are missing
    """
    try:
        import streamlit as st

        # Try to get from streamlit secrets
        if hasattr(st, "secrets") and "openai" in st.secrets:
            openai_secrets = st.secrets["openai"]
            api_key = openai_secrets.get("api_key", "")

            if not api_key:
                raise ValueError("openai.api_key not found in Streamlit secrets")

            llm_config = LLMConfig(
                api_key=api_key,
                model=openai_secrets.get("model", "gpt-4o"),
                base_url=openai_secrets.get("base_url"),
                temperature=float(openai_secrets.get("temperature", 0.8)),
                max_tokens=int(openai_secrets.get("max_tokens", 1500)),
            )

            return AppConfig(llm=llm_config)

    except ImportError:
        pass
    except Exception as e:
        logger.warning("Failed to load Streamlit secrets: %s", e)

    # Fall back to environment
    return load_config_from_env()


def get_config() -> AppConfig:
    """
    Get application configuration, trying Streamlit secrets first.

    Returns:
        AppConfig from best available source
    """
    try:
        return load_config_from_streamlit()
    except ValueError:
        return load_config_from_env()
