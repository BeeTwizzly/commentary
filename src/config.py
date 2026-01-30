"""Application configuration with validation and defaults."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with Streamlit secrets fallback."""
    # Try environment first
    value = os.environ.get(key, "")
    if value:
        return value

    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass

    return default


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = _get_env(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def _get_env_int(key: str, default: int) -> int:
    """Get integer environment variable."""
    value = _get_env(key, "")
    if value:
        try:
            return int(value)
        except ValueError:
            logger.warning("Invalid integer for %s: %s, using default %d", key, value, default)
    return default


def _get_env_float(key: str, default: float) -> float:
    """Get float environment variable."""
    value = _get_env(key, "")
    if value:
        try:
            return float(value)
        except ValueError:
            logger.warning("Invalid float for %s: %s, using default %f", key, value, default)
    return default


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
        retry_delay_seconds: Base delay between retries
        base_url: Optional custom API base URL (for enterprise endpoints)
    """
    api_key: str = ""
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    base_url: str | None = None

    def __post_init__(self):
        if not self.api_key:
            self.api_key = _get_env("OPENAI_API_KEY", "")

    @property
    def is_configured(self) -> bool:
        """
        Check if API key is set and appears valid.

        Accepts various API key formats:
        - Standard OpenAI keys (sk-...)
        - Azure OpenAI keys (hex strings)
        - OpenAI-compatible services (various formats)
        """
        if not self.api_key:
            return False
        # Minimum length check - most API keys are 20+ characters
        return len(self.api_key.strip()) >= 20


@dataclass
class UIConfig:
    """UI-specific configuration."""

    page_title: str = "Portfolio Commentary Generator"
    page_icon: str = "📊"
    layout: str = "wide"
    max_holdings_per_type: int = 5
    show_debug_info: bool = False
    enable_export_preview: bool = True


@dataclass
class GenerationConfig:
    """Commentary generation configuration."""

    variations_per_holding: int = 3
    target_word_count: int = 50
    word_count_tolerance: int = 15
    max_concurrent_requests: int = 3
    enable_caching: bool = True


@dataclass
class PathConfig:
    """File path configuration."""

    data_dir: Path = field(default_factory=lambda: Path("data"))
    thesis_file: Path = field(default_factory=lambda: Path("data/thesis_registry.csv"))
    exemplars_file: Path = field(default_factory=lambda: Path("data/exemplars/exemplars.json"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))

    def __post_init__(self):
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    """
    Main application configuration.

    Combines LLM, UI, generation, and path configurations.
    """

    llm: LLMConfig = field(default_factory=LLMConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Environment
    debug: bool = False
    log_level: str = "INFO"

    # Legacy compatibility
    num_variations: int = 3

    def __post_init__(self):
        self.debug = _get_env_bool("DEBUG", False)
        self.log_level = _get_env("LOG_LEVEL", "INFO").upper()

        if self.debug:
            self.ui.show_debug_info = True
            self.log_level = "DEBUG"

        # Sync num_variations
        self.num_variations = self.generation.variations_per_holding

    # Legacy property aliases for backward compatibility
    @property
    def data_dir(self) -> Path:
        return self.paths.data_dir

    @property
    def thesis_registry_path(self) -> Path:
        return self.paths.thesis_file

    @property
    def exemplars_path(self) -> Path:
        return self.paths.exemplars_file

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig(
                api_key=_get_env("OPENAI_API_KEY", ""),
                model=_get_env("OPENAI_MODEL", _get_env("LLM_MODEL", "gpt-4o")),
                temperature=_get_env_float("OPENAI_TEMPERATURE", _get_env_float("LLM_TEMPERATURE", 0.7)),
                max_tokens=_get_env_int("OPENAI_MAX_TOKENS", _get_env_int("LLM_MAX_TOKENS", 2000)),
                timeout_seconds=_get_env_float("LLM_TIMEOUT", 120.0),
                max_retries=_get_env_int("LLM_MAX_RETRIES", 3),
                base_url=_get_env("OPENAI_BASE_URL", "") or None,
            ),
            ui=UIConfig(
                max_holdings_per_type=_get_env_int("MAX_HOLDINGS", 5),
                show_debug_info=_get_env_bool("SHOW_DEBUG", False),
            ),
            generation=GenerationConfig(
                variations_per_holding=_get_env_int("VARIATIONS_COUNT", 3),
                target_word_count=_get_env_int("TARGET_WORDS", 50),
                max_concurrent_requests=_get_env_int("MAX_CONCURRENT", 3),
            ),
        )

    def validate(self) -> list[str]:
        """
        Validate configuration and return any errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.llm.is_configured:
            errors.append("OpenAI API key not configured")

        if self.llm.temperature < 0 or self.llm.temperature > 2:
            errors.append(f"Invalid temperature: {self.llm.temperature} (must be 0-2)")

        if self.llm.max_tokens < 100:
            errors.append(f"Max tokens too low: {self.llm.max_tokens}")

        if self.generation.variations_per_holding < 1:
            errors.append("Must generate at least 1 variation")

        if self.generation.variations_per_holding > 5:
            errors.append("Maximum 5 variations per holding")

        return errors


# Global config instance (lazy loaded)
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the application configuration (singleton)."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def reset_config() -> None:
    """Reset configuration (for testing)."""
    global _config
    _config = None


# Legacy function aliases for backward compatibility
def load_config_from_env() -> AppConfig:
    """Load configuration from environment variables."""
    return AppConfig.from_env()


def load_config_from_streamlit() -> AppConfig:
    """Load configuration from Streamlit secrets."""
    return AppConfig.from_env()
