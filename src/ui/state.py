"""Session state management for Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import streamlit as st

from src.models import ParsedWorkbook
from src.generation import GenerationResult


@dataclass
class GenerationState:
    """State for a single generation."""
    ticker: str
    strategy: str
    result: GenerationResult | None = None
    error: str | None = None
    selected_variation: str = "A"  # Which variation is selected for export
    edited_text: str | None = None  # User's edited version


@dataclass
class AppState:
    """
    Complete application state.

    This mirrors the session_state structure but provides
    type hints and defaults in a structured way.
    """
    # Data
    workbook: ParsedWorkbook | None = None
    selected_strategies: list[str] = field(default_factory=list)

    # Period
    quarter: str = "Q4"
    year: int = field(default_factory=lambda: date.today().year)

    # Generation
    generation_results: dict[str, GenerationState] = field(default_factory=dict)
    generation_in_progress: bool = False
    generation_progress: float = 0.0
    generation_status: str = ""

    # Costs
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    # Settings
    num_variations: int = 3


def get_state_key(strategy: str, ticker: str) -> str:
    """Generate a unique key for a holding."""
    return f"{strategy}|{ticker}"


def parse_state_key(key: str) -> tuple[str, str]:
    """Parse a state key into (strategy, ticker)."""
    parts = key.split("|", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", key


def reset_generation_state() -> None:
    """Reset all generation-related state."""
    st.session_state.generation_results = {}
    st.session_state.generation_in_progress = False
    st.session_state.generation_progress = 0.0
    st.session_state.generation_status = ""


def update_selected_variation(key: str, variation: str) -> None:
    """Update the selected variation for a holding."""
    if key in st.session_state.generation_results:
        st.session_state.generation_results[key]["selected_variation"] = variation


def update_edited_text(key: str, text: str) -> None:
    """Update the edited text for a holding."""
    if key in st.session_state.generation_results:
        st.session_state.generation_results[key]["edited_text"] = text
