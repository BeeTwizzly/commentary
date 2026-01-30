"""Reusable UI components for the Streamlit app."""

from __future__ import annotations

import streamlit as st
from src.models import HoldingData
from src.generation import GenerationResult


def holding_card(
    holding: HoldingData,
    result: GenerationResult | None = None,
    show_variations: bool = True,
) -> None:
    """
    Render a holding card with optional generation results.

    Args:
        holding: The holding data to display
        result: Optional generation result
        show_variations: Whether to show variation tabs
    """
    # Header with type indicator
    effect_str = ""
    if holding.total_attribution:
        sign = "+" if holding.total_attribution > 0 else ""
        effect_str = f"({sign}{holding.total_attribution:.1f} bps)"

    icon = "+" if holding.is_contributor else "-"

    st.markdown(f"### [{icon}] {holding.ticker} - {holding.company_name} {effect_str}")

    # Weight info
    if holding.avg_weight:
        st.caption(f"Avg Weight: {holding.avg_weight:.2f}%")

    # Results
    if result:
        if result.success and show_variations:
            tabs = st.tabs([f"Option {v.label}" for v in result.variations])
            for tab, variation in zip(tabs, result.variations):
                with tab:
                    st.write(variation.text)
                    st.caption(f"{variation.word_count} words")
        elif not result.success:
            st.error(f"Generation failed: {result.error_message}")


def progress_indicator(
    current: int,
    total: int,
    label: str = "Progress",
) -> None:
    """
    Render a progress indicator with label.

    Args:
        current: Current progress count
        total: Total items
        label: Label to display
    """
    if total == 0:
        progress = 1.0
    else:
        progress = current / total

    st.progress(progress, text=f"{label}: {current}/{total}")


def cost_display(
    cost_usd: float,
    tokens: int,
    label: str = "Cost",
) -> None:
    """
    Render cost and token usage display.

    Args:
        cost_usd: Cost in USD
        tokens: Token count
        label: Display label
    """
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label, f"${cost_usd:.4f}")
    with col2:
        st.metric("Tokens", f"{tokens:,}")


def strategy_summary(
    name: str,
    contributor_count: int,
    detractor_count: int,
    expanded: bool = False,
) -> None:
    """
    Render a strategy summary card.

    Args:
        name: Strategy name
        contributor_count: Number of contributors
        detractor_count: Number of detractors
        expanded: Whether to expand by default
    """
    with st.expander(f"{name}", expanded=expanded):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Contributors", contributor_count, delta=None)
        with col2:
            st.metric("Detractors", detractor_count, delta=None)
