"""Review UI for editing and approving generated commentary."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import streamlit as st

from src.models import HoldingData
from src.generation import GenerationResult
from src.generation.response_parser import ParsedVariation

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of a commentary approval."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_EDIT = "needs_edit"


@dataclass
class ReviewItem:
    """
    A single item in the review queue.

    Attributes:
        holding: Original holding data
        strategy: Strategy name
        result: LLM generation result
        selected_variation: Index of selected variation (0, 1, 2)
        edited_text: User-edited text (empty if using original)
        status: Current approval status
        notes: Reviewer notes
        thesis_found: Whether thesis was available during generation
    """
    holding: HoldingData
    strategy: str
    result: GenerationResult | None
    selected_variation: int = 0
    edited_text: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    notes: str = ""
    thesis_found: bool = False

    @property
    def ticker(self) -> str:
        return self.holding.ticker

    @property
    def company_name(self) -> str:
        return self.holding.company_name or self.holding.ticker

    @property
    def is_contributor(self) -> bool:
        return self.holding.is_contributor

    @property
    def has_valid_result(self) -> bool:
        return self.result is not None and self.result.success

    @property
    def variations(self) -> list[ParsedVariation]:
        if self.has_valid_result and self.result.parsed:
            return self.result.parsed.variations
        return []

    @property
    def current_text(self) -> str:
        """Get the current text (edited or selected variation)."""
        if self.edited_text:
            return self.edited_text
        if self.variations:
            idx = min(self.selected_variation, len(self.variations) - 1)
            return self.variations[idx].text
        return ""

    @property
    def current_word_count(self) -> int:
        """Get word count of current text."""
        return len(self.current_text.split())

    @property
    def is_complete(self) -> bool:
        """Check if this item is fully reviewed."""
        return self.status in (ApprovalStatus.APPROVED, ApprovalStatus.REJECTED)

    def select_variation(self, index: int) -> None:
        """Select a variation by index."""
        if 0 <= index < len(self.variations):
            self.selected_variation = index
            self.edited_text = ""  # Clear edits when selecting new variation

    def set_edited_text(self, text: str) -> None:
        """Set edited text."""
        self.edited_text = text.strip()

    def approve(self) -> None:
        """Mark as approved."""
        self.status = ApprovalStatus.APPROVED

    def reject(self) -> None:
        """Mark as rejected."""
        self.status = ApprovalStatus.REJECTED

    def mark_needs_edit(self) -> None:
        """Mark as needing edits."""
        self.status = ApprovalStatus.NEEDS_EDIT

    def reset(self) -> None:
        """Reset to pending state."""
        self.status = ApprovalStatus.PENDING
        self.edited_text = ""
        self.selected_variation = 0


@dataclass
class ReviewSession:
    """
    Complete review session state.

    Attributes:
        strategy: Strategy name
        quarter: Quarter string (e.g., "Q4 2025")
        items: List of review items
        current_index: Index of currently focused item
    """
    strategy: str
    quarter: str
    items: list[ReviewItem] = field(default_factory=list)
    current_index: int = 0

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def approved_count(self) -> int:
        return sum(1 for item in self.items if item.status == ApprovalStatus.APPROVED)

    @property
    def rejected_count(self) -> int:
        return sum(1 for item in self.items if item.status == ApprovalStatus.REJECTED)

    @property
    def pending_count(self) -> int:
        return sum(1 for item in self.items if item.status == ApprovalStatus.PENDING)

    @property
    def needs_edit_count(self) -> int:
        return sum(1 for item in self.items if item.status == ApprovalStatus.NEEDS_EDIT)

    @property
    def completion_pct(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.approved_count + self.rejected_count) / self.total_count * 100

    @property
    def is_complete(self) -> bool:
        return all(item.is_complete for item in self.items)

    @property
    def current_item(self) -> ReviewItem | None:
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index]
        return None

    @property
    def exportable_items(self) -> list[ReviewItem]:
        """Get items that are approved and ready for export."""
        return [item for item in self.items if item.status == ApprovalStatus.APPROVED]

    def get_item_by_ticker(self, ticker: str) -> ReviewItem | None:
        """Find item by ticker."""
        for item in self.items:
            if item.ticker.upper() == ticker.upper():
                return item
        return None

    def next_pending(self) -> int | None:
        """Get index of next pending item."""
        for i, item in enumerate(self.items):
            if item.status == ApprovalStatus.PENDING:
                return i
        return None

    def approve_all_pending(self) -> int:
        """Approve all pending items. Returns count approved."""
        count = 0
        for item in self.items:
            if item.status == ApprovalStatus.PENDING and item.has_valid_result:
                item.approve()
                count += 1
        return count


def init_review_session(
    strategy: str,
    quarter: str,
    generation_results: dict[str, dict],
) -> ReviewSession:
    """
    Initialize a review session from generation results.

    Args:
        strategy: Strategy name
        quarter: Quarter string
        generation_results: Dict of key -> {holding, strategy, result, thesis_found}

    Returns:
        Initialized ReviewSession
    """
    items = []

    for key, data in generation_results.items():
        if data.get("strategy") == strategy:
            holding = data.get("holding")
            result = data.get("result")
            thesis_found = data.get("thesis_found", False)

            if holding:
                items.append(ReviewItem(
                    holding=holding,
                    strategy=strategy,
                    result=result,
                    thesis_found=thesis_found,
                ))

    # Sort: contributors first, then by rank
    items.sort(key=lambda x: (not x.is_contributor, x.holding.rank))

    return ReviewSession(
        strategy=strategy,
        quarter=quarter,
        items=items,
    )


def render_review_header(session: ReviewSession) -> None:
    """Render the review page header with progress."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"## Review: {session.strategy}")
        st.caption(f"{session.quarter} | {session.total_count} holdings")

    with col2:
        # Progress metric
        st.metric(
            "Progress",
            f"{session.completion_pct:.0f}%",
            delta=f"{session.approved_count} approved",
        )

    # Progress bar
    st.progress(session.completion_pct / 100)

    # Status summary
    cols = st.columns(4)

    with cols[0]:
        st.markdown(f"**{session.approved_count}** Approved")
    with cols[1]:
        st.markdown(f"**{session.pending_count}** Pending")
    with cols[2]:
        st.markdown(f"**{session.needs_edit_count}** Needs Edit")
    with cols[3]:
        st.markdown(f"**{session.rejected_count}** Rejected")

    st.divider()


def render_review_controls(session: ReviewSession) -> None:
    """Render bulk action controls."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Approve All Pending", use_container_width=True):
            count = session.approve_all_pending()
            st.success(f"Approved {count} items")
            st.rerun()

    with col2:
        if st.button("Reset All", use_container_width=True):
            for item in session.items:
                item.reset()
            st.rerun()

    with col3:
        # Jump to next pending
        next_idx = session.next_pending()
        if next_idx is not None:
            if st.button("Next Pending", use_container_width=True):
                session.current_index = next_idx
                st.rerun()
        else:
            st.button("Next Pending", use_container_width=True, disabled=True)

    with col4:
        # Filter selector
        filter_options = ["All", "Pending", "Approved", "Needs Edit", "Rejected"]
        st.selectbox("Filter", filter_options, key="review_filter", label_visibility="collapsed")


def render_review_list(session: ReviewSession) -> None:
    """Render the list of items to review."""
    # Get filter
    filter_val = st.session_state.get("review_filter", "All")

    # Filter items
    if filter_val == "Pending":
        items = [i for i in session.items if i.status == ApprovalStatus.PENDING]
    elif filter_val == "Approved":
        items = [i for i in session.items if i.status == ApprovalStatus.APPROVED]
    elif filter_val == "Needs Edit":
        items = [i for i in session.items if i.status == ApprovalStatus.NEEDS_EDIT]
    elif filter_val == "Rejected":
        items = [i for i in session.items if i.status == ApprovalStatus.REJECTED]
    else:
        items = session.items

    if not items:
        st.info("No items match the current filter.")
        return

    # Render each item
    for i, item in enumerate(items):
        render_review_item(item, expanded=(i == 0))


def render_review_item(item: ReviewItem, expanded: bool = False) -> None:
    """Render a single review item."""
    # Determine status icon
    status_icons = {
        ApprovalStatus.PENDING: "[ ]",
        ApprovalStatus.APPROVED: "[+]",
        ApprovalStatus.REJECTED: "[x]",
        ApprovalStatus.NEEDS_EDIT: "[~]",
    }
    status_icon = status_icons.get(item.status, "[?]")

    # Direction indicator
    direction = "+" if item.is_contributor else "-"

    # Effect display
    effect = item.holding.total_attribution
    effect_str = f"{effect:+.1f} bps" if effect else ""

    # Build header
    header = f"{status_icon} [{direction}] **{item.ticker}** - {item.company_name}"
    if effect_str:
        header += f" ({effect_str})"

    with st.expander(header, expanded=expanded):
        if not item.has_valid_result:
            error_msg = item.result.error_message if item.result else "No result"
            st.error(f"Generation failed: {error_msg}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Regenerate", key=f"regen_{item.ticker}"):
                    st.session_state.regenerate_request = {
                        "strategy": item.strategy,
                        "ticker": item.ticker,
                    }
                    st.rerun()
            with col2:
                if st.button("Skip", key=f"skip_{item.ticker}"):
                    item.reject()
                    st.rerun()
            return

        # Thesis indicator
        if item.thesis_found:
            st.caption("Thesis context available")
        else:
            st.caption("No thesis context")

        # Variation selector
        render_variation_tabs(item)

        st.divider()

        # Action buttons
        render_item_actions(item)


def render_variation_tabs(item: ReviewItem) -> None:
    """Render variation selection tabs with editing."""
    variations = item.variations

    if not variations:
        st.warning("No variations available")
        return

    # Create tab labels
    tab_labels = []
    for i, v in enumerate(variations):
        label = f"Option {v.label}"
        if i == item.selected_variation and not item.edited_text:
            label += " *"
        tab_labels.append(label)

    tabs = st.tabs(tab_labels)

    for i, (tab, variation) in enumerate(zip(tabs, variations)):
        with tab:
            # Show original text
            st.markdown("**Original:**")
            st.markdown(f"> {variation.text}")
            st.caption(f"{variation.word_count} words")

            # Select button
            is_selected = (i == item.selected_variation and not item.edited_text)

            if st.button(
                "Use This Version" if not is_selected else "Selected",
                key=f"select_{item.ticker}_{i}",
                type="primary" if is_selected else "secondary",
                disabled=is_selected,
            ):
                item.select_variation(i)
                st.rerun()

    # Edit section (below tabs)
    st.divider()
    st.markdown("**Edit Commentary:**")

    # Get current text for editing
    current = item.current_text

    edited = st.text_area(
        "Edit text",
        value=current,
        height=120,
        key=f"edit_{item.ticker}",
        label_visibility="collapsed",
    )

    # Track if edited
    original_variation_text = variations[item.selected_variation].text if variations else ""

    if edited != original_variation_text:
        item.set_edited_text(edited)
        word_count = len(edited.split())
        st.caption(f"Edited | {word_count} words")
    else:
        item.edited_text = ""
        st.caption(f"{len(current.split())} words")


def render_item_actions(item: ReviewItem) -> None:
    """Render action buttons for a review item."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        approve_label = "Approved" if item.status == ApprovalStatus.APPROVED else "Approve"
        if st.button(
            approve_label,
            key=f"approve_{item.ticker}",
            type="primary" if item.status != ApprovalStatus.APPROVED else "secondary",
            use_container_width=True,
        ):
            item.approve()
            st.rerun()

    with col2:
        if st.button(
            "Needs Edit",
            key=f"needsedit_{item.ticker}",
            type="primary" if item.status == ApprovalStatus.NEEDS_EDIT else "secondary",
            use_container_width=True,
        ):
            item.mark_needs_edit()
            st.rerun()

    with col3:
        if st.button(
            "Reject",
            key=f"reject_{item.ticker}",
            type="secondary",
            use_container_width=True,
        ):
            item.reject()
            st.rerun()

    with col4:
        if st.button(
            "Reset",
            key=f"reset_{item.ticker}",
            type="secondary",
            use_container_width=True,
        ):
            item.reset()
            st.rerun()

    with col5:
        if st.button(
            "Regenerate",
            key=f"regen_success_{item.ticker}",
            type="secondary",
            use_container_width=True,
            help="Generate new commentary variations",
        ):
            st.session_state.regenerate_request = {
                "strategy": item.strategy,
                "ticker": item.ticker,
            }
            st.rerun()

    # Notes section
    with st.expander("Add Notes", expanded=False):
        notes = st.text_input(
            "Notes",
            value=item.notes,
            key=f"notes_{item.ticker}",
            label_visibility="collapsed",
            placeholder="Add reviewer notes...",
        )
        item.notes = notes


def render_export_section(session: ReviewSession) -> None:
    """Render the export section."""
    from src.ui.export_panel import render_export_panel

    st.divider()
    render_export_panel(session)


def render_review_sidebar(session: ReviewSession) -> None:
    """Render the review sidebar with summary."""
    st.markdown("### Summary")

    # Contributors vs Detractors
    contributors = [i for i in session.items if i.is_contributor]
    detractors = [i for i in session.items if not i.is_contributor]

    st.markdown("**Contributors**")
    approved_c = sum(1 for i in contributors if i.status == ApprovalStatus.APPROVED)
    st.progress(approved_c / len(contributors) if contributors else 0)
    st.caption(f"{approved_c}/{len(contributors)} approved")

    st.markdown("**Detractors**")
    approved_d = sum(1 for i in detractors if i.status == ApprovalStatus.APPROVED)
    st.progress(approved_d / len(detractors) if detractors else 0)
    st.caption(f"{approved_d}/{len(detractors)} approved")

    st.divider()

    # Quick stats
    st.markdown("### Statistics")

    # Average word count
    word_counts = [i.current_word_count for i in session.items if i.has_valid_result]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    st.metric("Avg Words", f"{avg_words:.0f}")

    # Items with edits
    edited_count = sum(1 for i in session.items if i.edited_text)
    st.metric("Edited", edited_count)

    # Total cost
    total_cost = sum(i.result.cost_usd for i in session.items if i.result)
    st.metric("Total Cost", f"${total_cost:.4f}")


def render_review_page(session: ReviewSession) -> None:
    """
    Main entry point for rendering the review page.

    Args:
        session: The review session to render
    """
    render_review_header(session)
    render_review_controls(session)

    st.divider()

    # Main review area in two columns
    col1, col2 = st.columns([3, 1])

    with col1:
        render_review_list(session)

    with col2:
        render_review_sidebar(session)

    render_export_section(session)
