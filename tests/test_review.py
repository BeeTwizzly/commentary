"""Tests for the review UI module."""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from src.ui.review import (
    ApprovalStatus,
    ReviewItem,
    ReviewSession,
    init_review_session,
)
from src.models import HoldingData
from src.generation import GenerationResult
from src.generation.response_parser import ParsedResponse, ParsedVariation


@pytest.fixture
def sample_holding() -> HoldingData:
    """Create sample holding data."""
    return HoldingData(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        strategy="Large Cap Growth",
        avg_weight=5.25,
        begin_weight=4.80,
        end_weight=5.70,
        benchmark_weight=3.50,
        benchmark_return=12.5,
        total_attribution=85.0,
        selection_effect=65.0,
        allocation_effect=20.0,
        rank=1,
        is_contributor=True,
    )


@pytest.fixture
def sample_detractor() -> HoldingData:
    """Create sample detractor holding."""
    return HoldingData(
        ticker="INTC",
        company_name="Intel Corporation",
        strategy="Large Cap Growth",
        avg_weight=2.0,
        begin_weight=2.2,
        end_weight=1.8,
        benchmark_weight=1.5,
        benchmark_return=-5.0,
        total_attribution=-35.0,
        selection_effect=-25.0,
        allocation_effect=-10.0,
        rank=1,
        is_contributor=False,
    )


@pytest.fixture
def sample_variations() -> list[ParsedVariation]:
    """Create sample parsed variations."""
    return [
        ParsedVariation(
            label="A",
            text="NVIDIA contributed positively as datacenter revenue exceeded expectations. The AI infrastructure thesis remains intact with strong demand visibility.",
            word_count=21,
        ),
        ParsedVariation(
            label="B",
            text="Datacenter strength drove NVIDIA's contribution this quarter. GPU demand for AI workloads validates our investment thesis.",
            word_count=18,
        ),
        ParsedVariation(
            label="C",
            text="NVIDIA added to returns on AI-related demand. Position maintained given secular tailwinds in compute infrastructure.",
            word_count=17,
        ),
    ]


@pytest.fixture
def sample_generation_result(sample_variations) -> GenerationResult:
    """Create sample generation result."""
    from src.generation.llm_client import TokenUsage

    parsed = ParsedResponse(
        variations=sample_variations,
        raw_response="[A] NVIDIA contributed... [B] Datacenter... [C] NVIDIA added...",
        parse_warnings=[],
    )
    return GenerationResult(
        success=True,
        parsed=parsed,
        cost_usd=0.0025,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        latency_seconds=1.5,
        model="gpt-4o-mini",
        error_message="",
    )


@pytest.fixture
def sample_review_item(sample_holding, sample_generation_result) -> ReviewItem:
    """Create sample review item."""
    return ReviewItem(
        holding=sample_holding,
        strategy="Large Cap Growth",
        result=sample_generation_result,
        thesis_found=True,
    )


@pytest.fixture
def sample_review_session(sample_holding, sample_detractor, sample_generation_result) -> ReviewSession:
    """Create sample review session with multiple items."""
    items = [
        ReviewItem(
            holding=sample_holding,
            strategy="Large Cap Growth",
            result=sample_generation_result,
            thesis_found=True,
        ),
        ReviewItem(
            holding=sample_detractor,
            strategy="Large Cap Growth",
            result=sample_generation_result,
            thesis_found=False,
        ),
    ]
    return ReviewSession(
        strategy="Large Cap Growth",
        quarter="Q4 2025",
        items=items,
    )


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_enum_values(self):
        """Should have all expected status values."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.NEEDS_EDIT.value == "needs_edit"

    def test_all_statuses_distinct(self):
        """All status values should be unique."""
        values = [s.value for s in ApprovalStatus]
        assert len(values) == len(set(values))


class TestReviewItem:
    """Tests for ReviewItem class."""

    def test_creates_with_defaults(self, sample_holding, sample_generation_result):
        """Should create item with proper defaults."""
        item = ReviewItem(
            holding=sample_holding,
            strategy="Test Strategy",
            result=sample_generation_result,
        )

        assert item.selected_variation == 0
        assert item.edited_text == ""
        assert item.status == ApprovalStatus.PENDING
        assert item.notes == ""
        assert item.thesis_found is False

    def test_ticker_property(self, sample_review_item):
        """Should expose ticker from holding."""
        assert sample_review_item.ticker == "NVDA"

    def test_company_name_property(self, sample_review_item):
        """Should expose company name from holding."""
        assert sample_review_item.company_name == "NVIDIA Corporation"

    def test_is_contributor_property(self, sample_review_item, sample_detractor):
        """Should correctly identify contributors and detractors."""
        assert sample_review_item.is_contributor is True

        detractor_item = ReviewItem(
            holding=sample_detractor,
            strategy="Test",
            result=None,
        )
        assert detractor_item.is_contributor is False

    def test_has_valid_result_true(self, sample_review_item):
        """Should return True when result exists and is successful."""
        assert sample_review_item.has_valid_result is True

    def test_has_valid_result_false_no_result(self, sample_holding):
        """Should return False when no result."""
        item = ReviewItem(
            holding=sample_holding,
            strategy="Test",
            result=None,
        )
        assert item.has_valid_result is False

    def test_has_valid_result_false_failed_result(self, sample_holding):
        """Should return False when result failed."""
        from src.generation.llm_client import TokenUsage

        failed_result = GenerationResult(
            success=False,
            parsed=None,
            cost_usd=0.0,
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            latency_seconds=0.0,
            model="gpt-4o-mini",
            error_message="API error",
        )
        item = ReviewItem(
            holding=sample_holding,
            strategy="Test",
            result=failed_result,
        )
        assert item.has_valid_result is False

    def test_variations_property(self, sample_review_item):
        """Should return variations from parsed result."""
        variations = sample_review_item.variations
        assert len(variations) == 3
        assert variations[0].label == "A"

    def test_variations_empty_when_no_result(self, sample_holding):
        """Should return empty list when no result."""
        item = ReviewItem(
            holding=sample_holding,
            strategy="Test",
            result=None,
        )
        assert item.variations == []

    def test_current_text_returns_selected_variation(self, sample_review_item):
        """Should return text of selected variation."""
        text = sample_review_item.current_text
        assert "NVIDIA contributed" in text

    def test_current_text_returns_edited_when_set(self, sample_review_item):
        """Should return edited text when present."""
        sample_review_item.set_edited_text("Custom edited text here.")
        assert sample_review_item.current_text == "Custom edited text here."

    def test_current_word_count(self, sample_review_item):
        """Should calculate word count correctly."""
        count = sample_review_item.current_word_count
        assert count > 0
        assert count == len(sample_review_item.current_text.split())

    def test_is_complete_when_approved(self, sample_review_item):
        """Should be complete when approved."""
        sample_review_item.approve()
        assert sample_review_item.is_complete is True

    def test_is_complete_when_rejected(self, sample_review_item):
        """Should be complete when rejected."""
        sample_review_item.reject()
        assert sample_review_item.is_complete is True

    def test_is_not_complete_when_pending(self, sample_review_item):
        """Should not be complete when pending."""
        assert sample_review_item.is_complete is False

    def test_is_not_complete_when_needs_edit(self, sample_review_item):
        """Should not be complete when needs edit."""
        sample_review_item.mark_needs_edit()
        assert sample_review_item.is_complete is False

    def test_select_variation(self, sample_review_item):
        """Should select variation and clear edits."""
        sample_review_item.set_edited_text("Some edit")
        sample_review_item.select_variation(1)

        assert sample_review_item.selected_variation == 1
        assert sample_review_item.edited_text == ""

    def test_select_variation_ignores_invalid_index(self, sample_review_item):
        """Should ignore invalid variation index."""
        sample_review_item.select_variation(99)
        assert sample_review_item.selected_variation == 0  # unchanged

    def test_set_edited_text_strips_whitespace(self, sample_review_item):
        """Should strip whitespace from edited text."""
        sample_review_item.set_edited_text("  Trimmed text  ")
        assert sample_review_item.edited_text == "Trimmed text"

    def test_approve(self, sample_review_item):
        """Should mark as approved."""
        sample_review_item.approve()
        assert sample_review_item.status == ApprovalStatus.APPROVED

    def test_reject(self, sample_review_item):
        """Should mark as rejected."""
        sample_review_item.reject()
        assert sample_review_item.status == ApprovalStatus.REJECTED

    def test_mark_needs_edit(self, sample_review_item):
        """Should mark as needs edit."""
        sample_review_item.mark_needs_edit()
        assert sample_review_item.status == ApprovalStatus.NEEDS_EDIT

    def test_reset(self, sample_review_item):
        """Should reset all state."""
        sample_review_item.select_variation(2)
        sample_review_item.set_edited_text("Edited")
        sample_review_item.approve()

        sample_review_item.reset()

        assert sample_review_item.status == ApprovalStatus.PENDING
        assert sample_review_item.edited_text == ""
        assert sample_review_item.selected_variation == 0


class TestReviewSession:
    """Tests for ReviewSession class."""

    def test_creates_with_defaults(self):
        """Should create session with proper defaults."""
        session = ReviewSession(
            strategy="Test Strategy",
            quarter="Q4 2025",
        )

        assert session.strategy == "Test Strategy"
        assert session.quarter == "Q4 2025"
        assert session.items == []
        assert session.current_index == 0

    def test_total_count(self, sample_review_session):
        """Should return total item count."""
        assert sample_review_session.total_count == 2

    def test_approved_count(self, sample_review_session):
        """Should count approved items."""
        assert sample_review_session.approved_count == 0

        sample_review_session.items[0].approve()
        assert sample_review_session.approved_count == 1

    def test_rejected_count(self, sample_review_session):
        """Should count rejected items."""
        assert sample_review_session.rejected_count == 0

        sample_review_session.items[0].reject()
        assert sample_review_session.rejected_count == 1

    def test_pending_count(self, sample_review_session):
        """Should count pending items."""
        assert sample_review_session.pending_count == 2

        sample_review_session.items[0].approve()
        assert sample_review_session.pending_count == 1

    def test_needs_edit_count(self, sample_review_session):
        """Should count needs edit items."""
        assert sample_review_session.needs_edit_count == 0

        sample_review_session.items[0].mark_needs_edit()
        assert sample_review_session.needs_edit_count == 1

    def test_completion_pct_zero(self, sample_review_session):
        """Should return 0% when all pending."""
        assert sample_review_session.completion_pct == 0.0

    def test_completion_pct_partial(self, sample_review_session):
        """Should calculate partial completion."""
        sample_review_session.items[0].approve()
        assert sample_review_session.completion_pct == 50.0

    def test_completion_pct_full(self, sample_review_session):
        """Should return 100% when all complete."""
        sample_review_session.items[0].approve()
        sample_review_session.items[1].reject()
        assert sample_review_session.completion_pct == 100.0

    def test_completion_pct_empty_session(self):
        """Should handle empty session."""
        session = ReviewSession(strategy="Test", quarter="Q4")
        assert session.completion_pct == 0.0

    def test_is_complete_false(self, sample_review_session):
        """Should return False when items pending."""
        assert sample_review_session.is_complete is False

    def test_is_complete_true(self, sample_review_session):
        """Should return True when all complete."""
        sample_review_session.items[0].approve()
        sample_review_session.items[1].approve()
        assert sample_review_session.is_complete is True

    def test_current_item(self, sample_review_session):
        """Should return item at current index."""
        item = sample_review_session.current_item
        assert item is not None
        assert item.ticker == "NVDA"

    def test_current_item_none_on_empty(self):
        """Should return None for empty session."""
        session = ReviewSession(strategy="Test", quarter="Q4")
        assert session.current_item is None

    def test_exportable_items(self, sample_review_session):
        """Should return only approved items."""
        assert len(sample_review_session.exportable_items) == 0

        sample_review_session.items[0].approve()
        assert len(sample_review_session.exportable_items) == 1

        sample_review_session.items[1].reject()
        assert len(sample_review_session.exportable_items) == 1  # rejected not included

    def test_get_item_by_ticker(self, sample_review_session):
        """Should find item by ticker."""
        item = sample_review_session.get_item_by_ticker("NVDA")
        assert item is not None
        assert item.ticker == "NVDA"

    def test_get_item_by_ticker_case_insensitive(self, sample_review_session):
        """Should find item regardless of case."""
        item = sample_review_session.get_item_by_ticker("nvda")
        assert item is not None

    def test_get_item_by_ticker_not_found(self, sample_review_session):
        """Should return None for unknown ticker."""
        assert sample_review_session.get_item_by_ticker("UNKNOWN") is None

    def test_next_pending(self, sample_review_session):
        """Should return index of first pending item."""
        assert sample_review_session.next_pending() == 0

        sample_review_session.items[0].approve()
        assert sample_review_session.next_pending() == 1

    def test_next_pending_none(self, sample_review_session):
        """Should return None when no pending items."""
        sample_review_session.items[0].approve()
        sample_review_session.items[1].approve()
        assert sample_review_session.next_pending() is None

    def test_approve_all_pending(self, sample_review_session):
        """Should approve all pending items with valid results."""
        count = sample_review_session.approve_all_pending()

        assert count == 2
        assert sample_review_session.approved_count == 2

    def test_approve_all_pending_skips_invalid(self, sample_holding):
        """Should skip items without valid results."""
        items = [
            ReviewItem(
                holding=sample_holding,
                strategy="Test",
                result=None,  # No result
            ),
        ]
        session = ReviewSession(
            strategy="Test",
            quarter="Q4",
            items=items,
        )

        count = session.approve_all_pending()

        assert count == 0
        assert session.pending_count == 1


class TestInitReviewSession:
    """Tests for init_review_session function."""

    def test_creates_session_from_results(self, sample_holding, sample_generation_result):
        """Should create session from generation results."""
        results = {
            "Large Cap Growth|NVDA": {
                "holding": sample_holding,
                "strategy": "Large Cap Growth",
                "result": sample_generation_result,
                "thesis_found": True,
            },
        }

        session = init_review_session(
            strategy="Large Cap Growth",
            quarter="Q4 2025",
            generation_results=results,
        )

        assert session.strategy == "Large Cap Growth"
        assert session.quarter == "Q4 2025"
        assert session.total_count == 1
        assert session.items[0].ticker == "NVDA"
        assert session.items[0].thesis_found is True

    def test_filters_by_strategy(self, sample_holding, sample_detractor, sample_generation_result):
        """Should only include items matching strategy."""
        results = {
            "Large Cap Growth|NVDA": {
                "holding": sample_holding,
                "strategy": "Large Cap Growth",
                "result": sample_generation_result,
                "thesis_found": True,
            },
            "Small Cap Value|INTC": {
                "holding": sample_detractor,
                "strategy": "Small Cap Value",
                "result": sample_generation_result,
                "thesis_found": False,
            },
        }

        session = init_review_session(
            strategy="Large Cap Growth",
            quarter="Q4 2025",
            generation_results=results,
        )

        assert session.total_count == 1
        assert session.items[0].ticker == "NVDA"

    def test_sorts_contributors_first(self, sample_holding, sample_detractor, sample_generation_result):
        """Should sort contributors before detractors."""
        # Put detractor first in input
        results = {
            "Test|INTC": {
                "holding": sample_detractor,
                "strategy": "Test",
                "result": sample_generation_result,
                "thesis_found": False,
            },
            "Test|NVDA": {
                "holding": sample_holding,
                "strategy": "Test",
                "result": sample_generation_result,
                "thesis_found": True,
            },
        }

        session = init_review_session(
            strategy="Test",
            quarter="Q4 2025",
            generation_results=results,
        )

        # Contributor should be first
        assert session.items[0].is_contributor is True
        assert session.items[1].is_contributor is False

    def test_handles_empty_results(self):
        """Should handle empty results dict."""
        session = init_review_session(
            strategy="Test",
            quarter="Q4 2025",
            generation_results={},
        )

        assert session.total_count == 0

    def test_handles_missing_holding(self, sample_generation_result):
        """Should skip entries without holding."""
        results = {
            "Test|NVDA": {
                "strategy": "Test",
                "result": sample_generation_result,
                "thesis_found": True,
                # No holding key
            },
        }

        session = init_review_session(
            strategy="Test",
            quarter="Q4 2025",
            generation_results=results,
        )

        assert session.total_count == 0
