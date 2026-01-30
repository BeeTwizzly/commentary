"""Tests for the prompt builder module."""

import pytest
from datetime import date

from src.generation.prompt_builder import (
    PromptBuilder,
    PromptContext,
    AssembledPrompt,
    create_prompt_context,
)
from src.generation.response_parser import (
    parse_llm_response,
    ParsedResponse,
    ParsedVariation,
)
from src.models import (
    HoldingData,
    ThesisEntry,
    ThesisLookupResult,
    ExemplarBlurb,
    ExemplarSelection,
)


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
def sample_thesis() -> ThesisLookupResult:
    """Create sample thesis lookup result."""
    entry = ThesisEntry(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        thesis_summary="AI infrastructure leader with dominant datacenter positioning. CUDA ecosystem creates high switching costs. Different risk profile at current scale but secular demand provides multi-year runway.",
        last_updated=date.today(),
        analyst="JSmith",
    )
    return ThesisLookupResult(
        ticker="NVDA",
        found=True,
        entry=entry,
        placeholder_text="",
    )


@pytest.fixture
def sample_exemplars() -> ExemplarSelection:
    """Create sample few-shot selection."""
    blurb1 = ExemplarBlurb(
        ticker="AAPL",
        company_name="Apple Inc.",
        blurb_text="Apple contributed positively as Services revenue exceeded expectations. The ecosystem monetization thesis remains intact with strong retention metrics. We maintain conviction in the long-term compounding story.",
        quarter="Q3",
        year=2025,
        blurb_type="contributor",
        word_count=38,
        source_file="Q3_2025.docx",
    )
    blurb2 = ExemplarBlurb(
        ticker="MSFT",
        company_name="Microsoft Corporation",
        blurb_text="Microsoft added to returns as Azure growth reaccelerated. Copilot monetization is beginning to materialize, providing visibility into the AI revenue opportunity. The balance sheet remains a strategic asset.",
        quarter="Q3",
        year=2025,
        blurb_type="contributor",
        word_count=37,
        source_file="Q3_2025.docx",
    )
    return ExemplarSelection(
        target_ticker="NVDA",
        target_is_contributor=True,
        same_ticker_exemplar=None,
        similar_exemplars=[blurb1, blurb2],
    )


@pytest.fixture
def sample_context(sample_holding, sample_thesis, sample_exemplars) -> PromptContext:
    """Create sample prompt context."""
    return PromptContext(
        holding=sample_holding,
        thesis=sample_thesis,
        exemplars=sample_exemplars,
        quarter="Q4 2025",
        strategy_name="Large Cap Growth",
        num_variations=3,
    )


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    def test_builds_complete_prompt(self, sample_context):
        """Should build prompt with all components."""
        builder = PromptBuilder()
        prompt = builder.build(sample_context)

        assert isinstance(prompt, AssembledPrompt)
        assert len(prompt.system_prompt) > 100
        assert len(prompt.user_prompt) > 100

    def test_system_prompt_contains_guidelines(self, sample_context):
        """System prompt should contain style guidelines."""
        builder = PromptBuilder()
        prompt = builder.build(sample_context)

        assert "4-6 sentences" in prompt.system_prompt
        assert "50-100 words" in prompt.system_prompt
        assert "[A]" in prompt.system_prompt or "variations" in prompt.system_prompt.lower()

    def test_user_prompt_contains_holding_data(self, sample_context):
        """User prompt should include holding performance data."""
        builder = PromptBuilder()
        prompt = builder.build(sample_context)

        assert "NVDA" in prompt.user_prompt
        assert "NVIDIA" in prompt.user_prompt
        assert "5.25" in prompt.user_prompt  # average weight
        assert "85.0" in prompt.user_prompt  # total attribution

    def test_user_prompt_contains_thesis(self, sample_context):
        """User prompt should include thesis text."""
        builder = PromptBuilder()
        prompt = builder.build(sample_context)

        assert "AI infrastructure" in prompt.user_prompt
        assert "CUDA" in prompt.user_prompt

    def test_user_prompt_contains_exemplars(self, sample_context):
        """User prompt should include few-shot examples."""
        builder = PromptBuilder()
        prompt = builder.build(sample_context)

        assert "Apple" in prompt.user_prompt
        assert "Microsoft" in prompt.user_prompt
        assert "Example" in prompt.user_prompt or "style" in prompt.user_prompt.lower()

    def test_metadata_populated(self, sample_context):
        """Prompt metadata should be populated correctly."""
        builder = PromptBuilder()
        prompt = builder.build(sample_context)

        assert prompt.metadata["ticker"] == "NVDA"
        assert prompt.metadata["strategy"] == "Large Cap Growth"
        assert prompt.metadata["is_contributor"] is True
        assert prompt.metadata["has_thesis"] is True
        assert prompt.metadata["exemplar_count"] == 2

    def test_handles_missing_thesis(self, sample_holding, sample_exemplars):
        """Should handle missing thesis gracefully."""
        missing_thesis = ThesisLookupResult(
            ticker="NVDA",
            found=False,
            entry=None,
            placeholder_text="[No thesis available]",
        )
        context = PromptContext(
            holding=sample_holding,
            thesis=missing_thesis,
            exemplars=sample_exemplars,
            quarter="Q4 2025",
            strategy_name="Large Cap Growth",
            num_variations=3,
        )

        builder = PromptBuilder()
        prompt = builder.build(context)

        assert "not available" in prompt.user_prompt.lower() or "general market" in prompt.user_prompt.lower()
        assert prompt.metadata["has_thesis"] is False

    def test_handles_detractor(self, sample_thesis, sample_exemplars):
        """Should adjust tone guidance for detractors."""
        detractor_holding = HoldingData(
            ticker="INTC",
            company_name="Intel Corporation",
            strategy="Large Cap Value",
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
        context = PromptContext(
            holding=detractor_holding,
            thesis=sample_thesis,
            exemplars=sample_exemplars,
            quarter="Q4 2025",
            strategy_name="Large Cap Value",
            num_variations=3,
        )

        builder = PromptBuilder()
        prompt = builder.build(context)

        assert "DETRACTED" in prompt.user_prompt or "detractor" in prompt.user_prompt.lower()


class TestResponseParser:
    """Tests for response parsing."""

    def test_parses_labeled_variations(self):
        """Should parse standard [A], [B], [C] format."""
        response = """
        [A] Apple contributed positively as Services revenue beat expectations. The ecosystem thesis remains intact with strong retention metrics. We maintain the position and expect continued growth.

        [B] Services growth drove Apple's contribution this quarter. Retention metrics support the long-term thesis and validate our investment approach. Position unchanged as fundamentals remain strong.

        [C] Apple added to returns on Services strength. The installed base monetization continues to exceed expectations. We remain constructive on the opportunity ahead.
        """

        parsed = parse_llm_response(response, expected_count=3)

        assert parsed.variation_count == 3
        assert parsed.get_variation("A") is not None
        assert parsed.get_variation("B") is not None
        assert parsed.get_variation("C") is not None

    def test_handles_alternative_formats(self):
        """Should handle A), B), C) format."""
        response = """
        A) First variation of commentary text that discusses the stock performance in detail with enough words to be valid.

        B) Second variation with different emphasis on the investment thesis and forward outlook for the position going forward.

        C) Third variation focusing on forward positioning and the conviction level we maintain in the holding.
        """

        parsed = parse_llm_response(response, expected_count=3)

        assert parsed.variation_count >= 3

    def test_calculates_word_count(self):
        """Should calculate word count for each variation."""
        response = """
        [A] This is a test variation with exactly ten words here now.

        [B] Another variation that also contains ten words in total.

        [C] Final test with a specific count of words included.
        """

        parsed = parse_llm_response(response, expected_count=3)

        for v in parsed.variations:
            assert v.word_count > 0
            assert v.word_count == len(v.text.split())

    def test_validates_variation_length(self):
        """Should identify valid vs invalid length variations."""
        short_var = ParsedVariation(label="A", text="Too short.", word_count=2)
        good_var = ParsedVariation(
            label="B",
            text="This variation has an appropriate length with enough words to be considered valid commentary for our purposes.",
            word_count=18,
        )

        assert short_var.is_valid(min_words=10) is False
        assert good_var.is_valid(min_words=10) is True

    def test_warns_on_missing_variations(self):
        """Should warn if fewer variations than expected."""
        response = """
        [A] Only one variation provided here with enough content to be parsed correctly.
        """

        parsed = parse_llm_response(response, expected_count=3)

        assert len(parsed.parse_warnings) > 0
        assert any("Expected" in w or "found" in w.lower() for w in parsed.parse_warnings)

    def test_detects_duplicate_variations(self):
        """Should detect and warn about duplicates."""
        response = """
        [A] This is the exact same text repeated with enough words to pass validation checks.

        [B] This is the exact same text repeated with enough words to pass validation checks.

        [C] This is different and unique text here with completely different content for variation.
        """

        parsed = parse_llm_response(response, expected_count=3)

        assert any("duplicate" in w.lower() for w in parsed.parse_warnings)

    def test_cleans_text_properly(self):
        """Should clean variation text of artifacts."""
        response = """
        [A]   Extra whitespace   and    spaces   should be cleaned up properly in this test case.
        """

        parsed = parse_llm_response(response, expected_count=1)

        assert "  " not in parsed.variations[0].text  # No double spaces
        assert not parsed.variations[0].text.startswith(" ")
        assert not parsed.variations[0].text.endswith(" ")


class TestCreatePromptContext:
    """Tests for the convenience function."""

    def test_creates_valid_context(self, sample_holding, sample_thesis, sample_exemplars):
        """Should create valid PromptContext."""
        context = create_prompt_context(
            holding=sample_holding,
            thesis=sample_thesis,
            exemplars=sample_exemplars,
            quarter="Q4 2025",
            strategy_name="Test Strategy",
            num_variations=3,
        )

        assert context.holding == sample_holding
        assert context.thesis == sample_thesis
        assert context.quarter == "Q4 2025"
        assert context.num_variations == 3
