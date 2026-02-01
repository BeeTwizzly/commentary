"""Prompt builder for assembling LLM generation requests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.models import (
    HoldingData,
    ThesisLookupResult,
    ExemplarSelection,
)

logger = logging.getLogger(__name__)

# Default paths for prompt templates
DEFAULT_SYSTEM_TEMPLATE = Path(__file__).parent / "prompts" / "system.txt"
DEFAULT_USER_TEMPLATE = Path(__file__).parent / "prompts" / "user_template.txt"


@dataclass(frozen=True)
class PromptContext:
    """
    All context needed to generate commentary for a single holding.

    Attributes:
        holding: Performance data for the stock
        thesis: Investment thesis lookup result
        exemplars: Selected few-shot examples
        quarter: Current quarter (e.g., "Q4 2025")
        strategy_name: Name of the strategy
        num_variations: Number of variations to generate
    """
    holding: HoldingData
    thesis: ThesisLookupResult
    exemplars: ExemplarSelection
    quarter: str
    strategy_name: str
    num_variations: int = 3


@dataclass(frozen=True)
class AssembledPrompt:
    """
    Complete prompt ready for LLM submission.

    Attributes:
        system_prompt: System/instruction content
        user_prompt: User message content
        context: Original PromptContext for reference
        metadata: Additional info for logging/debugging
    """
    system_prompt: str
    user_prompt: str
    context: PromptContext
    metadata: dict


class PromptBuilder:
    """
    Assembles prompts for commentary generation.

    Combines holding data, thesis context, and few-shot exemplars into
    structured prompts that guide the LLM to produce consistent, high-quality
    commentary variations.

    Usage:
        builder = PromptBuilder()
        prompt = builder.build(context)
        # prompt.system_prompt, prompt.user_prompt ready for LLM
    """

    def __init__(
        self,
        system_template_path: Path | str | None = None,
        user_template_path: Path | str | None = None,
    ):
        """
        Initialize builder with optional custom templates.

        Args:
            system_template_path: Path to system prompt template
            user_template_path: Path to user prompt template
        """
        self._system_template = self._load_template(
            system_template_path or DEFAULT_SYSTEM_TEMPLATE
        )
        self._user_template = self._load_template(
            user_template_path or DEFAULT_USER_TEMPLATE
        )

        logger.info("PromptBuilder initialized")

    def build(self, context: PromptContext) -> AssembledPrompt:
        """
        Build complete prompt from context.

        Args:
            context: PromptContext with all required data

        Returns:
            AssembledPrompt ready for LLM submission
        """
        # Build system prompt (mostly static)
        system_prompt = self._build_system_prompt(context)

        # Build user prompt (dynamic per holding)
        user_prompt = self._build_user_prompt(context)

        metadata = {
            "ticker": context.holding.ticker,
            "strategy": context.strategy_name,
            "quarter": context.quarter,
            "is_contributor": context.holding.is_contributor,
            "has_thesis": context.thesis.found,
            "exemplar_count": len(context.exemplars.get_all_exemplars()),
            "num_variations": context.num_variations,
        }

        logger.debug(
            "Built prompt for %s: %d system chars, %d user chars",
            context.holding.ticker,
            len(system_prompt),
            len(user_prompt),
        )

        return AssembledPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            metadata=metadata,
        )

    def _build_system_prompt(self, context: PromptContext) -> str:
        """Build the system prompt with style guidelines."""
        return self._system_template.format(
            num_variations=context.num_variations,
        )

    def _build_user_prompt(self, context: PromptContext) -> str:
        """Build the user prompt with holding-specific data."""
        holding = context.holding

        # Format performance data
        performance_summary = self._format_performance_summary(holding)

        # Format thesis
        thesis_text = self._format_thesis(context.thesis)

        # Format exemplars
        exemplars_text = context.exemplars.format_for_prompt()

        # Determine tone guidance based on contributor/detractor
        if holding.is_contributor:
            tone_guidance = "This stock CONTRIBUTED to performance. Frame positively while remaining balanced."
        else:
            tone_guidance = "This stock DETRACTED from performance. Acknowledge challenges while maintaining constructive tone."

        return self._user_template.format(
            quarter=context.quarter,
            strategy_name=context.strategy_name,
            ticker=holding.ticker,
            company_name=context.thesis.company_name,
            performance_summary=performance_summary,
            thesis_text=thesis_text,
            exemplars_text=exemplars_text,
            tone_guidance=tone_guidance,
            num_variations=context.num_variations,
            contribution_type="contributor" if holding.is_contributor else "detractor",
        )

    def _format_performance_summary(self, holding: HoldingData) -> str:
        """Format holding performance data for prompt injection."""
        lines = [
            f"Ticker: {holding.ticker}",
            f"Rank: #{holding.rank} {'contributor' if holding.is_contributor else 'detractor'}",
        ]

        # Weight information
        lines.append(f"Average Weight: {holding.avg_weight:.2f}%")
        lines.append(f"End Weight: {holding.end_weight:.2f}%")

        active = holding.avg_weight - holding.benchmark_weight
        lines.append(f"Benchmark Weight: {holding.benchmark_weight:.2f}% (Active: {active:+.2f}%)")

        # Attribution (in basis points)
        lines.append(f"Total Attribution: {holding.total_attribution:+.1f} bps")
        lines.append(f"Selection Effect: {holding.selection_effect:+.1f} bps")
        lines.append(f"Allocation Effect: {holding.allocation_effect:+.1f} bps")

        return "\n".join(lines)

    def _format_thesis(self, thesis: ThesisLookupResult) -> str:
        """Format thesis for prompt injection."""
        if thesis.found and thesis.entry:
            age_days = thesis.entry.age_days()
            freshness = "current" if age_days < 90 else "may need refresh" if age_days < 180 else "stale"

            return (
                f"Investment Thesis ({freshness}, last updated {age_days} days ago):\n"
                f"{thesis.entry.thesis_summary}"
            )
        else:
            return (
                "Investment Thesis: Not available in registry.\n"
                "Generate commentary based on performance data and general market knowledge."
            )

    def _load_template(self, path: Path | str) -> str:
        """Load template from file, with fallback to embedded default."""
        path = Path(path)

        if path.exists():
            content = path.read_text(encoding="utf-8")
            logger.debug("Loaded template from %s", path)
            return content

        logger.warning("Template not found at %s, using embedded default", path)

        # Return embedded defaults
        if "system" in str(path).lower():
            return self._default_system_template()
        else:
            return self._default_user_template()

    def _default_system_template(self) -> str:
        """Embedded default system template."""
        return '''You are an expert portfolio commentary writer for an institutional investment firm.

Your task is to write quarterly stock commentary blurbs that will appear in client-facing portfolio reports.

## Style Guidelines

1. **Length**: Each blurb should be 4-6 sentences, approximately 50-100 words.

2. **Structure**:
   - Lead with performance context (contributed/detracted and why)
   - Reference the investment thesis or key drivers
   - Include a forward-looking statement on positioning
   - End with conviction level or action taken

3. **Tone**:
   - Professional but accessible (avoid excessive jargon)
   - Confident but not hyperbolic
   - Balanced—acknowledge both positives and risks
   - Active voice preferred

4. **Avoid**:
   - Hedging language ("we believe", "we think", "arguably")
   - Excessive qualifiers ("very", "extremely", "significantly")
   - Generic statements that could apply to any stock
   - Forward-looking predictions with specific numbers

## Output Format

Provide exactly {num_variations} distinct variations. Use ONLY a bracket label followed IMMEDIATELY by the commentary text. Do NOT add any descriptive text, variation titles, or explanatory prefixes after the label.

CORRECT format:
[A] The stock contributed positively to portfolio performance...

INCORRECT format (do NOT do this):
[A] First variation — Thesis confirmation: The stock contributed...
[A] Variation 1 (Conservative): The stock contributed...

Each variation should take a slightly different angle or emphasis while maintaining the same factual accuracy and professional tone.'''

    def _default_user_template(self) -> str:
        """Embedded default user template."""
        return '''## Current Assignment

Write commentary for the following holding:

**Quarter**: {quarter}
**Strategy**: {strategy_name}
**Stock**: {company_name} ({ticker})
**Type**: {contribution_type}

## Performance Data

{performance_summary}

## Investment Context

{thesis_text}

## Style Examples

Study these examples from prior quarters to match our writing style:

{exemplars_text}

## Instructions

{tone_guidance}

Write {num_variations} distinct variations of commentary for {company_name} ({ticker}).

Each variation should:
- Be 4-6 sentences (50-100 words)
- Reference the performance data provided
- Connect to the investment thesis
- Include forward-looking positioning

Label each variation [A], [B], [C] and start the commentary IMMEDIATELY after the label. Do not add any descriptive prefixes, variation titles, or category labels.'''


def create_prompt_context(
    holding: HoldingData,
    thesis: ThesisLookupResult,
    exemplars: ExemplarSelection,
    quarter: str,
    strategy_name: str,
    num_variations: int = 3,
) -> PromptContext:
    """
    Convenience function to create PromptContext.

    Args:
        holding: Performance data for the stock
        thesis: Investment thesis lookup result
        exemplars: Selected few-shot examples
        quarter: Current quarter (e.g., "Q4 2025")
        strategy_name: Name of the strategy
        num_variations: Number of variations to generate

    Returns:
        PromptContext ready for prompt building
    """
    return PromptContext(
        holding=holding,
        thesis=thesis,
        exemplars=exemplars,
        quarter=quarter,
        strategy_name=strategy_name,
        num_variations=num_variations,
    )
