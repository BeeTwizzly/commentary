"""Parser for extracting variations from LLM responses."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Pattern to match variation labels [A], [B], [C], etc.
VARIATION_PATTERN = re.compile(
    r'\[([A-Z])\]\s*(.+?)(?=\[([A-Z])\]|$)',
    re.DOTALL
)

# Alternative patterns for less strict formatting
ALT_PATTERNS = [
    re.compile(r'(?:Variation\s+)?([A-Z])[\):\.]?\s*(.+?)(?=(?:Variation\s+)?[A-Z][\):\.]|$)', re.DOTALL | re.IGNORECASE),
    re.compile(r'(\d)[\):\.]?\s*(.+?)(?=\d[\):\.]|$)', re.DOTALL),
]


@dataclass
class ParsedVariation:
    """
    A single parsed variation from LLM response.

    Attributes:
        label: The variation label (A, B, C, or 1, 2, 3)
        text: The commentary text
        word_count: Number of words in the text
    """
    label: str
    text: str
    word_count: int

    def is_valid(self, min_words: int = 30, max_words: int = 150) -> bool:
        """Check if variation meets length requirements."""
        return min_words <= self.word_count <= max_words


@dataclass
class ParsedResponse:
    """
    Complete parsed response with all variations.

    Attributes:
        variations: List of parsed variations
        raw_response: Original LLM response text
        parse_warnings: Any issues encountered during parsing
    """
    variations: list[ParsedVariation]
    raw_response: str
    parse_warnings: list[str]

    @property
    def variation_count(self) -> int:
        return len(self.variations)

    def get_variation(self, label: str) -> ParsedVariation | None:
        """Get variation by label (case-insensitive)."""
        label = label.upper()
        for v in self.variations:
            if v.label.upper() == label:
                return v
        return None

    def get_valid_variations(self, min_words: int = 30, max_words: int = 150) -> list[ParsedVariation]:
        """Return only variations meeting length requirements."""
        return [v for v in self.variations if v.is_valid(min_words, max_words)]

    def has_all_expected(self, expected_count: int = 3) -> bool:
        """Check if we got the expected number of valid variations."""
        return len(self.get_valid_variations()) >= expected_count


def parse_llm_response(response_text: str, expected_count: int = 3) -> ParsedResponse:
    """
    Parse LLM response to extract labeled variations.

    Args:
        response_text: Raw text from LLM
        expected_count: Expected number of variations

    Returns:
        ParsedResponse with extracted variations
    """
    warnings = []
    variations = []

    # Try primary pattern first: [A], [B], [C]
    matches = VARIATION_PATTERN.findall(response_text)

    if matches:
        for match in matches:
            label = match[0]
            text = _clean_text(match[1])

            if text:
                variations.append(ParsedVariation(
                    label=label,
                    text=text,
                    word_count=len(text.split()),
                ))

    # If primary pattern failed, try alternatives
    if len(variations) < expected_count:
        for alt_pattern in ALT_PATTERNS:
            alt_matches = alt_pattern.findall(response_text)

            if len(alt_matches) >= expected_count:
                variations = []
                for match in alt_matches:
                    label = str(match[0]).upper()
                    text = _clean_text(match[1])

                    if text:
                        variations.append(ParsedVariation(
                            label=label,
                            text=text,
                            word_count=len(text.split()),
                        ))

                warnings.append("Used alternative parsing pattern")
                break

    # If still not enough, try splitting by blank lines
    if len(variations) < expected_count:
        paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]

        # Filter to reasonable length paragraphs
        valid_paragraphs = [
            p for p in paragraphs
            if 30 <= len(p.split()) <= 200
        ]

        if len(valid_paragraphs) >= expected_count:
            variations = []
            labels = ['A', 'B', 'C', 'D', 'E']

            for i, para in enumerate(valid_paragraphs[:expected_count]):
                # Strip any existing label
                text = re.sub(r'^\[?[A-Z]\]?[\):\.]?\s*', '', para)
                text = _clean_text(text)

                variations.append(ParsedVariation(
                    label=labels[i] if i < len(labels) else str(i + 1),
                    text=text,
                    word_count=len(text.split()),
                ))

            warnings.append("Fell back to paragraph splitting")

    # Validate results
    if len(variations) < expected_count:
        warnings.append(f"Expected {expected_count} variations, found {len(variations)}")

    # Check for duplicates
    seen_texts = set()
    unique_variations = []
    for v in variations:
        # Normalize for comparison
        normalized = v.text.lower()[:100]
        if normalized not in seen_texts:
            seen_texts.add(normalized)
            unique_variations.append(v)
        else:
            warnings.append(f"Duplicate variation detected: {v.label}")

    if len(unique_variations) < len(variations):
        variations = unique_variations

    logger.debug(
        "Parsed %d variations from response (%d chars), %d warnings",
        len(variations),
        len(response_text),
        len(warnings),
    )

    return ParsedResponse(
        variations=variations,
        raw_response=response_text,
        parse_warnings=warnings,
    )


def _clean_text(text: str) -> str:
    """Clean and normalize variation text."""
    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove any remaining label artifacts at the start
    text = re.sub(r'^[\[\(]?[A-Za-z0-9][\]\):]?\s*', '', text)

    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove trailing incomplete sentences if present
    if text and text[-1] not in '.!?"\'':
        # Check if last sentence seems complete
        last_period = text.rfind('.')
        if last_period > len(text) * 0.7:  # If period in last 30%, keep as is
            pass
        elif last_period > 0:
            # Truncate to last complete sentence
            text = text[:last_period + 1]

    return text
