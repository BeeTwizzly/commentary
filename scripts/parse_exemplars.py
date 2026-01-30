#!/usr/bin/env python3
"""
Script to parse exemplar Word documents and generate JSON index.

Usage:
    python scripts/parse_exemplars.py --input-dir data/exemplar_docs --output data/exemplars/exemplars.json

If no input directory exists, creates a sample JSON with synthetic exemplars for testing.
"""

import argparse
import logging
from pathlib import Path

from src.parsers.exemplar_parser import parse_directory, save_exemplars_json
from src.models import ExemplarBlurb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_synthetic_exemplars() -> list[ExemplarBlurb]:
    """Create synthetic exemplars for testing when no real docs available."""

    synthetic_data = [
        # Contributors
        ("NVDA", "NVIDIA Corporation", "contributor", "Q3", 2025,
         "NVIDIA continued its exceptional performance as datacenter revenue grew 120% year-over-year, "
         "driven by insatiable demand for AI training infrastructure. Management raised guidance for the "
         "third consecutive quarter, citing enterprise adoption broadening beyond hyperscalers. While "
         "valuation appears extended, the duration of the AI investment cycle supports our overweight position."),

        ("AAPL", "Apple Inc.", "contributor", "Q3", 2025,
         "Apple contributed positively as Services revenue inflected higher, growing 18% and demonstrating "
         "the recurring nature of its ecosystem monetization. iPhone 16 early demand data suggests a stronger "
         "upgrade cycle than feared. The capital return program continues to provide valuation support. We "
         "maintain our position."),

        ("MSFT", "Microsoft Corporation", "contributor", "Q2", 2025,
         "Microsoft delivered robust results with Azure growth reaccelerating to 32%, ahead of expectations. "
         "Copilot adoption across the enterprise suite is gaining traction with Fortune 500 customers. The "
         "combination of cloud scale and AI positioning creates a compelling long-term growth profile."),

        ("AMZN", "Amazon.com Inc.", "contributor", "Q2", 2025,
         "Amazon added to performance as AWS margins expanded meaningfully while maintaining 20%+ growth. "
         "The advertising business continues to compound at impressive rates with minimal incremental investment. "
         "Retail profitability is normalizing faster than anticipated. We remain constructive."),

        ("META", "Meta Platforms Inc.", "contributor", "Q1", 2025,
         "Meta contributed as engagement metrics across the family of apps reached all-time highs. Reels "
         "monetization is closing the gap with Stories faster than expected. The company's AI capabilities "
         "are underappreciated by the market. Cost discipline has been maintained following efficiency initiatives."),

        # Detractors
        ("INTC", "Intel Corporation", "detractor", "Q3", 2025,
         "Intel detracted from returns as the foundry roadmap faced additional delays. The 18A process node "
         "timeline slipped by two quarters, raising questions about competitive positioning. While valuation "
         "is undemanding, execution risk remains elevated. We have trimmed the position."),

        ("PYPL", "PayPal Holdings Inc.", "detractor", "Q3", 2025,
         "PayPal weighed on performance amid intensifying competition in digital payments. Take rates continue "
         "to face pressure from both traditional card networks and newer fintech entrants. Management's strategic "
         "pivot shows early signs of progress but will require patience. We are monitoring closely."),

        ("DIS", "The Walt Disney Company", "detractor", "Q2", 2025,
         "Disney detracted as streaming losses persisted longer than anticipated. Parks demand showed signs of "
         "normalization following exceptional post-pandemic strength. Content costs remain elevated given "
         "competitive dynamics. The path to profitability in direct-to-consumer is taking longer than expected."),

        ("BA", "Boeing Company", "detractor", "Q2", 2025,
         "Boeing hurt returns as production challenges in the 737 MAX program continued to constrain deliveries. "
         "Supply chain issues have proven more persistent than management guided. The defense backlog provides "
         "some stability but commercial execution is critical. Position sized appropriately for ongoing uncertainty."),

        ("MRNA", "Moderna Inc.", "detractor", "Q1", 2025,
         "Moderna detracted as COVID vaccine demand continued its secular decline. The pipeline of non-COVID "
         "products remains promising but is several years from meaningful revenue contribution. Cash burn has "
         "increased as R&D investments expand. We await clearer clinical catalysts before adding to the position."),
    ]

    blurbs = []
    for ticker, company, blurb_type, quarter, year, text in synthetic_data:
        blurbs.append(ExemplarBlurb(
            ticker=ticker,
            company_name=company,
            blurb_text=text,
            quarter=quarter,
            year=year,
            blurb_type=blurb_type,
            word_count=len(text.split()),
            source_file="synthetic_exemplars.docx",
        ))

    return blurbs


def main():
    parser = argparse.ArgumentParser(description="Parse exemplar Word documents")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/exemplar_docs"),
        help="Directory containing Word documents to parse",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exemplars/exemplars.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create synthetic exemplars (for testing without real docs)",
    )

    args = parser.parse_args()

    if args.synthetic or not args.input_dir.exists():
        if not args.synthetic:
            logger.warning(
                "Input directory %s not found, creating synthetic exemplars",
                args.input_dir
            )

        blurbs = create_synthetic_exemplars()
        logger.info("Created %d synthetic exemplars", len(blurbs))
    else:
        blurbs = parse_directory(args.input_dir)

    if blurbs:
        save_exemplars_json(blurbs, args.output)
        logger.info("Saved %d blurbs to %s", len(blurbs), args.output)
    else:
        logger.warning("No blurbs found to save")


if __name__ == "__main__":
    main()
