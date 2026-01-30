"""Exemplar selection logic for few-shot prompting."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from src.models import ExemplarBlurb, ExemplarSelection
from src.parsers.exemplar_parser import load_exemplars_json

logger = logging.getLogger(__name__)


class ExemplarSelector:
    """
    Selects relevant exemplar blurbs for few-shot prompting.

    Selection strategy:
    1. If we have prior blurbs for the same ticker, include the most recent one
    2. Add 2-3 additional blurbs of the same type (contributor/detractor)
    3. Prefer variety in tickers for the additional blurbs

    Usage:
        selector = ExemplarSelector.load("data/exemplars/exemplars.json")
        selection = selector.select("NVDA", is_contributor=True, count=3)
        prompt_text = selection.format_for_prompt()
    """

    def __init__(self, blurbs: list[ExemplarBlurb]):
        """
        Initialize selector with loaded blurbs.

        Args:
            blurbs: List of all available exemplar blurbs
        """
        self._blurbs = blurbs
        self._by_ticker: dict[str, list[ExemplarBlurb]] = {}
        self._contributors: list[ExemplarBlurb] = []
        self._detractors: list[ExemplarBlurb] = []

        # Index blurbs
        for blurb in blurbs:
            # By ticker
            if blurb.ticker not in self._by_ticker:
                self._by_ticker[blurb.ticker] = []
            self._by_ticker[blurb.ticker].append(blurb)

            # By type
            if blurb.blurb_type == "contributor":
                self._contributors.append(blurb)
            elif blurb.blurb_type == "detractor":
                self._detractors.append(blurb)

        # Sort each ticker's blurbs by recency
        for ticker in self._by_ticker:
            self._by_ticker[ticker].sort(
                key=lambda b: (b.year, b.quarter),
                reverse=True
            )

        logger.info(
            "ExemplarSelector initialized: %d blurbs, %d tickers, %d contributors, %d detractors",
            len(blurbs), len(self._by_ticker), len(self._contributors), len(self._detractors)
        )

    @classmethod
    def load(cls, json_path: Path | str) -> ExemplarSelector:
        """
        Load selector from JSON file.

        Args:
            json_path: Path to exemplars JSON file

        Returns:
            ExemplarSelector ready for use
        """
        blurbs = load_exemplars_json(json_path)
        return cls(blurbs)

    def select(
        self,
        ticker: str,
        is_contributor: bool,
        count: int = 3,
        exclude_tickers: list[str] | None = None,
    ) -> ExemplarSelection:
        """
        Select exemplars for a given ticker and type.

        Args:
            ticker: Target ticker symbol
            is_contributor: True for contributor, False for detractor
            count: Total number of exemplars to include (including same-ticker)
            exclude_tickers: Tickers to exclude from similar exemplars

        Returns:
            ExemplarSelection with selected blurbs
        """
        ticker = ticker.upper()
        exclude_tickers = set(t.upper() for t in (exclude_tickers or []))

        # Get same-ticker exemplar (most recent of matching type)
        same_ticker_exemplar = self._get_same_ticker_exemplar(ticker, is_contributor)

        # Determine how many additional exemplars we need
        additional_count = count
        if same_ticker_exemplar:
            additional_count -= 1

        # Get similar exemplars (same type, different tickers)
        similar = self._get_similar_exemplars(
            is_contributor=is_contributor,
            exclude_tickers=exclude_tickers | {ticker},
            count=additional_count,
        )

        return ExemplarSelection(
            target_ticker=ticker,
            target_is_contributor=is_contributor,
            same_ticker_exemplar=same_ticker_exemplar,
            similar_exemplars=similar,
        )

    def _get_same_ticker_exemplar(
        self,
        ticker: str,
        is_contributor: bool,
    ) -> ExemplarBlurb | None:
        """Get most recent blurb for same ticker matching type."""
        if ticker not in self._by_ticker:
            return None

        for blurb in self._by_ticker[ticker]:
            if blurb.matches_type(is_contributor):
                return blurb

        return None

    def _get_similar_exemplars(
        self,
        is_contributor: bool,
        exclude_tickers: set[str],
        count: int,
    ) -> list[ExemplarBlurb]:
        """
        Get exemplars of the same type from different tickers.

        Prefers variety - one blurb per ticker.
        """
        pool = self._contributors if is_contributor else self._detractors

        # Filter out excluded tickers
        candidates = [b for b in pool if b.ticker not in exclude_tickers]

        if not candidates:
            return []

        # Group by ticker to ensure variety
        by_ticker: dict[str, ExemplarBlurb] = {}
        for blurb in candidates:
            if blurb.ticker not in by_ticker:
                by_ticker[blurb.ticker] = blurb

        # Select randomly from unique tickers
        unique_blurbs = list(by_ticker.values())

        if len(unique_blurbs) <= count:
            return unique_blurbs

        # Random sample for variety
        return random.sample(unique_blurbs, count)

    def get_ticker_history(self, ticker: str) -> list[ExemplarBlurb]:
        """Get all historical blurbs for a ticker, most recent first."""
        return self._by_ticker.get(ticker.upper(), [])

    def has_exemplars_for(self, ticker: str) -> bool:
        """Check if we have any exemplars for this ticker."""
        return ticker.upper() in self._by_ticker

    @property
    def available_tickers(self) -> list[str]:
        """Return sorted list of tickers with exemplars."""
        return sorted(self._by_ticker.keys())

    @property
    def total_blurbs(self) -> int:
        """Total number of blurbs available."""
        return len(self._blurbs)


def create_empty_exemplars_json(output_path: Path | str) -> None:
    """
    Create an empty exemplars JSON file with proper structure.

    Args:
        output_path: Path for new JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "total_blurbs": 0,
            "unique_tickers": 0,
            "contributors": 0,
            "detractors": 0,
        },
        "blurbs_by_ticker": {},
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Created empty exemplars file at %s", output_path)
