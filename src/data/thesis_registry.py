"""Thesis registry for investment thesis storage and lookup."""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterator

from src.models import ThesisEntry, ThesisLookupResult

logger = logging.getLogger(__name__)

# Default placeholder when thesis is not found
DEFAULT_PLACEHOLDER = (
    "[No thesis on file for {ticker}. "
    "Please add thesis context for better commentary quality.]"
)


class ThesisRegistry:
    """
    CSV-backed registry of investment theses per ticker.

    The registry loads theses from a CSV file and provides fast lookup by ticker.
    Missing tickers return a structured placeholder rather than raising errors,
    allowing commentary generation to proceed with degraded quality.

    CSV format:
        ticker,company_name,thesis_summary,last_updated,analyst
        NVDA,NVIDIA Corporation,"AI infrastructure play...",2026-01-15,JSmith

    Usage:
        registry = ThesisRegistry.load("data/thesis_registry.csv")
        result = registry.lookup("NVDA")
        if result.found:
            print(result.entry.thesis_summary)
        else:
            print(result.placeholder_text)
    """

    def __init__(self, entries: dict[str, ThesisEntry] | None = None):
        """
        Initialize registry with optional pre-loaded entries.

        Args:
            entries: Dict mapping ticker -> ThesisEntry. If None, starts empty.
        """
        self._entries: dict[str, ThesisEntry] = entries or {}
        self._source_path: Path | None = None

    @classmethod
    def load(cls, path: Path | str) -> ThesisRegistry:
        """
        Load thesis registry from CSV file.

        Args:
            path: Path to CSV file

        Returns:
            ThesisRegistry populated with entries from file

        Raises:
            FileNotFoundError: If CSV file does not exist
            ValueError: If CSV format is invalid
        """
        path = Path(path)

        if not path.exists():
            logger.warning("Thesis registry file not found: %s", path)
            raise FileNotFoundError(f"Thesis registry not found: {path}")

        entries: dict[str, ThesisEntry] = {}

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            # Validate required columns
            required = {"ticker", "company_name", "thesis_summary", "last_updated", "analyst"}
            if reader.fieldnames is None:
                raise ValueError("CSV file is empty or has no headers")

            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}")

            for row_num, row in enumerate(reader, start=2):
                try:
                    entry = _parse_row(row)
                    entries[entry.ticker] = entry
                except Exception as e:
                    logger.warning("Skipping invalid row %d: %s", row_num, e)
                    continue

        registry = cls(entries)
        registry._source_path = path

        logger.info("Loaded thesis registry: %d entries from %s", len(entries), path)
        return registry

    def lookup(self, ticker: str) -> ThesisLookupResult:
        """
        Look up thesis for a ticker.

        Args:
            ticker: Stock ticker symbol (case-insensitive)

        Returns:
            ThesisLookupResult with entry if found, placeholder if not
        """
        ticker = ticker.upper().strip()
        entry = self._entries.get(ticker)

        if entry:
            logger.debug("Thesis found for %s", ticker)
            return ThesisLookupResult(
                ticker=ticker,
                found=True,
                entry=entry,
                placeholder_text="",
            )

        logger.debug("No thesis found for %s", ticker)
        return ThesisLookupResult(
            ticker=ticker,
            found=False,
            entry=None,
            placeholder_text=DEFAULT_PLACEHOLDER.format(ticker=ticker),
        )

    def lookup_many(self, tickers: list[str]) -> dict[str, ThesisLookupResult]:
        """
        Look up theses for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dict mapping ticker -> ThesisLookupResult
        """
        return {ticker: self.lookup(ticker) for ticker in tickers}

    def add(self, entry: ThesisEntry) -> None:
        """
        Add or update a thesis entry.

        Args:
            entry: ThesisEntry to add/update
        """
        ticker = entry.ticker.upper().strip()
        self._entries[ticker] = ThesisEntry(
            ticker=ticker,
            company_name=entry.company_name,
            thesis_summary=entry.thesis_summary,
            last_updated=entry.last_updated,
            analyst=entry.analyst,
        )
        logger.info("Added/updated thesis for %s", ticker)

    def remove(self, ticker: str) -> bool:
        """
        Remove a thesis entry.

        Args:
            ticker: Ticker to remove

        Returns:
            True if entry was removed, False if not found
        """
        ticker = ticker.upper().strip()
        if ticker in self._entries:
            del self._entries[ticker]
            logger.info("Removed thesis for %s", ticker)
            return True
        return False

    def save(self, path: Path | str | None = None) -> None:
        """
        Save registry to CSV file.

        Args:
            path: Output path. If None, uses original source path.

        Raises:
            ValueError: If no path specified and registry wasn't loaded from file
        """
        if path is None:
            path = self._source_path
        if path is None:
            raise ValueError("No path specified and registry has no source path")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["ticker", "company_name", "thesis_summary", "last_updated", "analyst"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

            for entry in sorted(self._entries.values(), key=lambda e: e.ticker):
                writer.writerow({
                    "ticker": entry.ticker,
                    "company_name": entry.company_name,
                    "thesis_summary": entry.thesis_summary,
                    "last_updated": entry.last_updated.isoformat(),
                    "analyst": entry.analyst,
                })

        logger.info("Saved thesis registry: %d entries to %s", len(self._entries), path)

    def get_stale_entries(self, days: int = 180) -> list[ThesisEntry]:
        """
        Get entries that haven't been updated in given number of days.

        Args:
            days: Threshold for staleness

        Returns:
            List of stale ThesisEntry objects
        """
        return [e for e in self._entries.values() if e.is_stale(days)]

    def get_missing_tickers(self, tickers: list[str]) -> list[str]:
        """
        Return tickers from input list that are not in registry.

        Args:
            tickers: List of tickers to check

        Returns:
            List of tickers not found in registry
        """
        return [t for t in tickers if t.upper().strip() not in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, ticker: str) -> bool:
        return ticker.upper().strip() in self._entries

    def __iter__(self) -> Iterator[ThesisEntry]:
        return iter(self._entries.values())

    @property
    def tickers(self) -> list[str]:
        """Return sorted list of all tickers in registry."""
        return sorted(self._entries.keys())


def _parse_row(row: dict[str, str]) -> ThesisEntry:
    """Parse a CSV row into a ThesisEntry."""
    ticker = row["ticker"].upper().strip()
    if not ticker:
        raise ValueError("Empty ticker")

    company_name = row["company_name"].strip()
    thesis_summary = row["thesis_summary"].strip()
    analyst = row["analyst"].strip()

    # Parse date - accept ISO format (YYYY-MM-DD)
    date_str = row["last_updated"].strip()
    try:
        last_updated = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD")

    return ThesisEntry(
        ticker=ticker,
        company_name=company_name,
        thesis_summary=thesis_summary,
        last_updated=last_updated,
        analyst=analyst,
    )


def create_empty_registry(path: Path | str) -> ThesisRegistry:
    """
    Create a new empty registry file with headers.

    Args:
        path: Path for new CSV file

    Returns:
        Empty ThesisRegistry with source path set
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["ticker", "company_name", "thesis_summary", "last_updated", "analyst"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    registry = ThesisRegistry()
    registry._source_path = path

    logger.info("Created empty thesis registry at %s", path)
    return registry
