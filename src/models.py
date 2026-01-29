"""Data models for the Portfolio Commentary Generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class HoldingData:
    """
    Represents a single holding extracted from the performance Excel.

    Attributes:
        ticker: Stock ticker symbol (e.g., "NVDA", "BRK.B")
        company_name: Full company name
        strategy: Strategy name (from Excel tab)
        avg_weight: Average portfolio weight during period (%)
        begin_weight: Weight at period start (%)
        end_weight: Weight at period end (%)
        benchmark_weight: Weight in benchmark (%)
        benchmark_return: Benchmark return for this security (%)
        total_attribution: Total Brinson attribution (bps)
        selection_effect: Brinson selection effect (bps)
        allocation_effect: Brinson allocation effect (bps)
        rank: Rank within strategy (1 = highest attribution)
        is_contributor: True if top 5, False if bottom 5
    """
    ticker: str
    company_name: str
    strategy: str
    avg_weight: float
    begin_weight: float
    end_weight: float
    benchmark_weight: float
    benchmark_return: float
    total_attribution: float
    selection_effect: float
    allocation_effect: float
    rank: int
    is_contributor: bool

    def attribution_description(self) -> str:
        """Human-readable attribution summary."""
        direction = "added" if self.total_attribution > 0 else "detracted"
        return f"{direction} {abs(self.total_attribution):.0f} bps"


@dataclass
class StrategyHoldings:
    """
    Container for parsed holdings from a single strategy.

    Attributes:
        strategy_name: Name of the strategy (Excel tab name)
        contributors: Top 5 holdings by attribution (descending)
        detractors: Bottom 5 holdings by attribution (ascending)
        all_holdings: Complete list of valid holdings parsed
        parse_warnings: Any warnings generated during parsing
    """
    strategy_name: str
    contributors: list[HoldingData] = field(default_factory=list)
    detractors: list[HoldingData] = field(default_factory=list)
    all_holdings: list[HoldingData] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)


@dataclass
class ParsedWorkbook:
    """
    Container for all parsed data from an Excel workbook.

    Attributes:
        strategies: List of parsed strategy holdings
        total_holdings: Total holdings across all strategies
        total_for_commentary: Holdings selected for commentary (top/bottom 5 per strategy)
        errors: Any errors encountered during parsing
    """
    strategies: list[StrategyHoldings] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_holdings(self) -> int:
        return sum(len(s.all_holdings) for s in self.strategies)

    @property
    def total_for_commentary(self) -> int:
        return sum(len(s.contributors) + len(s.detractors) for s in self.strategies)

    def get_all_commentary_holdings(self) -> list[HoldingData]:
        """Flatten all contributors and detractors across strategies."""
        holdings = []
        for strategy in self.strategies:
            holdings.extend(strategy.contributors)
            holdings.extend(strategy.detractors)
        return holdings


@dataclass(frozen=True)
class ThesisEntry:
    """
    Investment thesis for a single holding.

    Attributes:
        ticker: Stock ticker symbol
        company_name: Full company name
        thesis_summary: 2-4 sentence investment thesis
        last_updated: Date thesis was last reviewed/updated
        analyst: Analyst who owns this thesis
    """
    ticker: str
    company_name: str
    thesis_summary: str
    last_updated: date
    analyst: str

    def is_stale(self, days: int = 180) -> bool:
        """Check if thesis hasn't been updated in given number of days."""
        age = (date.today() - self.last_updated).days
        return age > days

    def age_days(self) -> int:
        """Return age of thesis in days."""
        return (date.today() - self.last_updated).days


@dataclass(frozen=True)
class ThesisLookupResult:
    """
    Result of a thesis lookup operation.

    Attributes:
        ticker: Requested ticker
        found: Whether thesis was found in registry
        entry: ThesisEntry if found, None otherwise
        placeholder_text: Fallback text if not found
    """
    ticker: str
    found: bool
    entry: Optional[ThesisEntry]
    placeholder_text: str

    @property
    def thesis_text(self) -> str:
        """Return thesis summary or placeholder."""
        if self.entry:
            return self.entry.thesis_summary
        return self.placeholder_text

    @property
    def company_name(self) -> str:
        """Return company name or ticker as fallback."""
        if self.entry:
            return self.entry.company_name
        return self.ticker
