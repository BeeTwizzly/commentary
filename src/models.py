"""Data models for the Portfolio Commentary Generator."""

from __future__ import annotations

from dataclasses import dataclass, field


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
