"""Tests for the thesis registry module."""

import pytest
from datetime import date, timedelta
from pathlib import Path

from src.data.thesis_registry import (
    ThesisRegistry,
    create_empty_registry,
    DEFAULT_PLACEHOLDER,
)
from src.models import ThesisEntry, ThesisLookupResult


@pytest.fixture
def sample_csv(tmp_path) -> Path:
    """Create a sample thesis registry CSV for testing."""
    csv_path = tmp_path / "thesis_registry.csv"
    csv_content = """ticker,company_name,thesis_summary,last_updated,analyst
NVDA,NVIDIA Corporation,"AI infrastructure picks-and-shovels play with datacenter dominance. CUDA moat creates high switching costs.",2026-01-15,JSmith
AAPL,Apple Inc.,"Quality compounder with ecosystem lock-in. Services growth provides margin expansion runway. Capital return program supports floor valuation.",2025-12-01,JSmith
MSFT,Microsoft Corporation,"Cloud leader with Azure growth runway. AI integration across product suite. Enterprise relationships provide durability.",2025-11-15,ABrown
AMZN,Amazon.com Inc.,"AWS profit engine funds retail growth. Advertising emerging as high-margin contributor. Logistics moat widening.",2025-10-20,ABrown
META,Meta Platforms Inc.,"Social engagement monopoly with underappreciated AI capabilities. Reels monetization inflecting. Cost discipline restored.",2025-09-10,JSmith
"""
    csv_path.write_text(csv_content)
    return csv_path


@pytest.fixture
def stale_csv(tmp_path) -> Path:
    """Create a CSV with some stale entries."""
    csv_path = tmp_path / "stale_registry.csv"
    old_date = (date.today() - timedelta(days=200)).isoformat()
    recent_date = (date.today() - timedelta(days=30)).isoformat()

    csv_content = f"""ticker,company_name,thesis_summary,last_updated,analyst
OLD1,Old Company One,"Stale thesis.",{old_date},JSmith
OLD2,Old Company Two,"Another stale thesis.",{old_date},JSmith
NEW1,New Company One,"Fresh thesis.",{recent_date},ABrown
"""
    csv_path.write_text(csv_content)
    return csv_path


class TestThesisRegistryLoad:
    """Tests for loading thesis registry from CSV."""

    def test_loads_valid_csv(self, sample_csv):
        """Should load all entries from valid CSV."""
        registry = ThesisRegistry.load(sample_csv)

        assert len(registry) == 5
        assert "NVDA" in registry
        assert "AAPL" in registry
        assert "MSFT" in registry

    def test_raises_on_missing_file(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            ThesisRegistry.load(tmp_path / "nonexistent.csv")

    def test_raises_on_missing_columns(self, tmp_path):
        """Should raise ValueError if required columns missing."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("ticker,company_name\nAAPL,Apple")

        with pytest.raises(ValueError, match="missing required columns"):
            ThesisRegistry.load(bad_csv)

    def test_skips_invalid_rows(self, tmp_path):
        """Should skip rows with invalid data and continue."""
        csv_path = tmp_path / "partial.csv"
        csv_content = """ticker,company_name,thesis_summary,last_updated,analyst
GOOD,Good Company,"Valid thesis.",2026-01-15,JSmith
BAD,Bad Company,"Invalid date.",not-a-date,JSmith
ALSO_GOOD,Also Good,"Another valid one.",2026-01-10,ABrown
"""
        csv_path.write_text(csv_content)

        registry = ThesisRegistry.load(csv_path)

        assert len(registry) == 2
        assert "GOOD" in registry
        assert "ALSO_GOOD" in registry
        assert "BAD" not in registry


class TestThesisRegistryLookup:
    """Tests for thesis lookup operations."""

    def test_lookup_found(self, sample_csv):
        """Should return entry when ticker found."""
        registry = ThesisRegistry.load(sample_csv)
        result = registry.lookup("NVDA")

        assert result.found is True
        assert result.entry is not None
        assert result.entry.ticker == "NVDA"
        assert result.entry.company_name == "NVIDIA Corporation"
        assert "AI infrastructure" in result.entry.thesis_summary
        assert result.thesis_text == result.entry.thesis_summary

    def test_lookup_not_found(self, sample_csv):
        """Should return placeholder when ticker not found."""
        registry = ThesisRegistry.load(sample_csv)
        result = registry.lookup("UNKNOWN")

        assert result.found is False
        assert result.entry is None
        assert "UNKNOWN" in result.placeholder_text
        assert result.thesis_text == result.placeholder_text

    def test_lookup_case_insensitive(self, sample_csv):
        """Should find ticker regardless of case."""
        registry = ThesisRegistry.load(sample_csv)

        assert registry.lookup("nvda").found is True
        assert registry.lookup("Nvda").found is True
        assert registry.lookup("NVDA").found is True

    def test_lookup_strips_whitespace(self, sample_csv):
        """Should handle whitespace in ticker."""
        registry = ThesisRegistry.load(sample_csv)

        assert registry.lookup("  NVDA  ").found is True

    def test_lookup_many(self, sample_csv):
        """Should return results for multiple tickers."""
        registry = ThesisRegistry.load(sample_csv)
        results = registry.lookup_many(["NVDA", "AAPL", "UNKNOWN"])

        assert len(results) == 3
        assert results["NVDA"].found is True
        assert results["AAPL"].found is True
        assert results["UNKNOWN"].found is False

    def test_company_name_fallback(self, sample_csv):
        """Should return ticker as company name when not found."""
        registry = ThesisRegistry.load(sample_csv)

        found = registry.lookup("NVDA")
        not_found = registry.lookup("UNKNOWN")

        assert found.company_name == "NVIDIA Corporation"
        assert not_found.company_name == "UNKNOWN"


class TestThesisRegistryCRUD:
    """Tests for add/remove/save operations."""

    def test_add_new_entry(self, sample_csv):
        """Should add new entry to registry."""
        registry = ThesisRegistry.load(sample_csv)

        new_entry = ThesisEntry(
            ticker="GOOGL",
            company_name="Alphabet Inc.",
            thesis_summary="Search monopoly with AI optionality.",
            last_updated=date.today(),
            analyst="JSmith",
        )

        registry.add(new_entry)

        assert "GOOGL" in registry
        assert len(registry) == 6
        assert registry.lookup("GOOGL").entry.thesis_summary == "Search monopoly with AI optionality."

    def test_add_updates_existing(self, sample_csv):
        """Should update existing entry when adding same ticker."""
        registry = ThesisRegistry.load(sample_csv)
        original_thesis = registry.lookup("NVDA").entry.thesis_summary

        updated_entry = ThesisEntry(
            ticker="NVDA",
            company_name="NVIDIA Corporation",
            thesis_summary="Updated thesis text.",
            last_updated=date.today(),
            analyst="ABrown",
        )

        registry.add(updated_entry)

        assert len(registry) == 5  # Same count
        assert registry.lookup("NVDA").entry.thesis_summary == "Updated thesis text."
        assert registry.lookup("NVDA").entry.analyst == "ABrown"

    def test_remove_existing(self, sample_csv):
        """Should remove existing entry."""
        registry = ThesisRegistry.load(sample_csv)

        result = registry.remove("NVDA")

        assert result is True
        assert "NVDA" not in registry
        assert len(registry) == 4

    def test_remove_nonexistent(self, sample_csv):
        """Should return False when removing nonexistent ticker."""
        registry = ThesisRegistry.load(sample_csv)

        result = registry.remove("UNKNOWN")

        assert result is False
        assert len(registry) == 5

    def test_save_and_reload(self, sample_csv, tmp_path):
        """Should save registry and reload correctly."""
        registry = ThesisRegistry.load(sample_csv)

        # Add new entry
        registry.add(ThesisEntry(
            ticker="NEW",
            company_name="New Company",
            thesis_summary="New thesis.",
            last_updated=date.today(),
            analyst="Test",
        ))

        # Save to new location
        new_path = tmp_path / "saved_registry.csv"
        registry.save(new_path)

        # Reload and verify
        reloaded = ThesisRegistry.load(new_path)

        assert len(reloaded) == 6
        assert "NEW" in reloaded
        assert reloaded.lookup("NEW").entry.thesis_summary == "New thesis."


class TestThesisRegistryUtilities:
    """Tests for utility methods."""

    def test_get_stale_entries(self, stale_csv):
        """Should identify entries older than threshold."""
        registry = ThesisRegistry.load(stale_csv)

        stale = registry.get_stale_entries(days=180)

        assert len(stale) == 2
        tickers = {e.ticker for e in stale}
        assert tickers == {"OLD1", "OLD2"}

    def test_get_missing_tickers(self, sample_csv):
        """Should identify tickers not in registry."""
        registry = ThesisRegistry.load(sample_csv)

        missing = registry.get_missing_tickers(["NVDA", "AAPL", "UNKNOWN1", "UNKNOWN2"])

        assert set(missing) == {"UNKNOWN1", "UNKNOWN2"}

    def test_tickers_property(self, sample_csv):
        """Should return sorted list of tickers."""
        registry = ThesisRegistry.load(sample_csv)

        tickers = registry.tickers

        assert tickers == ["AAPL", "AMZN", "META", "MSFT", "NVDA"]

    def test_iteration(self, sample_csv):
        """Should support iteration over entries."""
        registry = ThesisRegistry.load(sample_csv)

        entries = list(registry)

        assert len(entries) == 5
        assert all(isinstance(e, ThesisEntry) for e in entries)


class TestThesisEntry:
    """Tests for ThesisEntry model."""

    def test_is_stale(self):
        """Should correctly identify stale entries."""
        old_entry = ThesisEntry(
            ticker="OLD",
            company_name="Old Co",
            thesis_summary="Stale",
            last_updated=date.today() - timedelta(days=200),
            analyst="Test",
        )

        new_entry = ThesisEntry(
            ticker="NEW",
            company_name="New Co",
            thesis_summary="Fresh",
            last_updated=date.today() - timedelta(days=30),
            analyst="Test",
        )

        assert old_entry.is_stale(days=180) is True
        assert new_entry.is_stale(days=180) is False

    def test_age_days(self):
        """Should return correct age in days."""
        entry = ThesisEntry(
            ticker="TEST",
            company_name="Test Co",
            thesis_summary="Test",
            last_updated=date.today() - timedelta(days=45),
            analyst="Test",
        )

        assert entry.age_days() == 45


class TestCreateEmptyRegistry:
    """Tests for creating new empty registry."""

    def test_creates_file_with_headers(self, tmp_path):
        """Should create CSV with headers only."""
        path = tmp_path / "new_registry.csv"

        registry = create_empty_registry(path)

        assert path.exists()
        assert len(registry) == 0

        # Verify headers
        content = path.read_text()
        assert "ticker,company_name,thesis_summary,last_updated,analyst" in content

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories if needed."""
        path = tmp_path / "nested" / "dir" / "registry.csv"

        create_empty_registry(path)

        assert path.exists()
