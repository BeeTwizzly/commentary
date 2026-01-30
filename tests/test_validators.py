"""Tests for validation utilities."""

import pytest

from src.utils.validators import (
    ValidationResult,
    validate_ticker,
    validate_strategy_name,
    validate_quarter,
    validate_api_key,
)


class TestValidateTicker:
    """Tests for ticker validation."""

    def test_valid_ticker(self):
        result = validate_ticker("AAPL")
        assert result.valid is True

    def test_valid_ticker_lowercase(self):
        result = validate_ticker("aapl")
        assert result.valid is True

    def test_valid_ticker_with_dot(self):
        result = validate_ticker("BRK.A")
        assert result.valid is True

    def test_empty_ticker(self):
        result = validate_ticker("")
        assert result.valid is False
        assert "empty" in result.error_message.lower()

    def test_invalid_ticker_numbers(self):
        result = validate_ticker("123")
        assert result.valid is False

    def test_invalid_ticker_too_long(self):
        result = validate_ticker("ABCDEFG")
        assert result.valid is False


class TestValidateStrategyName:
    """Tests for strategy name validation."""

    def test_valid_name(self):
        result = validate_strategy_name("Growth Equity")
        assert result.valid is True

    def test_empty_name(self):
        result = validate_strategy_name("")
        assert result.valid is False

    def test_too_short(self):
        result = validate_strategy_name("A")
        assert result.valid is False

    def test_too_long(self):
        result = validate_strategy_name("A" * 150)
        assert result.valid is False


class TestValidateQuarter:
    """Tests for quarter validation."""

    def test_valid_quarter(self):
        result = validate_quarter("Q4 2025")
        assert result.valid is True

    def test_valid_quarter_no_space(self):
        result = validate_quarter("Q12025")
        assert result.valid is True

    def test_empty_quarter(self):
        result = validate_quarter("")
        assert result.valid is False

    def test_invalid_quarter_number(self):
        result = validate_quarter("Q5 2025")
        assert result.valid is False

    def test_invalid_quarter_format(self):
        result = validate_quarter("2025 Q4")
        assert result.valid is False


class TestValidateApiKey:
    """Tests for API key validation."""

    def test_valid_key(self):
        result = validate_api_key("sk-" + "a" * 48)
        assert result.valid is True

    def test_empty_key(self):
        result = validate_api_key("")
        assert result.valid is False

    def test_invalid_prefix(self):
        result = validate_api_key("pk-" + "a" * 48)
        assert result.valid is False

    def test_too_short(self):
        result = validate_api_key("sk-abc")
        assert result.valid is False


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.error_message == ""
        assert result.warnings == []

    def test_invalid_result(self):
        result = ValidationResult(valid=False, error_message="Test error")
        assert result.valid is False
        assert result.error_message == "Test error"

    def test_result_with_warnings(self):
        result = ValidationResult(valid=True, warnings=["Warning 1"])
        assert result.valid is True
        assert len(result.warnings) == 1
