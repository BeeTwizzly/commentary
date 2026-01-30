"""Tests for error handling utilities."""

import pytest

from src.utils.error_handler import (
    AppError,
    ErrorSeverity,
    get_user_message,
    safe_execute,
)


class TestAppError:
    """Tests for AppError dataclass."""

    def test_basic_error(self):
        error = AppError(message="Test error")
        assert error.message == "Test error"
        assert error.severity == ErrorSeverity.ERROR
        assert error.recoverable is True

    def test_display_message_fallback(self):
        error = AppError(message="Internal error")
        assert error.display_message == "Internal error"

    def test_display_message_custom(self):
        error = AppError(
            message="Internal error",
            user_message="Something went wrong",
        )
        assert error.display_message == "Something went wrong"

    def test_error_with_severity(self):
        error = AppError(message="Warning", severity=ErrorSeverity.WARNING)
        assert error.severity == ErrorSeverity.WARNING

    def test_error_with_details(self):
        error = AppError(message="Error", details="Stack trace here")
        assert error.details == "Stack trace here"


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_severity_values(self):
        assert ErrorSeverity.INFO.value == "info"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestGetUserMessage:
    """Tests for user message lookup."""

    def test_known_key(self):
        msg = get_user_message("api_key_missing")
        assert "API key" in msg

    def test_unknown_key(self):
        msg = get_user_message("nonexistent_key")
        assert "unexpected error" in msg.lower()

    def test_rate_limit_key(self):
        msg = get_user_message("api_rate_limit")
        assert "rate limit" in msg.lower()

    def test_timeout_key(self):
        msg = get_user_message("api_timeout")
        assert "timed out" in msg.lower()


class TestSafeExecute:
    """Tests for safe_execute wrapper."""

    def test_successful_execution(self):
        def add(a, b):
            return a + b

        result = safe_execute(add, 1, 2, show_ui=False)
        assert result == 3

    def test_failed_execution_returns_default(self):
        def fail():
            raise ValueError("oops")

        result = safe_execute(fail, default="fallback", show_ui=False)
        assert result == "fallback"

    def test_failed_execution_default_none(self):
        def fail():
            raise RuntimeError("error")

        result = safe_execute(fail, show_ui=False)
        assert result is None

    def test_with_kwargs(self):
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = safe_execute(greet, "World", greeting="Hi", show_ui=False)
        assert result == "Hi, World!"

    def test_with_context(self):
        def fail():
            raise ValueError("test")

        result = safe_execute(fail, context="Testing", show_ui=False, default="ok")
        assert result == "ok"
