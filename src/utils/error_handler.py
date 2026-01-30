"""Centralized error handling utilities."""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

import streamlit as st

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorSeverity(Enum):
    """Error severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AppError:
    """Structured application error."""

    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    user_message: str | None = None
    details: str | None = None
    recoverable: bool = True

    @property
    def display_message(self) -> str:
        """Get message for user display."""
        return self.user_message or self.message


# User-friendly error messages
ERROR_MESSAGES = {
    "api_key_missing": (
        "OpenAI API key not configured. Please set OPENAI_API_KEY in your "
        "environment variables or .streamlit/secrets.toml file."
    ),
    "api_key_invalid": (
        "Invalid API key. Please check that your OpenAI API key is correct "
        "and has not expired."
    ),
    "api_rate_limit": (
        "Rate limit exceeded. Please wait a moment and try again. "
        "If this persists, check your OpenAI usage limits."
    ),
    "api_timeout": (
        "Request timed out. The AI service is taking longer than expected. "
        "Please try again in a few moments."
    ),
    "api_connection": (
        "Could not connect to the AI service. Please check your internet "
        "connection and try again."
    ),
    "file_parse_error": (
        "Could not read the uploaded file. Please ensure it's a valid Excel "
        "file (.xlsx) and is not corrupted or password-protected."
    ),
    "no_holdings_found": (
        "No holdings found in the uploaded file. Please ensure your Excel "
        "file contains the required columns (Ticker, Total Effect)."
    ),
    "generation_failed": (
        "Failed to generate commentary. Please try again. If the problem "
        "persists, try with fewer holdings or check your API quota."
    ),
    "export_failed": (
        "Export failed. Please try again or try a different export format."
    ),
    "session_expired": (
        "Your session has expired. Please refresh the page and start over."
    ),
    "unknown_error": (
        "An unexpected error occurred. Please refresh the page and try again. "
        "If the problem persists, contact support."
    ),
}


def get_user_message(error_key: str) -> str:
    """Get user-friendly error message by key."""
    return ERROR_MESSAGES.get(error_key, ERROR_MESSAGES["unknown_error"])


def handle_error(
    error: Exception,
    context: str = "",
    show_ui: bool = True,
    reraise: bool = False,
) -> AppError:
    """
    Handle an exception with logging and optional UI display.

    Args:
        error: The exception to handle
        context: Context string for logging
        show_ui: Whether to show error in Streamlit UI
        reraise: Whether to re-raise the exception

    Returns:
        AppError with structured error info
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Determine severity and user message based on error type
    severity = ErrorSeverity.ERROR
    user_message = ERROR_MESSAGES["unknown_error"]
    recoverable = True

    # Map specific errors to user messages
    if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
        user_message = ERROR_MESSAGES["api_key_invalid"]
    elif "rate limit" in error_msg.lower() or "429" in error_msg:
        user_message = ERROR_MESSAGES["api_rate_limit"]
        severity = ErrorSeverity.WARNING
    elif "timeout" in error_msg.lower():
        user_message = ERROR_MESSAGES["api_timeout"]
    elif "connection" in error_msg.lower():
        user_message = ERROR_MESSAGES["api_connection"]
    elif isinstance(error, FileNotFoundError):
        user_message = "File not found. Please upload the file again."
    elif isinstance(error, PermissionError):
        user_message = "Permission denied. Please check file permissions."
    elif isinstance(error, ValueError):
        user_message = f"Invalid value: {error_msg}"
        severity = ErrorSeverity.WARNING

    # Log the error
    log_msg = f"{context}: {error_type}: {error_msg}" if context else f"{error_type}: {error_msg}"

    if severity == ErrorSeverity.CRITICAL:
        logger.critical(log_msg, exc_info=True)
    elif severity == ErrorSeverity.ERROR:
        logger.error(log_msg, exc_info=True)
    elif severity == ErrorSeverity.WARNING:
        logger.warning(log_msg)
    else:
        logger.info(log_msg)

    # Create structured error
    app_error = AppError(
        message=error_msg,
        severity=severity,
        user_message=user_message,
        details=traceback.format_exc(),
        recoverable=recoverable,
    )

    # Show in UI
    if show_ui:
        display_error(app_error)

    # Re-raise if requested
    if reraise:
        raise error

    return app_error


def display_error(error: AppError) -> None:
    """Display an error in the Streamlit UI."""
    if error.severity == ErrorSeverity.WARNING:
        st.warning(f"{error.display_message}")
    elif error.severity == ErrorSeverity.INFO:
        st.info(f"{error.display_message}")
    else:
        st.error(f"{error.display_message}")


def safe_execute(
    func: Callable[..., T],
    *args,
    default: T | None = None,
    context: str = "",
    show_ui: bool = True,
    **kwargs,
) -> T | None:
    """
    Execute a function with error handling.

    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value on error
        context: Context for error logging
        show_ui: Whether to show errors in UI
        **kwargs: Keyword arguments

    Returns:
        Function result or default on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_error(e, context=context, show_ui=show_ui)
        return default
