"""Export module for generating Word documents from approved commentary."""

from .word_exporter import (
    WordExporter,
    ExportConfig,
    ExportResult,
    export_to_word,
)

__all__ = [
    "WordExporter",
    "ExportConfig",
    "ExportResult",
    "export_to_word",
]
