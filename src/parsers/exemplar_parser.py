"""Parser for extracting exemplar blurbs from historical Word documents."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from src.models import ExemplarBlurb

logger = logging.getLogger(__name__)

# Regex pattern to match "Company Name (TICKER):" format
# Handles various company name formats including those with periods, ampersands, commas
BLURB_PATTERN = re.compile(
    r"^([A-Za-z][A-Za-z0-9\s\.\,\&\'\-]+?)\s*\(([A-Z]{1,5}(?:\.[A-Z])?)\):\s*(.+)$",
    re.MULTILINE | re.DOTALL
)

# Section headers that indicate contributor vs detractor
CONTRIBUTOR_HEADERS = [
    "contributors",
    "top contributors",
    "positive contributors",
    "what worked",
]

DETRACTOR_HEADERS = [
    "detractors",
    "bottom detractors",
    "negative contributors",
    "what didn't work",
    "what hurt",
]


def parse_word_document(file_path: Path | str) -> list[ExemplarBlurb]:
    """
    Parse a Word document and extract all stock commentary blurbs.

    Args:
        file_path: Path to .docx file

    Returns:
        List of ExemplarBlurb objects extracted from document

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid Word document
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    try:
        doc = Document(file_path)
    except PackageNotFoundError:
        raise ValueError(f"Invalid Word document: {file_path}")

    # Extract quarter/year from filename if possible
    quarter, year = _parse_filename_metadata(file_path.name)

    # Extract full text with paragraph breaks preserved
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    blurbs = []
    current_section = "unknown"

    for i, para in enumerate(paragraphs):
        # Check for section headers
        section = _detect_section_header(para)
        if section:
            current_section = section
            continue

        # Try to extract blurb from paragraph
        blurb = _extract_blurb_from_paragraph(
            para,
            section_type=current_section,
            quarter=quarter,
            year=year,
            source_file=file_path.name,
        )

        if blurb:
            blurbs.append(blurb)

    # Also try combining consecutive paragraphs (some blurbs span multiple)
    combined_blurbs = _extract_multiline_blurbs(
        paragraphs,
        quarter=quarter,
        year=year,
        source_file=file_path.name,
    )

    # Merge, preferring longer blurbs for same ticker
    all_blurbs = _deduplicate_blurbs(blurbs + combined_blurbs)

    logger.info(
        "Parsed %s: found %d blurbs (%d contributors, %d detractors)",
        file_path.name,
        len(all_blurbs),
        sum(1 for b in all_blurbs if b.blurb_type == "contributor"),
        sum(1 for b in all_blurbs if b.blurb_type == "detractor"),
    )

    return all_blurbs


def parse_directory(dir_path: Path | str) -> list[ExemplarBlurb]:
    """
    Parse all Word documents in a directory.

    Args:
        dir_path: Path to directory containing .docx files

    Returns:
        Combined list of all blurbs from all documents
    """
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    all_blurbs = []

    for docx_file in sorted(dir_path.glob("*.docx")):
        # Skip temp files
        if docx_file.name.startswith("~"):
            continue

        try:
            blurbs = parse_word_document(docx_file)
            all_blurbs.extend(blurbs)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", docx_file.name, e)
            continue

    logger.info("Parsed directory %s: %d total blurbs from %d files",
                dir_path, len(all_blurbs), len(list(dir_path.glob("*.docx"))))

    return all_blurbs


def save_exemplars_json(blurbs: list[ExemplarBlurb], output_path: Path | str) -> None:
    """
    Save parsed blurbs to JSON file.

    Args:
        blurbs: List of ExemplarBlurb objects
        output_path: Output JSON file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group by ticker for easier lookup
    by_ticker: dict[str, list[dict]] = {}

    for blurb in blurbs:
        if blurb.ticker not in by_ticker:
            by_ticker[blurb.ticker] = []

        by_ticker[blurb.ticker].append({
            "ticker": blurb.ticker,
            "company_name": blurb.company_name,
            "blurb_text": blurb.blurb_text,
            "quarter": blurb.quarter,
            "year": blurb.year,
            "blurb_type": blurb.blurb_type,
            "word_count": blurb.word_count,
            "source_file": blurb.source_file,
        })

    # Sort entries within each ticker by year/quarter (most recent first)
    for ticker in by_ticker:
        by_ticker[ticker].sort(key=lambda x: (x["year"], x["quarter"]), reverse=True)

    data = {
        "metadata": {
            "total_blurbs": len(blurbs),
            "unique_tickers": len(by_ticker),
            "contributors": sum(1 for b in blurbs if b.blurb_type == "contributor"),
            "detractors": sum(1 for b in blurbs if b.blurb_type == "detractor"),
        },
        "blurbs_by_ticker": by_ticker,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d blurbs to %s", len(blurbs), output_path)


def load_exemplars_json(json_path: Path | str) -> list[ExemplarBlurb]:
    """
    Load parsed blurbs from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        List of ExemplarBlurb objects
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Exemplars file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    blurbs = []

    for ticker, entries in data.get("blurbs_by_ticker", {}).items():
        for entry in entries:
            blurbs.append(ExemplarBlurb(
                ticker=entry["ticker"],
                company_name=entry["company_name"],
                blurb_text=entry["blurb_text"],
                quarter=entry["quarter"],
                year=entry["year"],
                blurb_type=entry["blurb_type"],
                word_count=entry["word_count"],
                source_file=entry["source_file"],
            ))

    logger.info("Loaded %d blurbs from %s", len(blurbs), json_path)
    return blurbs


def _parse_filename_metadata(filename: str) -> tuple[str, int]:
    """
    Extract quarter and year from filename.

    Handles patterns like:
    - "Q3_2025_Commentary.docx"
    - "2025_Q3_Report.docx"
    - "Commentary_Q3_2025.docx"
    - "3Q25_Report.docx"

    Returns:
        Tuple of (quarter_str, year_int), defaults to ("Q4", 2025) if not found
    """
    filename_lower = filename.lower()

    # Try "Q3 2025" or "Q3_2025" pattern
    match = re.search(r"q([1-4])[\s_\-]*(20\d{2})", filename_lower)
    if match:
        return f"Q{match.group(1)}", int(match.group(2))

    # Try "2025 Q3" or "2025_Q3" pattern
    match = re.search(r"(20\d{2})[\s_\-]*q([1-4])", filename_lower)
    if match:
        return f"Q{match.group(2)}", int(match.group(1))

    # Try "3Q25" pattern
    match = re.search(r"([1-4])q(\d{2})", filename_lower)
    if match:
        year = 2000 + int(match.group(2))
        return f"Q{match.group(1)}", year

    # Default
    return "Q4", 2025


def _detect_section_header(text: str) -> str | None:
    """
    Detect if text is a section header indicating contributor/detractor section.

    Returns:
        "contributor", "detractor", or None
    """
    text_lower = text.lower().strip()

    # Must be short (headers are typically 1-3 words)
    if len(text_lower.split()) > 5:
        return None

    for header in CONTRIBUTOR_HEADERS:
        if header in text_lower:
            return "contributor"

    for header in DETRACTOR_HEADERS:
        if header in text_lower:
            return "detractor"

    return None


def _extract_blurb_from_paragraph(
    para: str,
    section_type: str,
    quarter: str,
    year: int,
    source_file: str,
) -> ExemplarBlurb | None:
    """
    Try to extract a blurb from a single paragraph.

    Returns:
        ExemplarBlurb if pattern matches, None otherwise
    """
    # Must have minimum length to be a real blurb
    if len(para) < 100:
        return None

    match = BLURB_PATTERN.match(para)
    if not match:
        return None

    company_name = match.group(1).strip()
    ticker = match.group(2).upper()
    blurb_text = match.group(3).strip()

    # Clean up blurb text - remove extra whitespace
    blurb_text = " ".join(blurb_text.split())

    word_count = len(blurb_text.split())

    # Sanity check - blurbs should be 30-200 words
    if word_count < 30 or word_count > 200:
        return None

    return ExemplarBlurb(
        ticker=ticker,
        company_name=company_name,
        blurb_text=blurb_text,
        quarter=quarter,
        year=year,
        blurb_type=section_type if section_type in ("contributor", "detractor") else "unknown",
        word_count=word_count,
        source_file=source_file,
    )


def _extract_multiline_blurbs(
    paragraphs: list[str],
    quarter: str,
    year: int,
    source_file: str,
) -> list[ExemplarBlurb]:
    """
    Extract blurbs that may span multiple paragraphs.

    Some documents have the company header on one line and text on next.
    """
    blurbs = []
    current_section = "unknown"

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]

        # Check for section header
        section = _detect_section_header(para)
        if section:
            current_section = section
            i += 1
            continue

        # Check if this looks like a header line: "Company Name (TICKER):"
        header_match = re.match(
            r"^([A-Za-z][A-Za-z0-9\s\.\,\&\'\-]+?)\s*\(([A-Z]{1,5}(?:\.[A-Z])?)\):?\s*$",
            para
        )

        if header_match and i + 1 < len(paragraphs):
            # Next paragraph might be the blurb text
            next_para = paragraphs[i + 1]

            # Skip if next para is another header or section
            if _detect_section_header(next_para):
                i += 1
                continue

            if re.match(r"^[A-Za-z].*\([A-Z]{1,5}\)", next_para):
                i += 1
                continue

            if len(next_para) >= 100:
                company_name = header_match.group(1).strip()
                ticker = header_match.group(2).upper()
                blurb_text = " ".join(next_para.split())
                word_count = len(blurb_text.split())

                if 30 <= word_count <= 200:
                    blurbs.append(ExemplarBlurb(
                        ticker=ticker,
                        company_name=company_name,
                        blurb_text=blurb_text,
                        quarter=quarter,
                        year=year,
                        blurb_type=current_section if current_section in ("contributor", "detractor") else "unknown",
                        word_count=word_count,
                        source_file=source_file,
                    ))
                    i += 2
                    continue

        i += 1

    return blurbs


def _deduplicate_blurbs(blurbs: list[ExemplarBlurb]) -> list[ExemplarBlurb]:
    """
    Remove duplicate blurbs, keeping the longer version for each ticker+quarter.
    """
    seen: dict[tuple[str, str, int], ExemplarBlurb] = {}

    for blurb in blurbs:
        key = (blurb.ticker, blurb.quarter, blurb.year)

        if key not in seen or blurb.word_count > seen[key].word_count:
            seen[key] = blurb

    return list(seen.values())
