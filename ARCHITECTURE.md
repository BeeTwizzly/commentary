# Portfolio Commentary Generator - Architecture

## Overview

This application automates the quarterly process of writing investment commentary for portfolio performance reports. It transforms Excel-based attribution data into human-quality prose using LLM generation with few-shot prompting.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           STREAMLIT UI                                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Upload  │→ │  Review &    │→ │   Edit /     │→ │   Export     │     │
│  │  Excel   │  │  Select      │  │  Regenerate  │  │   to Word    │     │
│  └──────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
         │                                                      ▲
         ▼                                                      │
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING PIPELINE                             │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐   │
│  │   Excel    │    │  Thesis    │    │  Exemplar  │    │   Prompt   │   │
│  │   Parser   │──▶│  Lookup    │───▶│  Selector  │───▶│ Assembler │   │
│  └────────────┘    └────────────┘    └────────────┘    └────────────┘   │
│                                                               │         │
│                                                               ▼         │
│                                                        ┌────────────┐   │
│                                                        │  LLM API   │   │
│                                                        └────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### Excel Parser (`src/parsers/excel_parser.py`)
- Reads multi-tab Excel workbooks (one tab per strategy)
- Identifies header rows and data rows
- Extracts holdings with Brinson attribution data
- Ranks by total attribution, returns top 5 and bottom 5

### Thesis Registry (`src/data/thesis_registry.py`)
- CSV-backed storage of investment theses per ticker
- Simple lookup by ticker symbol
- Returns placeholder text for missing entries

### Exemplar Parser (`src/parsers/exemplar_parser.py`)
- Parses historical Word documents containing prior commentary
- Extracts individual blurbs using pattern: "Company Name (TICKER): blurb text"
- Stores parsed blurbs in JSON for few-shot retrieval

### Prompt Builder (`src/generation/prompt_builder.py`)
- Assembles complete prompts from:
  - System prompt with style guidelines
  - Holding performance data
  - Investment thesis context
  - Few-shot exemplar blurbs
- Requests 3 variations per holding

### LLM Client (`src/generation/llm_client.py`)
- Async HTTP client for OpenAI API
- Supports configurable base URL (for Enterprise)
- Retry logic with exponential backoff
- Response parsing to extract [A], [B], [C] variations

### Word Export (`src/export/word_export.py`)
- Generates Word document from approved selections
- Structures output with contributor/detractor sections

## Data Models

### HoldingData
```python
@dataclass
class HoldingData:
    ticker: str
    company_name: str
    strategy: str
    avg_weight: float
    benchmark_weight: float
    total_attribution: float
    selection_effect: float
    allocation_effect: float
    is_contributor: bool
```

### GeneratedDraft
```python
@dataclass
class GeneratedDraft:
    holding: HoldingData
    variations: list[str]  # 3 options
    selected_index: int | None
    edited_text: str | None
    approved: bool
```

## Configuration

Secrets are loaded from Streamlit's secrets management:
- `OPENAI_API_KEY`: API authentication
- `OPENAI_BASE_URL`: API endpoint (override for Enterprise)
- `OPENAI_MODEL`: Model identifier (default: gpt-4o)

## File Structure

```
portfolio-commentary/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── .streamlit/
│   ├── config.toml           # Theme configuration
│   └── secrets.toml          # API keys (not committed)
├── src/
│   ├── __init__.py
│   ├── models.py             # Data classes
│   ├── config.py             # Configuration loading
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── excel_parser.py
│   │   └── exemplar_parser.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── thesis_registry.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py
│   │   ├── llm_client.py
│   │   └── prompts/
│   │       ├── system.txt
│   │       └── user_template.txt
│   └── export/
│       ├── __init__.py
│       └── word_export.py
├── data/
│   ├── thesis_registry.csv
│   ├── exemplars/
│   │   └── exemplars.json
│   └── synthetic/
│       └── sample_performance.xlsx
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_excel_parser.py
    ├── test_thesis_registry.py
    ├── test_exemplar_parser.py
    ├── test_prompt_builder.py
    └── test_llm_client.py
```
