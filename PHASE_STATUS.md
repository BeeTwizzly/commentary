# Phase Status Tracker

This document tracks the build progress of the Portfolio Commentary Generator.

---

## Current Phase: 6 - Core UI

**Status:** ✅ Complete

---

## Phase History

### Phase 0: Scaffolding
- **Status:** ✅ Complete
- **Date:** 2026-01-29
- **What was built:**
  - Project directory structure
  - Git configuration (.gitignore)
  - Streamlit configuration and theming
  - Secrets template
  - Placeholder app.py
  - requirements.txt with pinned dependencies
  - Documentation files (README, ARCHITECTURE, PHASE_STATUS)
- **How to verify:**
  ```bash
  pip install -r requirements.txt
  streamlit run app.py
  ```
  Should display placeholder page with "Phase 0 - Scaffolding Complete" message.
- **Known issues:** None

---

### Phase 1: Excel Parser
- **Status:** ✅ Complete
- **Date:** 2026-01-29
- **What was built:**
  - `src/models.py` - Core data classes (HoldingData, StrategyHoldings, ParsedWorkbook)
  - `src/parsers/excel_parser.py` - Multi-tab Excel parser with top/bottom 5 extraction
  - `tests/test_excel_parser.py` - Comprehensive unit tests
  - `scripts/create_sample_excel.py` - Synthetic data generator
  - `data/synthetic/sample_performance.xlsx` - Test fixture
- **How to verify:**
  ```bash
  # Run tests
  pytest tests/test_excel_parser.py -v

  # Generate sample Excel (if not present)
  python scripts/create_sample_excel.py
  ```
- **Known issues:** None
- **Dependencies added:** None (openpyxl already in requirements.txt)

### Phase 2: Thesis Registry
- **Status:** ✅ Complete
- **Date:** 2026-01-29
- **What was built:**
  - Updated `src/models.py` with ThesisEntry and ThesisLookupResult classes
  - `src/data/thesis_registry.py` - CSV-backed thesis storage and lookup
  - `tests/test_thesis_registry.py` - Comprehensive unit tests (20+ test cases)
  - `data/thesis_registry.csv` - Initial registry with 15 sample theses
- **How to verify:**
  ```bash
  # Run tests
  pytest tests/test_thesis_registry.py -v

  # Quick smoke test
  python -c "
  from src.data.thesis_registry import ThesisRegistry
  r = ThesisRegistry.load('data/thesis_registry.csv')
  print(f'Loaded {len(r)} theses')
  print(r.lookup('NVDA').thesis_text[:50] + '...')
  "
  ```
- **Known issues:** None
- **Key decisions:**
  - CSV format chosen for simplicity and manual editability
  - Lookup returns structured result (not None) so generation always proceeds
  - Case-insensitive ticker matching
  - Stale thesis detection for maintenance visibility

### Phase 3: Exemplar Parser
- **Status:** ✅ Complete
- **Date:** 2026-01-30
- **What was built:**
  - Updated `src/models.py` with ExemplarBlurb and ExemplarSelection classes
  - `src/parsers/exemplar_parser.py` - Word document parser for blurb extraction
  - `src/parsers/exemplar_selector.py` - Few-shot exemplar selection logic
  - `tests/test_exemplar_parser.py` - Comprehensive unit tests (25+ test cases)
  - `scripts/parse_exemplars.py` - CLI tool for parsing docs and generating JSON
  - `data/exemplars/exemplars.json` - Initial exemplars (10 synthetic blurbs)
- **How to verify:**
  ```bash
  # Run tests
  pytest tests/test_exemplar_parser.py -v

  # Regenerate synthetic exemplars
  PYTHONPATH=. python scripts/parse_exemplars.py --synthetic

  # Quick smoke test
  python -c "
  from src.parsers.exemplar_selector import ExemplarSelector
  s = ExemplarSelector.load('data/exemplars/exemplars.json')
  print(f'Loaded {s.total_blurbs} blurbs for {len(s.available_tickers)} tickers')
  sel = s.select('NVDA', is_contributor=True)
  print(sel.format_for_prompt()[:200] + '...')
  "
  ```
- **Known issues:** None
- **Key decisions:**
  - Regex pattern handles "Company Name (TICKER):" format with variations
  - Selector prioritizes same-ticker exemplars for continuity
  - Synthetic exemplars provided for testing without real docs
  - JSON format enables fast loading without re-parsing Word docs

### Phase 4: Prompt Builder
- **Status:** ✅ Complete
- **Date:** 2026-01-30
- **What was built:**
  - `src/generation/prompt_builder.py` - PromptBuilder class with PromptContext and AssembledPrompt dataclasses
  - `src/generation/response_parser.py` - LLM response parsing with ParsedResponse and ParsedVariation classes
  - `src/generation/prompts/system.txt` - System prompt template with style guidelines
  - `src/generation/prompts/user_template.txt` - User prompt template with placeholders
  - `src/generation/__init__.py` - Module exports
  - `tests/test_prompt_builder.py` - Comprehensive unit tests (20+ test cases)
- **How to verify:**
  ```bash
  # Run tests
  pytest tests/test_prompt_builder.py -v

  # Quick smoke test
  python -c "
  from src.generation.prompt_builder import PromptBuilder, PromptContext, create_prompt_context
  from src.generation.response_parser import parse_llm_response
  from src.models import HoldingData, ThesisLookupResult, ExemplarSelection
  print('Prompt builder imports successfully')
  builder = PromptBuilder()
  print(f'PromptBuilder created: {builder}')
  "
  ```
- **Known issues:** None
- **Key decisions:**
  - Template-based system with external txt files for easy modification
  - Response parser handles multiple variation formats ([A], A), etc.)
  - Duplicate detection and word count validation in parser
  - Metadata tracking for debugging and analytics

### Phase 5: LLM Client
- **Status:** ✅ Complete
- **Date:** 2026-01-30
- **What was built:**
  - `src/config.py` - Configuration management (env vars + Streamlit secrets)
  - `src/generation/llm_client.py` - Async OpenAI client with retry logic
  - `tests/test_llm_client.py` - Unit tests with mocked API responses (15+ tests)
  - Updated `.streamlit/secrets.toml.example` with OpenAI config
  - Updated `src/generation/__init__.py` with new exports
- **How to verify:**
  ```bash
  # Run tests (no API key needed - uses mocks)
  pytest tests/test_llm_client.py -v

  # Test config loading (set env var first)
  export OPENAI_API_KEY="sk-test-key"
  python -c "
  from src.config import load_config_from_env
  config = load_config_from_env()
  print(f'Model: {config.llm.model}')
  print(f'Temperature: {config.llm.temperature}')
  print('Config loaded successfully!')
  "
  ```
- **Known issues:** None
- **Key decisions:**
  - Async-first with sync wrapper for Streamlit compatibility
  - Exponential backoff retry (2^attempt seconds)
  - Rate limit handling via Retry-After header
  - Token cost tracking with model-specific pricing
  - Config from Streamlit secrets with env var fallback

### Phase 6: Core UI
- **Status:** ✅ Complete
- **Date:** 2026-01-30
- **What was built:**
  - `app.py` - Main Streamlit application (upload, select, generate, preview)
  - `src/ui/__init__.py` - UI module initialization
  - `src/ui/components.py` - Reusable UI components (holding_card, progress, etc.)
  - `src/ui/state.py` - Session state management helpers
  - `.streamlit/config.toml` - Updated Streamlit theme and server configuration
- **How to verify:**
  ```bash
  # Start the app (requires OPENAI_API_KEY for generation)
  export OPENAI_API_KEY="sk-your-key"
  streamlit run app.py

  # Or without API key (upload/preview only)
  streamlit run app.py

  # The app should:
  # 1. Show file upload section
  # 2. Accept Excel file and parse strategies
  # 3. Allow strategy selection
  # 4. Show generation controls (disabled without API key)
  # 5. Display results after generation
  ```
- **Known issues:**
  - Export button disabled (Phase 8)
  - Review/edit UI is basic (Phase 7 will enhance)
- **Key features:**
  - Session state persistence across reruns
  - Progress tracking during generation
  - Cost/token tracking in sidebar
  - Expandable results by strategy
  - Variation tabs for each holding

### Phase 7: Review UI
- **Status:** 🔲 Not started
- **Planned deliverables:**
  - Review dashboard
  - Selection and editing interface

### Phase 8: Export
- **Status:** 🔲 Not started
- **Planned deliverables:**
  - `src/export/word_export.py`
  - Download functionality

### Phase 9: Polish & Harden
- **Status:** 🔲 Not started
- **Planned deliverables:**
  - Error handling
  - Loading states
  - Edge case handling
