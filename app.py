"""
Portfolio Commentary Generator - Main Streamlit Application.

A tool for generating quarterly portfolio commentary using AI.
Upload performance data, configure settings, and generate
professional investment commentary with human review.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import streamlit as st

from src.config import get_config, AppConfig, LLMConfig
from src.models import ParsedWorkbook, HoldingData, StrategyHoldings
from src.parsers.excel_parser import parse_workbook
from src.parsers.exemplar_selector import ExemplarSelector
from src.data.thesis_registry import ThesisRegistry
from src.generation import (
    PromptBuilder,
    create_prompt_context,
    LLMClient,
    GenerationResult,
)
from src.ui.review import (
    ReviewSession,
    ReviewItem,
    ApprovalStatus,
    init_review_session,
    render_review_page,
)
from src.ui.export_panel import increment_export_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Portfolio Commentary Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Path to custom CSS file
CSS_FILE = Path(__file__).parent / "assets" / "streamlit_polish.css"


def load_custom_css() -> None:
    """
    Load and inject custom CSS styles.

    Call this inside main() to apply custom styling.
    CSS is loaded from assets/streamlit_polish.css.
    """
    if CSS_FILE.exists():
        try:
            css_content = CSS_FILE.read_text(encoding="utf-8")
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        except Exception as e:
            logger.warning("Failed to load custom CSS: %s", e)
    else:
        logger.debug("Custom CSS file not found: %s", CSS_FILE)


def init_session_state() -> None:
    """Initialize session state variables."""
    defaults = {
        # Data state
        "workbook": None,  # ParsedWorkbook
        "selected_strategies": [],  # List of strategy names
        "quarter": get_default_quarter(),
        "year": get_reporting_year(),

        # Generation state
        "generation_results": {},  # {key: {holding, strategy, result, thesis_found}}
        "generation_in_progress": False,
        "generation_progress": 0.0,
        "generation_status": "",

        # Cost tracking
        "total_cost_usd": 0.0,
        "total_tokens": 0,

        # Configuration
        "num_variations": 3,
        "config_loaded": False,

        # View state
        "current_view": "upload",  # upload, review

        # Review state
        "review_sessions": {},  # strategy_name -> ReviewSession
        "active_review_strategy": None,

        # Export state
        "trigger_export": False,

        # Regeneration state
        "regenerate_request": None,  # {"strategy": str, "ticker": str}
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get_default_quarter() -> str:
    """Get the most likely quarter being reported (previous quarter)."""
    month = date.today().month
    if month <= 3:
        return "Q4"  # Reporting Q4 of previous year
    elif month <= 6:
        return "Q1"
    elif month <= 9:
        return "Q2"
    else:
        return "Q3"


def get_reporting_year() -> int:
    """Get the year for the quarter being reported."""
    month = date.today().month
    year = date.today().year
    # If we're in Q1, we're likely reporting Q4 of last year
    if month <= 3:
        return year - 1
    return year


@st.cache_resource
def load_thesis_registry() -> ThesisRegistry | None:
    """Load and cache thesis registry."""
    try:
        config = get_config()
        registry = ThesisRegistry.load(config.thesis_registry_path)
        return registry
    except Exception as e:
        logger.warning("Failed to load thesis registry: %s", e)
        return None


@st.cache_resource
def load_exemplars() -> ExemplarSelector | None:
    """Load and cache exemplar corpus."""
    try:
        config = get_config()
        return ExemplarSelector.load(config.exemplars_path)
    except Exception as e:
        logger.warning("Failed to load exemplars: %s", e)
        return None


def render_sidebar() -> None:
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("Navigation")

        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Upload",
                use_container_width=True,
                type="primary" if st.session_state.current_view == "upload" else "secondary",
            ):
                st.session_state.current_view = "upload"
                st.rerun()
        with col2:
            has_reviews = bool(st.session_state.review_sessions)
            if st.button(
                "Review",
                use_container_width=True,
                type="primary" if st.session_state.current_view == "review" else "secondary",
                disabled=not has_reviews,
            ):
                st.session_state.current_view = "review"
                st.rerun()

        st.divider()

        st.header("Configuration")

        # Quarter selection
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.quarter = st.selectbox(
                "Quarter",
                options=["Q1", "Q2", "Q3", "Q4"],
                index=["Q1", "Q2", "Q3", "Q4"].index(st.session_state.quarter),
            )
        with col2:
            st.session_state.year = st.number_input(
                "Year",
                min_value=2020,
                max_value=2030,
                value=st.session_state.year,
            )

        st.divider()

        # Generation settings
        st.subheader("Generation Settings")
        st.session_state.num_variations = st.slider(
            "Variations per holding",
            min_value=1,
            max_value=5,
            value=st.session_state.num_variations,
            help="Number of commentary variations to generate for each holding",
        )

        # Show loaded resources status
        st.divider()
        st.subheader("Resources")

        registry = load_thesis_registry()
        exemplars = load_exemplars()

        if registry and len(registry) > 0:
            st.success(f"Thesis registry: {len(registry)} entries")
        else:
            st.warning("No thesis registry loaded")

        if exemplars and exemplars.total_blurbs > 0:
            st.success(f"Exemplars: {exemplars.total_blurbs} blurbs")
        else:
            st.warning("No exemplars loaded")

        # Review progress
        if st.session_state.review_sessions:
            st.divider()
            st.subheader("Review Progress")
            for strategy_name, session in st.session_state.review_sessions.items():
                st.progress(session.completion_pct / 100)
                st.caption(f"{strategy_name}: {session.approved_count}/{session.total_count}")

        # Cost display
        if st.session_state.total_cost_usd > 0:
            st.divider()
            st.subheader("Session Costs")
            st.metric(
                "Total Cost",
                f"${st.session_state.total_cost_usd:.4f}",
            )
            st.caption(f"Tokens: {st.session_state.total_tokens:,}")


def render_upload_section() -> None:
    """Render the file upload section."""
    st.header("Upload Performance Data")

    uploaded_file = st.file_uploader(
        "Upload Excel workbook with performance attribution data",
        type=["xlsx", "xls"],
        help="Excel file should contain strategy tabs with ticker, weights, and attribution data",
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Parsing Excel file..."):
                workbook = parse_workbook(uploaded_file)
                st.session_state.workbook = workbook

                st.success(f"Loaded {len(workbook.strategies)} strategies with {workbook.total_holdings} holdings")

                # Show strategy summary
                with st.expander("Strategy Summary", expanded=True):
                    for strategy in workbook.strategies:
                        contributors = len(strategy.contributors)
                        detractors = len(strategy.detractors)
                        st.write(f"**{strategy.strategy_name}**: {contributors} contributors, {detractors} detractors")

        except Exception as e:
            st.error(f"Error parsing file: {e}")
            logger.exception("Excel parsing failed")
            st.session_state.workbook = None


def render_strategy_selection() -> None:
    """Render strategy selection interface."""
    workbook = st.session_state.workbook
    if workbook is None:
        return

    st.header("Select Strategies")

    strategy_names = [s.strategy_name for s in workbook.strategies]

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.multiselect(
            "Choose strategies to generate commentary for",
            options=strategy_names,
            default=st.session_state.selected_strategies if st.session_state.selected_strategies else strategy_names[:1],
            help="Select one or more strategies",
        )
        st.session_state.selected_strategies = selected

    with col2:
        if st.button("Select All", use_container_width=True):
            st.session_state.selected_strategies = strategy_names
            st.rerun()
        if st.button("Clear All", use_container_width=True):
            st.session_state.selected_strategies = []
            st.rerun()

    # Show holdings preview for selected strategies
    if selected:
        with st.expander(f"Preview Holdings ({len(selected)} strategies)", expanded=False):
            for strategy_name in selected:
                strategy = next((s for s in workbook.strategies if s.strategy_name == strategy_name), None)
                if strategy:
                    st.subheader(strategy_name)

                    # Top contributors
                    if strategy.contributors:
                        st.write("**Top Contributors:**")
                        for h in strategy.contributors[:5]:
                            effect_str = f"+{h.total_attribution:.1f} bps" if h.total_attribution else ""
                            st.write(f"  - {h.ticker} ({h.company_name}) {effect_str}")

                    # Top detractors
                    if strategy.detractors:
                        st.write("**Top Detractors:**")
                        for h in strategy.detractors[:5]:
                            effect_str = f"{h.total_attribution:.1f} bps" if h.total_attribution else ""
                            st.write(f"  - {h.ticker} ({h.company_name}) {effect_str}")

                    st.divider()


def get_holdings_to_generate() -> list[tuple[str, HoldingData]]:
    """Get list of (strategy_name, holding) tuples to generate."""
    workbook = st.session_state.workbook
    selected = st.session_state.selected_strategies

    if not workbook or not selected:
        return []

    holdings = []
    for strategy_name in selected:
        strategy = next((s for s in workbook.strategies if s.strategy_name == strategy_name), None)
        if strategy:
            # Get top 5 contributors and top 5 detractors
            for h in strategy.contributors[:5]:
                holdings.append((strategy_name, h))
            for h in strategy.detractors[:5]:
                holdings.append((strategy_name, h))

    return holdings


def render_generation_section() -> None:
    """Render the generation control section."""
    if not st.session_state.selected_strategies:
        return

    st.header("Generate Commentary")

    holdings = get_holdings_to_generate()
    num_holdings = len(holdings)

    if num_holdings == 0:
        st.warning("No holdings to generate. Check your strategy selection.")
        return

    # Estimate display
    estimated_cost = num_holdings * 0.01  # Rough estimate: ~$0.01 per holding
    quarter_str = f"{st.session_state.quarter} {st.session_state.year}"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Holdings", num_holdings)
    with col2:
        st.metric("Variations", st.session_state.num_variations)
    with col3:
        st.metric("Est. Cost", f"~${estimated_cost:.2f}")

    st.info(f"Generating commentary for **{quarter_str}**")

    # Generation button
    if st.session_state.generation_in_progress:
        st.warning("Generation in progress...")
        st.progress(st.session_state.generation_progress)
        st.caption(st.session_state.generation_status)
    else:
        if st.button(
            f"Generate Commentary for {num_holdings} Holdings",
            type="primary",
            use_container_width=True,
        ):
            run_generation(holdings)


def run_generation(holdings: list[tuple[str, HoldingData]]) -> None:
    """Run commentary generation for selected holdings."""
    st.session_state.generation_in_progress = True
    st.session_state.generation_results = {}

    # Load resources
    registry = load_thesis_registry()
    exemplars = load_exemplars()
    prompt_builder = PromptBuilder()

    quarter_str = f"{st.session_state.quarter} {st.session_state.year}"
    num_variations = st.session_state.num_variations

    # Try to get LLM config
    try:
        config = get_config()
        client = LLMClient(config.llm)
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        st.info("Set OPENAI_API_KEY environment variable or configure in .streamlit/secrets.toml")
        st.session_state.generation_in_progress = False
        return

    # Progress tracking
    progress_container = st.empty()
    status_container = st.empty()

    total = len(holdings)
    completed = 0
    failed = 0
    total_cost = 0.0
    total_tokens = 0

    for strategy_name, holding in holdings:
        # Update progress
        st.session_state.generation_progress = completed / total
        st.session_state.generation_status = f"Generating: {holding.ticker} ({strategy_name})"

        progress_container.progress(completed / total, text=f"Processing {completed + 1}/{total}")
        status_container.caption(f"{holding.ticker} - {holding.company_name}")

        try:
            # Look up thesis
            thesis_result = registry.lookup(holding.ticker) if registry else None
            if thesis_result is None:
                from src.models import ThesisLookupResult
                thesis_result = ThesisLookupResult(
                    ticker=holding.ticker,
                    found=False,
                    entry=None,
                    placeholder_text=f"[No thesis on file for {holding.ticker}]",
                )

            # Get exemplars
            exemplar_selection = exemplars.select(
                ticker=holding.ticker,
                is_contributor=holding.is_contributor,
            ) if exemplars else None

            if exemplar_selection is None:
                from src.models import ExemplarSelection
                exemplar_selection = ExemplarSelection(
                    target_ticker=holding.ticker,
                    target_is_contributor=holding.is_contributor,
                    same_ticker_exemplar=None,
                    similar_exemplars=[],
                )

            # Build prompt
            context = create_prompt_context(
                holding=holding,
                thesis=thesis_result,
                exemplars=exemplar_selection,
                quarter=quarter_str,
                strategy_name=strategy_name,
                num_variations=num_variations,
            )
            prompt = prompt_builder.build(context)

            # Generate (sync call)
            result = client.generate_sync(prompt, num_variations)

            # Store result
            key = f"{strategy_name}|{holding.ticker}"
            st.session_state.generation_results[key] = {
                "holding": holding,
                "strategy": strategy_name,
                "result": result,
                "thesis_found": thesis_result.found,
            }

            # Track costs
            total_cost += result.cost_usd
            total_tokens += result.usage.total_tokens

            if result.success:
                completed += 1
            else:
                failed += 1
                logger.warning(
                    "Generation failed for %s: %s",
                    holding.ticker,
                    result.error_message,
                )

        except Exception as e:
            failed += 1
            logger.exception("Error generating for %s", holding.ticker)

            # Store error result
            key = f"{strategy_name}|{holding.ticker}"
            st.session_state.generation_results[key] = {
                "holding": holding,
                "strategy": strategy_name,
                "result": None,
                "error": str(e),
            }

    # Update session totals
    st.session_state.total_cost_usd += total_cost
    st.session_state.total_tokens += total_tokens

    # Complete
    st.session_state.generation_in_progress = False
    st.session_state.generation_progress = 1.0
    st.session_state.generation_status = "Complete!"

    progress_container.progress(1.0, text="Complete!")
    status_container.success(
        f"Generated {completed}/{total} holdings. "
        f"Cost: ${total_cost:.4f}, Tokens: {total_tokens:,}"
    )

    # Trigger rerun to show results
    increment_export_key()  # Refresh download button keys
    st.rerun()


def regenerate_single_item(strategy_name: str, ticker: str) -> bool:
    """
    Regenerate commentary for a single holding.

    Args:
        strategy_name: Strategy the holding belongs to
        ticker: Ticker symbol of holding to regenerate

    Returns:
        True if regeneration succeeded, False otherwise
    """
    # Find the review session and item
    session = st.session_state.review_sessions.get(strategy_name)
    if not session:
        st.error(f"Review session not found for strategy: {strategy_name}")
        return False

    item = session.get_item_by_ticker(ticker)
    if not item:
        st.error(f"Item not found: {ticker}")
        return False

    holding = item.holding

    # Load resources (these are cached)
    registry = load_thesis_registry()
    exemplars = load_exemplars()
    prompt_builder = PromptBuilder()

    quarter_str = f"{st.session_state.quarter} {st.session_state.year}"
    num_variations = st.session_state.num_variations

    # Get LLM client
    try:
        config = get_config()
        if not config.llm.is_configured:
            st.error("API key not configured. Set OPENAI_API_KEY.")
            return False
        client = LLMClient(config.llm)
    except Exception as e:
        st.error(f"Configuration error: {e}")
        return False

    try:
        # Look up thesis
        thesis_result = registry.lookup(holding.ticker) if registry else None
        if thesis_result is None:
            from src.models import ThesisLookupResult
            thesis_result = ThesisLookupResult(
                ticker=holding.ticker,
                found=False,
                entry=None,
                placeholder_text=f"[No thesis on file for {holding.ticker}]",
            )

        # Get exemplars
        exemplar_selection = exemplars.select(
            ticker=holding.ticker,
            is_contributor=holding.is_contributor,
        ) if exemplars else None

        if exemplar_selection is None:
            from src.models import ExemplarSelection
            exemplar_selection = ExemplarSelection(
                target_ticker=holding.ticker,
                target_is_contributor=holding.is_contributor,
                same_ticker_exemplar=None,
                similar_exemplars=[],
            )

        # Build prompt
        context = create_prompt_context(
            holding=holding,
            thesis=thesis_result,
            exemplars=exemplar_selection,
            quarter=quarter_str,
            strategy_name=strategy_name,
            num_variations=num_variations,
        )
        prompt = prompt_builder.build(context)

        # Generate (sync call)
        result = client.generate_sync(prompt, num_variations)

        # Update the review item with new result
        item.result = result
        item.thesis_found = thesis_result.found
        item.selected_variation = 0
        item.edited_text = ""
        item.status = ApprovalStatus.PENDING

        # Track costs
        st.session_state.total_cost_usd += result.cost_usd
        st.session_state.total_tokens += result.usage.total_tokens

        # Also update generation_results for consistency
        key = f"{strategy_name}|{ticker}"
        st.session_state.generation_results[key] = {
            "holding": holding,
            "strategy": strategy_name,
            "result": result,
            "thesis_found": thesis_result.found,
        }

        if result.success:
            logger.info("Regenerated %s successfully", ticker)
            return True
        else:
            st.error(f"Regeneration failed: {result.error_message}")
            logger.warning("Regeneration failed for %s: %s", ticker, result.error_message)
            return False

    except Exception as e:
        st.error(f"Error regenerating {ticker}: {e}")
        logger.exception("Regeneration error for %s", ticker)
        return False


def render_results_preview() -> None:
    """Render a preview of generation results."""
    results = st.session_state.generation_results

    if not results:
        return

    st.header("Generated Commentary")

    # Summary stats
    successful = sum(1 for r in results.values() if r.get("result") and r["result"].success)
    failed = len(results) - successful

    if failed > 0:
        st.warning(f"{failed} generation(s) failed. Check logs for details.")

    st.success(f"{successful} commentaries ready for review")

    # Group by strategy
    by_strategy = {}
    for key, data in results.items():
        strategy = data["strategy"]
        if strategy not in by_strategy:
            by_strategy[strategy] = []
        by_strategy[strategy].append(data)

    # Display results
    for strategy_name, holdings_data in by_strategy.items():
        with st.expander(f"{strategy_name} ({len(holdings_data)} holdings)", expanded=True):
            for data in holdings_data:
                holding = data["holding"]
                result = data.get("result")

                # Holding header
                effect_str = f"{'+' if holding.total_attribution > 0 else ''}{holding.total_attribution:.1f} bps"
                holding_type = "+" if holding.is_contributor else "-"

                st.subheader(f"[{holding_type}] {holding.ticker} - {holding.company_name} ({effect_str})")

                # Thesis indicator
                if data.get("thesis_found"):
                    st.caption("Thesis context available")
                else:
                    st.caption("No thesis context")

                # Show variations or error
                if result and result.success and result.variations:
                    tabs = st.tabs([f"Variation {v.label}" for v in result.variations])
                    for tab, variation in zip(tabs, result.variations):
                        with tab:
                            st.write(variation.text)
                            st.caption(f"{variation.word_count} words")
                elif result and result.success:
                    # Success but no parseable variations
                    st.warning("No valid variations could be parsed from the response")
                    if result.parsed and result.parsed.raw_response:
                        with st.expander("Raw response"):
                            st.text(result.parsed.raw_response[:500])
                elif data.get("error"):
                    st.error(f"Error: {data['error']}")
                elif result:
                    st.error(f"Generation failed: {result.error_message}")

                st.divider()

    # Action buttons
    st.subheader("Next Steps")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Regenerate All", use_container_width=True):
            st.session_state.generation_results = {}
            st.rerun()
    with col2:
        if st.button("Go to Review", type="primary", use_container_width=True):
            transition_to_review()
            st.rerun()
    with col3:
        st.button(
            "Export to Word",
            use_container_width=True,
            disabled=True,
            help="Coming in Phase 8",
        )


def transition_to_review() -> None:
    """Transition from generation results to review view."""
    results = st.session_state.generation_results
    if not results:
        return

    quarter_str = f"{st.session_state.quarter} {st.session_state.year}"

    # Group results by strategy
    by_strategy = {}
    for key, data in results.items():
        strategy = data.get("strategy")
        if strategy:
            if strategy not in by_strategy:
                by_strategy[strategy] = {}
            by_strategy[strategy][key] = data

    # Create review sessions for each strategy
    for strategy_name, strategy_results in by_strategy.items():
        session = init_review_session(
            strategy=strategy_name,
            quarter=quarter_str,
            generation_results=strategy_results,
        )
        st.session_state.review_sessions[strategy_name] = session

    # Set active strategy to first one
    if by_strategy:
        st.session_state.active_review_strategy = list(by_strategy.keys())[0]

    st.session_state.current_view = "review"


def render_review_view() -> None:
    """Render the review page."""
    sessions = st.session_state.review_sessions

    if not sessions:
        st.warning("No review sessions available. Generate commentary first.")
        if st.button("Go to Upload"):
            st.session_state.current_view = "upload"
            st.rerun()
        return

    # Check for export trigger
    if st.session_state.get("trigger_export"):
        st.session_state.trigger_export = False
        st.info("Export functionality coming in Phase 8!")

    # Strategy selector if multiple strategies
    if len(sessions) > 1:
        strategy_names = list(sessions.keys())
        active = st.session_state.active_review_strategy
        if active not in strategy_names:
            active = strategy_names[0]

        selected = st.selectbox(
            "Select Strategy to Review",
            options=strategy_names,
            index=strategy_names.index(active) if active in strategy_names else 0,
        )
        st.session_state.active_review_strategy = selected
        session = sessions[selected]
    else:
        session = list(sessions.values())[0]
        st.session_state.active_review_strategy = list(sessions.keys())[0]

    render_review_page(session)


def main() -> None:
    """Main application entry point."""
    init_session_state()

    # Load custom CSS styling (safe to disable by removing this line)
    load_custom_css()

    # Handle regeneration request at top level before any rendering
    # This ensures the request is processed on the very next rerun
    regen_request = st.session_state.get("regenerate_request")
    if regen_request:
        strategy = regen_request.get("strategy")
        ticker = regen_request.get("ticker")
        st.session_state.regenerate_request = None  # Clear before processing

        if strategy and ticker:
            with st.spinner(f"Regenerating commentary for {ticker}..."):
                success = regenerate_single_item(strategy, ticker)
            if success:
                st.toast(f"Regenerated commentary for {ticker}")
            increment_export_key()  # Refresh download button keys to prevent stale file refs
            st.rerun()

    # Title
    st.title("Portfolio Commentary Generator")
    st.caption("Generate professional quarterly commentary with AI assistance")

    # Sidebar
    render_sidebar()

    # Route based on current view
    if st.session_state.current_view == "review":
        render_review_view()
    else:
        # Upload view
        render_upload_section()

        if st.session_state.workbook:
            render_strategy_selection()
            render_generation_section()

        # Results
        render_results_preview()

    # Footer
    st.divider()
    st.caption(
        "Portfolio Commentary Generator v0.1 | "
        f"Session: {st.session_state.quarter} {st.session_state.year}"
    )


if __name__ == "__main__":
    main()
