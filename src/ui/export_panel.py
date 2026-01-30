"""Export panel UI component for Streamlit."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import streamlit as st

from src.ui.review import ReviewSession
from src.export import WordExporter, ExportConfig
from src.export.formats import export_to_csv, export_to_json, export_summary_stats

if TYPE_CHECKING:
    from src.ui.review import ReviewItem

logger = logging.getLogger(__name__)


def render_export_panel(session: ReviewSession) -> None:
    """
    Render the export panel with download options.

    Args:
        session: Review session with items to export
    """
    st.markdown("## Export Commentary")

    # Check for exportable items
    exportable = session.exportable_items

    if not exportable:
        st.warning(
            "No approved commentaries to export. "
            "Approve items in the review section above."
        )
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Ready to Export", len(exportable))

    with col2:
        contributors = sum(1 for i in exportable if i.is_contributor)
        st.metric("Contributors", contributors)

    with col3:
        detractors = sum(1 for i in exportable if not i.is_contributor)
        st.metric("Detractors", detractors)

    with col4:
        avg_words = sum(i.current_word_count for i in exportable) / len(exportable)
        st.metric("Avg Words", f"{avg_words:.0f}")

    st.divider()

    # Export options
    st.markdown("### Export Options")

    col1, col2 = st.columns(2)

    with col1:
        include_metadata = st.checkbox("Include metadata header", value=True, key="exp_metadata")
        include_stats = st.checkbox("Include statistics page", value=True, key="exp_stats")

    with col2:
        group_by_type = st.checkbox("Group by contributor/detractor", value=True, key="exp_group")
        include_effects = st.checkbox("Include attribution effects", value=True, key="exp_effects")

    st.divider()

    # Download buttons
    st.markdown("### Download")

    col1, col2, col3, col4 = st.columns(4)

    # Build config from options
    config = ExportConfig(
        include_metadata=include_metadata,
        include_statistics=include_stats,
        group_by_type=group_by_type,
        include_effects=include_effects,
    )

    # Generate filenames
    date_str = datetime.now().strftime("%Y%m%d")
    strategy_clean = "".join(c if c.isalnum() else "_" for c in session.strategy)
    quarter_clean = session.quarter.replace(" ", "_")
    base_filename = f"{strategy_clean}_{quarter_clean}_{date_str}"

    # Word export
    with col1:
        exporter = WordExporter(config)
        result = exporter.export(session)

        if result.success:
            st.download_button(
                label="Word",
                data=result.buffer.getvalue(),
                file_name=result.filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                use_container_width=True,
            )
        else:
            st.error(f"Export failed: {result.error_message}")

    # Plain text export
    with col2:
        text_content = exporter.export_to_text(session)
        text_filename = f"{base_filename}.txt"

        st.download_button(
            label="Text",
            data=text_content,
            file_name=text_filename,
            mime="text/plain",
            use_container_width=True,
        )

    # CSV export
    with col3:
        csv_content = export_to_csv(session)
        csv_filename = f"{base_filename}.csv"

        st.download_button(
            label="CSV",
            data=csv_content,
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True,
        )

    # JSON export
    with col4:
        json_content = export_to_json(session)
        json_filename = f"{base_filename}.json"

        st.download_button(
            label="JSON",
            data=json_content,
            file_name=json_filename,
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    # Preview section
    with st.expander("Export Preview", expanded=False):
        render_export_preview(session, exportable)

    # Statistics section
    with st.expander("Export Statistics", expanded=False):
        render_export_stats(session)


def render_export_preview(session: ReviewSession, items: list) -> None:
    """Render a preview of what will be exported."""
    st.markdown(f"**{session.strategy}** | {session.quarter}")
    st.markdown("---")

    # Group by type
    contributors = [i for i in items if i.is_contributor]
    detractors = [i for i in items if not i.is_contributor]

    # Sort by effect
    contributors.sort(key=lambda x: abs(x.holding.total_attribution or 0), reverse=True)
    detractors.sort(key=lambda x: abs(x.holding.total_attribution or 0), reverse=True)

    if contributors:
        st.markdown("**Top Contributors**")
        for item in contributors[:3]:  # Show first 3
            effect = item.holding.total_attribution
            effect_str = f" [{effect:+.1f} bps]" if effect else ""
            st.markdown(f"**{item.company_name}** ({item.ticker}){effect_str}")
            preview_text = item.current_text[:150] + "..." if len(item.current_text) > 150 else item.current_text
            st.markdown(f"> {preview_text}")
            st.caption(f"{item.current_word_count} words")
        if len(contributors) > 3:
            st.caption(f"...and {len(contributors) - 3} more contributors")
        st.markdown("")

    if detractors:
        st.markdown("**Top Detractors**")
        for item in detractors[:3]:  # Show first 3
            effect = item.holding.total_attribution
            effect_str = f" [{effect:+.1f} bps]" if effect else ""
            st.markdown(f"**{item.company_name}** ({item.ticker}){effect_str}")
            preview_text = item.current_text[:150] + "..." if len(item.current_text) > 150 else item.current_text
            st.markdown(f"> {preview_text}")
            st.caption(f"{item.current_word_count} words")
        if len(detractors) > 3:
            st.caption(f"...and {len(detractors) - 3} more detractors")


def render_export_stats(session: ReviewSession) -> None:
    """Render export statistics."""
    stats = export_summary_stats(session)

    if "error" in stats:
        st.warning(stats["error"])
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Generation**")
        st.write(f"Total Generated: {stats['total_generated']}")
        st.write(f"Approved: {stats['total_approved']}")
        st.write(f"Rejected: {stats['total_rejected']}")
        st.write(f"Approval Rate: {stats['approval_rate']:.1f}%")

    with col2:
        st.markdown("**Content**")
        st.write(f"Contributors: {stats['contributors_approved']}")
        st.write(f"Detractors: {stats['detractors_approved']}")
        st.write(f"Items Edited: {stats['items_edited']}")

    with col3:
        st.markdown("**Word Counts**")
        st.write(f"Average: {stats['avg_word_count']:.0f}")
        st.write(f"Min: {stats['min_word_count']}")
        st.write(f"Max: {stats['max_word_count']}")
        st.write(f"Total Cost: ${stats['total_cost_usd']:.4f}")
