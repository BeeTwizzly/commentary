"""Additional export format utilities."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from io import StringIO
from typing import Any

from src.ui.review import ReviewSession


def export_to_csv(session: ReviewSession) -> str:
    """
    Export approved items to CSV format.

    Args:
        session: Review session with approved items

    Returns:
        CSV string
    """
    items = session.exportable_items

    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Ticker",
        "Company",
        "Type",
        "Effect (bps)",
        "Weight (%)",
        "Word Count",
        "Was Edited",
        "Commentary",
    ])

    # Data rows
    for item in items:
        writer.writerow([
            item.ticker,
            item.company_name,
            "Contributor" if item.is_contributor else "Detractor",
            f"{item.holding.total_attribution:+.1f}" if item.holding.total_attribution else "",
            f"{item.holding.avg_weight:.2f}" if item.holding.avg_weight else "",
            item.current_word_count,
            "Yes" if item.edited_text else "No",
            item.current_text,
        ])

    return output.getvalue()


def export_to_json(session: ReviewSession) -> str:
    """
    Export approved items to JSON format.

    Args:
        session: Review session with approved items

    Returns:
        JSON string
    """
    items = session.exportable_items

    data = {
        "metadata": {
            "strategy": session.strategy,
            "quarter": session.quarter,
            "exported_at": datetime.now().isoformat(),
            "total_items": len(items),
            "contributors": sum(1 for i in items if i.is_contributor),
            "detractors": sum(1 for i in items if not i.is_contributor),
        },
        "commentary": [
            {
                "ticker": item.ticker,
                "company_name": item.company_name,
                "type": "contributor" if item.is_contributor else "detractor",
                "effect_bps": item.holding.total_attribution,
                "weight_pct": item.holding.avg_weight,
                "word_count": item.current_word_count,
                "text": item.current_text,
                "was_edited": bool(item.edited_text),
                "selected_variation": item.selected_variation,
            }
            for item in items
        ],
    }

    return json.dumps(data, indent=2)


def export_summary_stats(session: ReviewSession) -> dict[str, Any]:
    """
    Generate summary statistics for export.

    Args:
        session: Review session

    Returns:
        Dictionary of statistics
    """
    items = session.exportable_items
    all_items = session.items

    if not items:
        return {"error": "No approved items"}

    word_counts = [i.current_word_count for i in items]

    return {
        "strategy": session.strategy,
        "quarter": session.quarter,
        "total_generated": len(all_items),
        "total_approved": len(items),
        "total_rejected": session.rejected_count,
        "approval_rate": len(items) / len(all_items) * 100 if all_items else 0,
        "contributors_approved": sum(1 for i in items if i.is_contributor),
        "detractors_approved": sum(1 for i in items if not i.is_contributor),
        "avg_word_count": sum(word_counts) / len(word_counts) if word_counts else 0,
        "min_word_count": min(word_counts) if word_counts else 0,
        "max_word_count": max(word_counts) if word_counts else 0,
        "items_edited": sum(1 for i in items if i.edited_text),
        "total_cost_usd": sum(i.result.cost_usd for i in all_items if i.result),
    }
