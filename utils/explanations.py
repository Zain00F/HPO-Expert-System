"""Human-readable formatting for Recommendation facts."""

from __future__ import annotations

from typing import Any


def format_recommendation(rec: Any) -> str:
    """Format a single Recommendation fact."""
    lines = [
        f"[{rec['priority'].upper()}] {rec['category']}: {rec['recommendation']}",
        f"  Confidence: {rec.get('confidence', 'medium')}",
        f"  Summary: {rec['justification']}",
    ]
    reasons = rec.get("reasons") or []
    if reasons:
        lines.append("  Because:")
        for r in reasons:
            lines.append(f"    - {r}")
    return "\n".join(lines)


def format_report(recommendations: list[Any], title: str = "HPO Expert Report") -> str:
    """Build a full text report from collected recommendations."""
    if not recommendations:
        return f"{title}\n\n(No recommendations produced — declare more input facts.)"

    by_category: dict[str, list[Any]] = {}
    for rec in recommendations:
        by_category.setdefault(rec["category"], []).append(rec)

    parts = [title, "=" * len(title), ""]
    for category in sorted(by_category):
        parts.append(f"## {category.replace('_', ' ').title()}")
        parts.append("")
        for rec in by_category[category]:
            parts.append(format_recommendation(rec))
            parts.append("")
    return "\n".join(parts).rstrip()
