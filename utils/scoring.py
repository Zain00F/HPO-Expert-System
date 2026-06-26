"""Map evidence strength to confidence labels for recommendations."""

from __future__ import annotations

from hpo_expert.utils.enums import Confidence


def confidence_from_score(score: int, max_score: int = 6) -> str:
    """
    Convert an integer evidence score to low / medium / high.

    Rules increment score when multiple independent constraints align.
    """
    ratio = score / max(max_score, 1)
    if ratio >= 0.75:
        return Confidence.HIGH.value
    if ratio >= 0.45:
        return Confidence.MEDIUM.value
    return Confidence.LOW.value
