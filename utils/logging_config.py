"""Configure experta watchers for debugging rule activation."""

from __future__ import annotations

import logging


def setup_debug_logging(level: int = logging.INFO) -> None:
    """Enable RULES and ACTIVATIONS watchers (use for development only)."""
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")
    try:
        from experta import watch

        watch("RULES", "ACTIVATIONS")
    except ImportError:
        pass
