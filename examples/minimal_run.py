#!/usr/bin/env python
"""Minimal non-interactive demo of staged HPO expert reasoning."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without pip install (-m or direct script)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hpo_expert  # noqa: F401 — Python 3.10+ experta compatibility

from hpo_expert.engines.hpo_engine import HPOExpertEngine, run_consultation
from hpo_expert.examples.scenarios import SCENARIOS
from hpo_expert.utils.explanations import format_report


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "expensive_deep"
    if name not in SCENARIOS:
        print(f"Unknown scenario '{name}'. Choose from: {', '.join(SCENARIOS)}")
        sys.exit(1)

    facts = SCENARIOS[name]()
    engine = run_consultation(HPOExpertEngine(), facts)
    recs = engine.recommendations_as_dicts()

    print(format_report(recs, title=f"HPO Expert — scenario: {name}"))
    print()
    print("--- Derived facts (audit trail) ---")
    for key, fact in sorted(engine.facts.items()):
        label = type(fact).__name__
        if label in {
            "InitialFact",
            "Recommendation",
            "ProjectContext",
            "ComputeConstraints",
            "OptimizationBudget",
            "ModelArchitecture",
            "DatasetProfile",
        }:
            continue
        print(f"  {label}: {dict(fact)}")


if __name__ == "__main__":
    main()
