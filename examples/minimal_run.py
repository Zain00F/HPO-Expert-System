#!/usr/bin/env python
"""Minimal non-interactive demo of staged HPO expert reasoning."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hpo_expert  # noqa: F401 — Python 3.10+ experta compatibility

from hpo_expert.engines.hpo_engine import HPOExpertEngine, run_consultation
from hpo_expert.examples.diagnosis_scenarios import DIAGNOSIS_SCENARIOS
from hpo_expert.examples.scenarios import SCENARIOS
from hpo_expert.utils.explanations import format_report


def main() -> None:
    diagnosis_mode = "--diagnosis" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--diagnosis"]
    name = args[0] if args else ("overfitting_gap" if diagnosis_mode else "expensive_deep")

    if diagnosis_mode:
        scenarios = DIAGNOSIS_SCENARIOS
        title_prefix = "Post-Training Diagnosis"
    else:
        scenarios = SCENARIOS
        title_prefix = "HPO Expert"

    if name not in scenarios:
        print(f"Unknown scenario '{name}'. Choose from: {', '.join(scenarios)}")
        sys.exit(1)

    facts = scenarios[name]()
    engine = run_consultation(HPOExpertEngine(), facts)
    recs = engine.recommendations_as_dicts()

    print(format_report(recs, title=f"{title_prefix} — scenario: {name}"))
    print()
    print("--- Derived facts (audit trail) ---")
    skip = {
        "InitialFact",
        "Recommendation",
        "ProjectContext",
        "ComputeConstraints",
        "OptimizationBudget",
        "ModelArchitecture",
        "DatasetProfile",
        "ConsultationType",
        "TrainingObservation",
        "CurrentTrainingConfig",
    }
    for key, fact in sorted(engine.facts.items()):
        label = type(fact).__name__
        if label in skip:
            continue
        print(f"  {label}: {dict(fact)}")


if __name__ == "__main__":
    main()
