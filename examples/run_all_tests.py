#!/usr/bin/env python
"""
Run all pre-training and diagnosis scenarios with assertions.

Usage (from repo root):
  python hpo_expert/examples/run_all_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hpo_expert  # noqa: F401

from hpo_expert.engines.hpo_engine import HPOExpertEngine, run_consultation
from hpo_expert.examples.diagnosis_scenarios import DIAGNOSIS_SCENARIOS
from hpo_expert.examples.scenarios import SCENARIOS
from hpo_expert.facts.consultation import ConsultationType
from hpo_expert.facts.context import ComputeConstraints, OptimizationBudget, ProjectContext
from hpo_expert.facts.model import DatasetProfile, ModelArchitecture
from hpo_expert.utils.enums import ConsultationTypeId


def _facts_by_type(engine) -> dict[str, list]:
    out: dict[str, list] = {}
    for fact in engine.facts.values():
        name = type(fact).__name__
        out.setdefault(name, []).append(dict(fact))
    return out


def _rec_categories(engine) -> dict[str, list[str]]:
    cats: dict[str, list[str]] = {}
    for fact in engine.facts.values():
        if type(fact).__name__ == "Recommendation":
            cat = fact["category"]
            cats.setdefault(cat, []).append(fact["recommendation"])
    return cats


def _stages(engine) -> list[str]:
    return [f["current"] for f in _facts_by_type(engine).get("ReasoningStage", [])]


def _run_case(name: str, facts: list, checks: dict) -> list[str]:
    errors: list[str] = []
    engine = run_consultation(HPOExpertEngine(), facts)
    by_type = _facts_by_type(engine)
    recs = _rec_categories(engine)

    for fact_type, predicate in checks.get("must_have_facts", {}).items():
        items = by_type.get(fact_type, [])
        if not items:
            errors.append(f"{name}: missing fact {fact_type}")
        elif callable(predicate) and not predicate(items):
            errors.append(f"{name}: fact {fact_type} failed check: {items}")

    for fact_type in checks.get("must_not_have_facts", []):
        if fact_type in by_type:
            errors.append(f"{name}: unexpected fact {fact_type}: {by_type[fact_type]}")

    for cat, expected in checks.get("recommendations", {}).items():
        actual = recs.get(cat, [])
        for label in expected:
            if label not in actual:
                errors.append(f"{name}: missing {cat} recommendation '{label}' (got {actual})")

    for cat in checks.get("must_not_have_recommendations", []):
        if cat in recs:
            errors.append(f"{name}: unexpected {cat} recommendations: {recs[cat]}")

    for forbidden_stage in checks.get("must_not_reach_stages", []):
        if forbidden_stage in _stages(engine):
            errors.append(f"{name}: reached forbidden stage '{forbidden_stage}' (stages={_stages(engine)})")

    final_stage = checks.get("final_stage")
    if final_stage and final_stage not in _stages(engine):
        errors.append(f"{name}: expected final stage '{final_stage}', got {_stages(engine)}")

    if checks.get("require_range_suggestions") and not recs.get("range_suggestion"):
        errors.append(f"{name}: expected range_suggestion recommendations, got none")

    return errors


PRE_TRAINING_CHECKS = {
    "small_fast": {
        "must_have_facts": {
            "DatasetSize": lambda items: items[0].get("size") == "small",
            "TrialCost": lambda items: items[0].get("level") == "low",
            "ComputeBudget": lambda items: items[0].get("level") == "high",
            "HPOMethodChoice": lambda items: items[0].get("method") == "grid_search",
            "RangeChoice": lambda items: len(items) >= 1,
        },
        "recommendations": {
            "hpo_method": ["Grid Search"],
            "optimizer_specific": ["Adam"],
            "search_space": ["Extreme Focus Mode"],
            "range_suggestion": ["Adam Standard Ranges"],
        },
        "final_stage": "ranges",
        "require_range_suggestions": True,
    },
    "expensive_deep": {
        "must_have_facts": {
            "DatasetSize": lambda items: items[0].get("size") == "large",
            "TrialCost": lambda items: items[0].get("level") == "high",
            "ComputeBudget": lambda items: items[0].get("level") == "high",
            "HPOMethodChoice": lambda items: items[0].get("method") == "bayesian_optimization",
        },
        "recommendations": {
            "hpo_method": ["Bayesian Optimization"],
            "optimizer_specific": ["AdamW (Decoupled Weight Decay)"],
            "advanced_strategy(for hpo method)": ["Two-Stage Hybrid Tuning"],
            "range_suggestion": ["AdamW Range (Sequential Context)"],
        },
        "final_stage": "ranges",
        "require_range_suggestions": True,
    },
    "tight_budget": {
        "must_have_facts": {
            "TrialCost": lambda items: items[0].get("level") == "high",
            "ComputeBudget": lambda items: items[0].get("level") == "low",
            "HPOMethodChoice": lambda items: items[0].get("method") == "hyperband",
        },
        "recommendations": {
            "hpo_method": ["Hyperband"],
            "optimizer_specific": ["Adam"],
            "search_space": ["Spatial Standard Mode"],
            "range_suggestion": ["Adam Standard Ranges"],
        },
        "final_stage": "ranges",
        "require_range_suggestions": True,
    },
}

DIAGNOSIS_CHECKS = {
    "overfitting_gap": {
        "must_have_facts": {
            "Diagnosis": lambda items: items[0].get("issue") == "overfitting",
            "DatasetSize": lambda items: items[0].get("size") == "small",
            "PossibleCause": lambda items: len(items) >= 1,
        },
        "must_not_have_facts": ["SearchSpaceChoice", "RangeChoice"],
        "recommendations": {
            "diagnosis": ["Overfitting"],
            "diagnosis_recommendation": ["Increase dropout"],
        },
        "must_not_have_recommendations": ["search_space", "range_suggestion"],
        "must_not_reach_stages": ["search_space", "ranges"],
        "final_stage": "diagnosis_recommend",
    },
    "underfitting_high_loss": {
        "must_have_facts": {
            "Diagnosis": lambda items: items[0].get("issue") == "underfitting",
        },
        "must_not_have_facts": ["SearchSpaceChoice"],
        "recommendations": {
            "diagnosis": ["Underfitting"],
            "diagnosis_recommendation": ["Increase model capacity"],
        },
        "must_not_have_recommendations": ["search_space"],
        "final_stage": "diagnosis_recommend",
    },
    "nan_loss": {
        "must_have_facts": {
            "Diagnosis": lambda items: items[0].get("issue") == "nan_loss",
        },
        "recommendations": {
            "diagnosis": ["NaN Loss"],
            "diagnosis_recommendation": ["Lower learning rate"],
        },
        "must_not_have_recommendations": ["search_space", "hpo_method"],
        "final_stage": "diagnosis_recommend",
    },
    "inf_loss": {
        "must_have_facts": {
            "Diagnosis": lambda items: items[0].get("issue") == "inf_loss",
        },
        "recommendations": {
            "diagnosis": ["Infinite (Inf) Loss"],
        },
        "must_not_have_recommendations": ["search_space", "hpo_method"],
        "final_stage": "diagnosis_recommend",
    },
}


def _edge_derived_dataset_per_class() -> list[str]:
    """50 samples/class → small (classification path)."""
    facts = [
        ConsultationType(mode=ConsultationTypeId.PRE_TRAINING.value),
        ProjectContext(optimization_goal="balanced", deployment_type="research", optimization_priority="medium"),
        ComputeConstraints(has_gpu=True, time_budget_hours=24.0, search_space_dimensions=3, trial_time_minutes=10.0),
        OptimizationBudget(max_trials=10, max_total_runtime_hours=24.0),
        ModelArchitecture(architecture_type="cnn", num_layers=4, parameter_count=100_000, model_scale="small"),
        DatasetProfile(data_type="image", sample_count=500, dataset_noise="low", class_balance="balanced", classification_categories=10),
    ]
    engine = run_consultation(HPOExpertEngine(), facts)
    by_type = _facts_by_type(engine)
    size = by_type.get("DatasetSize", [{}])[0].get("size")
    if size != "small":
        return [f"edge_per_class: expected DatasetSize=small, got {size}"]
    return []


def _edge_compute_budget_medium() -> list[str]:
    """Required time ~90% of available → medium compute budget."""
    facts = [
        ConsultationType(mode=ConsultationTypeId.PRE_TRAINING.value),
        ProjectContext(optimization_goal="balanced", deployment_type="research", optimization_priority="medium"),
        ComputeConstraints(has_gpu=True, time_budget_hours=10.0, search_space_dimensions=3, trial_time_minutes=30.0),
        OptimizationBudget(max_trials=18, max_total_runtime_hours=10.0),
        ModelArchitecture(architecture_type="mlp", num_layers=3, parameter_count=50_000, model_scale="small"),
        DatasetProfile(data_type="tabular", sample_count=20_000, dataset_noise="low", class_balance="balanced"),
    ]
    engine = run_consultation(HPOExpertEngine(), facts)
    by_type = _facts_by_type(engine)
    level = by_type.get("ComputeBudget", [{}])[0].get("level")
    required = by_type.get("RequiredSearchTime", [{}])[0].get("hours")
    if level != "medium":
        return [f"edge_compute_medium: expected ComputeBudget=medium (required={required}h), got {level}"]
    return []


EDGE_CASES = [
    ("derived_dataset_per_class", _edge_derived_dataset_per_class),
    ("derived_compute_budget_medium", _edge_compute_budget_medium),
]


def main() -> int:
    all_errors: list[str] = []

    print("=== Pre-Training Scenarios ===")
    for name, builder in SCENARIOS.items():
        checks = PRE_TRAINING_CHECKS.get(name, {})
        errors = _run_case(name, builder(), checks)
        status = "PASS" if not errors else "FAIL"
        print(f"  [{status}] {name}")
        all_errors.extend(errors)

    print("\n=== Diagnosis Scenarios ===")
    for name, builder in DIAGNOSIS_SCENARIOS.items():
        checks = DIAGNOSIS_CHECKS.get(name, {})
        errors = _run_case(name, builder(), checks)
        status = "PASS" if not errors else "FAIL"
        print(f"  [{status}] {name}")
        all_errors.extend(errors)

    print("\n=== Edge Cases ===")
    for name, fn in EDGE_CASES:
        errors = fn()
        status = "PASS" if not errors else "FAIL"
        print(f"  [{status}] {name}")
        all_errors.extend(errors)

    print()
    if all_errors:
        print(f"FAILED — {len(all_errors)} issue(s):\n")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    total = len(SCENARIOS) + len(DIAGNOSIS_SCENARIOS) + len(EDGE_CASES)
    print(f"ALL {total} SCENARIOS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
