#!/usr/bin/env python
"""
Interactive CLI — questionnaire converts answers into Facts.

Run from repo root:
  python -m hpo_expert.cli.interactive
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hpo_expert  # noqa: F401

from hpo_expert.engines.hpo_engine import HPOExpertEngine, run_consultation
from hpo_expert.facts.context import ComputeConstraints, OptimizationBudget, ProjectContext
from hpo_expert.facts.model import DatasetProfile, ModelArchitecture
from hpo_expert.utils.explanations import format_report


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def _ask_int(prompt: str, default: int) -> int:
    raw = _ask(prompt, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _ask_float(prompt: str, default: float) -> float:
    raw = _ask(prompt, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _ask_bool(prompt: str, default: bool = True) -> bool:
    default_s = "y" if default else "n"
    raw = _ask(f"{prompt} (y/n)", default_s).lower()
    return raw in {"y", "yes", "1", "true"}


def gather_facts() -> list:
    print("\n=== HPO Expert Consultant ===\n")
    print("Answer a few questions. The system will narrow choices in stages.\n")

    goal = _ask("Optimization goal (maximize_accuracy / minimize_training_time / balanced)", "balanced")
    deploy = _ask("Deployment type (research / production / educational)", "educational")
    priority = _ask("Optimization priority (low / medium / high)", "medium")

    dataset_size = _ask("Dataset size (small / medium / large)", "medium")
    compute_budget = _ask("Compute budget (low / medium / high)", "medium")
    has_gpu = _ask_bool("GPU available?", True)
    time_hours = _ask_float("Time budget (hours)", 24.0)
    dimensions = _ask_int("Number of hyperparameters to tune", 5)
    trial_cost = _ask("Relative trial cost (low / medium / high)", "medium")
    max_trials = _ask_int("Maximum trials", 30)

    arch = _ask("Architecture (mlp / cnn / rnn / transformer / llm)", "cnn")
    layers = _ask_int("Number of layers", 6)
    params = _ask_int("Approximate parameter count", 1_000_000)
    scale = _ask("Model scale (small / medium / large)", "medium")
    batch_norm = _ask_bool("Uses batch normalization?", False)
    attention = _ask_bool("Uses attention blocks?", False)

    data_type = _ask("Data type (image / text / tabular / audio)", "image")
    samples = _ask_int("Training sample count", 50_000)
    noise = _ask("Dataset noise (low / medium / high)", "low")
    balance = _ask("Class balance (balanced / imbalanced)", "balanced")

    return [
        ProjectContext(
            optimization_goal=goal,
            deployment_type=deploy,
            optimization_priority=priority,
        ),
        ComputeConstraints(
            dataset_size=dataset_size,
            compute_budget=compute_budget,
            has_gpu=has_gpu,
            time_budget_hours=time_hours,
            search_space_dimensions=dimensions,
            trial_cost=trial_cost,
        ),
        OptimizationBudget(
            max_trials=max_trials,
            max_total_runtime_hours=time_hours,
        ),
        ModelArchitecture(
            architecture_type=arch,
            num_layers=layers,
            parameter_count=params,
            model_scale=scale,
            uses_batch_norm=batch_norm,
            uses_attention=attention,
        ),
        DatasetProfile(
            data_type=data_type,
            sample_count=samples,
            dataset_noise=noise,
            class_balance=balance,
        ),
    ]


def main() -> None:
    facts = gather_facts()
    engine = run_consultation(HPOExpertEngine(), facts)
    print("\n" + format_report(engine.recommendations_as_dicts()))
    print("\n(Derived audit facts available via engine.facts in programmatic use.)\n")


if __name__ == "__main__":
    main()
