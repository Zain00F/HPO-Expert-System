from __future__ import annotations

from hpo_expert.facts.context import ComputeConstraints, OptimizationBudget, ProjectContext
from hpo_expert.facts.model import DatasetProfile, ModelArchitecture


def scenario_small_fast() -> list:
    return [
        ProjectContext(
            optimization_goal="balanced",
            deployment_type="educational",
            optimization_priority="medium",
        ),
        ComputeConstraints(
            has_gpu=False,
            time_budget_hours=8.0,
            search_space_dimensions=3,
            trial_time_minutes=3.0,
        ),
        OptimizationBudget(max_trials=20, max_total_runtime_hours=8.0),
        ModelArchitecture(
            architecture_type="mlp",
            num_layers=3,
            parameter_count=50_000,
            model_scale="small",
        ),
        DatasetProfile(
            data_type="tabular",
            sample_count=5_000,
            dataset_noise="low",
            class_balance="balanced",
        ),
    ]


def scenario_expensive_deep() -> list:
    return [
        ProjectContext(
            optimization_goal="maximize_accuracy",
            deployment_type="research",
            optimization_priority="high",
        ),
        ComputeConstraints(
            has_gpu=True,
            gpu_memory_gb=24.0,
            time_budget_hours=48.0,
            search_space_dimensions=8,
            trial_time_minutes=45.0,
        ),
        OptimizationBudget(max_trials=50, max_total_runtime_hours=48.0),
        ModelArchitecture(
            architecture_type="transformer",
            num_layers=12,
            parameter_count=25_000_000,
            model_scale="large",
            uses_attention=True,
        ),
        DatasetProfile(
            data_type="text",
            sample_count=500_000,
            dataset_noise="medium",
            class_balance="balanced",
        ),
    ]


def scenario_tight_budget() -> list:
    return [
        ProjectContext(
            optimization_goal="minimize_training_time",
            deployment_type="production",
            optimization_priority="high",
        ),
        ComputeConstraints(
            has_gpu=True,
            time_budget_hours=12.0,
            search_space_dimensions=5,
            trial_time_minutes=45.0,
        ),
        OptimizationBudget(max_trials=25, max_total_runtime_hours=12.0),
        ModelArchitecture(
            architecture_type="cnn",
            num_layers=8,
            parameter_count=2_000_000,
            model_scale="medium",
            uses_batch_norm=True,
        ),
        DatasetProfile(
            data_type="image",
            sample_count=40_000,
            dataset_noise="low",
            class_balance="balanced",
            classification_categories=10,
        ),
    ]


SCENARIOS = {
    "small_fast": scenario_small_fast,
    "expensive_deep": scenario_expensive_deep,
    "tight_budget": scenario_tight_budget,
}
