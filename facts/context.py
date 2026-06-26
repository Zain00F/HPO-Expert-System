"""Project goals, compute limits, and optimization budget."""

from experta import Fact, Field


class ProjectContext(Fact):
    """
    High-level intent of the tuning exercise.

    optimization_goal: maximize_accuracy | minimize_training_time | balanced
    deployment_type: research | production | educational
    optimization_priority: low | medium | high
    """

    optimization_goal = Field(str, mandatory=True)
    deployment_type = Field(str, mandatory=True)
    optimization_priority = Field(str, mandatory=True)


class ComputeConstraints(Fact):
    """
    Hardware and time limits shaping feasible HPO strategies.

    dataset_size: small | medium | large
    compute_budget: low | medium | high
    has_gpu: whether GPU training is available
    gpu_memory_gb: optional GPU VRAM
    ram_gb: system RAM
    time_budget_hours: wall-clock budget for all trials
    search_space_dimensions: number of hyperparameters to tune
    trial_cost: low | medium | high (relative cost of one training run)
    """

    dataset_size = Field(str, mandatory=True)
    compute_budget = Field(str, mandatory=True)
    has_gpu = Field(bool, mandatory=True, default=True)
    gpu_memory_gb = Field(float, default=8.0)
    ram_gb = Field(float, default=16.0)
    time_budget_hours = Field(float, mandatory=True)
    search_space_dimensions = Field(int, mandatory=True)
    trial_cost = Field(str, mandatory=True)


class OptimizationBudget(Fact):
    """Explicit caps on search effort."""

    max_trials = Field(int, mandatory=True)
    early_stopping_enabled = Field(bool, default=True)
    max_total_runtime_hours = Field(float, mandatory=True)
