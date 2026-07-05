from experta import Fact, Field

class ProjectContext(Fact):
    optimization_goal = Field(str, mandatory=True)
    deployment_type = Field(str, mandatory=True)
    optimization_priority = Field(str, mandatory=True)


class ComputeConstraints(Fact):
    dataset_size = Field(str, mandatory=True)
    compute_budget = Field(str, mandatory=True)
    has_gpu = Field(bool, mandatory=True, default=True)
    gpu_memory_gb = Field(float, default=8.0)
    ram_gb = Field(float, default=16.0)
    time_budget_hours = Field(float, mandatory=True)
    search_space_dimensions = Field(int, mandatory=True)
    trial_cost = Field(str, mandatory=True)


class OptimizationBudget(Fact):
    max_trials = Field(int, mandatory=True)
    early_stopping_enabled = Field(bool, default=True)
    max_total_runtime_hours = Field(float, mandatory=True)
