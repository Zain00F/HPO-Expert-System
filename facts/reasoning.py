
from experta import Fact, Field


class ReasoningStage(Fact):
    """
    current: init | hpo_method | optimizer_family | optimizer_specific | ...
    """
    current = Field(str, mandatory=True)


class HPOMethodCategoryChoice(Fact):
    """Stage 1a — coarse HPO family before naming a specific algorithm."""

    category = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class HPOMethodChoice(Fact):
    """Stage 1b — specific HPO algorithm recommendation."""

    method = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class OptimizerFamilyChoice(Fact):
    """Stage 2a — adaptive vs non-adaptive."""

    family = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class OptimizerChoice(Fact):
    """Stage 2b — concrete optimizer (Adam, SGD, …)."""

    optimizer = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class Diagnosis(Fact):
    """Identified training issue (expanded in future diagnosis rules)."""

    issue = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class Recommendation(Fact):
    """
    Explainable advice shown to the user.

    category: hpo_method | optimizer_family | optimizer | strategy | ...
    recommendation: human-readable label
    justification: one-line summary
    reasons: list of bullet strings (WHY)
    priority: low | medium | high
    confidence: low | medium | high
    """

    category = Field(str, mandatory=True)
    recommendation = Field(str, mandatory=True)
    justification = Field(str, mandatory=True)
    reasons = Field(list, default=list)
    priority = Field(str, mandatory=True)
    confidence = Field(str, default="medium")


class SearchSpaceChoice(Fact):
    """
    profile_mode: bn_safe | extreme_focus | accuracy_fanova | speed_fanova | transformer_balanced | cnn_high_risk | cnn_standard
    """
    mode = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class RangeChoice(Fact):
    """
    Stage 4 — hyperparameter range choices and stage completion tracking.
    
    parameter: all | optimizer_bound | batch_size | dropout
    strategy: frozen_bn_safe | frozen_extreme_focus | adamw_sequential | adamw_spatial | adam_standard | sgd_momentum | high_performance | resource_constrained | high_regularization | standard_regularization
    status: active | complete
    """
    parameter = Field(str, default="")
    strategy = Field(str, default="")
    confidence = Field(str, default="medium")
    status = Field(str, default="active")    