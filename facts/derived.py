"""Derived expert knowledge inferred from objective measurements."""

from experta import Fact, Field


class DatasetSize(Fact):
    """Inferred dataset scale from sample count and class count."""

    size = Field(str, mandatory=True)


class TrialCost(Fact):
    """Inferred relative cost of one HPO trial from average training duration."""

    level = Field(str, mandatory=True)


class RequiredSearchTime(Fact):
    """Total wall-clock hours needed to run all planned trials."""

    hours = Field(float, mandatory=True)


class ComputeBudget(Fact):
    """Inferred HPO affordability from required vs available search time."""

    level = Field(str, mandatory=True)
