"""
Stage gating — ensures progressive narrowing.

Salience 100: bootstrap and transitions fire before content rules in the same cycle.
"""

from experta import AS, NOT, Rule

from hpo_expert.facts.context import OptimizationBudget
from hpo_expert.facts.reasoning import (
    HPOMethodChoice,
    OptimizerChoice,
    OptimizerFamilyChoice,
    ReasoningStage,
    SearchSpaceChoice
)
from hpo_expert.utils.enums import ReasoningStageId


class StageControlRules:
    """Mixin: attach to HPOEngine via multiple inheritance."""

    @Rule(NOT(ReasoningStage()), salience=100)
    def begin_hpo_method_stage(self):
        """Start consultation at HPO method selection."""
        self.declare(ReasoningStage(current=ReasoningStageId.HPO_METHOD.value))

    @Rule(
        AS.stage << ReasoningStage(current=ReasoningStageId.HPO_METHOD.value),
        HPOMethodChoice(),
        NOT(ReasoningStage(current=ReasoningStageId.OPTIMIZER_FAMILY.value)),
        salience=95,
    )
    def advance_to_optimizer_family(self, stage):
        self.retract(stage)
        self.declare(ReasoningStage(current=ReasoningStageId.OPTIMIZER_FAMILY.value))

    @Rule(
        AS.stage << ReasoningStage(current=ReasoningStageId.OPTIMIZER_FAMILY.value),
        OptimizerFamilyChoice(),
        NOT(ReasoningStage(current=ReasoningStageId.OPTIMIZER_SPECIFIC.value)),
        salience=95,
    )
    def advance_to_optimizer_specific(self, stage):
        self.retract(stage)
        self.declare(ReasoningStage(current=ReasoningStageId.OPTIMIZER_SPECIFIC.value))

    @Rule(
        ReasoningStage(current=ReasoningStageId.OPTIMIZER_SPECIFIC.value),
        OptimizerChoice(),
        salience=90,
    )
    def mark_optimizer_stage_complete(self):
        """Placeholder — future transition to search_space stage."""
        pass


    @Rule(
        AS.stage << ReasoningStage(current=ReasoningStageId.OPTIMIZER_SPECIFIC.value),
        OptimizerChoice(),
        NOT(ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value)),
        salience=90,
    )
    def advance_to_search_space(self, stage):
        self.retract(stage)
        self.declare(ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value))    


    @Rule(
        AS.stage << ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        SearchSpaceChoice(),
        NOT(ReasoningStage(current=ReasoningStageId.RANGES.value)),
        salience=85,
    )
    def advance_to_ranges(self, stage):
        """Transition from Search Space stage to Range Suggestion stage."""
        self.retract(stage)
        self.declare(ReasoningStage(current=ReasoningStageId.RANGES.value))   