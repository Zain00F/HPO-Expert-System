
from experta import AS, NOT, OR, Rule

from hpo_expert.facts.consultation import ConsultationType
from hpo_expert.facts.reasoning import (
    Diagnosis,
    HPOMethodChoice,
    OptimizerChoice,
    OptimizerFamilyChoice,
    PossibleCause,
    ReasoningStage,
    SearchSpaceChoice
)
from hpo_expert.utils.enums import ConsultationTypeId, ReasoningStageId


class StageControlRules:
    @Rule(
        NOT(ReasoningStage()),
        OR(
            ConsultationType(mode=ConsultationTypeId.PRE_TRAINING.value),
            NOT(ConsultationType()),
        ),
        salience=100,
    )
    def begin_pre_training(self):
        self.declare(ReasoningStage(current=ReasoningStageId.HPO_METHOD.value))

    @Rule(
        NOT(ReasoningStage()),
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        salience=100,
    )
    def begin_post_training(self):
        self.declare(ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value))

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
        AS.stage << ReasoningStage(current=ReasoningStageId.OPTIMIZER_SPECIFIC.value),
        OptimizerChoice(),
        NOT(ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value)),
        salience=90,
    )
    def advance_to_search_space(self, stage):
        self.retract(stage)
        self.declare(ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value))

    @Rule(
        AS.stage << ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        Diagnosis(),
        NOT(ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value)),
        salience=50,
    )
    def advance_to_diagnosis_cause(self, stage):
        self.retract(stage)
        self.declare(ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value))

    @Rule(
        AS.stage << ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(),
        NOT(ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value)),
        salience=50,
    )
    def advance_to_diagnosis_recommend(self, stage):
        self.retract(stage)
        self.declare(ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value))


    @Rule(
        AS.stage << ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        SearchSpaceChoice(),
        NOT(ReasoningStage(current=ReasoningStageId.RANGES.value)),
        salience=85,
    )
    def advance_to_ranges(self, stage):
        self.retract(stage)
        self.declare(ReasoningStage(current=ReasoningStageId.RANGES.value))   
