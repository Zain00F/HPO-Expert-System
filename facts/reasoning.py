from experta import Fact, Field

class ReasoningStage(Fact):
    current = Field(str, mandatory=True)


class HPOMethodCategoryChoice(Fact):
    category = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class HPOMethodChoice(Fact):
    method = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class OptimizerFamilyChoice(Fact):
    family = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class OptimizerChoice(Fact):

    optimizer = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class Diagnosis(Fact):
    issue = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class PossibleCause(Fact):

    diagnosis = Field(str, mandatory=True)
    cause = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class Recommendation(Fact):
    category = Field(str, mandatory=True)
    recommendation = Field(str, mandatory=True)
    justification = Field(str, mandatory=True)
    reasons = Field(list, default=list)
    priority = Field(str, mandatory=True)
    confidence = Field(str, default="medium")


class SearchSpaceChoice(Fact):
    mode = Field(str, mandatory=True)
    confidence = Field(str, mandatory=True)


class RangeChoice(Fact):
    parameter = Field(str, default="")
    strategy = Field(str, default="")
    confidence = Field(str, default="medium")
    status = Field(str, default="active")    