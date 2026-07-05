from experta import Fact, Field

class ConsultationType(Fact):
    mode = Field(str, mandatory=True)
