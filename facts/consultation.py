"""Consultation mode routing."""

from experta import Fact, Field


class ConsultationType(Fact):
    """
    Selects which reasoning branch executes.

    mode: pre_training | post_training
    """

    mode = Field(str, mandatory=True)
