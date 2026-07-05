from experta import Fact, Field


class CurrentTrainingConfig(Fact):
    optimizer = Field(str, mandatory=True)
    learning_rate = Field(float, mandatory=True)
    batch_size = Field(int, mandatory=True)
    weight_decay = Field(float, default=0.0)
    dropout_rate = Field(float, default=0.0)
    scheduler = Field(str, default="none")
    gradient_clipping = Field(bool, default=False)


class TrainingObservation(Fact):
    """
    Raw post-training observations supplied by the client.
    The expert system infers diagnosis — the client never declares it.

    Accuracies are percentages in [0, 100] (e.g. 85.0 for 85%).
    See docs/KNOWLEDGE_ACQUISITION_DIAGNOSIS.md for evidence definitions.
    """

    training_loss = Field(float, mandatory=False, default=None)
    validation_loss = Field(float, mandatory=False, default=None)
    training_accuracy = Field(float, mandatory=False, default=None)
    validation_accuracy = Field(float, mandatory=False, default=None)
    nan_detected = Field(bool, default=False)
    inf_detected = Field(bool, default=False)
    epochs_completed = Field(int, mandatory=False, default=None)
