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
    training_loss = Field(float, mandatory=False, default=None)
    validation_loss = Field(float, mandatory=False, default=None)
    training_accuracy = Field(float, mandatory=False, default=None)
    validation_accuracy = Field(float, mandatory=False, default=None)
    nan_detected = Field(bool, default=False)
    inf_detected = Field(bool, default=False)
    epochs_completed = Field(int, mandatory=False, default=None)
