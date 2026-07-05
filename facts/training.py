from experta import Fact, Field

from hpo_expert.utils.heuristic_thresholds import HeuristicThresholds as HT


class CurrentTrainingConfig(Fact):
    optimizer = Field(str, mandatory=True)
    learning_rate = Field(float, mandatory=True)
    batch_size = Field(int, mandatory=True)
    weight_decay = Field(float, default=0.0)
    dropout_rate = Field(float, default=0.0)
    scheduler = Field(str, default="none")
    gradient_clipping = Field(bool, default=False)


class TrainingObservation(Fact):
    training_loss = Field(float, mandatory=False, default=HT.DEFAULT_TRAINING_LOSS)
    validation_loss = Field(float, mandatory=False, default=HT.DEFAULT_VALIDATION_LOSS)
    training_accuracy = Field(float, mandatory=False, default=HT.DEFAULT_TRAINING_ACCURACY)
    validation_accuracy = Field(float, mandatory=False, default=HT.DEFAULT_VALIDATION_ACCURACY)
    nan_detected = Field(bool, default=False)
    inf_detected = Field(bool, default=False)
    epochs_completed = Field(int, mandatory=False, default=HT.DEFAULT_EPOCHS_COMPLETED)
