from experta import Fact, Field

class CurrentTrainingConfig(Fact):
    optimizer = Field(str, mandatory=True)
    learning_rate = Field(float, mandatory=True)
    batch_size = Field(int, mandatory=True)
    weight_decay = Field(float, default=0.0)
    dropout_rate = Field(float, default=0.0)
    scheduler = Field(str, default="none")
    gradient_clipping = Field(bool, default=False)


# class TrainingObservation(Fact):

#     training_diverged = Field(bool, default=False)
#     nan_detected = Field(bool, default=False)
#     overfitting_detected = Field(bool, default=False)
#     underfitting_detected = Field(bool, default=False)
#     oscillating_loss = Field(bool, default=False)
#     plateau_detected = Field(bool, default=False)
#     slow_convergence = Field(bool, default=False)
#     high_variance_training = Field(bool, default=False)
#     vanishing_gradients = Field(bool, default=False)
#     exploding_gradients = Field(bool, default=False)
