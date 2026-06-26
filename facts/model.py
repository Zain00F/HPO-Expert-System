"""Model architecture and dataset characteristics."""

from experta import Fact, Field


class ModelArchitecture(Fact):
    """
    Neural network structure influencing optimizer and search choices.

    architecture_type: mlp | cnn | rnn | transformer | llm
    num_layers: depth of the network
    parameter_count: approximate trainable parameters
    model_scale: small | medium | large
    uses_batch_norm: batch normalization present
    uses_attention: attention-based blocks present
    """

    architecture_type = Field(str, mandatory=True)
    num_layers = Field(int, mandatory=True)
    parameter_count = Field(int, mandatory=True)
    model_scale = Field(str, mandatory=True)
    uses_batch_norm = Field(bool, default=False)
    uses_attention = Field(bool, default=False)


class DatasetProfile(Fact):
    """
    Data properties affecting regularization and search complexity.

    data_type: image | text | tabular | audio | multimodal
    sample_count: number of training samples
    num_classes: classification classes (optional)
    dataset_noise: low | medium | high
    class_balance: balanced | imbalanced
    """

    data_type = Field(str, mandatory=True)
    sample_count = Field(int, mandatory=True)
    dataset_noise = Field(str, mandatory=True)
    class_balance = Field(str, mandatory=True)
    num_classes = Field(int, default=None)
