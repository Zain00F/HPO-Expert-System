from experta import Fact, Field

class ModelArchitecture(Fact):
    architecture_type = Field(str, mandatory=True)
    num_layers = Field(int, mandatory=True)
    parameter_count = Field(int, mandatory=True)
    model_scale = Field(str, mandatory=True)
    uses_batch_norm = Field(bool, default=False)
    uses_attention = Field(bool, default=False)


class DatasetProfile(Fact):
    data_type = Field(str, mandatory=True)
    sample_count = Field(int, mandatory=True)
    dataset_noise = Field(str, mandatory=True)
    class_balance = Field(str, mandatory=True)
    num_classes = Field(int, default=None)
    classification_categories = Field(int, default=None)
