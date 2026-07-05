from enum import Enum


class ReasoningStageId(str, Enum):
    INIT = "init"
    HPO_METHOD = "hpo_method"
    OPTIMIZER_FAMILY = "optimizer_family"
    OPTIMIZER_SPECIFIC = "optimizer_specific"
    SEARCH_SPACE = "search_space"
    RANGES = "ranges"
    DIAGNOSIS = "diagnosis"
    DIAGNOSIS_IDENTIFY = "diagnosis_identify"
    DIAGNOSIS_CAUSE = "diagnosis_cause"
    DIAGNOSIS_RECOMMEND = "diagnosis_recommend"


class ConsultationTypeId(str, Enum):
    PRE_TRAINING = "pre_training"
    POST_TRAINING = "post_training"


class DiagnosisIssue(str, Enum):
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    NAN_LOSS = "nan_loss"
    INF_LOSS = "inf_loss"


class DiagnosisCause(str, Enum):
    MODEL_TOO_COMPLEX = "model_too_complex"
    INSUFFICIENT_DATA = "insufficient_data"
    REGULARIZATION_TOO_WEAK = "regularization_too_weak"
    TRAINED_TOO_LONG = "trained_too_long"
    MODEL_TOO_SIMPLE = "model_too_simple"
    LEARNING_RATE_TOO_LOW = "learning_rate_too_low"
    TRAINING_STOPPED_EARLY = "training_stopped_early"
    EXCESSIVE_REGULARIZATION = "excessive_regularization"
    LEARNING_RATE_TOO_HIGH = "learning_rate_too_high"
    NUMERICAL_INSTABILITY = "numerical_instability"
    GRADIENT_EXPLOSION = "gradient_explosion"
    INVALID_INPUT_VALUES = "invalid_input_values"
    BAD_INITIALIZATION = "bad_initialization"
    OVERFLOW = "overflow"
    DIVERGING_GRADIENTS = "diverging_gradients"
    INVALID_LOSS_COMPUTATION = "invalid_loss_computation"
    GENERAL_OVERFITTING = "general_overfitting"
    GENERAL_UNDERFITTING = "general_underfitting"
    GENERAL_NAN_LOSS = "general_nan_loss"
    GENERAL_INF_LOSS = "general_inf_loss"


class HPOMethodCategory(str, Enum):
    EXHAUSTIVE = "exhaustive"  # grid
    SAMPLING = "sampling"  # random
    MODEL_BASED = "model_based"  # bayesian
    MULTIFIDELITY = "multifidelity"  # hyperband
    POPULATION = "population"  # PBT


class HPOMethod(str, Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    HYPERBAND = "hyperband"
    POPULATION_BASED_TRAINING = "population_based_training"


class OptimizerFamily(str, Enum):
    ADAPTIVE = "adaptive"
    NON_ADAPTIVE = "non_adaptive"


class Optimizer(str, Enum):
    SGD = "sgd"
    MOMENTUM_SGD = "momentum_sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DatasetSize(str, Enum):
    SMALL = "small"  # < 10k samples
    MEDIUM = "medium"
    LARGE = "large"


class ComputeBudget(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ArchitectureType(str, Enum):
    MLP = "mlp"
    CNN = "cnn"
    RNN = "rnn"
    TRANSFORMER = "transformer"
    LLM = "llm"
