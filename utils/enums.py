"""Canonical values for facts, rules, and CLI — single source of truth."""

from enum import Enum


class ReasoningStageId(str, Enum):
    INIT = "init"
    HPO_METHOD = "hpo_method"
    OPTIMIZER_FAMILY = "optimizer_family"
    OPTIMIZER_SPECIFIC = "optimizer_specific"
    SEARCH_SPACE = "search_space"
    RANGES = "ranges"
    DIAGNOSIS = "diagnosis"


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
