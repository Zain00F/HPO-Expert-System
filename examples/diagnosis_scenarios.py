from __future__ import annotations

from hpo_expert.facts.consultation import ConsultationType
from hpo_expert.facts.context import ComputeConstraints
from hpo_expert.facts.model import ModelArchitecture
from hpo_expert.facts.training import CurrentTrainingConfig, TrainingObservation
from hpo_expert.utils.enums import ConsultationTypeId


def scenario_overfitting_gap() -> list:
    return [
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        TrainingObservation(training_loss=0.2, validation_loss=0.9),
        CurrentTrainingConfig(
            optimizer="adam",
            learning_rate=0.001,
            batch_size=32,
            dropout_rate=0.0,
            weight_decay=0.0,
        ),
        ModelArchitecture(
            architecture_type="cnn",
            num_layers=8,
            parameter_count=2_000_000,
            model_scale="large",
        ),
        ComputeConstraints(
            dataset_size="small",
            compute_budget="medium",
            has_gpu=True,
            time_budget_hours=24.0,
            search_space_dimensions=5,
            trial_cost="medium",
        ),
    ]


def scenario_underfitting_high_loss() -> list:
    return [
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        TrainingObservation(
            training_loss=2.5,
            validation_loss=2.4,
            training_accuracy=40.0,
            validation_accuracy=38.0,
            epochs_completed=5,
        ),
        CurrentTrainingConfig(
            optimizer="adam",
            learning_rate=1e-5,
            batch_size=32,
            dropout_rate=0.6,
            weight_decay=0.2,
        ),
        ModelArchitecture(
            architecture_type="mlp",
            num_layers=2,
            parameter_count=50_000,
            model_scale="small",
        ),
    ]


def scenario_nan_loss() -> list:
    return [
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        TrainingObservation(nan_detected=True),
        CurrentTrainingConfig(
            optimizer="sgd",
            learning_rate=0.1,
            batch_size=64,
            gradient_clipping=False,
        ),
    ]


def scenario_inf_loss() -> list:
    return [
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        TrainingObservation(inf_detected=True),
        CurrentTrainingConfig(
            optimizer="sgd",
            learning_rate=0.01,
            batch_size=64,
            gradient_clipping=False,
        ),
    ]


DIAGNOSIS_SCENARIOS = {
    "overfitting_gap": scenario_overfitting_gap,
    "underfitting_high_loss": scenario_underfitting_high_loss,
    "nan_loss": scenario_nan_loss,
    "inf_loss": scenario_inf_loss,
}
