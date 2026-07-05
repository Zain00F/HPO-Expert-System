
from experta import MATCH, NOT, TEST, Rule

from hpo_expert.facts.consultation import ConsultationType
from hpo_expert.facts.derived import DatasetSize
from hpo_expert.facts.model import ModelArchitecture
from hpo_expert.facts.reasoning import Diagnosis, PossibleCause, ReasoningStage, Recommendation
from hpo_expert.facts.training import CurrentTrainingConfig, TrainingObservation
from hpo_expert.utils.enums import (
    ConsultationTypeId,
    DiagnosisCause,
    DiagnosisIssue,
    Priority,
    ReasoningStageId,
)
from hpo_expert.utils.heuristic_thresholds import HeuristicThresholds as HT
from hpo_expert.utils.scoring import confidence_from_score


class DiagnosisRules:
    @Rule(
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        TrainingObservation(nan_detected=True),
        NOT(Diagnosis()),
        salience=90,
    )
    def diagnose_nan_loss(self):
        self.declare(
            Diagnosis(
                issue=DiagnosisIssue.NAN_LOSS.value,
                confidence=confidence_from_score(5),
            )
        )

    @Rule(
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        TrainingObservation(inf_detected=True),
        NOT(Diagnosis()),
        salience=89,
    )
    def diagnose_inf_loss(self):
        self.declare(
            Diagnosis(
                issue=DiagnosisIssue.INF_LOSS.value,
                confidence=confidence_from_score(5),
            )
        )

    @Rule(
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        TrainingObservation(
            training_loss=MATCH.tl,
            validation_loss=MATCH.vl,
        ),
        TEST(
            lambda tl, vl: (
                tl is not None
                and vl is not None
                and vl > tl * HT.MAX_HEALTHY_LOSS_RATIO
            )
        ),
        NOT(Diagnosis()),
        salience=70,
    )
    def diagnose_overfitting_loss_gap(self):
        self.declare(
            Diagnosis(
                issue=DiagnosisIssue.OVERFITTING.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        TrainingObservation(
            training_accuracy=MATCH.ta,
            validation_accuracy=MATCH.va,
        ),
        TEST(
            lambda ta, va: (
                ta is not None
                and va is not None
                and (ta - va) >= HT.MIN_OVERFIT_ACC_GAP
            )
        ),
        NOT(Diagnosis()),
        salience=69,
    )
    def diagnose_overfitting_acc_gap(self):
        self.declare(
            Diagnosis(
                issue=DiagnosisIssue.OVERFITTING.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        TrainingObservation(
            training_loss=MATCH.tl,
            validation_loss=MATCH.vl,
        ),
        TEST(
            lambda tl, vl: (
                tl is not None
                and vl is not None
                and tl >= HT.MIN_POOR_TRAINING_LOSS
                and abs(vl - tl) / max(tl, 1e-9) <= HT.MAX_UNDERFIT_LOSS_REL_GAP
            )
        ),
        NOT(Diagnosis()),
        salience=60,
    )
    def diagnose_underfitting_high_loss(self):
        self.declare(
            Diagnosis(
                issue=DiagnosisIssue.UNDERFITTING.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        TrainingObservation(
            training_accuracy=MATCH.ta,
            validation_accuracy=MATCH.va,
        ),
        TEST(
            lambda ta, va: (
                ta is not None
                and va is not None
                and ta < HT.POOR_ACCURACY_CEILING
                and va < HT.POOR_ACCURACY_CEILING
                and abs(ta - va) <= HT.MAX_UNDERFIT_ACC_GAP
            )
        ),
        NOT(Diagnosis()),
        salience=59,
    )
    def diagnose_underfitting_low_accuracy(self):
        self.declare(
            Diagnosis(
                issue=DiagnosisIssue.UNDERFITTING.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ConsultationType(mode=ConsultationTypeId.POST_TRAINING.value),
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        NOT(Diagnosis()),
        salience=10,
    )
    def diagnose_healthy_training(self):
        self.declare(
            Diagnosis(
                issue=DiagnosisIssue.HEALTHY_TRAINING.value,
                confidence=confidence_from_score(3),
            )
        )

    # ------------------------------------------------------------------
    # Stage: diagnosis_identify — explain inferred Diagnosis
    # ------------------------------------------------------------------

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        Diagnosis(issue=DiagnosisIssue.NAN_LOSS.value),
        NOT(Recommendation(category="diagnosis")),
        salience=96,
    )
    def explain_diagnosis_nan(self):
        self.declare(
            Recommendation(
                category="diagnosis",
                recommendation="NaN Loss",
                justification="Training produced non-finite (NaN) loss values.",
                reasons=[
                    "NaN loss indicates numerical failure during forward or backward pass",
                    "Common triggers include excessive learning rate or unstable gradients",
                ],
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        Diagnosis(issue=DiagnosisIssue.INF_LOSS.value),
        NOT(Recommendation(category="diagnosis")),
        salience=96,
    )
    def explain_diagnosis_inf(self):
        self.declare(
            Recommendation(
                category="diagnosis",
                recommendation="Infinite (Inf) Loss",
                justification="Training produced infinite loss values.",
                reasons=[
                    "Inf loss typically signals overflow or diverging optimization",
                    "Verify loss computation and gradient magnitudes",
                ],
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        Diagnosis(issue=DiagnosisIssue.OVERFITTING.value),
        NOT(Recommendation(category="diagnosis")),
        salience=96,
    )
    def explain_diagnosis_overfitting(self):
        self.declare(
            Recommendation(
                category="diagnosis",
                recommendation="Overfitting",
                justification="Generalization gap: model fits training data better than validation data.",
                reasons=[
                    "Validation metric is significantly worse than training (generalization gap)",
                    "Pattern consistent with memorizing training-specific features",
                    "Same signal early stopping monitors on learning curves (Prechelt)",
                ],
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        Diagnosis(issue=DiagnosisIssue.UNDERFITTING.value),
        NOT(Recommendation(category="diagnosis")),
        salience=96,
    )
    def explain_diagnosis_underfitting(self):
        self.declare(
            Recommendation(
                category="diagnosis",
                recommendation="Underfitting",
                justification="Model performs poorly on both training and validation with little gap.",
                reasons=[
                    "High loss (or low accuracy) on both sets with similar values",
                    "Little generalization gap — model has not learned the signal adequately",
                    "Consistent with insufficient capacity or optimization (Goodfellow et al.)",
                ],
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_IDENTIFY.value),
        Diagnosis(issue=DiagnosisIssue.HEALTHY_TRAINING.value),
        NOT(Recommendation(category="diagnosis")),
        salience=96,
    )
    def explain_diagnosis_healthy(self):
        self.declare(
            Recommendation(
                category="diagnosis",
                recommendation="Healthy Training",
                justification=(
                    "No overfitting, underfitting, or numerical failure detected "
                    "from the supplied observations."
                ),
                reasons=[
                    "Training and validation losses show a modest generalization gap",
                    "Accuracy gap is within normal bounds",
                    "No NaN or Inf loss reported",
                ],
                priority=Priority.MEDIUM.value,
                confidence=confidence_from_score(3),
            )
        )

    # ------------------------------------------------------------------
    # Stage: diagnosis_cause — overfitting causes
    # ------------------------------------------------------------------

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.OVERFITTING.value),
        ModelArchitecture(model_scale="large"),
        NOT(PossibleCause(cause=DiagnosisCause.MODEL_TOO_COMPLEX.value)),
        salience=80,
    )
    def cause_overfitting_model_complex(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.OVERFITTING.value,
                cause=DiagnosisCause.MODEL_TOO_COMPLEX.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.OVERFITTING.value),
        ModelArchitecture(parameter_count=MATCH.params),
        TEST(lambda params: params >= HT.LARGE_PARAMETER_COUNT),
        NOT(PossibleCause(cause=DiagnosisCause.MODEL_TOO_COMPLEX.value)),
        salience=79,
    )
    def cause_overfitting_large_params(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.OVERFITTING.value,
                cause=DiagnosisCause.MODEL_TOO_COMPLEX.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.OVERFITTING.value),
        DatasetSize(size="small"),
        NOT(PossibleCause(cause=DiagnosisCause.INSUFFICIENT_DATA.value)),
        salience=78,
    )
    def cause_overfitting_insufficient_data(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.OVERFITTING.value,
                cause=DiagnosisCause.INSUFFICIENT_DATA.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.OVERFITTING.value),
        CurrentTrainingConfig(dropout_rate=MATCH.d),
        TEST(lambda d: d < HT.WEAK_DROPOUT_RATE),
        NOT(PossibleCause(cause=DiagnosisCause.REGULARIZATION_TOO_WEAK.value)),
        salience=77,
    )
    def cause_overfitting_weak_dropout(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.OVERFITTING.value,
                cause=DiagnosisCause.REGULARIZATION_TOO_WEAK.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.OVERFITTING.value),
        CurrentTrainingConfig(weight_decay=MATCH.w),
        TEST(lambda w: w < HT.WEAK_WEIGHT_DECAY),
        NOT(PossibleCause(cause=DiagnosisCause.REGULARIZATION_TOO_WEAK.value)),
        salience=76,
    )
    def cause_overfitting_weak_weight_decay(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.OVERFITTING.value,
                cause=DiagnosisCause.REGULARIZATION_TOO_WEAK.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.OVERFITTING.value),
        TrainingObservation(epochs_completed=MATCH.e),
        TEST(lambda e: e is not None and e > HT.TOO_MANY_EPOCHS),
        NOT(PossibleCause(cause=DiagnosisCause.TRAINED_TOO_LONG.value)),
        salience=75,
    )
    def cause_overfitting_trained_too_long(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.OVERFITTING.value,
                cause=DiagnosisCause.TRAINED_TOO_LONG.value,
                confidence=confidence_from_score(3),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.OVERFITTING.value),
        NOT(PossibleCause(diagnosis=DiagnosisIssue.OVERFITTING.value)),
        salience=10,
    )
    def cause_overfitting_fallback(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.OVERFITTING.value,
                cause=DiagnosisCause.GENERAL_OVERFITTING.value,
                confidence=confidence_from_score(2),
            )
        )

    # ------------------------------------------------------------------
    # Stage: diagnosis_cause — underfitting causes
    # ------------------------------------------------------------------

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.UNDERFITTING.value),
        ModelArchitecture(model_scale="small"),
        NOT(PossibleCause(cause=DiagnosisCause.MODEL_TOO_SIMPLE.value)),
        salience=80,
    )
    def cause_underfitting_model_simple(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.UNDERFITTING.value,
                cause=DiagnosisCause.MODEL_TOO_SIMPLE.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.UNDERFITTING.value),
        CurrentTrainingConfig(learning_rate=MATCH.lr),
        TEST(lambda lr: lr < HT.LR_TOO_LOW),
        NOT(PossibleCause(cause=DiagnosisCause.LEARNING_RATE_TOO_LOW.value)),
        salience=79,
    )
    def cause_underfitting_lr_low(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.UNDERFITTING.value,
                cause=DiagnosisCause.LEARNING_RATE_TOO_LOW.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.UNDERFITTING.value),
        TrainingObservation(epochs_completed=MATCH.e),
        TEST(lambda e: e is not None and e < HT.TOO_FEW_EPOCHS),
        NOT(PossibleCause(cause=DiagnosisCause.TRAINING_STOPPED_EARLY.value)),
        salience=78,
    )
    def cause_underfitting_stopped_early(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.UNDERFITTING.value,
                cause=DiagnosisCause.TRAINING_STOPPED_EARLY.value,
                confidence=confidence_from_score(3),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.UNDERFITTING.value),
        CurrentTrainingConfig(dropout_rate=MATCH.d),
        TEST(lambda d: d > HT.HIGH_DROPOUT_RATE),
        NOT(PossibleCause(cause=DiagnosisCause.EXCESSIVE_REGULARIZATION.value)),
        salience=77,
    )
    def cause_underfitting_high_dropout(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.UNDERFITTING.value,
                cause=DiagnosisCause.EXCESSIVE_REGULARIZATION.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.UNDERFITTING.value),
        CurrentTrainingConfig(weight_decay=MATCH.w),
        TEST(lambda w: w > HT.HIGH_WEIGHT_DECAY),
        NOT(PossibleCause(cause=DiagnosisCause.EXCESSIVE_REGULARIZATION.value)),
        salience=76,
    )
    def cause_underfitting_high_weight_decay(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.UNDERFITTING.value,
                cause=DiagnosisCause.EXCESSIVE_REGULARIZATION.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.UNDERFITTING.value),
        NOT(PossibleCause(diagnosis=DiagnosisIssue.UNDERFITTING.value)),
        salience=10,
    )
    def cause_underfitting_fallback(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.UNDERFITTING.value,
                cause=DiagnosisCause.GENERAL_UNDERFITTING.value,
                confidence=confidence_from_score(2),
            )
        )

    # ------------------------------------------------------------------
    # Stage: diagnosis_cause — NaN causes
    # ------------------------------------------------------------------

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.NAN_LOSS.value),
        CurrentTrainingConfig(learning_rate=MATCH.lr),
        TEST(lambda lr: lr > HT.LR_TOO_HIGH_NAN),
        NOT(PossibleCause(cause=DiagnosisCause.LEARNING_RATE_TOO_HIGH.value)),
        salience=80,
    )
    def cause_nan_lr_high(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.NAN_LOSS.value,
                cause=DiagnosisCause.LEARNING_RATE_TOO_HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.NAN_LOSS.value),
        CurrentTrainingConfig(gradient_clipping=False),
        NOT(PossibleCause(cause=DiagnosisCause.GRADIENT_EXPLOSION.value)),
        salience=79,
    )
    def cause_nan_gradient_explosion(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.NAN_LOSS.value,
                cause=DiagnosisCause.GRADIENT_EXPLOSION.value,
                confidence=confidence_from_score(3),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.NAN_LOSS.value),
        NOT(PossibleCause(cause=DiagnosisCause.NUMERICAL_INSTABILITY.value)),
        salience=78,
    )
    def cause_nan_numerical_instability(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.NAN_LOSS.value,
                cause=DiagnosisCause.NUMERICAL_INSTABILITY.value,
                confidence=confidence_from_score(3),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.NAN_LOSS.value),
        NOT(PossibleCause(cause=DiagnosisCause.INVALID_INPUT_VALUES.value)),
        salience=77,
    )
    def cause_nan_invalid_inputs(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.NAN_LOSS.value,
                cause=DiagnosisCause.INVALID_INPUT_VALUES.value,
                confidence=confidence_from_score(3),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.NAN_LOSS.value),
        NOT(PossibleCause(cause=DiagnosisCause.BAD_INITIALIZATION.value)),
        salience=76,
    )
    def cause_nan_bad_initialization(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.NAN_LOSS.value,
                cause=DiagnosisCause.BAD_INITIALIZATION.value,
                confidence=confidence_from_score(2),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.NAN_LOSS.value),
        NOT(PossibleCause(diagnosis=DiagnosisIssue.NAN_LOSS.value)),
        salience=10,
    )
    def cause_nan_fallback(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.NAN_LOSS.value,
                cause=DiagnosisCause.GENERAL_NAN_LOSS.value,
                confidence=confidence_from_score(2),
            )
        )

    # ------------------------------------------------------------------
    # Stage: diagnosis_cause — Inf causes
    # ------------------------------------------------------------------

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.INF_LOSS.value),
        NOT(PossibleCause(cause=DiagnosisCause.OVERFLOW.value)),
        salience=80,
    )
    def cause_inf_overflow(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.INF_LOSS.value,
                cause=DiagnosisCause.OVERFLOW.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.INF_LOSS.value),
        CurrentTrainingConfig(learning_rate=MATCH.lr),
        TEST(lambda lr: lr > HT.LR_TOO_HIGH_INF),
        NOT(PossibleCause(cause=DiagnosisCause.DIVERGING_GRADIENTS.value)),
        salience=79,
    )
    def cause_inf_diverging_gradients(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.INF_LOSS.value,
                cause=DiagnosisCause.DIVERGING_GRADIENTS.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.INF_LOSS.value),
        NOT(PossibleCause(cause=DiagnosisCause.INVALID_LOSS_COMPUTATION.value)),
        salience=78,
    )
    def cause_inf_invalid_loss(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.INF_LOSS.value,
                cause=DiagnosisCause.INVALID_LOSS_COMPUTATION.value,
                confidence=confidence_from_score(3),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.INF_LOSS.value),
        NOT(PossibleCause(diagnosis=DiagnosisIssue.INF_LOSS.value)),
        salience=10,
    )
    def cause_inf_fallback(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.INF_LOSS.value,
                cause=DiagnosisCause.GENERAL_INF_LOSS.value,
                confidence=confidence_from_score(2),
            )
        )

    # ------------------------------------------------------------------
    # Stage: diagnosis_cause — healthy training
    # ------------------------------------------------------------------

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        Diagnosis(issue=DiagnosisIssue.HEALTHY_TRAINING.value),
        NOT(PossibleCause(diagnosis=DiagnosisIssue.HEALTHY_TRAINING.value)),
        salience=10,
    )
    def cause_healthy_metrics_normal(self):
        self.declare(
            PossibleCause(
                diagnosis=DiagnosisIssue.HEALTHY_TRAINING.value,
                cause=DiagnosisCause.METRICS_WITHIN_NORMAL.value,
                confidence=confidence_from_score(3),
            )
        )

    # ------------------------------------------------------------------
    # Stage: diagnosis_cause — explain PossibleCause to user
    # ------------------------------------------------------------------

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.MODEL_TOO_COMPLEX.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Model too complex")),
        salience=60,
    )
    def explain_cause_model_complex(self, d):
        self._declare_possible_cause_rec(
            d, "Model too complex", "The model capacity may exceed what the data supports."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.INSUFFICIENT_DATA.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Insufficient data")),
        salience=60,
    )
    def explain_cause_insufficient_data(self, d):
        self._declare_possible_cause_rec(
            d, "Insufficient data", "Limited training data increases memorization risk."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.REGULARIZATION_TOO_WEAK.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Regularization too weak")),
        salience=60,
    )
    def explain_cause_weak_regularization(self, d):
        self._declare_possible_cause_rec(
            d, "Regularization too weak", "Dropout or weight decay may be insufficient."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.TRAINED_TOO_LONG.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Trained too long")),
        salience=60,
    )
    def explain_cause_trained_too_long(self, d):
        self._declare_possible_cause_rec(
            d, "Trained too long", "Training may have continued after validation stopped improving."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.MODEL_TOO_SIMPLE.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Model too simple")),
        salience=60,
    )
    def explain_cause_model_simple(self, d):
        self._declare_possible_cause_rec(
            d, "Model too simple", "Model capacity may be too limited for the task complexity."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.LEARNING_RATE_TOO_LOW.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Learning rate too low")),
        salience=60,
    )
    def explain_cause_lr_low(self, d):
        self._declare_possible_cause_rec(
            d, "Learning rate too low", "Optimization steps may be too small to learn effectively."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.TRAINING_STOPPED_EARLY.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Training stopped too early")),
        salience=60,
    )
    def explain_cause_stopped_early(self, d):
        self._declare_possible_cause_rec(
            d, "Training stopped too early", "The model may not have had enough epochs to converge."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.EXCESSIVE_REGULARIZATION.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Excessive regularization")),
        salience=60,
    )
    def explain_cause_excessive_reg(self, d):
        self._declare_possible_cause_rec(
            d, "Excessive regularization", "Strong dropout or weight decay may suppress learning."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.LEARNING_RATE_TOO_HIGH.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Learning rate too high")),
        salience=60,
    )
    def explain_cause_lr_high(self, d):
        self._declare_possible_cause_rec(
            d, "Learning rate too high", "Large update steps can destabilize training numerics."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.GRADIENT_EXPLOSION.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Gradient explosion")),
        salience=60,
    )
    def explain_cause_gradient_explosion(self, d):
        self._declare_possible_cause_rec(
            d, "Gradient explosion", "Unbounded gradients can produce NaN values during backprop."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.NUMERICAL_INSTABILITY.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Numerical instability")),
        salience=60,
    )
    def explain_cause_numerical_instability(self, d):
        self._declare_possible_cause_rec(
            d, "Numerical instability", "Floating-point operations may be producing invalid values."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.INVALID_INPUT_VALUES.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Invalid input values")),
        salience=60,
    )
    def explain_cause_invalid_inputs(self, d):
        self._declare_possible_cause_rec(
            d, "Invalid input values", "The dataset may contain NaN or extreme feature values."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.BAD_INITIALIZATION.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Bad initialization")),
        salience=60,
    )
    def explain_cause_bad_init(self, d):
        self._declare_possible_cause_rec(
            d, "Bad initialization", "Poor weight initialization can prevent stable early training."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.OVERFLOW.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Numerical overflow")),
        salience=60,
    )
    def explain_cause_overflow(self, d):
        self._declare_possible_cause_rec(
            d, "Numerical overflow", "Loss or activations exceeded floating-point range."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.DIVERGING_GRADIENTS.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Diverging gradients")),
        salience=60,
    )
    def explain_cause_diverging_gradients(self, d):
        self._declare_possible_cause_rec(
            d, "Diverging gradients", "Optimization is moving away from a stable minimum."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.INVALID_LOSS_COMPUTATION.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Invalid loss computation")),
        salience=60,
    )
    def explain_cause_invalid_loss(self, d):
        self._declare_possible_cause_rec(
            d, "Invalid loss computation", "The loss function may produce Inf for certain inputs."
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_CAUSE.value),
        PossibleCause(cause=DiagnosisCause.METRICS_WITHIN_NORMAL.value, diagnosis=MATCH.d),
        NOT(Recommendation(category="possible_cause", recommendation="Metrics within normal range")),
        salience=60,
    )
    def explain_cause_healthy_metrics(self, d):
        self._declare_possible_cause_rec(
            d,
            "Metrics within normal range",
            "Observed training and validation metrics are consistent with stable learning.",
        )

    def _declare_possible_cause_rec(self, diagnosis: str, label: str, justification: str) -> None:
        self.declare(
            Recommendation(
                category="possible_cause",
                recommendation=label,
                justification=justification,
                reasons=[f"Linked to inferred diagnosis: {diagnosis}"],
                priority=Priority.MEDIUM.value,
                confidence=confidence_from_score(3),
            )
        )

    # ------------------------------------------------------------------
    # Stage: diagnosis_recommend — actionable recommendations per cause
    # ------------------------------------------------------------------

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.MODEL_TOO_COMPLEX.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Reduce model complexity")),
        salience=80,
    )
    def recommend_reduce_complexity(self):
        self._declare_action_rec(
            "Reduce model complexity",
            "Use fewer layers or parameters to improve generalization.",
            ["Large model capacity relative to data increases overfitting risk"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.INSUFFICIENT_DATA.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Increase dataset size")),
        salience=80,
    )
    def recommend_increase_dataset(self):
        self._declare_action_rec(
            "Increase dataset size",
            "Collect more training data or use data augmentation.",
            ["Small datasets amplify overfitting on complex models"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.REGULARIZATION_TOO_WEAK.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Increase dropout")),
        salience=79,
    )
    def recommend_increase_dropout(self):
        self._declare_action_rec(
            "Increase dropout",
            "Raise dropout rate to penalize co-adaptation of neurons.",
            ["Weak regularization allows the model to memorize training noise"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.REGULARIZATION_TOO_WEAK.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Increase weight decay")),
        salience=78,
    )
    def recommend_increase_weight_decay(self):
        self._declare_action_rec(
            "Increase weight decay",
            "Apply stronger L2 regularization on weights.",
            ["Weak weight decay permits large weight magnitudes and overfitting"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.TRAINED_TOO_LONG.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Apply early stopping")),
        salience=77,
    )
    def recommend_early_stopping(self):
        self._declare_action_rec(
            "Apply early stopping",
            "Stop training when validation loss stops improving (Prechelt).",
            ["Continued training past the generalization point causes overfitting"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.GENERAL_OVERFITTING.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Use data augmentation")),
        salience=30,
    )
    def recommend_data_augmentation(self):
        self._declare_action_rec(
            "Use data augmentation",
            "Expand effective training set diversity with augmentation.",
            ["General overfitting signal — broaden training distribution"],
            Priority.MEDIUM,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.MODEL_TOO_SIMPLE.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Increase model capacity")),
        salience=80,
    )
    def recommend_increase_capacity(self):
        self._declare_action_rec(
            "Increase model capacity",
            "Add layers or widen hidden dimensions to capture more patterns.",
            ["Underfitting often indicates insufficient representational power"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.LEARNING_RATE_TOO_LOW.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Increase learning rate")),
        salience=79,
    )
    def recommend_increase_lr(self):
        self._declare_action_rec(
            "Increase learning rate",
            "Raise the learning rate to accelerate convergence.",
            ["Very small steps prevent the model from learning effectively"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.TRAINING_STOPPED_EARLY.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Train longer")),
        salience=78,
    )
    def recommend_train_longer(self):
        self._declare_action_rec(
            "Train longer",
            "Increase the number of training epochs.",
            ["Model may need more iterations to fit the training signal"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.EXCESSIVE_REGULARIZATION.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Reduce regularization")),
        salience=77,
    )
    def recommend_reduce_regularization(self):
        self._declare_action_rec(
            "Reduce regularization",
            "Lower dropout rate or weight decay to allow more learning.",
            ["Excessive regularization can suppress useful model capacity"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.GENERAL_UNDERFITTING.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Review model architecture")),
        salience=30,
    )
    def recommend_review_architecture(self):
        self._declare_action_rec(
            "Review model architecture",
            "Re-evaluate whether the architecture matches task complexity.",
            ["General underfitting — verify capacity and training schedule"],
            Priority.MEDIUM,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.LEARNING_RATE_TOO_HIGH.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Lower learning rate")),
        salience=80,
    )
    def recommend_lower_lr(self):
        self._declare_action_rec(
            "Lower learning rate",
            "Reduce the learning rate by 10× and retry training.",
            ["Excessive learning rate is a primary cause of NaN loss"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.GRADIENT_EXPLOSION.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Enable gradient clipping")),
        salience=79,
    )
    def recommend_gradient_clipping(self):
        self._declare_action_rec(
            "Enable gradient clipping",
            "Clip global gradient norm to a stable threshold (e.g. 1.0).",
            ["Unbounded gradients frequently produce NaN or Inf values"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.INVALID_INPUT_VALUES.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Normalize inputs")),
        salience=78,
    )
    def recommend_normalize_inputs(self):
        self._declare_action_rec(
            "Normalize inputs",
            "Standardize or scale features to a stable numeric range.",
            ["Extreme input magnitudes can destabilize forward passes"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.INVALID_INPUT_VALUES.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Check dataset for NaN values")),
        salience=77,
    )
    def recommend_check_dataset_nan(self):
        self._declare_action_rec(
            "Check dataset for NaN values",
            "Scan features and labels for missing or invalid entries.",
            ["NaN in inputs propagates directly to NaN loss"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.BAD_INITIALIZATION.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Use a stable optimizer")),
        salience=76,
    )
    def recommend_stable_optimizer(self):
        self._declare_action_rec(
            "Use a stable optimizer",
            "Switch to Adam or AdamW with a conservative learning rate.",
            ["Adaptive optimizers tolerate difficult loss landscapes better early on"],
            Priority.MEDIUM,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.NUMERICAL_INSTABILITY.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Enable mixed-precision safeguards")),
        salience=75,
    )
    def recommend_mixed_precision_safeguards(self):
        self._declare_action_rec(
            "Enable mixed-precision safeguards",
            "Use loss scaling or full float32 for unstable layers.",
            ["Mixed precision without scaling can produce numerical failures"],
            Priority.MEDIUM,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.OVERFLOW.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Lower learning rate")),
        salience=74,
    )
    def recommend_lower_lr_inf(self):
        self._declare_action_rec(
            "Lower learning rate",
            "Reduce learning rate to prevent loss overflow.",
            ["Inf loss often follows unchecked parameter updates"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.DIVERGING_GRADIENTS.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Enable gradient clipping")),
        salience=73,
    )
    def recommend_gradient_clipping_inf(self):
        self._declare_action_rec(
            "Enable gradient clipping",
            "Cap gradient norms to prevent runaway updates.",
            ["Diverging gradients are a common source of Inf loss"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.INVALID_LOSS_COMPUTATION.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Verify loss implementation")),
        salience=72,
    )
    def recommend_verify_loss(self):
        self._declare_action_rec(
            "Verify loss implementation",
            "Audit loss function for log(0), division by zero, or unchecked exponentials.",
            ["Implementation bugs can produce Inf even with valid inputs"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.GENERAL_NAN_LOSS.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Lower learning rate")),
        salience=20,
    )
    def recommend_lower_lr_nan_fallback(self):
        self._declare_action_rec(
            "Lower learning rate",
            "Start with a smaller learning rate and increase gradually.",
            ["First-line remediation for unexplained NaN loss"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.GENERAL_INF_LOSS.value),
        NOT(Recommendation(category="diagnosis_recommendation", recommendation="Verify loss implementation")),
        salience=20,
    )
    def recommend_verify_loss_inf_fallback(self):
        self._declare_action_rec(
            "Verify loss implementation",
            "Inspect loss computation and intermediate activations for overflow.",
            ["First-line remediation for unexplained Inf loss"],
            Priority.HIGH,
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.DIAGNOSIS_RECOMMEND.value),
        PossibleCause(cause=DiagnosisCause.METRICS_WITHIN_NORMAL.value),
        NOT(
            Recommendation(
                category="diagnosis_recommendation",
                recommendation="Continue training and monitor validation metrics",
            )
        ),
        salience=30,
    )
    def recommend_continue_healthy_training(self):
        self._declare_action_rec(
            "Continue training and monitor validation metrics",
            "No corrective action required based on current observations.",
            [
                "Loss and accuracy gaps are within healthy bounds",
                "Re-check if performance plateaus or validation metrics degrade",
                "Consider hyperparameter tuning only if you need further gains",
            ],
            Priority.MEDIUM,
        )

    def _declare_action_rec(
        self,
        label: str,
        justification: str,
        reasons: list[str],
        priority: Priority,
    ) -> None:
        self.declare(
            Recommendation(
                category="diagnosis_recommendation",
                recommendation=label,
                justification=justification,
                reasons=reasons,
                priority=priority.value,
                confidence=confidence_from_score(4),
            )
        )
