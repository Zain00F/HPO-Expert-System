"""
Stage 4: Range Suggestion Rules.
Suggests hyperparameter search spaces and fixed scalar values based on previous stages.
"""

from experta import MATCH, NOT, TEST, Rule

from hpo_expert.facts.context import ComputeConstraints
from hpo_expert.facts.model import DatasetProfile, ModelArchitecture
from hpo_expert.facts.reasoning import (
    OptimizerChoice,
    RangeChoice,
    ReasoningStage,
    Recommendation,
    SearchSpaceChoice,
)
from hpo_expert.utils.enums import ArchitectureType, Optimizer, Priority, ReasoningStageId
from hpo_expert.utils.scoring import confidence_from_score

_SEQUENTIAL_ARCHITECTURES = {
    ArchitectureType.TRANSFORMER.value,
    ArchitectureType.LLM.value,
    ArchitectureType.RNN.value,
}


class RangeRules:
    """Mixin: Contains expert rules for hyperparameter range suggestion and constraints."""

    # 1. OPTIMIZER-BOUND PARAMETERS BRANCH 
    @Rule(
        ReasoningStage(current=ReasoningStageId.RANGES.value),
        OptimizerChoice(optimizer=Optimizer.ADAMW.value),
        ModelArchitecture(architecture_type=MATCH.arch),
        TEST(lambda arch: arch in _SEQUENTIAL_ARCHITECTURES),
        salience=70,
    )
    def range_adamw_sequential(self, arch):
        """AdamW ranges for Sequential/Attention contexts."""
        conf = confidence_from_score(5)
        self.declare(RangeChoice(parameter="optimizer_bound", strategy="adamw_sequential", confidence=conf))
        self.declare(
            Recommendation(
                category="range_suggestion",
                recommendation="AdamW Range (Sequential Context)",
                justification=f"Optimizing '{arch}' with AdamW using conservative LR for sequential/attention-style models.",
                reasons=[
                    "lr: [1e-5, 1e-3] (Log-Uniform Distribution) — lower LR for stability in deep sequential/attention models",
                    "weight_decay: [1e-2, 1e-1] (Log-Uniform Distribution) — stronger regularization commonly used in large-scale pretraining setups",
                    "Fixed Constants: beta1=0.9, beta2=0.999, epsilon=1e-8",
                ],
                priority=Priority.HIGH.value,
                confidence=conf,
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.RANGES.value),
        OptimizerChoice(optimizer=Optimizer.ADAMW.value),
        ModelArchitecture(architecture_type=MATCH.arch),
        TEST(lambda arch: arch not in _SEQUENTIAL_ARCHITECTURES),
        salience=70,
    )
    def range_adamw_spatial(self, arch):
        """AdamW ranges for CNN/MLP/Other spatial contexts."""
        conf = confidence_from_score(5)
        self.declare(RangeChoice(parameter="optimizer_bound", strategy="adamw_spatial", confidence=conf))
        self.declare(
            Recommendation(
                category="range_suggestion",
                recommendation="AdamW Range (Spatial/Standard Context)",
                justification=f"Standard AdamW tuning for '{arch}' with general-purpose vision/ML architectures.",

                reasons=[
                    "lr: [1e-4, 1e-2] (Log-Uniform Distribution) — commonly used learning-rate range for general-purpose CNN/MLP training with AdamW",
                    "weight_decay: [1e-4, 1e-2] (Log-Uniform Distribution) — balanced regularization for general training regimes",
                    "Fixed Constants: beta1=0.9, beta2=0.999, epsilon=1e-8",
                ],
                priority=Priority.HIGH.value,
                confidence=conf,
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.RANGES.value),
        OptimizerChoice(optimizer=Optimizer.ADAM.value),
        salience=70,
    )
    def range_adam_standard(self):
        """Standard Adam boundaries."""
        conf = confidence_from_score(5)
        self.declare(RangeChoice(parameter="optimizer_bound", strategy="adam_standard", confidence=conf))
        self.declare(
            Recommendation(
                category="range_suggestion",
                recommendation="Adam Standard Ranges",
                justification="Standard Adam configuration for general-purpose training without architecture-specific tuning assumptions.",

                reasons=[
                    "lr: [1e-4, 1e-2] (Log-Uniform Distribution) — typical adaptive optimization range",
                    "weight_decay: [1e-6, 1e-3] (Log-Uniform Distribution) — conservative regularization due to coupled decay behavior in Adam",
                    "Fixed Constants: beta1=0.9, beta2=0.999, epsilon=1e-8",
                ],
                priority=Priority.HIGH.value,
                confidence=conf,
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.RANGES.value),
        OptimizerChoice(optimizer=MATCH.opt),
        TEST(lambda opt: opt in {Optimizer.SGD.value, Optimizer.MOMENTUM_SGD.value}),
        salience=70,
    )
    def range_sgd_momentum(self, opt):
        """Standard SGD + Momentum boundaries."""
        conf = confidence_from_score(5)
        self.declare(RangeChoice(parameter="optimizer_bound", strategy="sgd_momentum", confidence=conf))
        self.declare(
            Recommendation(
                category="range_suggestion",
                recommendation="SGD with Momentum Ranges",
                justification="Classic SGD with momentum typically benefits from higher learning rates than adaptive optimizers.",

                reasons=[
                    "lr: [1e-3, 1e-1] (Log-Uniform Distribution) — required due to absence of adaptive scaling",
                    "momentum: 0.9 (Standard value for velocity smoothing)",
                    "weight_decay: [1e-4, 1e-2] (Log-Uniform Distribution) — standard regularization range for CNN/MLP training",
                ],
                priority=Priority.HIGH.value,
                confidence=conf,
            )
        )

    # 2. CONTEXT-BOUND PARAMETERS BRANCH 

    #Batch Size
    @Rule(
        ReasoningStage(current=ReasoningStageId.RANGES.value),
        ComputeConstraints(has_gpu=True),
        salience=70,
    )
    def range_batch_size_high_perf(self):
        """High-performance batch size range."""
        conf = confidence_from_score(4)
        self.declare(RangeChoice(parameter="batch_size", strategy="high_performance", confidence=conf))
        self.declare(
            Recommendation(
                category="range_suggestion",
                recommendation="Batch Size: High-Performance Setup",
                justification="Batch size selection under high compute availability for throughput-optimized training.",

                reasons=[
                    "Range: [32, 256]",
                    "Scale Strategy: Powers of 2 — improves GPU utilization efficiency",
                ],
                priority=Priority.MEDIUM.value,
                confidence=conf,
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.RANGES.value),
        ComputeConstraints(has_gpu=False),
        salience=70,
    )
    def range_batch_size_constrained(self):
        """Resource-constrained batch size range."""
        conf = confidence_from_score(4)
        self.declare(RangeChoice(parameter="batch_size", strategy="resource_constrained", confidence=conf))
        self.declare(
            Recommendation(
                category="range_suggestion",
                recommendation="Batch Size: Resource-Constrained Setup",
                justification="Batch size selection under limited compute resources prioritizing memory efficiency.",

                reasons=[
                    "Range: [16, 64] — recommended for standard resource-constrained setups with a GPU",
                    "Scale Strategy: Powers of 2 — practical hardware alignment (16, 32, 64)",
                    "Emergency Note: In Extreme Focus Mode (CPU training / severe budget), override the range and strictly use a fixed batch size of 8 (or 16 if Batch Normalization is active) to prevent runtime crashes.",
                ],
                priority=Priority.MEDIUM.value,
                confidence=conf,
            )
        )

    # Dropout
    @Rule(
        ReasoningStage(current=ReasoningStageId.RANGES.value),
        ModelArchitecture(model_scale="large"),
        DatasetProfile(data_type=MATCH.dtype, sample_count=MATCH.count),
        TEST(lambda dtype, count: (
            (dtype == "image" and count < 2000) or
            (dtype == "text" and count < 5000) or
            (dtype == "tabular" and count < 1000) or
            (dtype == "audio" and count < 1000) or
            (dtype == "multimodal" and count < 2000)
        )),
        salience=70,
    )
    def range_dropout_high_regularization(self, dtype, count):
        """High overfitting risk (Large Model + Small Dataset) -> High Dropout range."""
        conf = confidence_from_score(5)
        self.declare(RangeChoice(parameter="dropout", strategy="high_regularization", confidence=conf))
        self.declare(
            Recommendation(
                category="range_suggestion",
                recommendation="Dropout: High Regularization Space",
                justification=f"Increased regularization because the dataset is small for '{dtype}' data ({count} samples) while utilizing a 'large' model architecture.",
                reasons=[
                    "Range: [0.3, 0.6]",
                    "Scale Strategy: Uniform Distribution",
                ],
                priority=Priority.HIGH.value,
                confidence=conf,
            )
        )


    @Rule(
        ReasoningStage(current=ReasoningStageId.RANGES.value),
        ModelArchitecture(model_scale=MATCH.scale),
        DatasetProfile(data_type=MATCH.dtype, sample_count=MATCH.count),
        TEST(lambda scale, dtype, count: not (
            scale == "large" and (
                (dtype == "image" and count < 2000) or
                (dtype == "text" and count < 5000) or
                (dtype == "tabular" and count < 1000) or
                (dtype == "audio" and count < 1000) or
                (dtype == "multimodal" and count < 2000)
            )
        )),
        salience=70,
    )
    def range_dropout_standard(self, scale, dtype, count):
        """Standard overfitting risk -> Normal Dropout range."""
        conf = confidence_from_score(4)
        self.declare(RangeChoice(parameter="dropout", strategy="standard_regularization", confidence=conf))
        self.declare(
            Recommendation(
                category="range_suggestion",
                recommendation="Dropout: Standard Space",
                justification=f"Default regularization regime. The combination of model scale '{scale}' and dataset size ({count} samples for '{dtype}') presents a balanced overfitting risk.",
                reasons=[
                    "Range: [0.0, 0.3]",
                    "Scale Strategy: Uniform Distribution",
                ],
                priority=Priority.MEDIUM.value,
                confidence=conf,
            )
        )

        
    # 3. STAGE COMPLETION ANCHOR 

    @Rule(
        ReasoningStage(current=ReasoningStageId.RANGES.value),
        NOT(RangeChoice(status="complete")),
        salience=60,
    )
    def complete_ranges_stage(self):
        """Triggers after all parallel parameter rules have finished firing."""
        self.declare(RangeChoice(status="complete"))