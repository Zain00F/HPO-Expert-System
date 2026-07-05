from experta import MATCH, NOT, TEST, Rule ,AS

from hpo_expert.facts.model import ModelArchitecture
from hpo_expert.facts.reasoning import (
    OptimizerChoice,
    OptimizerFamilyChoice,
    ReasoningStage,
    Recommendation,  
)
from hpo_expert.utils.enums import (
    ArchitectureType,
    Optimizer,
    OptimizerFamily,
    Priority,
    ReasoningStageId,
)
from hpo_expert.utils.scoring import confidence_from_score

_ADAPTIVE_ARCHITECTURES = {
    ArchitectureType.TRANSFORMER.value,
    ArchitectureType.LLM.value,
    ArchitectureType.RNN.value,
}


class OptimizerRules:
    @Rule(
        ReasoningStage(current=ReasoningStageId.OPTIMIZER_FAMILY.value),
        ModelArchitecture(architecture_type=MATCH.arch, parameter_count=MATCH.params),
        TEST(lambda arch, params: arch in _ADAPTIVE_ARCHITECTURES or params >= 1_000_000),
        NOT(OptimizerFamilyChoice()),
        salience=80,
    )
    def family_adaptive(self, arch, params):
        reasons = [
            f"Architecture '{arch}' or scale ({params:,} parameters) benefits from per-parameter adaptive rates",
            "Adaptive methods (Adam family) reduce manual learning-rate scheduling early in tuning",
        ]
        
        self.declare(
            OptimizerFamilyChoice(
                family=OptimizerFamily.ADAPTIVE.value,
                confidence=confidence_from_score(5),
            )
        )
        
        self.declare(
            Recommendation(
                category="optimizer_family",
                recommendation="Adaptive Family",
                justification=f"Large scale model or adaptive architecture ({arch}) requires per-parameter learning rates.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5)
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.OPTIMIZER_FAMILY.value),
        ModelArchitecture(architecture_type=MATCH.arch),
        TEST(lambda arch: arch in {ArchitectureType.CNN.value, ArchitectureType.MLP.value}),
        NOT(OptimizerFamilyChoice()),
        salience=70,
    )
    def family_adaptive_cnn_mlp_default(self, arch):
        """CNN/MLP still default to adaptive unless user needs fine SGD control (future rule)."""
        reasons = [
            f"Classic '{arch}' pipelines commonly start with Adam for faster convergence",
            "You can still switch to SGD+momentum after a strong baseline if generalization requires it",
        ]
        
        self.declare(
            OptimizerFamilyChoice(
                family=OptimizerFamily.ADAPTIVE.value,
                confidence=confidence_from_score(4),
            )
        )
        
        self.declare(
            Recommendation(
                category="optimizer_family",
                recommendation="Adaptive Family (Default)",
                justification=f"Standard baseline for '{arch}' usually defaults to adaptive methods for speed.",
                reasons=reasons,
                priority=Priority.MEDIUM.value,
                confidence=confidence_from_score(4)
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.OPTIMIZER_FAMILY.value),
        NOT(OptimizerFamilyChoice()),
        salience=40,
    )
    def family_non_adaptive_fallback(self):
        reasons = [
            "No strong signal for adaptive methods — non-adaptive SGD family offers interpretable control",
        ]
        
        self.declare(
            OptimizerFamilyChoice(
                family=OptimizerFamily.NON_ADAPTIVE.value,
                confidence=confidence_from_score(2),
            )
        )
        
        self.declare(
            Recommendation(
                category="optimizer_family",
                recommendation="Non-Adaptive Family (SGD)",
                justification="Falling back to standard SGD as no explicitly complex or deep structures were detected.",
                reasons=reasons,
                priority=Priority.LOW.value,
                confidence=confidence_from_score(2)
            )
        )

 # --- Specific optimizers (stage optimizer_specific) ---

    @Rule(
        ReasoningStage(current=ReasoningStageId.OPTIMIZER_SPECIFIC.value),
        OptimizerFamilyChoice(family=OptimizerFamily.ADAPTIVE.value),
        ModelArchitecture(
            architecture_type=MATCH.arch,
            uses_attention=MATCH.attn,
        ),
        TEST(lambda arch, attn: arch in _ADAPTIVE_ARCHITECTURES or attn is True),
        NOT(OptimizerChoice()),
        salience=75,
    )
    def optimizer_adamw(self, arch, attn):
        reasons = [
            f"Architecture '{arch}' with decoupled weight decay (AdamW) is standard for transformers",
            "AdamW separates L2 regularization from the adaptive step — better generalization than classic Adam",
        ]
        
        self.declare(
            OptimizerChoice(
                optimizer=Optimizer.ADAMW.value,
                confidence=confidence_from_score(5),
            )
        )
        
        self.declare(
            Recommendation(
                category="optimizer_specific",
                recommendation="AdamW (Decoupled Weight Decay)",
                justification=f"Crucial for stability in attention-based blocks or '{arch}' structures.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5)
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.OPTIMIZER_SPECIFIC.value),
        OptimizerFamilyChoice(family=OptimizerFamily.ADAPTIVE.value),
        NOT(OptimizerChoice()),
        salience=70,
    )
    def optimizer_adam_default(self):
        reasons = [
            "General-purpose adaptive baseline with well-understood behavior",
            "Strong starting point before trying RMSProp or Adagrad variants",
        ]
        
        self.declare(
            OptimizerChoice(
                optimizer=Optimizer.ADAM.value,
                confidence=confidence_from_score(4),
            )
        )
        
        self.declare(
            Recommendation(
                category="optimizer_specific",
                recommendation="Adam",
                justification="Safe, standard general-purpose adaptive optimizer for initial benchmarking.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(4)
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.OPTIMIZER_SPECIFIC.value),
        OptimizerFamilyChoice(family=OptimizerFamily.NON_ADAPTIVE.value),
        NOT(OptimizerChoice()),
        salience=70,
    )
    def optimizer_momentum_sgd(self):
        reasons = [
            "SGD with momentum is the standard non-adaptive choice for vision models",
            "Often achieves better generalization than Adam when tuned with care",
        ]
        
        self.declare(
            OptimizerChoice(
                optimizer=Optimizer.MOMENTUM_SGD.value,
                confidence=confidence_from_score(4),
            )
        )
        
        self.declare(
            Recommendation(
                category="optimizer_specific",
                recommendation="SGD with Momentum",
                justification="Highly recommended for deep vision pipelines where generalization out-values convergence speed.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(4)
            )
        )

    