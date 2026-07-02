"""
Search Space Prioritization Rules.
Implements Google Playbook & academic 2025 constraints for active search and freezing.
"""

from experta import MATCH, NOT, TEST, Rule

from hpo_expert.facts.context import ComputeConstraints, OptimizationBudget, ProjectContext
from hpo_expert.facts.model import DatasetProfile, ModelArchitecture
from hpo_expert.facts.reasoning import ReasoningStage, Recommendation ,SearchSpaceChoice
from hpo_expert.utils.enums import Priority, ReasoningStageId
from hpo_expert.utils.scoring import confidence_from_score


class SearchSpaceRules:
    """Mixin: Contains expert rules for partitioning and prioritizing hyperparameter search spaces."""

    # 1. BRANCH 1: STRICT RESOURCE CHECK (Yes) -> BatchNorm Check
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=MATCH.has_gpu),
        OptimizationBudget(max_trials=MATCH.trials),
        ModelArchitecture(uses_batch_norm=True, architecture_type="cnn"),
        TEST(lambda has_gpu, trials: has_gpu is False or trials <= 20),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=85,
    )
    def space_strict_resource_bn_safe(self):
        """[BN_Safe Node] Strict resources + BatchNorm inside a CNN structure."""
        reasons = [
            "Strict budget condition met (GPU unavailable OR max_trials <= 20).",
            "Model architecture uses Batch Normalization within a CNN topology.",
            "Freezing batch size at 16 to stabilize internal BatchNorm statistics without memory failure risk.",
            "Freezing learning rate schedule to 'constant' and dropout to 0 to reduce dimensionality and avoid BatchNorm conflict.",
            "Priority 1: Isolate active search to the Learning Rate (lr).",
            "Priority 2: Maintain Weight Decay (wd) tuning safely to prevent catastrophic overfitting.",
        ]

        self.declare(
            SearchSpaceChoice(
                mode="resource_bn_safe",
                confidence=confidence_from_score(5),
            )
        )

        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Resource & BN Constraint Mode",
                justification="Tight compute parameters with BatchNorm present requires locking batch size and schedule to safe defaults.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=MATCH.has_gpu),
        OptimizationBudget(max_trials=MATCH.trials),
        ModelArchitecture(uses_batch_norm=MATCH.bn),
        TEST(lambda has_gpu, trials, bn: (has_gpu is False or trials <= 20) and bn is False),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=84,
    )
    def space_strict_resource_extreme_focus(self):
        """[Safe_Def Node] Strict resources without BatchNorm."""
        reasons = [
            "Strict budget condition met (GPU unavailable OR max_trials <= 20).",
            "No active Batch Normalization constraints detected.",
            "Aggressively reducing search dimensions to avoid sub-optimal convergence.",
            "Freezing: batch size at 8 or 16, weight decay (wd), dropout, and schedule to 'constant'.",
            "Priority 1: Invest 100% of active search updates exclusively on the Learning Rate (lr).",
        ]

        self.declare(
            SearchSpaceChoice(
                mode="extreme_focus",
                confidence=confidence_from_score(5),
            )
        )

        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Extreme Focus Mode",
                justification="Severe hardware or temporal constraints require freezing all dimensions except the core learning rate.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    # BRANCH 2: SUFFICIENT RESOURCES (No to C1) -> Optimization Goal Routing
    
    # 2a. Maximize Accuracy Node
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=True),
        OptimizationBudget(max_trials=MATCH.trials),
        ProjectContext(optimization_goal="maximize_accuracy"),
        TEST(lambda trials: trials > 20),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=80,
    )
    def space_goal_accuracy_fanova(self):
        """[Goal_Acc Node] High resource + Maximize Accuracy."""
        reasons = [
            "Sufficient resources verified (GPU enabled AND max_trials > 20).",
            "Objective is absolute accuracy maximization.",
            "fANOVA sensitivity analysis indicates Batch Size shares a 65% joint interaction effect with LR.",
            "Priority 1: Tune Batch Size via active search.",
            "Priority 2: Tune Learning Rate (lr) via active search concurrently.",
            "Priority 3: Tune LR Schedule style (Decay Style).",
            "Freezing: Dropout order and Weight Decay (wd) at stable baseline configurations.",
        ]

        self.declare(
            SearchSpaceChoice(
                mode="accuracy_fanova",
                confidence=confidence_from_score(5),
            )
        )

        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Accuracy Mode (fANOVA-Driven)",
                justification="Maximizing performance requires co-tuning interaction parameters (batch, lr) followed by the decay schedule.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    # 2b. Minimize Training Time Node
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=True),
        OptimizationBudget(max_trials=MATCH.trials),
        ProjectContext(optimization_goal="minimize_training_time"),
        TEST(lambda trials: trials > 20),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=79,
    )
    def space_goal_time_fanova(self):
        """[Goal_Time Node] High resource + Minimize Training Time."""
        reasons = [
            "Sufficient resources verified (GPU enabled AND max_trials > 20).",
            "Objective is minimizing wall-clock runtime / maximizing training speed.",
            "fANOVA validates that the Learning Rate holds a 54% direct standalone effect on convergence speed.",
            "Priority 1: Isolate active search strictly to the Learning Rate (lr).",
            "Freezing: Batch size to maximum available hardware limits, weight decay, dropout, and schedule to 'fixed_linear'.",
        ]

        self.declare(
            SearchSpaceChoice(
                mode="speed_fanova",
                confidence=confidence_from_score(5),
            )
        )

        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Speed Mode (fANOVA-Driven)",
                justification="Tuning for speed requires isolating search to LR while pinning batch size high to exploit parallelism.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    # BRANCH 3: BALANCED GOAL Routing -> Architecture Check
    
    # 3a. Transformer + Warmup Node
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=True),
        OptimizationBudget(max_trials=MATCH.trials),
        ProjectContext(optimization_goal="balanced"),
        ModelArchitecture(architecture_type="transformer"),
        TEST(lambda trials: trials > 20),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=75,
    )
    def space_balanced_transformer_warmup(self):
        """[Arch_Trans Node] High resource + Balanced + Transformer."""
        reasons = [
            "Sufficient resources available with a balanced optimization intent.",
            "Architecture is attention-based (Transformer).",
            "Priority 1: Active Search on Learning Rate (lr).",
            "Priority 2: Active Search on Weight Decay (wd) due to high transformer sensitivity to decoupled regularization.",
            "Freezing: Batch size and dropout order at verified default states.",
            "Mandatory: An LR Warmup period must be explicitly injected during trials to avoid early gradient explosion.",
        ]

        self.declare(
            SearchSpaceChoice(
                mode="transformer_balanced",
                confidence=confidence_from_score(5),
            )
        )

        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Adaptive Context Focus",
                justification="Transformers under balanced conditions require co-tuning lr and regularizers with mandatory learning rate warmup.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    # 3b. CNN + High Overfitting Risk Node (Large model + Small Dataset)
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=True),
        OptimizationBudget(max_trials=MATCH.trials),
        ProjectContext(optimization_goal="balanced"),
        ModelArchitecture(architecture_type="cnn", model_scale="large"),
        DatasetProfile(dataset_size="small"),  
        TEST(lambda trials: trials > 20),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=72,
    )
    def space_balanced_cnn_high_overfitting(self):
        """[CNN_Reg Node] High resource + Balanced + CNN + High Risk (Large model + Small Dataset)."""
        reasons = [
            "Sufficient resources with a balanced objective on a large-scale CNN vision architecture.",
            "High Overfitting Risk detected: Large parameter capacity operating on a small dataset profile.",
            "Priority 1: Active Search on Learning Rate (lr).",
            "Priority 2: Active Search on Batch Size to find regularizing noise surfaces.",
            "Priority 3: Activate active search for both Weight Decay (wd) and Dropout to structurally suppress overfitting.",
        ]
        
        self.declare(
            SearchSpaceChoice(
                mode="cnn_high_risk",
                confidence=confidence_from_score(5),
            )
        )

        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Spatial Regularization Mode",
                justification="High overfitting vulnerability due to small dataset size requires tuning both optimization and regularizer limits.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    # 3c. CNN + Low Overfitting Risk Node (Standard Falling back)
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=True),
        OptimizationBudget(max_trials=MATCH.trials),
        ProjectContext(optimization_goal="balanced"),
        ModelArchitecture(architecture_type="cnn"),  
        TEST(lambda trials: trials > 20),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=70,
    )
    def space_balanced_cnn_standard(self):
        """[CNN_Standard Node] High resource + Balanced + CNN + Normal Risk."""
        reasons = [
            "Sufficient resources with a balanced objective on a standard CNN vision model.",
            "Low/Normal Overfitting Risk profile: dataset capacity matches model footprint.",
            "Priority 1: Active Search on Learning Rate (lr).",
            "Priority 2: Active Search on Batch Size.",
            "Freezing: Weight decay (wd) and dropout at standard safe defaults to preserve sample efficiency.",
        ]
        
        self.declare(
            SearchSpaceChoice(
                mode="cnn_standard",
                confidence=confidence_from_score(4),
            )
        )

        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Spatial Standard Mode",
                justification="Standard computer vision setups optimize performance by focusing trials on lr and batch size while pinning constraints.",
                reasons=reasons,
                priority=Priority.MEDIUM.value,
                confidence=confidence_from_score(4),
            )
        )


    # 4. FALLBACK CATCH-ALL: SAFE DEFAULT MODE
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        NOT(SearchSpaceChoice()),  
        salience=50,               
    )
    def space_fallback_safe_default(self):
        """[Fallback Node] General safe default when no specific resource/architecture branch matches."""
        reasons = [
            "No specific resource configuration or architecture patterns were explicitly matched.",
            "Falling back to a robust, general-purpose machine learning search strategy.",
            "Priority 1: Isolate active search primarily to the Learning Rate (lr) as the highest impact parameter.",
            "Priority 2: Include Batch Size in active search at standard boundaries to handle generic convergence.",
            "Freezing: Keeping weight decay, dropout, and schedule at safe baseline defaults to avoid over-parameterization.",
        ]
        
        self.declare(
            SearchSpaceChoice(
                mode="safe_default",
                confidence=confidence_from_score(3),  
            )
        )

        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Safe Default Space Mode",
                justification="A standard general-purpose configuration is applied to guarantee system progression when strict context rules are unmapped.",
                reasons=reasons,
                priority=Priority.MEDIUM.value,
                confidence=confidence_from_score(3),
            )
        )
