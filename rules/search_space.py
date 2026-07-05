
from experta import MATCH, NOT, TEST, Rule

from hpo_expert.facts.context import ComputeConstraints, OptimizationBudget, ProjectContext
from hpo_expert.facts.model import DatasetProfile, ModelArchitecture
from hpo_expert.facts.reasoning import ReasoningStage, Recommendation ,SearchSpaceChoice,OptimizerChoice
from hpo_expert.utils.enums import Priority, ReasoningStageId ,Optimizer
from hpo_expert.utils.scoring import confidence_from_score
_SEQUENTIAL_ARCHITECTURES = {"transformer", "llm", "rnn"}
_SPATIAL_ARCHITECTURES = {"cnn", "mlp"}


class SearchSpaceRules:

    # 1. BRANCH 1: STRICT RESOURCE CHECK (Yes) -> BatchNorm Check
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=MATCH.has_gpu),
        OptimizationBudget(max_trials=MATCH.trials),
        ModelArchitecture(uses_batch_norm=True, architecture_type=MATCH.arch),
        TEST(lambda has_gpu, trials, arch: (has_gpu is False or trials <= 20) and arch in _SPATIAL_ARCHITECTURES),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=85,
    )
    def space_strict_resource_bn_safe(self, arch):
        """[BN_Safe Node] Strict resources + BatchNorm inside a CNN/MLP structure."""
        reasons = [
            "Strict budget condition met (GPU unavailable OR max_trials <= 20).",
            f"Architecture '{arch}' uses Batch Normalization.",
            "Freezing batch size at 16 to stabilize internal BatchNorm statistics without memory failure risk.",
            "Freezing learning rate schedule to 'constant' and dropout to 0 to reduce dimensionality and avoid BatchNorm conflict (Karpathy, 2019).",
            "Priority 1: Isolate active search to the Learning Rate (lr).",
            "Priority 2: Weight Decay (wd) — set to a small value (recommended: 1e-4 or lower). "
            "Batch Normalization provides its own regularization effect, reducing the need for large wd values. "
            "Large wd with BN can cause underfitting — keep it small but non-zero.",
        ]

        self.declare(SearchSpaceChoice(mode="resource_bn_safe", confidence=confidence_from_score(5)))
        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Resource & BN Constraint Mode",
                justification=f"Tight compute with BatchNorm in '{arch}' requires locking batch size and schedule to safe defaults.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=MATCH.has_gpu),
        OptimizationBudget(max_trials=MATCH.trials),
        ModelArchitecture(uses_batch_norm=MATCH.bn, architecture_type=MATCH.arch),
        TEST(lambda has_gpu, trials, bn, arch: (has_gpu is False or trials <= 20) and (bn is False or arch not in _SPATIAL_ARCHITECTURES)),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=84,
    )
    def space_strict_resource_extreme_focus(self, arch):
        reasons = [
            "Strict budget condition met (GPU unavailable OR max_trials <= 20).",
            f"Architecture '{arch}' — No active Batch Normalization constraints detected.",
            "Aggressively reducing search dimensions to avoid sub-optimal convergence.",
            "Freezing: batch size at 8 or 16, weight decay (wd), dropout, and schedule to 'constant'.",
            "Priority 1: Invest 100% of active search updates exclusively on the Learning Rate (lr).",
        ]

        self.declare(SearchSpaceChoice(mode="extreme_focus", confidence=confidence_from_score(5)))
        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Extreme Focus Mode",
                justification=f"Severe constraints on '{arch}' require freezing all dimensions except the core learning rate.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    
    # BRANCH 2: SUFFICIENT RESOURCES -> Architecture Check
    # 2a. Transformer/LLM/RNN + Warmup Node
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=True),
        OptimizationBudget(max_trials=MATCH.trials),
        ModelArchitecture(architecture_type=MATCH.arch),
        TEST(lambda trials, arch: trials > 20 and arch in _SEQUENTIAL_ARCHITECTURES),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=75,
    )
    def space_sequential_warmup(self, arch):
        reasons = [
            "Sufficient resources verified (GPU enabled AND max_trials > 20).",
            f"Architecture '{arch}' is sequential/attention-based.",
            "Priority 1: Active Search on Learning Rate (lr).",
            "Priority 2: Active Search on Weight Decay (wd) — AdamW decoupled regularization is a first-class hyperparameter (Loshchilov & Hutter, 2019).",
            "Freezing: Batch size and dropout at verified default states.",
            "Mandatory: LR Warmup period must be injected during trials to avoid early gradient explosion (Vaswani et al., 2017).",
        ]

        self.declare(SearchSpaceChoice(mode="transformer_balanced", confidence=confidence_from_score(5)))
        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Adaptive Context Focus",
                justification=f"Architecture '{arch}' requires co-tuning lr and wd with mandatory learning rate warmup.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    # 2b. CNN/MLP + High Overfitting Risk Node
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=True),
        OptimizationBudget(max_trials=MATCH.trials),
        ModelArchitecture(architecture_type=MATCH.arch, model_scale="large"),
        DatasetProfile(data_type=MATCH.dtype, sample_count=MATCH.count),
        TEST(lambda dtype, count: 
            (dtype == "image" and count < 2000) or 
            (dtype == "text" and count < 5000) or
            (dtype == "tabular" and count < 1000)
        ),
        TEST(lambda trials, arch: trials > 20 and arch in _SPATIAL_ARCHITECTURES),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=72,
    )
    def space_spatial_high_overfitting(self, arch):
        reasons = [
            "Sufficient resources verified (GPU enabled AND max_trials > 20).",
            f"High Overfitting Risk: Large '{arch}' capacity operating on a small dataset profile.",
            "Priority 1: Active Search on Learning Rate (lr).",
            f"Priority 2: Active Search on Dropout — directly suppresses overfitting in '{arch}' (Taram et al., 2024).",
            "Priority 3: Active Search on Batch Size for regularizing noise control.",
            "Priority 4: Weight Decay (wd) — lower priority for non-AdamW optimizers.",
        ]

        self.declare(SearchSpaceChoice(mode="cnn_high_risk", confidence=confidence_from_score(5)))
        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Spatial Regularization Mode",
                justification=f"High overfitting risk in '{arch}' requires prioritizing dropout and batch noise before weight decay.",
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    # 2c. CNN/MLP + Standard Node
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        ComputeConstraints(has_gpu=True),
        OptimizationBudget(max_trials=MATCH.trials),
        ModelArchitecture(architecture_type=MATCH.arch),
        TEST(lambda trials, arch: trials > 20 and arch in _SPATIAL_ARCHITECTURES),
        NOT(SearchSpaceChoice()),
        NOT(Recommendation(category="search_space")),
        salience=70,
    )
    def space_spatial_standard(self, arch):
        reasons = [
            "Sufficient resources verified (GPU enabled AND max_trials > 20).",
            f"Standard '{arch}' profile: dataset capacity matches model footprint.",
            "Priority 1: Active Search on Learning Rate (lr).",
            f"Priority 2: Active Search on Dropout — higher sensitivity than batch size in '{arch}' (Taram et al., 2024).",
            "Priority 3: Active Search on Batch Size.",
            "Freezing: Weight Decay (wd) at safe defaults — low impact for non-AdamW optimizers.",
        ]

        self.declare(SearchSpaceChoice(mode="cnn_standard", confidence=confidence_from_score(4)))
        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Spatial Standard Mode",
                justification=f"Standard '{arch}' setups prioritize lr, dropout, then batch size based on fANOVA sensitivity ordering.",
                reasons=reasons,
                priority=Priority.MEDIUM.value,
                confidence=confidence_from_score(4),
            )
        )

    # Fallback
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        NOT(SearchSpaceChoice()),
        salience=50,
    )
    def space_fallback_safe_default(self):
        reasons = [
            "No specific architecture or resource pattern matched.",
            "Falling back to general-purpose search strategy.",
            "Priority 1: Active Search on Learning Rate (lr) — highest impact parameter across all architectures.",
            "Priority 2: Active Search on Batch Size — second most impactful parameter in general deep learning settings.",
            "Freezing: Weight decay, dropout, and schedule at safe baseline defaults.",
        ]

        self.declare(SearchSpaceChoice(mode="safe_default", confidence=confidence_from_score(3)))
        self.declare(
            Recommendation(
                category="search_space",
                recommendation="Safe Default Space Mode",
                justification="General-purpose configuration applied when no strict context rules match.",
                reasons=reasons,
                priority=Priority.MEDIUM.value,
                confidence=confidence_from_score(3),
            )
        )

    # AdamW Upgrade
    @Rule(
        ReasoningStage(current=ReasoningStageId.SEARCH_SPACE.value),
        SearchSpaceChoice(mode=MATCH.mode),
        OptimizerChoice(optimizer=Optimizer.ADAMW.value),
        TEST(lambda mode: mode in {
            "cnn_standard",
            "cnn_high_risk",
            "safe_default",
        }),
        salience=60,
    )
    def upgrade_wd_for_adamw(self, mode):
        self.declare(
            Recommendation(
                category="search_space_wd_note",
                recommendation="AdamW Weight Decay Upgrade",
                justification="AdamW decoupled wd is meaningful — consider adding it to active search.",
                reasons=[
                    f"Current mode '{mode}' has wd frozen or deprioritized.",
                    "OptimizerChoice is AdamW — decoupled weight decay is a first-class hyperparameter.",
                    "Recommend promoting wd to active search if trial budget allows.",
                ],
                priority=Priority.MEDIUM.value,
                confidence=confidence_from_score(3),
            )
        )
