from experta import MATCH, NOT, TEST, Rule

from hpo_expert.facts.context import ComputeConstraints, OptimizationBudget, ProjectContext
from hpo_expert.facts.derived import ComputeBudget, TrialCost
from hpo_expert.facts.reasoning import (
    HPOMethodChoice,
    ReasoningStage,
    Recommendation,
)
from hpo_expert.utils.enums import (
    ComputeBudget as ComputeBudgetLevel,
    HPOMethod,
    Priority,
    ReasoningStageId,
    TrialCostLevel,
)
from hpo_expert.utils.scoring import confidence_from_score


class HPOMethodRules:
    @Rule(
        ReasoningStage(current=ReasoningStageId.HPO_METHOD.value),
        ComputeConstraints(
            search_space_dimensions=MATCH.d,
            time_budget_hours=MATCH.hours,
        ),
        TrialCost(level=TrialCostLevel.LOW.value),
        TEST(lambda d, hours: d <= 4 and hours >= 2),
        NOT(HPOMethodChoice()),
        salience=85,
    )
    def method_grid_search(self, d, hours):
        reasons = [
            "Search space has few dimensions (<=4)",
            "Each trial is relatively cheap (derived from short average training time)",
            "Exhaustive grid is feasible within your time budget",
        ]

        self.declare(
            HPOMethodChoice(
                method=HPOMethod.GRID_SEARCH.value,
                confidence=confidence_from_score(5),
            )
        )

        self.declare(
            Recommendation(
                category="hpo_method",
                recommendation="Grid Search",
                justification=(
                    "Small search space and low per-trial cost "
                    "favor exhaustive search."
                ),
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.HPO_METHOD.value),
        ComputeConstraints(
            search_space_dimensions=MATCH.d,
            time_budget_hours=MATCH.h,
        ),
        TrialCost(level=MATCH.cost),
        ComputeBudget(level=MATCH.budget),
        TEST(
            lambda d, cost, budget: (
                d >= 5
                and (
                    cost == TrialCostLevel.HIGH.value
                    or budget == ComputeBudgetLevel.HIGH.value
                )
            )
        ),
        NOT(HPOMethodChoice()),
        salience=84,
    )
    def method_bayesian(self, d, cost, budget, h):
        reasons = [
            f"Search space is large ({d} dimensions) "
            f"or trials are expensive ({cost})",
            "Bayesian optimization samples promising regions "
            "instead of uniform random points",
            f"Limited wall-clock budget ({h}h) benefits "
            "sample-efficient search",
        ]

        self.declare(
            HPOMethodChoice(
                method=HPOMethod.BAYESIAN_OPTIMIZATION.value,
                confidence=confidence_from_score(5),
            )
        )

        self.declare(
            Recommendation(
                category="hpo_method",
                recommendation="Bayesian Optimization",
                justification=(
                    "Sample-efficient search for expensive, "
                    "high-dimensional tuning."
                ),
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(5),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.HPO_METHOD.value),
        TrialCost(level=TrialCostLevel.HIGH.value),
        OptimizationBudget(max_trials=MATCH.trials),
        TEST(lambda trials: trials <= 40),
        NOT(HPOMethodChoice()),
        salience=86,
    )
    def method_hyperband_trial_cost(self, trials):
        reasons = [
            "Training runs are costly relative to your budget",
            "Hyperband allocates more trials to promising "
            "configurations via successive halving",
            "Pairs well with early stopping between brackets",
        ]

        self.declare(
            HPOMethodChoice(
                method=HPOMethod.HYPERBAND.value,
                confidence=confidence_from_score(4),
            )
        )

        self.declare(
            Recommendation(
                category="hpo_method",
                recommendation="Hyperband",
                justification=(
                    "Multi-fidelity scheduling when trials are "
                    "expensive and budgets are tight."
                ),
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.HPO_METHOD.value),
        TrialCost(level=TrialCostLevel.HIGH.value),
        ComputeConstraints(time_budget_hours=MATCH.h),
        TEST(lambda h: h < 48),
        NOT(HPOMethodChoice()),
        salience=82,
    )
    def method_hyperband_time(self, h):
        reasons = [
            "Training runs are costly relative to your budget",
            "Hyperband allocates more trials to promising "
            "configurations via successive halving",
            "Limited wall-clock budget benefits early stopping",
        ]

        self.declare(
            HPOMethodChoice(
                method=HPOMethod.HYPERBAND.value,
                confidence=confidence_from_score(4),
            )
        )

        self.declare(
            Recommendation(
                category="hpo_method",
                recommendation="Hyperband",
                justification=(
                    "Multi-fidelity scheduling when training time "
                    "is constrained."
                ),
                reasons=reasons,
                priority=Priority.HIGH.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ReasoningStage(current=ReasoningStageId.HPO_METHOD.value),
        NOT(HPOMethodChoice()),
        salience=50,
    )
    def method_random_search(self):
        reasons = [
            "Moderate constraints — random search explores broadly "
            "with simple implementation",
            "Often outperforms grid search in high dimensions "
            "(Bergstra & Bengio, 2012)",
            "Good baseline before investing in Bayesian or "
            "bandit methods",
        ]

        self.declare(
            HPOMethodChoice(
                method=HPOMethod.RANDOM_SEARCH.value,
                confidence=confidence_from_score(4),
            )
        )

        self.declare(
            Recommendation(
                category="hpo_method",
                recommendation="Random Search",
                justification=(
                    "Balanced default when neither exhaustive "
                    "search nor Bayesian optimization is clearly best."
                ),
                reasons=reasons,
                priority=Priority.MEDIUM.value,
                confidence=confidence_from_score(4),
            )
        )

    @Rule(
        ProjectContext(optimization_goal="maximize_accuracy"),
        ComputeBudget(level=ComputeBudgetLevel.HIGH.value),
        HPOMethodChoice(method=MATCH.selected_method),
        TEST(
            lambda selected_method: selected_method
            in ["bayesian_optimization", "random_search"]
        ),
        salience=20,
    )
    def recommend_hybrid_refinement(self, selected_method):
        reasons = [
            "Derived compute budget is high — planned search fits comfortably within available time.",
            f"After {selected_method} locates the optimal region, a narrow local Grid Search eliminates stochastic noise.",
        ]

        self.declare(
            Recommendation(
                category="advanced_strategy(for hpo method)",
                recommendation="Two-Stage Hybrid Tuning",
                justification=(
                    f"Maximize performance by running a localized, dense Grid Search "
                    f"around the best hyperparameter configuration found by {selected_method}."
                ),
                reasons=reasons,
                priority=Priority.MEDIUM.value,
                confidence=confidence_from_score(4),
            )
        )
