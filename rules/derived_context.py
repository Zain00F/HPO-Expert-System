from experta import MATCH, NOT, TEST, Rule

from hpo_expert.facts.context import ComputeConstraints, OptimizationBudget
from hpo_expert.facts.derived import ComputeBudget, DatasetSize, RequiredSearchTime, TrialCost
from hpo_expert.facts.model import DatasetProfile
from hpo_expert.utils.enums import ComputeBudget as ComputeBudgetLevel
from hpo_expert.utils.enums import DatasetSize as DatasetSizeLevel
from hpo_expert.utils.enums import TrialCostLevel
from hpo_expert.utils.heuristic_thresholds import HeuristicThresholds as HT


def _class_count(categories, legacy) -> int | None:
    if categories is not None and categories > 0:
        return categories
    if legacy is not None and legacy > 0:
        return legacy
    return None


class DerivedContextRules:
    """Mixin: infer DatasetSize, TrialCost, and ComputeBudget from observations."""

    # ------------------------------------------------------------------
    # Trial cost from average training duration
    # ------------------------------------------------------------------

    @Rule(
        ComputeConstraints(trial_time_minutes=MATCH.t),
        TEST(lambda t: t < HT.TRIAL_TIME_LOW_MAX),
        NOT(TrialCost()),
        salience=210,
    )
    def derive_trial_cost_low(self):
        self.declare(TrialCost(level=TrialCostLevel.LOW.value))

    @Rule(
        ComputeConstraints(trial_time_minutes=MATCH.t),
        TEST(lambda t: HT.TRIAL_TIME_LOW_MAX <= t <= HT.TRIAL_TIME_MEDIUM_MAX),
        NOT(TrialCost()),
        salience=210,
    )
    def derive_trial_cost_medium(self):
        self.declare(TrialCost(level=TrialCostLevel.MEDIUM.value))

    @Rule(
        ComputeConstraints(trial_time_minutes=MATCH.t),
        TEST(lambda t: t > HT.TRIAL_TIME_MEDIUM_MAX),
        NOT(TrialCost()),
        salience=210,
    )
    def derive_trial_cost_high(self):
        self.declare(TrialCost(level=TrialCostLevel.HIGH.value))

    # ------------------------------------------------------------------
    # Required search time (intermediate)
    # ------------------------------------------------------------------

    @Rule(
        ComputeConstraints(trial_time_minutes=MATCH.t),
        OptimizationBudget(max_trials=MATCH.n),
        NOT(RequiredSearchTime()),
        salience=209,
    )
    def derive_required_search_time(self, t, n):
        self.declare(RequiredSearchTime(hours=(t * n) / 60.0))

    # ------------------------------------------------------------------
    # Compute budget from required vs available time
    # ------------------------------------------------------------------

    @Rule(
        RequiredSearchTime(hours=MATCH.req),
        ComputeConstraints(time_budget_hours=MATCH.avail),
        TEST(
            lambda req, avail: req > avail
        ),
        NOT(ComputeBudget()),
        salience=208,
    )
    def derive_compute_budget_low(self):
        self.declare(ComputeBudget(level=ComputeBudgetLevel.LOW.value))

    @Rule(
        RequiredSearchTime(hours=MATCH.req),
        ComputeConstraints(time_budget_hours=MATCH.avail),
        TEST(
            lambda req, avail: (
                req <= avail
                and req >= avail * (1.0 - HT.COMPUTE_BUDGET_HEADROOM)
            )
        ),
        NOT(ComputeBudget()),
        salience=207,
    )
    def derive_compute_budget_medium(self):
        self.declare(ComputeBudget(level=ComputeBudgetLevel.MEDIUM.value))

    @Rule(
        RequiredSearchTime(hours=MATCH.req),
        ComputeConstraints(time_budget_hours=MATCH.avail),
        TEST(
            lambda req, avail: req < avail * (1.0 - HT.COMPUTE_BUDGET_HEADROOM)
        ),
        NOT(ComputeBudget()),
        salience=206,
    )
    def derive_compute_budget_high(self):
        self.declare(ComputeBudget(level=ComputeBudgetLevel.HIGH.value))

    # ------------------------------------------------------------------
    # Dataset size — per-class path (classification)
    # ------------------------------------------------------------------

    @Rule(
        DatasetProfile(
            sample_count=MATCH.n,
            classification_categories=MATCH.c,
            num_classes=MATCH.legacy,
        ),
        TEST(lambda n, c, legacy: (_class_count(c, legacy) or 0) > 0),
        TEST(
            lambda n, c, legacy: (n / _class_count(c, legacy)) < HT.PER_CLASS_SMALL
        ),
        NOT(DatasetSize()),
        salience=205,
    )
    def derive_dataset_size_small_per_class(self):
        self.declare(DatasetSize(size=DatasetSizeLevel.SMALL.value))

    @Rule(
        DatasetProfile(
            sample_count=MATCH.n,
            classification_categories=MATCH.c,
            num_classes=MATCH.legacy,
        ),
        TEST(lambda n, c, legacy: (_class_count(c, legacy) or 0) > 0),
        TEST(
            lambda n, c, legacy: HT.PER_CLASS_SMALL
            <= (n / _class_count(c, legacy))
            <= HT.PER_CLASS_MEDIUM
        ),
        NOT(DatasetSize()),
        salience=205,
    )
    def derive_dataset_size_medium_per_class(self):
        self.declare(DatasetSize(size=DatasetSizeLevel.MEDIUM.value))

    @Rule(
        DatasetProfile(
            sample_count=MATCH.n,
            classification_categories=MATCH.c,
            num_classes=MATCH.legacy,
        ),
        TEST(lambda n, c, legacy: (_class_count(c, legacy) or 0) > 0),
        TEST(
            lambda n, c, legacy: (n / _class_count(c, legacy)) > HT.PER_CLASS_MEDIUM
        ),
        NOT(DatasetSize()),
        salience=205,
    )
    def derive_dataset_size_large_per_class(self):
        self.declare(DatasetSize(size=DatasetSizeLevel.LARGE.value))

    # ------------------------------------------------------------------
    # Dataset size — total-sample fallback
    # ------------------------------------------------------------------

    @Rule(
        DatasetProfile(
            sample_count=MATCH.n,
            classification_categories=MATCH.c,
            num_classes=MATCH.legacy,
        ),
        TEST(lambda n, c, legacy: _class_count(c, legacy) is None),
        TEST(lambda n, c, legacy: n < HT.TOTAL_SAMPLES_SMALL),
        NOT(DatasetSize()),
        salience=204,
    )
    def derive_dataset_size_small_total(self):
        self.declare(DatasetSize(size=DatasetSizeLevel.SMALL.value))

    @Rule(
        DatasetProfile(
            sample_count=MATCH.n,
            classification_categories=MATCH.c,
            num_classes=MATCH.legacy,
        ),
        TEST(lambda n, c, legacy: _class_count(c, legacy) is None),
        TEST(
            lambda n, c, legacy: HT.TOTAL_SAMPLES_SMALL
            <= n
            <= HT.TOTAL_SAMPLES_MEDIUM
        ),
        NOT(DatasetSize()),
        salience=204,
    )
    def derive_dataset_size_medium_total(self):
        self.declare(DatasetSize(size=DatasetSizeLevel.MEDIUM.value))

    @Rule(
        DatasetProfile(
            sample_count=MATCH.n,
            classification_categories=MATCH.c,
            num_classes=MATCH.legacy,
        ),
        TEST(lambda n, c, legacy: _class_count(c, legacy) is None),
        TEST(lambda n, c, legacy: n > HT.TOTAL_SAMPLES_MEDIUM),
        NOT(DatasetSize()),
        salience=204,
    )
    def derive_dataset_size_large_total(self):
        self.declare(DatasetSize(size=DatasetSizeLevel.LARGE.value))
