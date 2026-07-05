class HeuristicThresholds:
    MAX_HEALTHY_LOSS_RATIO = 1.25
    MIN_OVERFIT_ACC_GAP = 10.0
    POOR_ACCURACY_CEILING = 60.0
    MAX_UNDERFIT_ACC_GAP = 5.0
    MIN_POOR_TRAINING_LOSS = 1.0
    MAX_UNDERFIT_LOSS_REL_GAP = 0.15

    # --- Derived dataset size (engineering heuristics; domain-dependent) ---
    # Per-class thresholds when classification_categories is known
    PER_CLASS_SMALL = 100
    PER_CLASS_MEDIUM = 1000
    # Total-sample fallback when class count is unavailable
    TOTAL_SAMPLES_SMALL = 10_000
    TOTAL_SAMPLES_MEDIUM = 100_000

    # --- Derived trial cost from average training time (minutes) ---
    TRIAL_TIME_LOW_MAX = 5.0
    TRIAL_TIME_MEDIUM_MAX = 30.0

    # --- Derived compute budget: required vs available search time ---
    # High = comfortable headroom; Medium = tight but fits; Low = exceeds budget
    COMPUTE_BUDGET_HEADROOM = 0.15

    # --- Overfitting causes (context facts) ---
    WEAK_DROPOUT_RATE = 0.1
    WEAK_WEIGHT_DECAY = 1e-4
    TOO_MANY_EPOCHS = 50
    LARGE_PARAMETER_COUNT = 1_000_000

    # --- Underfitting causes (context facts) ---
    LR_TOO_LOW = 1e-4
    TOO_FEW_EPOCHS = 10
    HIGH_DROPOUT_RATE = 0.5
    HIGH_WEIGHT_DECAY = 0.1

    # --- NaN / Inf causes (context facts) ---
    LR_TOO_HIGH_NAN = 0.01
    LR_TOO_HIGH_INF = 0.001
