"""
Implementation heuristics for post-training diagnosis rules.

Expert concepts (generalization gap, poor performance on both train and val)
are defined in docs/KNOWLEDGE_ACQUISITION_DIAGNOSIS.md and standard ML texts.
The values here are discrete decision boundaries required by Experta's matcher —
they are NOT universal laws of machine learning.
"""


class HeuristicThresholds:
    """
    Translate fuzzy expert concepts into discrete trigger points for the KBS.

    Theory vs implementation
    --------------------------
    * Theory:   validation worse than training → generalization gap → overfitting
                (Goodfellow et al.; early-stopping literature)
    * Heuristic: MAX_HEALTHY_LOSS_RATIO — chosen so the engine can fire a rule

    * Theory:   poor performance on train AND val with small gap → underfitting
    * Heuristic: POOR_ACCURACY_CEILING, MAX_UNDERFIT_ACC_GAP — practitioner defaults
    """

    # --- Overfitting identification ---
    # Val loss noticeably worse than train loss (generalization gap).
    MAX_HEALTHY_LOSS_RATIO = 1.25

    # Train accuracy exceeds val accuracy by this many percentage points.
    MIN_OVERFIT_ACC_GAP = 10.0

    # --- Underfitting identification ---
    # Both accuracies below this (%) are considered poor performance.
    POOR_ACCURACY_CEILING = 60.0

    # Train–val accuracy gap below this (%) → "similar" poor performance.
    MAX_UNDERFIT_ACC_GAP = 5.0

    # Unnormalized loss floor for "remains high" (task-dependent; documented heuristic).
    MIN_POOR_TRAINING_LOSS = 1.0

    # Relative train–val loss gap below this fraction → "similar" losses.
    MAX_UNDERFIT_LOSS_REL_GAP = 0.15

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
