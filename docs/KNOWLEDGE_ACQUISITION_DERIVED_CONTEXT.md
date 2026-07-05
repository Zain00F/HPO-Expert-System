# Knowledge Acquisition: Derived Dataset Size and Compute Budget

## Principle

The client provides **objective measurements**. The expert system **derives** categorical expert knowledge using Experta rules.

Subjective labels (small/medium/large, low/medium/high) never come from the user.

Implementation thresholds live in `utils/heuristic_thresholds.py` and are documented as **engineering heuristics**, not universal constants.

---

## Part 1 — Dataset Size

### Observations (client input)

| Field | Fact |
|-------|------|
| Total sample count | `DatasetProfile.sample_count` |
| Number of classes (optional) | `DatasetProfile.classification_categories` |

### Expert reasoning

Dataset scale is domain-dependent. When class count is known, **samples per class** is the meaningful signal for classification tasks. Otherwise, fall back to total sample count conventions inspired by common benchmark scales.

### Derivation chain

```
DatasetProfile
    ↓
DatasetSize(size = small | medium | large)
```

### Rules

**Classification path** (when `classification_categories` or legacy `num_classes` > 0):

| Samples per class | Derived size |
|------------------:|--------------|
| < 100 | small |
| 100 – 1,000 | medium |
| > 1,000 | large |

**Fallback path** (no class count):

| Total samples | Derived size |
|--------------:|--------------|
| < 10,000 | small |
| 10,000 – 100,000 | medium |
| > 100,000 | large |

---

## Part 2 — Trial Cost and Compute Budget

### Observations (client input)

| Field | Fact |
|-------|------|
| Average training time per trial (minutes) | `ComputeConstraints.trial_time_minutes` |
| Time budget for HPO (hours) | `ComputeConstraints.time_budget_hours` |
| Maximum trials | `OptimizationBudget.max_trials` |

### Expert reasoning

Compute budget represents how much hyperparameter search is realistically affordable. Hardware specs are poor proxies — **trial duration × trial count vs available time** summarizes affordability regardless of GPU model.

### Derivation chain

```
ComputeConstraints.trial_time_minutes
    ↓
TrialCost(level = low | medium | high)

ComputeConstraints.trial_time_minutes × OptimizationBudget.max_trials
    ↓
RequiredSearchTime(hours)

RequiredSearchTime vs ComputeConstraints.time_budget_hours
    ↓
ComputeBudget(level = low | medium | high)
```

### Trial cost rules

| Average trial time | Derived trial cost |
|-------------------:|--------------------|
| < 5 minutes | low |
| 5 – 30 minutes | medium |
| > 30 minutes | high |

### Compute budget rules

Let `required = trial_time_minutes × max_trials / 60` and `available = time_budget_hours`.

| Condition | Derived compute budget |
|-----------|------------------------|
| `required > available` | low — search exceeds time budget |
| `available × 0.85 ≤ required ≤ available` | medium — tight but feasible |
| `required < available × 0.85` | high — comfortable headroom |

The 15% headroom (`COMPUTE_BUDGET_HEADROOM`) is an implementation tolerance for “approximately equal” vs “comfortably fits”.

---

## Downstream usage

Pre-training and diagnosis rules match on derived facts:

- `DatasetSize(size=...)` — e.g. insufficient data cause, search-space risk
- `TrialCost(level=...)` — e.g. Grid Search vs Hyperband selection
- `ComputeBudget(level=...)` — e.g. Bayesian optimization, hybrid refinement

Never match subjective fields on `ComputeConstraints` directly.
