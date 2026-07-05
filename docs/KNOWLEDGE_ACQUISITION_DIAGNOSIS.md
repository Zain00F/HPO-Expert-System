# Knowledge Acquisition: Post-Training Diagnosis

## Reasoning chain

```
Training observations (loss, accuracy, NaN, Inf, epochs)
        ↓
Diagnosis          ← inferred only from observations
        ↓
Possible cause     ← inferred from context facts (model, config, dataset)
        ↓
Recommendation     ← linked to each cause
```

Observations never skip directly to recommendations.

---

## 1. Overfitting

### Expert reasoning

A model overfits when it improves on training data while validation performance stalls or degrades — it has learned training-specific patterns rather than general features. The strongest single-snapshot signal is the **generalization gap**.

Stronger evidence (learning-curve view, as in early stopping): training loss still decreasing while validation loss increases.

### Evidence (observations only)

| Signal | Expert basis |
|--------|----------------|
| Validation loss >> training loss | Generalization gap |
| Training accuracy >> validation accuracy | Same gap, classification view |

### Implementation heuristics

| Constant | Role |
|----------|------|
| `MAX_HEALTHY_LOSS_RATIO` | Val loss > 25% worse than train → gap large enough to flag |
| `MIN_OVERFIT_ACC_GAP` | ≥ 10 percentage points train−val accuracy |

### Causes (context facts — not evidence of overfitting)

| Context | Cause |
|---------|--------|
| Large model / high parameter count | Model too complex |
| Small dataset | Insufficient data |
| Low dropout / weight decay | Weak regularization |
| Many epochs completed | Trained too long |

### Recommendations

Increase dropout · Increase weight decay · Reduce model complexity · Enable early stopping · Collect more data · Use data augmentation

---

## 2. Underfitting

### Expert reasoning

Underfitting: the model cannot fit even the training data. Train and validation performance are both poor with **little generalization gap**.

### Evidence (observations only)

| Signal | Expert basis |
|--------|----------------|
| Training loss high, validation loss similar | Poor everywhere, small gap |
| Training and validation accuracy both low, similar | Same pattern |

### Implementation heuristics

| Constant | Role |
|----------|------|
| `MIN_POOR_TRAINING_LOSS` | Loss “remains high” (unnormalized; task-dependent) |
| `MAX_UNDERFIT_LOSS_REL_GAP` | Val within 15% of train → “similar” |
| `POOR_ACCURACY_CEILING` | Both accuracies below 60% |
| `MAX_UNDERFIT_ACC_GAP` | Gap < 5 percentage points |

### Causes (context facts)

Model too simple · Learning rate too small · Too few epochs · Excessive regularization · Optimization not converged

### Recommendations

Increase model capacity · Train longer · Increase learning rate · Reduce regularization · Use a better optimizer

---

## 3. NaN loss

### Expert reasoning

NaN is **observed**, not inferred. It indicates numerical instability has already occurred.

### Evidence

`nan_detected = True`

### Causes (context)

Learning rate too high · Exploding gradients · Invalid inputs · Bad initialization · Corrupted data

### Recommendations

Reduce learning rate · Enable gradient clipping · Normalize data · Check dataset · Verify loss implementation

---

## 4. Infinite loss

### Expert reasoning

Inf is **observed**. Indicates overflow or unstable optimization.

### Evidence

`inf_detected = True`

### Causes (context)

Exploding gradients · Overflow · Invalid loss function · Numerical instability

### Recommendations

Reduce learning rate · Enable gradient clipping · Check numerical stability · Inspect loss implementation

---

## CLI observations (user input)

| Observation | Purpose |
|-------------|---------|
| Training loss | Compare with validation loss |
| Validation loss | Detect generalization gap |
| Training accuracy (0–100%) | Compare with validation accuracy |
| Validation accuracy (0–100%) | Detect generalization gap |
| NaN encountered? | Direct diagnosis |
| Inf encountered? | Direct diagnosis |
| Epochs completed | Cause: trained too long / too few epochs |

Context facts (learning rate, dropout, parameter count, dataset size) are declared separately and used **only in cause rules**, not in diagnosis identification.
