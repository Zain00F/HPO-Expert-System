# Decision Trees (Prototype v0.1)

## HPO method narrowing

```mermaid
flowchart TD
    Start([ComputeConstraints known]) --> D{dimensions <= 4 AND trial_cost low?}
    D -->|yes| G[Grid Search]
    D -->|no| E{trial_cost high AND max_trials <= 40?}
    E -->|yes| H[Hyperband]
    E -->|no| B{dimensions >= 6 AND costly/high budget?}
    B -->|yes| BO[Bayesian Optimization]
    B -->|no| R[Random Search]
```

## Optimizer narrowing

```mermaid
flowchart TD
    Start([ModelArchitecture known]) --> F{transformer/llm/rnn OR params >= 1M?}
    F -->|yes| AD[Adaptive family]
    F -->|no| C{cnn or mlp?}
    C -->|yes| AD2[Adaptive default]
    C -->|no| NA[Non-adaptive SGD family]
    AD --> T{transformer + attention?}
    T -->|yes| AW[AdamW]
    T -->|no| A[Adam]
    NA --> M[SGD + Momentum]
```

## Planned stages (not yet implemented)

- Search-space prioritization (lr, batch, wd, dropout order)
- Range suggestion per optimizer
- Diagnosis branch from `TrainingObservation`
