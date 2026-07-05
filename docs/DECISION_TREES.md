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



## Search-space prioritization (lr, batch, wd, dropout order, lr_schedule , lr_warmup)
flowchart TD
    Start([System Inputs Known]) --> C1{has_gpu == False OR max_trials <= 20?<br>Strict Resource Check}
    
    %% Strict Resource Branch
    C1 -->|Yes| R1{uses_batch_norm == True AND architecture_type == cnn?}
    R1 -->|Yes| BN_Safe["[Resource & BN Constraint Mode]<br>Freeze: batch = 16, schedule = constant, dropout = 0<br>Priority 1: lr<br>Priority 2: wd"]
    R1 -->|No| Safe_Def["[Extreme Focus Mode]<br>Freeze: batch = 8/16, wd, dropout, schedule = constant<br>Priority 1: lr"]
    
    %% High Resource Branch
    C1 -->|No| G1{Optimization Goal?}
    
    G1 -->|maximize_accuracy| Goal_Acc["[Accuracy Mode - fANOVA 65% Effect]<br>Priority 1: batch<br>Priority 2: lr<br>Priority 3: lr_schedule<br>Freeze: dropout order, wd"]
    
    G1 -->|minimize_training_time| Goal_Time["[Speed Mode - fANOVA 54% Effect]<br>Priority 1: lr<br>Freeze: batch to max hardware limits, wd, dropout, schedule = fixed_linear"]
    
    %% Balanced Goal Branch
    G1 -->|balanced| A1{Architecture Type?}
    
    A1 -->|transformer| Arch_Trans["[Adaptive Context Focus]<br>Priority 1: lr<br>Priority 2: wd<br>Freeze: batch, dropout order<br>Mandatory: Include LR Warmup"]
    
    A1 -->|cnn| O1{High Overfitting Risk?<br>model_scale == large AND dataset_size == small}
    O1 -->|Yes| CNN_Reg["[Spatial Regularization Mode]<br>Priority 1: lr<br>Priority 2: batch<br>Priority 3: wd & dropout"]
    O1 -->|No| CNN_Standard["[Spatial Standard Mode]<br>Priority 1: lr<br>Priority 2: batch<br>Freeze: wd, dropout"]

    %% Catch-All Safe Fallback Mechanism
    A1 -->|other/unmapped| FallbackNode["[Safe Default Space Mode]<br>Priority 1: lr<br>Priority 2: batch<br>Freeze: wd, dropout, schedule"]
    G1 -->|other/unmapped| FallbackNode

    
## Planned stages (not yet implemented)

- Range suggestion per optimizer
- Diagnosis branch from `TrainingObservation`

## Post-training diagnosis (v0.2)

Knowledge acquisition: see [KNOWLEDGE_ACQUISITION_DIAGNOSIS.md](KNOWLEDGE_ACQUISITION_DIAGNOSIS.md).
Thresholds: `utils/heuristic_thresholds.py`.

```mermaid
flowchart TD
    Start([TrainingObservation declared]) --> N{nan_detected?}
    N -->|yes| NAN[Diagnosis: NaN Loss]
    N -->|no| I{inf_detected?}
    I -->|yes| INF[Diagnosis: Inf Loss]
    I -->|no| O{val_loss > train_loss × HT.MAX_HEALTHY_LOSS_RATIO?}
    O -->|yes| OF[Diagnosis: Overfitting]
    O -->|no| U{high loss + similar val OR poor similar acc?}
    U -->|yes| UF[Diagnosis: Underfitting]
    NAN --> C[PossibleCause rules + context facts]
    INF --> C
    OF --> C
    UF --> C
    C --> R[Recommendation: diagnosis_recommendation]
```