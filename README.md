# HPO Expert System

Rule-based expert consultant for **neural network hyperparameter optimization** — built with Python and [experta](https://github.com/nilp0inter/experta).

This is **not** an AutoML runner. It mimics how an ML expert **reasons**, **narrows** choices in stages, and **explains why**.

## Quick start

```bash
cd "d:\VS Code\kbs"
pip install -r hpo_expert/requirements.txt
python hpo_expert/examples/minimal_run.py expensive_deep
python -m hpo_expert.cli.interactive
```

## Project layout

See [ARCHITECTURE.md](ARCHITECTURE.md) for inference flow, conflict resolution, evaluation plan, and roadmap.

| Package | Role |
|---------|------|
| `facts/` | Evidence + derived conclusions |
| `rules/` | Modular rule mixins (salience-gated) |
| `engines/` | `HPOExpertEngine` composes mixins |
| `utils/` | Enums, scoring, explanations, logging |
| `cli/` | Interactive questionnaire |
| `examples/` | Scenario fixtures + minimal demo |

## Prototype scope (v0.1)

- Staged reasoning: **HPO method** → **optimizer family** → **specific optimizer**
- Explainable `Recommendation` facts with bullet reasons
- Three test scenarios in `examples/scenarios.py`

## Next increments

1. Learning-rate strategy + range rules (`rules/ranges.py`)
2. Training diagnosis (`rules/diagnosis.py`)
3. Search-space prioritization
4. JSON scenario regression tests

## Academic use

- Document rules with domain citations in comments
- Compare engine output to expert rubric on fixed scenarios
- Export `engine.facts` as audit trail for explainability grading
