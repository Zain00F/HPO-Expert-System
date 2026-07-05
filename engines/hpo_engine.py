from __future__ import annotations

from typing import Any, Iterable

import hpo_expert  # noqa: F401 — apply collections.Mapping patch before experta

from experta import KnowledgeEngine

from hpo_expert.facts.reasoning import Recommendation
from hpo_expert.rules.diagnosis import DiagnosisRules
from hpo_expert.rules.hpo_methods import HPOMethodRules
from hpo_expert.rules.optimizers import OptimizerRules
from hpo_expert.rules.stage_control import StageControlRules
from hpo_expert.rules.search_space import SearchSpaceRules


class HPOExpertEngine(
    StageControlRules,
    DiagnosisRules,
    HPOMethodRules,
    OptimizerRules,
    SearchSpaceRules,
    KnowledgeEngine,
):


    def recommendations_as_dicts(self) -> list[dict]:
        out = []
        for fact in self.facts.values():
            if type(fact).__name__ == "Recommendation":
                out.append(
                    {
                        "category": fact["category"],
                        "recommendation": fact["recommendation"],
                        "justification": fact["justification"],
                        "reasons": list(fact.get("reasons") or []),
                        "priority": fact["priority"],
                        "confidence": fact.get("confidence", "medium"),
                    }
                )
        return out


def run_consultation(engine: KnowledgeEngine, facts: Iterable[Any]) -> KnowledgeEngine:
    engine.reset()
    for fact in facts:
        engine.declare(fact)
    engine.run()
    return engine