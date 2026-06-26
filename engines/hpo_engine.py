from __future__ import annotations

from typing import Any, Iterable

import hpo_expert  # noqa: F401 — apply collections.Mapping patch before experta

from experta import KnowledgeEngine

from hpo_expert.facts.reasoning import Recommendation
from hpo_expert.rules.hpo_methods import HPOMethodRules
from hpo_expert.rules.optimizers import OptimizerRules
from hpo_expert.rules.stage_control import StageControlRules


class HPOExpertEngine(StageControlRules, HPOMethodRules, KnowledgeEngine):

    def recommendations_as_dicts(self) -> list[dict]:
        """Serialize recommendations for reporting."""
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