from __future__ import annotations

from typing import Any

from genetic_algorithm import GeneticAlgorithmAssignmentOptimizer
from hybrid_hs_ga import GuidedMutationSeededGeneticAlgorithmAssignmentOptimizer
from models import OptimizerOptions


class SkillGuidedGeneticAlgorithmAssignmentOptimizer(
    GuidedMutationSeededGeneticAlgorithmAssignmentOptimizer
):
    """Standalone GA baseline with TA-3 skill-guided mutation only.

    Initialization and all non-mutation GA operators are inherited from the
    standalone GA baseline. This isolates the effect of choosing mutation genes
    using task skill weakness, without an HS phase or transferred candidates.
    """

    def __init__(
        self,
        tasks: list[dict[str, Any]],
        resources: list[dict[str, Any]],
        evaluator: Any,
        options: OptimizerOptions,
    ) -> None:
        super().__init__(
            tasks=tasks,
            resources=resources,
            evaluator=evaluator,
            options=options,
            seed_candidates=[],
        )

    def _initial_population(self):
        # Use the exact baseline GA initialization, including its greedy seed.
        return GeneticAlgorithmAssignmentOptimizer._initial_population(self)
