from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator import AssignmentEvaluator  # noqa: E402
from genetic_algorithm import GeneticAlgorithmAssignmentOptimizer  # noqa: E402
from harmony_search import HarmonySearchAssignmentOptimizer  # noqa: E402
from hybrid_hs_ga import (  # noqa: E402
    HybridHarmonySearchGAAssignmentOptimizer,
    HybridHarmonySearchGuidedMutationGAAssignmentOptimizer,
)
from models import OptimizerOptions, TaskSchedule  # noqa: E402


def make_evaluator() -> AssignmentEvaluator:
    schedule = {
        "T1": TaskSchedule("T1", 0.0, 2.0, 2.0, 0),
        "T2": TaskSchedule("T2", 2.0, 4.0, 2.0, 1),
    }
    tasks = [
        {
            "taskId": "T1",
            "estimatedHours": 2,
            "requiredSkills": [{"skillName": "Q1", "level": 1}],
            "kpiImpacts": [{"kpiCode": "PROJECT_COMPLETION", "weight": 0.5}],
        },
        {
            "taskId": "T2",
            "estimatedHours": 2,
            "dependencies": ["T1"],
            "requiredSkills": [{"skillName": "Q1", "level": 1}],
            "kpiImpacts": [{"kpiCode": "PROJECT_COMPLETION", "weight": 0.5}],
        },
    ]
    resources = [
        {
            "resourceId": "R1",
            "capacity": 100,
            "currentLoad": 0,
            "costPerHour": 10,
            "skills": [{"skillName": "Q1", "level": 1}],
        },
        {
            "resourceId": "R2",
            "capacity": 100,
            "currentLoad": 0,
            "costPerHour": 20,
            "skills": [{"skillName": "Q1", "level": 1}],
        },
    ]
    return AssignmentEvaluator(
        tasks=tasks,
        resources=resources,
        targets=[{"kpiCode": "PROJECT_COMPLETION", "targetValue": 100, "weight": 1}],
        definitions=[],
        schedule=schedule,
        predecessors={"T1": [], "T2": ["T1"]},
    )


class ColabSmokeTests(unittest.TestCase):
    def test_objective_uses_three_normalized_weights(self) -> None:
        evaluator = make_evaluator()
        weights = evaluator.strategy_weights()

        self.assertEqual({"kpi", "makespan", "cost"}, set(weights))
        self.assertAlmostEqual(1.0, sum(weights.values()), places=6)

    def test_main_optimizers_run_with_fixed_budget(self) -> None:
        optimizer_classes = [
            HarmonySearchAssignmentOptimizer,
            GeneticAlgorithmAssignmentOptimizer,
            HybridHarmonySearchGAAssignmentOptimizer,
            HybridHarmonySearchGuidedMutationGAAssignmentOptimizer,
        ]

        for optimizer_class in optimizer_classes:
            with self.subTest(optimizer=optimizer_class.__name__):
                evaluator = make_evaluator()
                result = optimizer_class(
                    tasks=evaluator.tasks,
                    resources=evaluator.resources,
                    evaluator=evaluator,
                    options=OptimizerOptions(
                        harmony_memory_size=3,
                        population_size=4,
                        max_iterations=100,
                        max_evaluations=20,
                        fixed_budget=True,
                        hybrid_hs_ratio=0.5,
                        seed=7,
                    ),
                ).run()

                self.assertEqual(20, evaluator.objective_evaluations)
                self.assertEqual({"T1", "T2"}, set(result.best.assignment))
                self.assertGreaterEqual(result.best.score.total_score, 0.0)
                self.assertLessEqual(result.best.score.total_score, 100.0)


if __name__ == "__main__":
    unittest.main()
