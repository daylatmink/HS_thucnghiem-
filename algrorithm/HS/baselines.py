from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any

try:
    from app.modules.assignment.algorithms.evaluator import AssignmentEvaluator
    from app.modules.assignment.algorithms.models import AssignmentCandidate, OptimizationResult, OptimizerOptions
except ModuleNotFoundError:
    from evaluator import AssignmentEvaluator
    from models import AssignmentCandidate, OptimizationResult, OptimizerOptions


class RandomMultiStartAssignmentOptimizer:
    def __init__(
        self,
        tasks: list[dict[str, Any]],
        resources: list[dict[str, Any]],
        evaluator: AssignmentEvaluator,
        options: OptimizerOptions,
    ) -> None:
        self.tasks = tasks
        self.resources = resources
        self.evaluator = evaluator
        self.options = options
        self.random = random.Random(options.seed)
        self.task_ids = [str(task.get("taskId")) for task in tasks if task.get("taskId")]
        self.resource_ids = [str(resource.get("resourceId")) for resource in resources if resource.get("resourceId")]

    def run(self) -> OptimizationResult:
        started = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        history = []
        memory: list[AssignmentCandidate] = []
        best: AssignmentCandidate | None = None
        iterations_run = 0

        for iteration in range(1, self.options.max_iterations + 1):
            if self.options.timeout and (time.perf_counter() - started_perf) >= self.options.timeout:
                break
            candidate = self._candidate(self._random_assignment())
            memory.append(candidate)
            memory.sort(key=lambda item: item.score.rank_key(), reverse=True)
            memory = memory[: self.options.top_candidates]
            if best is None or candidate.score.rank_key() > best.score.rank_key():
                best = candidate
            history.append(
                {
                    "iteration": iteration,
                    "bestScore": best.score.total_score,
                    "bestFeasible": best.score.feasible,
                    "currentScore": candidate.score.total_score,
                    "currentFeasible": candidate.score.feasible,
                    "violations": len(candidate.score.hard_violations),
                }
            )
            iterations_run = iteration

        if best is None:
            best = self._candidate({})
        return OptimizationResult(
            best=best,
            memory=memory or [best],
            history=history,
            iterations_run=iterations_run,
            converged=False,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )

    def _random_assignment(self) -> dict[str, str]:
        if not self.resource_ids:
            return {}
        return {task_id: self.random.choice(self.resource_ids) for task_id in self.task_ids}

    def _candidate(self, assignment: dict[str, str]) -> AssignmentCandidate:
        return AssignmentCandidate(assignment=assignment, score=self.evaluator.evaluate(assignment))


class GreedyAssignmentOptimizer:
    def __init__(
        self,
        tasks: list[dict[str, Any]],
        resources: list[dict[str, Any]],
        evaluator: AssignmentEvaluator,
        options: OptimizerOptions,
    ) -> None:
        self.tasks = tasks
        self.resources = resources
        self.evaluator = evaluator
        self.options = options
        self.task_ids = [str(task.get("taskId")) for task in tasks if task.get("taskId")]
        self.resource_ids = [str(resource.get("resourceId")) for resource in resources if resource.get("resourceId")]

    def run(self) -> OptimizationResult:
        started = datetime.now(timezone.utc)
        assignment = self._greedy_assignment()
        best = AssignmentCandidate(assignment=assignment, score=self.evaluator.evaluate(assignment))
        return OptimizationResult(
            best=best,
            memory=[best],
            history=[
                {
                    "iteration": 1,
                    "bestScore": best.score.total_score,
                    "bestFeasible": best.score.feasible,
                    "currentScore": best.score.total_score,
                    "currentFeasible": best.score.feasible,
                    "violations": len(best.score.hard_violations),
                }
            ],
            iterations_run=1,
            converged=True,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )

    def _greedy_assignment(self) -> dict[str, str]:
        assignment: dict[str, str] = {}
        loads = {resource_id: 0.0 for resource_id in self.resource_ids}
        ordered_task_ids = sorted(
            self.task_ids,
            key=lambda task_id: (
                self.evaluator.schedule.get(task_id).topo_level if task_id in self.evaluator.schedule else 0,
                self.evaluator.schedule.get(task_id).planned_start_hour if task_id in self.evaluator.schedule else 0.0,
                task_id,
            ),
        )

        for task_id in ordered_task_ids:
            best_resource = None
            best_score = None
            for resource_id in self.resource_ids:
                resource = self.evaluator.resource_by_id.get(resource_id, {})
                cost = float(resource.get("costPerHour") or 50.0)
                load = loads.get(resource_id, 0.0)
                capacity = float(resource.get("capacity") or 0.0)
                current_load = float(resource.get("currentLoad") or 0.0)
                remaining = capacity - current_load - load
                heuristic = self.evaluator.resource_assignment_heuristic_score(task_id, resource_id, remaining)
                score = (heuristic, -load, -cost, resource_id)
                if best_score is None or score > best_score:
                    best_score = score
                    best_resource = resource_id
            if best_resource is None:
                continue
            assignment[task_id] = best_resource
            schedule_item = self.evaluator.schedule.get(task_id)
            loads[best_resource] = loads.get(best_resource, 0.0) + (schedule_item.duration_hours if schedule_item else 0.0)
        return assignment
