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


def coerce_hs_options(options: dict[str, Any] | None) -> OptimizerOptions:
    data = options or {}

    def int_option(*names: str, default: int) -> int:
        for name in names:
            if name in data:
                try:
                    return max(1, int(data[name]))
                except (TypeError, ValueError):
                    return default
        return default

    def float_option(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
        try:
            value = float(data.get(name, default))
        except (TypeError, ValueError):
            value = default
        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        return value

    return OptimizerOptions(
        harmony_memory_size=int_option("harmonyMemorySize", "populationSize", default=20),
        hmcr=float_option("hmcr", 0.9, 0.0, 1.0),
        par=float_option("par", 0.25, 0.0, 1.0),
        max_iterations=int_option("maxIterations", "numIterations", default=500),
        top_candidates=int_option("topCandidates", default=3),
        seed=int_option("seed", default=42),
        simulation_iterations=int_option("simulationIterations", default=20),
        timeout=float_option("timeout", 0.0, 0.0) or None,
        convergence_threshold=float_option("convergenceThreshold", 0.001, 0.0),
    )


class HarmonySearchAssignmentOptimizer:
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
        self.candidate_pool = self._build_candidate_pool()

    def run(self) -> OptimizationResult:
        started = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        memory = self._initial_memory()
        history = []
        best = memory[0]
        previous_best_score = best.score.total_score
        stagnant_iterations = 0
        converged = False
        iterations_run = 0

        for iteration in range(1, self.options.max_iterations + 1):
            if self.options.timeout and (time.perf_counter() - started_perf) >= self.options.timeout:
                break

            new_assignment = self._improvise(memory)
            new_candidate = self._candidate(new_assignment)
            worst = memory[-1]
            if self._better(new_candidate, worst):
                memory[-1] = new_candidate
                memory.sort(key=lambda item: item.score.rank_key(), reverse=True)

            if self._better(new_candidate, best):
                best = new_candidate

            improvement = best.score.total_score - previous_best_score
            if abs(improvement) < self.options.convergence_threshold:
                stagnant_iterations += 1
            else:
                stagnant_iterations = 0
                previous_best_score = best.score.total_score
            if stagnant_iterations >= max(25, self.options.harmony_memory_size):
                converged = True
                iterations_run = iteration
                break

            history.append(
                {
                    "iteration": iteration,
                    "bestScore": best.score.total_score,
                    "bestFeasible": best.score.feasible,
                    "currentScore": new_candidate.score.total_score,
                    "currentFeasible": new_candidate.score.feasible,
                    "violations": len(new_candidate.score.hard_violations),
                }
            )
            iterations_run = iteration

        return OptimizationResult(
            best=best,
            memory=memory,
            history=history,
            iterations_run=iterations_run,
            converged=converged,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )

    def _initial_memory(self) -> list[AssignmentCandidate]:
        memory = [self._candidate(self._greedy_assignment("BALANCED"))]
        if len(memory) < self.options.harmony_memory_size:
            memory.append(self._candidate(self._greedy_assignment("SKILL")))
        if len(memory) < self.options.harmony_memory_size:
            memory.append(self._candidate(self._greedy_assignment("COST")))
        while len(memory) < self.options.harmony_memory_size:
            memory.append(self._candidate(self._random_assignment()))
        memory.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return memory

    def _build_candidate_pool(self) -> dict[str, list[str]]:
        return {
            task_id: list(self.resource_ids)
            for task_id in self.task_ids
            if self.resource_ids
        }

    def _random_assignment(self) -> dict[str, str]:
        return {
            task_id: self.random.choice(self.candidate_pool[task_id])
            for task_id in self.task_ids
        }

    def _greedy_assignment(self, mode: str) -> dict[str, str]:
        assignment: dict[str, str] = {}
        loads = {resource_id: 0.0 for resource_id in self.resource_ids}
        for task_id in self.task_ids:
            best_resource = None
            best_score = None
            for resource_id in self.candidate_pool[task_id]:
                resource = self.evaluator.resource_by_id.get(resource_id, {})
                cost = float(resource.get("costPerHour") or 50.0)
                load = loads.get(resource_id, 0.0)
                remaining = float(resource.get("capacity") or 0.0) - float(resource.get("currentLoad") or 0.0) - load
                heuristic = self.evaluator.resource_assignment_heuristic_score(task_id, resource_id, remaining)
                if mode == "COST":
                    score = (heuristic, -cost, -load)
                elif mode == "SKILL":
                    score = (heuristic, -load, -cost)
                else:
                    score = (heuristic, -load, -cost)
                if best_score is None or score > best_score:
                    best_score = score
                    best_resource = resource_id
            assignment[task_id] = best_resource or self.random.choice(self.candidate_pool[task_id])
            schedule_item = self.evaluator.schedule.get(task_id)
            loads[assignment[task_id]] = loads.get(assignment[task_id], 0.0) + (schedule_item.duration_hours if schedule_item else 0.0)
        return assignment

    def _improvise(self, memory: list[AssignmentCandidate]) -> dict[str, str]:
        assignment = {}
        for task_id in self.task_ids:
            options = self.candidate_pool[task_id]
            if self.random.random() < self.options.hmcr:
                source = self.random.choice(memory)
                resource_id = source.assignment.get(task_id)
                if resource_id not in options:
                    resource_id = self.random.choice(options)
                if self.random.random() < self.options.par:
                    resource_id = self._pitch_adjust(task_id, resource_id, options)
            else:
                resource_id = self.random.choice(options)
            assignment[task_id] = resource_id
        return assignment

    def _pitch_adjust(self, task_id: str, current_resource_id: str, options: list[str]) -> str:
        candidates = []
        current_score = self.evaluator.resource_assignment_heuristic_score(task_id, current_resource_id)
        for resource_id in options:
            resource = self.evaluator.resource_by_id.get(resource_id, {})
            cost = float(resource.get("costPerHour") or 50.0)
            score = self.evaluator.resource_assignment_heuristic_score(task_id, resource_id)
            distance = abs(score - current_score)
            candidates.append((score, -distance, -cost, resource_id))
        candidates.sort(reverse=True)
        shortlist = candidates[: max(1, min(3, len(candidates)))]
        return self.random.choice(shortlist)[3]

    def _candidate(self, assignment: dict[str, str]) -> AssignmentCandidate:
        return AssignmentCandidate(assignment=assignment, score=self.evaluator.evaluate(assignment))

    def _better(self, candidate: AssignmentCandidate, other: AssignmentCandidate) -> bool:
        return candidate.score.rank_key() > other.score.rank_key()
