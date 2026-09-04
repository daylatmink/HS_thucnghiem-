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

    def bool_option(*names: str, default: bool = False) -> bool:
        for name in names:
            if name not in data:
                continue
            value = data[name]
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"true", "1", "yes", "y"}:
                return True
            if text in {"false", "0", "no", "n"}:
                return False
        return default

    return OptimizerOptions(
        harmony_memory_size=int_option("harmonyMemorySize", "populationSize", default=20),
        hmcr=float_option("hmcr", 0.9, 0.0, 1.0),
        par=float_option("par", 0.25, 0.0, 1.0),
        max_iterations=int_option("maxIterations", "numIterations", default=500),
        top_candidates=int_option("topCandidates", default=3),
        seed=int_option("seed", default=42),
        simulation_iterations=int_option("simulationIterations", default=20),
        timeout=float_option("timeout", 0.0, 0.0) or None,
        max_evaluations=int_option("maxEvaluations", "max_evaluations", default=0) or None,
        fixed_budget=bool_option("fixedBudget", "fixed_budget", default=False),
        convergence_threshold=float_option("convergenceThreshold", 0.001, 0.0),
        population_size=int_option("populationSize", default=50),
        crossover_rate=float_option("crossoverRate", 0.8, 0.0, 1.0),
        mutation_rate=float_option("mutationRate", 0.05, 0.0, 1.0),
        elitism_count=int_option("elitismCount", default=2),
        tournament_size=int_option("tournamentSize", default=3),
        hybrid_hs_ratio=float_option("hybridHsRatio", 0.2, 0.01, 0.99),
        objective_guidance_probability=float_option("objectiveGuidanceProbability", 0.5, 0.0, 1.0),
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
        self.task_ids = self._ordered_task_ids()
        self.resource_ids = [str(resource.get("resourceId")) for resource in resources if resource.get("resourceId")]
        self.candidate_pool = self._build_candidate_pool()

    def run(self) -> OptimizationResult:
        started = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        memory = self._initial_memory()
        history = []
        best = memory[0]
        history.append(
            {
                "iteration": 0,
                "phase": "hs",
                "objectiveEvaluation": self.evaluator.objective_evaluations,
                "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
                "bestScore": best.score.total_score,
                "bestFeasible": best.score.feasible,
                "currentScore": best.score.total_score,
                "currentFeasible": best.score.feasible,
                "violations": len(best.score.hard_violations),
            }
        )
        previous_best_score = best.score.total_score
        stagnant_iterations = 0
        converged = False
        iterations_run = 0

        for iteration in range(1, self.options.max_iterations + 1):
            if self.options.timeout and (time.perf_counter() - started_perf) >= self.options.timeout:
                break
            if self._evaluation_budget_exhausted():
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
                if not self.options.fixed_budget:
                    break

            history.append(
                {
                    "iteration": iteration,
                    "phase": "hs",
                    "objectiveEvaluation": self.evaluator.objective_evaluations,
                    "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
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
        memory = []
        if not self._evaluation_budget_exhausted():
            memory.append(self._candidate(self._greedy_assignment("BALANCED")))
        if len(memory) < self.options.harmony_memory_size and not self._evaluation_budget_exhausted():
            memory.append(self._candidate(self._greedy_assignment("SKILL")))
        if len(memory) < self.options.harmony_memory_size and not self._evaluation_budget_exhausted():
            memory.append(self._candidate(self._greedy_assignment("COST")))
        while len(memory) < self.options.harmony_memory_size and not self._evaluation_budget_exhausted():
            memory.append(self._candidate(self._random_assignment()))
        if not memory:
            memory.append(self._candidate(self._greedy_assignment("BALANCED")))
        memory.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return memory

    def _ordered_task_ids(self) -> list[str]:
        task_ids = [str(task.get("taskId")) for task in self.tasks if task.get("taskId")]
        return sorted(
            task_ids,
            key=lambda task_id: (
                self.evaluator.schedule.get(task_id).topo_level if task_id in self.evaluator.schedule else 0,
                self.evaluator.schedule.get(task_id).planned_start_hour if task_id in self.evaluator.schedule else 0.0,
                task_id,
            ),
        )

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
        loads = {resource_id: 0.0 for resource_id in self.resource_ids}
        for task_id in self.task_ids:
            options = self.candidate_pool[task_id]
            if self.random.random() < self.options.hmcr:
                source = self.random.choice(memory)
                resource_id = source.assignment.get(task_id)
                if resource_id not in options:
                    resource_id = self.random.choice(options)
                if self.random.random() < self.options.par:
                    resource_id = self._pitch_adjust(task_id, resource_id, options, loads)
            else:
                resource_id = self.random.choice(options)
            assignment[task_id] = resource_id
            loads[resource_id] = loads.get(resource_id, 0.0) + self._task_duration(task_id)
        return assignment

    def _pitch_adjust(
        self,
        task_id: str,
        current_resource_id: str,
        options: list[str],
        loads: dict[str, float],
    ) -> str:
        candidates = []
        current_score = self.evaluator.resource_assignment_heuristic_score(
            task_id,
            current_resource_id,
            self._remaining_capacity(current_resource_id, loads),
        )
        for resource_id in options:
            resource = self.evaluator.resource_by_id.get(resource_id, {})
            cost = float(resource.get("costPerHour") or 50.0)
            score = self.evaluator.resource_assignment_heuristic_score(
                task_id,
                resource_id,
                self._remaining_capacity(resource_id, loads),
            )
            distance = abs(score - current_score)
            candidates.append((score, -distance, -cost, resource_id))
        candidates.sort(reverse=True)
        shortlist = candidates[: max(1, min(3, len(candidates)))]
        return self.random.choice(shortlist)[3]

    def _remaining_capacity(self, resource_id: str, loads: dict[str, float]) -> float:
        resource = self.evaluator.resource_by_id.get(resource_id, {})
        capacity = float(resource.get("capacity") or 0.0)
        current_load = float(resource.get("currentLoad") or 0.0)
        return capacity - current_load - loads.get(resource_id, 0.0)

    def _task_duration(self, task_id: str) -> float:
        schedule_item = self.evaluator.schedule.get(task_id)
        if schedule_item:
            return schedule_item.duration_hours
        task = self.evaluator.task_by_id.get(task_id, {})
        return float(task.get("estimatedHours") or 0.0)

    def _candidate(self, assignment: dict[str, str]) -> AssignmentCandidate:
        return AssignmentCandidate(assignment=assignment, score=self.evaluator.evaluate(assignment))

    def _better(self, candidate: AssignmentCandidate, other: AssignmentCandidate) -> bool:
        return candidate.score.rank_key() > other.score.rank_key()

    def _evaluation_budget_exhausted(self) -> bool:
        budget = self.options.max_evaluations
        return budget is not None and self.evaluator.objective_evaluations >= budget
