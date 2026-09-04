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


class GeneticAlgorithmAssignmentOptimizer:
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
        phase_start_evaluations = int(getattr(self, "phase_start_evaluations", self.evaluator.objective_evaluations))
        population = self._initial_population()
        history = []
        best = population[0]
        history.append(
            {
                "iteration": 0,
                "phase": "ga",
                "objectiveEvaluation": self.evaluator.objective_evaluations,
                "phaseEvaluation": self.evaluator.objective_evaluations - phase_start_evaluations,
                "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
                "bestScore": best.score.total_score,
                "bestFeasible": best.score.feasible,
                "currentScore": best.score.total_score,
                "currentFeasible": best.score.feasible,
                "violations": len(best.score.hard_violations),
                "populationDiversity": self._population_diversity(population),
                "uniquePopulationCount": self._unique_population_count(population),
                "populationSize": len(population),
            }
        )
        previous_best_score = best.score.total_score
        stagnant_generations = 0
        converged = False
        iterations_run = 0

        for generation in range(1, self.options.max_iterations + 1):
            if self.options.timeout and (time.perf_counter() - started_perf) >= self.options.timeout:
                break
            if self._evaluation_budget_exhausted():
                break

            population.sort(key=lambda item: item.score.rank_key(), reverse=True)
            if population[0].score.rank_key() > best.score.rank_key():
                best = population[0]

            improvement = best.score.total_score - previous_best_score
            if abs(improvement) < self.options.convergence_threshold:
                stagnant_generations += 1
            else:
                stagnant_generations = 0
                previous_best_score = best.score.total_score
            if stagnant_generations >= max(25, self.options.population_size):
                converged = True
                iterations_run = generation
                if not self.options.fixed_budget:
                    break

            history.append(
                {
                    "iteration": generation,
                    "phase": "ga",
                    "objectiveEvaluation": self.evaluator.objective_evaluations,
                    "phaseEvaluation": self.evaluator.objective_evaluations - phase_start_evaluations,
                    "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
                    "bestScore": best.score.total_score,
                    "bestFeasible": best.score.feasible,
                    "currentScore": population[0].score.total_score,
                    "currentFeasible": population[0].score.feasible,
                    "violations": len(population[0].score.hard_violations),
                    "populationDiversity": self._population_diversity(population),
                    "uniquePopulationCount": self._unique_population_count(population),
                    "populationSize": len(population),
                }
            )

            population = self._next_generation(population)
            iterations_run = generation

        population.sort(key=lambda item: item.score.rank_key(), reverse=True)
        if population and population[0].score.rank_key() > best.score.rank_key():
            best = population[0]
        memory = population[: self.options.top_candidates] if population else [best]
        return OptimizationResult(
            best=best,
            memory=memory,
            history=history,
            iterations_run=iterations_run,
            converged=converged,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )

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

    def _initial_population(self) -> list[AssignmentCandidate]:
        population_size = max(1, self.options.population_size)
        population = []
        if not self._evaluation_budget_exhausted():
            population.append(self._candidate(self._greedy_assignment()))
        while len(population) < population_size and not self._evaluation_budget_exhausted():
            population.append(self._candidate(self._random_assignment()))
        if not population:
            population.append(self._candidate(self._greedy_assignment()))
        population.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return population

    def _next_generation(self, population: list[AssignmentCandidate]) -> list[AssignmentCandidate]:
        population_size = max(1, self.options.population_size)
        elite_count = max(0, min(self.options.elitism_count, population_size, len(population)))
        next_population = population[:elite_count]

        while len(next_population) < population_size and not self._evaluation_budget_exhausted():
            parent_a = self._select_parent(population)
            parent_b = self._select_parent(population)
            if self.random.random() < self.options.crossover_rate:
                child_assignment = self._crossover(parent_a.assignment, parent_b.assignment)
            else:
                child_assignment = dict(parent_a.assignment)
            child_assignment = self._mutate(child_assignment)
            next_population.append(self._candidate(child_assignment))

        next_population.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return next_population

    def _select_parent(self, population: list[AssignmentCandidate]) -> AssignmentCandidate:
        tournament_size = max(1, min(self.options.tournament_size, len(population)))
        competitors = self.random.sample(population, tournament_size)
        competitors.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return competitors[0]

    def _crossover(self, first: dict[str, str], second: dict[str, str]) -> dict[str, str]:
        child = {}
        for task_id in self.task_ids:
            source = first if self.random.random() < 0.5 else second
            resource_id = source.get(task_id)
            if resource_id not in self.candidate_pool.get(task_id, []):
                resource_id = self.random.choice(self.candidate_pool[task_id])
            child[task_id] = resource_id
        return child

    def _mutate(self, assignment: dict[str, str]) -> dict[str, str]:
        mutated = dict(assignment)
        for task_id in self.task_ids:
            if self.random.random() >= self.options.mutation_rate:
                continue
            options = self.candidate_pool.get(task_id, [])
            if not options:
                continue
            current = mutated.get(task_id)
            alternatives = [item for item in options if item != current]
            mutated[task_id] = self.random.choice(alternatives or options)
        return mutated

    def _random_assignment(self) -> dict[str, str]:
        if not self.resource_ids:
            return {}
        return {
            task_id: self.random.choice(self.candidate_pool[task_id])
            for task_id in self.task_ids
        }

    def _greedy_assignment(self) -> dict[str, str]:
        assignment: dict[str, str] = {}
        loads = {resource_id: 0.0 for resource_id in self.resource_ids}
        for task_id in self.task_ids:
            best_resource = None
            best_score = None
            for resource_id in self.candidate_pool[task_id]:
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

    def _candidate(self, assignment: dict[str, str]) -> AssignmentCandidate:
        return AssignmentCandidate(assignment=assignment, score=self.evaluator.evaluate(assignment))

    def _evaluation_budget_exhausted(self) -> bool:
        budget = self.options.max_evaluations
        return budget is not None and self.evaluator.objective_evaluations >= budget

    def _assignment_key(self, assignment: dict[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(task_id), str(resource_id)) for task_id, resource_id in assignment.items()))

    def _unique_population_count(self, population: list[AssignmentCandidate]) -> int:
        return len({self._assignment_key(candidate.assignment) for candidate in population})

    def _normalized_hamming_distance(self, first: dict[str, str], second: dict[str, str]) -> float:
        if not self.task_ids:
            return 0.0
        different = sum(1 for task_id in self.task_ids if first.get(task_id) != second.get(task_id))
        return different / len(self.task_ids)

    def _population_diversity(self, population: list[AssignmentCandidate]) -> float:
        if len(population) < 2:
            return 0.0
        total = 0.0
        pairs = 0
        for index, first in enumerate(population):
            for second in population[index + 1:]:
                total += self._normalized_hamming_distance(first.assignment, second.assignment)
                pairs += 1
        return round(total / pairs, 6) if pairs else 0.0
