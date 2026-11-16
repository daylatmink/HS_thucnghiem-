from __future__ import annotations

import time
from statistics import mean
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

try:
    from app.modules.assignment.algorithms.genetic_algorithm import GeneticAlgorithmAssignmentOptimizer
    from app.modules.assignment.algorithms.harmony_search import HarmonySearchAssignmentOptimizer
    from app.modules.assignment.algorithms.models import AssignmentCandidate, OptimizationResult
except ModuleNotFoundError:
    from genetic_algorithm import GeneticAlgorithmAssignmentOptimizer
    from harmony_search import HarmonySearchAssignmentOptimizer
    from models import AssignmentCandidate, OptimizationResult


class SeededGeneticAlgorithmAssignmentOptimizer(GeneticAlgorithmAssignmentOptimizer):
    """GA variant that starts from candidates already evaluated by a previous phase."""

    def __init__(
        self,
        tasks: list[dict[str, Any]],
        resources: list[dict[str, Any]],
        evaluator,
        options,
        seed_candidates: list[AssignmentCandidate],
    ) -> None:
        super().__init__(tasks=tasks, resources=resources, evaluator=evaluator, options=options)
        self.seed_candidates = seed_candidates

    def _initial_population(self) -> list[AssignmentCandidate]:
        population_size = max(1, self.options.population_size)
        population: list[AssignmentCandidate] = []
        seen = set()
        for candidate in sorted(self.seed_candidates, key=lambda item: item.score.rank_key(), reverse=True):
            key = self._assignment_key(candidate.assignment)
            if key in seen:
                continue
            seen.add(key)
            population.append(candidate)
            if len(population) >= population_size:
                break

        while len(population) < population_size and not self._evaluation_budget_exhausted():
            candidate = self._candidate(self._random_assignment())
            key = self._assignment_key(candidate.assignment)
            if key in seen:
                continue
            seen.add(key)
            population.append(candidate)

        if not population:
            population.append(self._candidate(self._greedy_assignment()))
        population.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return population


class HybridHarmonySearchGAAssignmentOptimizer(HarmonySearchAssignmentOptimizer):
    """Run the standalone HS kernel first, then seed GA with the best HS memory."""

    def run(self) -> OptimizationResult:
        started = datetime.now(timezone.utc)
        started_perf = time.perf_counter()

        hs_ratio = self._hs_ratio()
        hs_iterations = max(1, int(self.options.max_iterations * hs_ratio))
        ga_iterations = max(1, self.options.max_iterations - hs_iterations)
        hs_result = self._run_hs_phase(hs_iterations)
        seed_candidates = self._seed_candidates_from_hs(hs_result.memory + [hs_result.best])

        ga_result = self._run_ga_phase(ga_iterations, seed_candidates, started_perf)
        best = hs_result.best
        if ga_result.best.score.rank_key() > best.score.rank_key():
            best = ga_result.best

        memory = sorted(
            self._deduplicate_candidates(ga_result.memory + seed_candidates + [best]),
            key=lambda item: item.score.rank_key(),
            reverse=True,
        )
        history = [
            {
                **item,
                "phase": "hs",
            }
            for item in hs_result.history
        ]
        history.extend(
            {
                **item,
                "iteration": hs_result.iterations_run + int(item.get("iteration") or 0),
                "phase": item.get("phase") or "ga",
            }
            for item in ga_result.history
        )

        return OptimizationResult(
            best=best,
            memory=memory[: max(1, self.options.top_candidates)],
            history=history,
            iterations_run=hs_result.iterations_run + ga_result.iterations_run,
            converged=hs_result.converged or ga_result.converged,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
            stopping_reason=ga_result.stopping_reason,
            last_improvement_evaluation=ga_result.last_improvement_evaluation,
        )

    def _run_hs_phase(self, hs_iterations: int) -> OptimizationResult:
        hs_timeout = None
        if self.options.timeout:
            hs_timeout = max(0.1, self.options.timeout * self._hs_ratio())
        hs_max_evaluations = self.options.max_evaluations
        if hs_max_evaluations is not None:
            hs_max_evaluations = max(1, int(hs_max_evaluations * self._hs_ratio()))
        hs_options = replace(
            self.options,
            max_iterations=hs_iterations,
            top_candidates=max(self.options.population_size, self.options.top_candidates),
            timeout=hs_timeout,
            max_evaluations=hs_max_evaluations,
        )
        return HarmonySearchAssignmentOptimizer(
            tasks=self.tasks,
            resources=self.resources,
            evaluator=self.evaluator,
            options=hs_options,
        ).run()

    def _run_ga_phase(
        self,
        ga_iterations: int,
        seed_candidates: list[AssignmentCandidate],
        started_perf: float,
    ) -> OptimizationResult:
        ga_timeout = None
        if self.options.timeout:
            elapsed = time.perf_counter() - started_perf
            ga_timeout = max(0.1, self.options.timeout - elapsed)
        ga_options = replace(
            self.options,
            max_iterations=ga_iterations,
            max_evaluations=self.options.max_evaluations,
        )
        if ga_timeout is not None:
            ga_options = replace(ga_options, timeout=ga_timeout)
        return SeededGeneticAlgorithmAssignmentOptimizer(
            tasks=self.tasks,
            resources=self.resources,
            evaluator=self.evaluator,
            options=ga_options,
            seed_candidates=seed_candidates,
        ).run()

    def _seed_candidates_from_hs(self, candidates: list[AssignmentCandidate]) -> list[AssignmentCandidate]:
        output = self._deduplicate_candidates(candidates)
        output.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return output[: max(1, self.options.population_size)]

    def _hs_ratio(self) -> float:
        return min(0.99, max(0.01, float(getattr(self.options, "hybrid_hs_ratio", 0.2))))

    def _deduplicate_candidates(self, candidates: list[AssignmentCandidate]) -> list[AssignmentCandidate]:
        output = []
        seen = set()
        for candidate in candidates:
            key = tuple(sorted((str(task_id), str(resource_id)) for task_id, resource_id in candidate.assignment.items()))
            if key in seen:
                continue
            seen.add(key)
            output.append(candidate)
        return output


class GuidedMutationSeededGeneticAlgorithmAssignmentOptimizer(SeededGeneticAlgorithmAssignmentOptimizer):
    """Seeded GA with contribution-guided gene selection for baseline mutation."""

    skill_weight = 1.0
    workload_weight = 0.0
    timeline_weight = 0.0
    weakness_epsilon = 1e-6

    def __init__(
        self,
        tasks: list[dict[str, Any]],
        resources: list[dict[str, Any]],
        evaluator,
        options,
        seed_candidates: list[AssignmentCandidate],
    ) -> None:
        super().__init__(
            tasks=tasks,
            resources=resources,
            evaluator=evaluator,
            options=options,
            seed_candidates=seed_candidates,
        )
        self.guided_mutation_history: list[dict[str, Any]] = []
        self._best: AssignmentCandidate | None = None
        self._current_generation = 0

    def run(self) -> OptimizationResult:
        started = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        phase_start_evaluations = int(getattr(self, "phase_start_evaluations", self.evaluator.objective_evaluations))
        population = self._initial_population()
        history = []
        best = population[0]
        self._best = best
        history.append(self._history_row(0, population, best, phase_start_evaluations))
        previous_best_score = best.score.total_score
        stagnant_generations = 0
        last_improvement_evaluation = int(
            best.score.diagnostics.get("objectiveEvaluation")
            or self.evaluator.objective_evaluations
        )
        converged = False
        stopping_reason = None
        iterations_run = 0

        for generation in range(1, self.options.max_iterations + 1):
            if self.options.timeout and (time.perf_counter() - started_perf) >= self.options.timeout:
                break
            if self._evaluation_budget_exhausted():
                break

            population.sort(key=lambda item: item.score.rank_key(), reverse=True)
            if population[0].score.rank_key() > best.score.rank_key():
                best = population[0]
                self._best = best

            improvement = best.score.total_score - previous_best_score
            if improvement >= self.options.convergence_threshold:
                stagnant_generations = 0
                previous_best_score = best.score.total_score
                last_improvement_evaluation = int(
                    best.score.diagnostics.get("objectiveEvaluation")
                    or self.evaluator.objective_evaluations
                )
            else:
                stagnant_generations += 1

            stagnation_evaluations = self.options.stagnation_evaluations
            if stagnation_evaluations is not None:
                if (
                    self.evaluator.objective_evaluations - last_improvement_evaluation
                    >= stagnation_evaluations
                ):
                    converged = True
                    stopping_reason = "nfe_stagnation"
                    iterations_run = generation
                    break
            elif stagnant_generations >= max(25, self.options.population_size):
                converged = True
                iterations_run = generation
                if not self.options.fixed_budget:
                    break

            history.append(self._history_row(generation, population, best, phase_start_evaluations))
            self._current_generation = generation
            population = self._next_generation(population)
            iterations_run = generation

        population.sort(key=lambda item: item.score.rank_key(), reverse=True)
        if population and population[0].score.rank_key() > best.score.rank_key():
            best = population[0]
            self._best = best
        memory = population[: self.options.top_candidates] if population else [best]
        full_history = sorted(
            history + self.guided_mutation_history,
            key=lambda item: (int(item.get("objectiveEvaluation") or 0), int(item.get("iteration") or 0)),
        )
        return OptimizationResult(
            best=best,
            memory=memory,
            history=full_history,
            iterations_run=iterations_run,
            converged=converged,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
            stopping_reason=stopping_reason,
            last_improvement_evaluation=last_improvement_evaluation,
        )

    def _history_row(
        self,
        iteration: int,
        population: list[AssignmentCandidate],
        best: AssignmentCandidate,
        phase_start_evaluations: int,
    ) -> dict[str, Any]:
        current = population[0]
        return {
            "iteration": iteration,
            "phase": "ga",
            "objectiveEvaluation": self.evaluator.objective_evaluations,
            "phaseEvaluation": self.evaluator.objective_evaluations - phase_start_evaluations,
            "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
            "bestScore": best.score.total_score,
            "bestFeasible": best.score.feasible,
            "currentScore": current.score.total_score,
            "currentFeasible": current.score.feasible,
            "violations": len(current.score.hard_violations),
            "populationDiversity": self._population_diversity(population),
            "uniquePopulationCount": self._unique_population_count(population),
            "populationSize": len(population),
            "mutationType": "contribution_guided",
            "baselineMutationRate": self.options.mutation_rate,
            "expectedMutationCount": self._expected_mutation_count(),
        }

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
            child_assignment, mutation_stats = self._guided_mutate(child_assignment)
            candidate = self._candidate(child_assignment)
            if self._best is None or candidate.score.rank_key() > self._best.score.rank_key():
                self._best = candidate
            self._record_guided_mutation(candidate, mutation_stats)
            next_population.append(candidate)

        next_population.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return next_population

    def _guided_mutate(self, assignment: dict[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
        mutated = dict(assignment)
        weaknesses = self._weakness_scores(assignment)
        selected_task_ids = self._sample_mutation_tasks(weaknesses, self._mutation_attempt_count())
        for task_id in selected_task_ids:
            options = self.candidate_pool.get(task_id, [])
            if not options:
                continue
            current = mutated.get(task_id)
            alternatives = [item for item in options if item != current]
            mutated[task_id] = self.random.choice(alternatives or options)

        selected_values = [weaknesses[task_id]["weakness"] for task_id in selected_task_ids if task_id in weaknesses]
        unselected_values = [
            item["weakness"]
            for task_id, item in weaknesses.items()
            if task_id not in set(selected_task_ids)
        ]
        selected_skill = [weaknesses[task_id]["skillWeakness"] for task_id in selected_task_ids if task_id in weaknesses]
        selected_workload = [weaknesses[task_id]["workloadPressure"] for task_id in selected_task_ids if task_id in weaknesses]
        selected_timeline = [weaknesses[task_id]["timelinePressure"] for task_id in selected_task_ids if task_id in weaknesses]
        all_values = [item["weakness"] for item in weaknesses.values()]
        stats = {
            "actualMutatedGeneCount": len(selected_task_ids),
            "meanWeakness": round(mean(all_values), 6) if all_values else 0.0,
            "maxWeakness": round(max(all_values), 6) if all_values else 0.0,
            "minWeakness": round(min(all_values), 6) if all_values else 0.0,
            "meanSelectedWeakness": round(mean(selected_values), 6) if selected_values else None,
            "meanUnselectedWeakness": round(mean(unselected_values), 6) if unselected_values else None,
            "selectedSkillWeaknessMean": round(mean(selected_skill), 6) if selected_skill else None,
            "selectedWorkloadPressureMean": round(mean(selected_workload), 6) if selected_workload else None,
            "selectedTimelinePressureMean": round(mean(selected_timeline), 6) if selected_timeline else None,
        }
        return mutated, stats

    def _weakness_scores(self, assignment: dict[str, str]) -> dict[str, dict[str, float]]:
        actual_schedule = self.evaluator.assignment_scheduler.build(assignment)
        durations = {
            task_id: (
                actual_schedule.get(task_id).duration_hours
                if task_id in actual_schedule
                else self.evaluator.schedule.get(task_id).duration_hours
                if task_id in self.evaluator.schedule
                else 0.0
            )
            for task_id in self.task_ids
        }
        loads: dict[str, float] = {}
        for task_id, resource_id in assignment.items():
            loads[resource_id] = loads.get(resource_id, 0.0) + durations.get(task_id, 0.0)
        max_load = max(loads.values(), default=0.0)
        makespan = max((item.planned_end_hour for item in actual_schedule.values()), default=0.0)

        output: dict[str, dict[str, float]] = {}
        for task_id in self.task_ids:
            resource_id = assignment.get(task_id)
            skill_match = self.evaluator.resource_skill_score(task_id, resource_id) if resource_id else 0.0
            skill_weakness = 1.0 - max(0.0, min(1.0, skill_match / 100.0))
            workload_pressure = (loads.get(resource_id, 0.0) / max_load) if resource_id and max_load > 0 else 0.0
            item = actual_schedule.get(task_id)
            duration = max(durations.get(task_id, 0.0), 1e-6)
            deadline = self._task_deadline(task_id)
            if item and deadline is not None:
                timeline_pressure = min(1.0, max(0.0, item.planned_end_hour - deadline) / duration)
            elif item and makespan > 0:
                timeline_pressure = max(0.0, min(1.0, item.planned_end_hour / makespan))
            else:
                timeline_pressure = 0.0
            weakness = (
                self.skill_weight * skill_weakness
                + self.workload_weight * workload_pressure
                + self.timeline_weight * timeline_pressure
            )
            output[task_id] = {
                "weakness": max(0.0, min(1.0, weakness)),
                "skillWeakness": skill_weakness,
                "workloadPressure": max(0.0, min(1.0, workload_pressure)),
                "timelinePressure": max(0.0, min(1.0, timeline_pressure)),
            }
        return output

    def _task_deadline(self, task_id: str) -> float | None:
        task = self.evaluator.task_by_id.get(task_id, {})
        for key in ("deadlineHour", "deadline", "dueHour", "dueDateHour"):
            value = task.get(key)
            try:
                if value not in ("", None):
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _sample_mutation_tasks(self, weaknesses: dict[str, dict[str, float]], count: int) -> list[str]:
        available = [task_id for task_id in self.task_ids if task_id in weaknesses]
        selected: list[str] = []
        count = max(0, min(count, len(available)))
        for _ in range(count):
            weights = [self.weakness_epsilon + weaknesses[task_id]["weakness"] for task_id in available]
            total = sum(weights)
            if total <= 0:
                index = self.random.randrange(len(available))
            else:
                pick = self.random.random() * total
                cumulative = 0.0
                index = len(available) - 1
                for candidate_index, weight in enumerate(weights):
                    cumulative += weight
                    if pick <= cumulative:
                        index = candidate_index
                        break
            selected.append(available.pop(index))
        return selected

    def _expected_mutation_count(self) -> float:
        return max(0.0, min(float(len(self.task_ids)), len(self.task_ids) * float(self.options.mutation_rate)))

    def _mutation_attempt_count(self) -> int:
        mutation_rate = max(0.0, min(1.0, float(self.options.mutation_rate)))
        return sum(1 for _task_id in self.task_ids if self.random.random() < mutation_rate)

    def _record_guided_mutation(self, candidate: AssignmentCandidate, stats: dict[str, Any]) -> None:
        best = self._best or candidate
        phase_start_evaluations = int(getattr(self, "phase_start_evaluations", 0))
        self.guided_mutation_history.append(
            {
                "iteration": self._current_generation,
                "phase": "ga",
                "objectiveEvaluation": self.evaluator.objective_evaluations,
                "phaseEvaluation": self.evaluator.objective_evaluations - phase_start_evaluations,
                "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
                "bestScore": best.score.total_score,
                "currentScore": candidate.score.total_score,
                "mutationType": "contribution_guided",
                "baselineMutationRate": self.options.mutation_rate,
                "expectedMutationCount": self._expected_mutation_count(),
                **stats,
            }
        )


class HybridHarmonySearchGuidedMutationGAAssignmentOptimizer(HybridHarmonySearchGAAssignmentOptimizer):
    """HS warm-start followed by GA with skill-guided mutation."""

    def _hs_ratio(self) -> float:
        return min(0.99, max(0.01, float(getattr(self.options, "hybrid_hs_ratio", 0.15))))

    def _run_ga_phase(
        self,
        ga_iterations: int,
        seed_candidates: list[AssignmentCandidate],
        started_perf: float,
    ) -> OptimizationResult:
        ga_timeout = None
        if self.options.timeout:
            elapsed = time.perf_counter() - started_perf
            ga_timeout = max(0.1, self.options.timeout - elapsed)
        ga_options = replace(
            self.options,
            max_iterations=ga_iterations,
            max_evaluations=self.options.max_evaluations,
        )
        if ga_timeout is not None:
            ga_options = replace(ga_options, timeout=ga_timeout)
        optimizer = GuidedMutationSeededGeneticAlgorithmAssignmentOptimizer(
            tasks=self.tasks,
            resources=self.resources,
            evaluator=self.evaluator,
            options=ga_options,
            seed_candidates=seed_candidates,
        )
        optimizer.phase_start_evaluations = self.evaluator.objective_evaluations
        return optimizer.run()
