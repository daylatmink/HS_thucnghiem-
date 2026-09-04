from __future__ import annotations

import time
from math import ceil
from statistics import mean, median
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


class AdaptiveMutationSeededGeneticAlgorithmAssignmentOptimizer(SeededGeneticAlgorithmAssignmentOptimizer):
    """Seeded GA with TA-1 success-rate adaptive mutation only."""

    success_window_size = 50

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
        self.baseline_mutation_rate = float(options.mutation_rate)
        self.mutation_min = max(0.0, 0.5 * self.baseline_mutation_rate)
        self.mutation_max = min(0.20, 2.0 * self.baseline_mutation_rate)
        self.current_mutation_rate = min(max(self.baseline_mutation_rate, self.mutation_min), self.mutation_max)
        self.success_window_count = 0
        self.success_window_successful = 0
        self.adaptation_history: list[dict[str, Any]] = []
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
                self._best = best

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

            history.append(self._history_row(generation, population, best, phase_start_evaluations))
            self._current_generation = generation
            population = self._next_generation(population)
            iterations_run = generation

        population.sort(key=lambda item: item.score.rank_key(), reverse=True)
        if population and population[0].score.rank_key() > best.score.rank_key():
            best = population[0]
            self._best = best
        if self.success_window_count:
            self._log_adaptive_window(final=True)
        memory = population[: self.options.top_candidates] if population else [best]
        full_history = sorted(
            history + self.adaptation_history,
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
            "mutationRate": self.current_mutation_rate,
            "baselineMutationRate": self.baseline_mutation_rate,
            "mutationMin": self.mutation_min,
            "mutationMax": self.mutation_max,
            "successWindowCount": self.success_window_count,
            "successWindowSuccessful": self.success_window_successful,
            "adaptiveUpdate": False,
        }

    def _next_generation(self, population: list[AssignmentCandidate]) -> list[AssignmentCandidate]:
        population_size = max(1, self.options.population_size)
        elite_count = max(0, min(self.options.elitism_count, population_size, len(population)))
        next_population = population[:elite_count]

        while len(next_population) < population_size and not self._evaluation_budget_exhausted():
            parent_a = self._select_parent(population)
            parent_b = self._select_parent(population)
            reference_parent = max((parent_a, parent_b), key=lambda item: item.score.rank_key())
            if self.random.random() < self.options.crossover_rate:
                child_assignment = self._crossover(parent_a.assignment, parent_b.assignment)
            else:
                child_assignment = dict(parent_a.assignment)
            mutation_rate = self.current_mutation_rate
            child_assignment = self._mutate_with_rate(child_assignment, mutation_rate)
            candidate = self._candidate(child_assignment)
            if self._best is None or candidate.score.rank_key() > self._best.score.rank_key():
                self._best = candidate
            self._record_offspring(candidate, reference_parent, mutation_rate)
            next_population.append(candidate)

        next_population.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return next_population

    def _mutate_with_rate(self, assignment: dict[str, str], mutation_rate: float) -> dict[str, str]:
        mutated = dict(assignment)
        for task_id in self.task_ids:
            if self.random.random() >= mutation_rate:
                continue
            options = self.candidate_pool.get(task_id, [])
            if not options:
                continue
            current = mutated.get(task_id)
            alternatives = [item for item in options if item != current]
            mutated[task_id] = self.random.choice(alternatives or options)
        return mutated

    def _record_offspring(
        self,
        candidate: AssignmentCandidate,
        reference_parent: AssignmentCandidate,
        mutation_rate: float,
    ) -> None:
        success = candidate.score.rank_key() > reference_parent.score.rank_key()
        self.success_window_count += 1
        if success:
            self.success_window_successful += 1
        best = self._best or candidate
        phase_start_evaluations = int(getattr(self, "phase_start_evaluations", 0))
        self.adaptation_history.append(
            {
                "iteration": self._current_generation,
                "phase": "ga",
                "objectiveEvaluation": self.evaluator.objective_evaluations,
                "phaseEvaluation": self.evaluator.objective_evaluations - phase_start_evaluations,
                "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
                "bestScore": best.score.total_score,
                "currentScore": candidate.score.total_score,
                "mutationRate": mutation_rate,
                "baselineMutationRate": self.baseline_mutation_rate,
                "mutationMin": self.mutation_min,
                "mutationMax": self.mutation_max,
                "offspringSuccess": success,
                "referenceParentScore": reference_parent.score.total_score,
                "offspringScore": candidate.score.total_score,
                "successWindowCount": self.success_window_count,
                "successWindowSuccessful": self.success_window_successful,
                "adaptiveUpdate": False,
            }
        )
        if self.success_window_count >= self.success_window_size:
            self._log_adaptive_window(final=False)

    def _log_adaptive_window(self, final: bool) -> None:
        if self.success_window_count <= 0:
            return
        success_rate = self.success_window_successful / self.success_window_count
        previous_rate = self.current_mutation_rate
        state = None
        new_rate = previous_rate
        if not final:
            new_rate, state = self.apply_success_rate_update(success_rate)
        best = self._best
        phase_start_evaluations = int(getattr(self, "phase_start_evaluations", 0))
        self.adaptation_history.append(
            {
                "iteration": self._current_generation,
                "phase": "adaptive_window_final" if final else "adaptive_update",
                "objectiveEvaluation": self.evaluator.objective_evaluations,
                "phaseEvaluation": self.evaluator.objective_evaluations - phase_start_evaluations,
                "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation") if best else None,
                "bestScore": best.score.total_score if best else None,
                "currentScore": best.score.total_score if best else None,
                "mutationRate": new_rate,
                "baselineMutationRate": self.baseline_mutation_rate,
                "mutationMin": self.mutation_min,
                "mutationMax": self.mutation_max,
                "successWindowCount": self.success_window_count,
                "successWindowSuccessful": self.success_window_successful,
                "offspringSuccessRate": success_rate,
                "adaptiveUpdate": not final,
                "adaptiveState": state,
                "previousMutationRate": previous_rate,
                "newMutationRate": new_rate,
            }
        )
        if not final:
            self.success_window_count = 0
            self.success_window_successful = 0

    def apply_success_rate_update(self, success_rate: float) -> tuple[float, str]:
        if success_rate < 0.10:
            state = "explore_more"
            updated = min(self.mutation_max, self.current_mutation_rate * 1.5)
        elif success_rate > 0.20:
            state = "exploit_more"
            updated = max(self.mutation_min, self.current_mutation_rate * 0.75)
        else:
            state = "baseline"
            updated = self.current_mutation_rate + 0.25 * (self.baseline_mutation_rate - self.current_mutation_rate)
            updated = min(self.mutation_max, max(self.mutation_min, updated))
        self.current_mutation_rate = updated
        return updated, state


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


class HybridHarmonySearchTA1GAAssignmentOptimizer(HybridHarmonySearchGAAssignmentOptimizer):
    """TA-1: HS150 warm-start followed by GA850 with success-rate adaptive mutation."""

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
        optimizer = AdaptiveMutationSeededGeneticAlgorithmAssignmentOptimizer(
            tasks=self.tasks,
            resources=self.resources,
            evaluator=self.evaluator,
            options=ga_options,
            seed_candidates=seed_candidates,
        )
        optimizer.phase_start_evaluations = self.evaluator.objective_evaluations
        return optimizer.run()


class LinkageAwareSeededGeneticAlgorithmAssignmentOptimizer(SeededGeneticAlgorithmAssignmentOptimizer):
    """Seeded GA with DAG/topological linkage-aware crossover only."""

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
        self.linkage_groups = self._build_linkage_groups()
        self.linkage_diagnostics = self._linkage_diagnostics()
        self.crossover_history: list[dict[str, Any]] = []
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
                self._best = best

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
            history + self.crossover_history,
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
            "crossoverType": "dag_linkage",
            **self.linkage_diagnostics,
        }

    def _next_generation(self, population: list[AssignmentCandidate]) -> list[AssignmentCandidate]:
        population_size = max(1, self.options.population_size)
        elite_count = max(0, min(self.options.elitism_count, population_size, len(population)))
        next_population = population[:elite_count]

        while len(next_population) < population_size and not self._evaluation_budget_exhausted():
            parent_a = self._select_parent(population)
            parent_b = self._select_parent(population)
            crossover_applied = self.random.random() < self.options.crossover_rate
            if crossover_applied:
                child_assignment, groups_from_a, groups_from_b = self._linkage_crossover(
                    parent_a.assignment,
                    parent_b.assignment,
                )
            else:
                child_assignment = dict(parent_a.assignment)
                groups_from_a = len(self.linkage_groups)
                groups_from_b = 0
            child_assignment = self._mutate(child_assignment)
            candidate = self._candidate(child_assignment)
            if self._best is None or candidate.score.rank_key() > self._best.score.rank_key():
                self._best = candidate
            self._record_crossover(
                candidate=candidate,
                parent_a=parent_a,
                parent_b=parent_b,
                crossover_applied=crossover_applied,
                groups_from_a=groups_from_a,
                groups_from_b=groups_from_b,
            )
            next_population.append(candidate)

        next_population.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return next_population

    def _linkage_crossover(
        self,
        parent_a: dict[str, str],
        parent_b: dict[str, str],
    ) -> tuple[dict[str, str], int, int]:
        child: dict[str, str] = {}
        groups_from_a = 0
        groups_from_b = 0
        for group in self.linkage_groups:
            source = parent_a if self.random.random() < 0.5 else parent_b
            if source is parent_a:
                groups_from_a += 1
            else:
                groups_from_b += 1
            for task_id in group:
                resource_id = source.get(task_id)
                if resource_id not in self.candidate_pool.get(task_id, []):
                    resource_id = self.random.choice(self.candidate_pool[task_id])
                child[task_id] = resource_id
        for task_id in self.task_ids:
            if task_id not in child:
                child[task_id] = parent_a.get(task_id) or self.random.choice(self.candidate_pool[task_id])
        return child, groups_from_a, groups_from_b

    def _build_linkage_groups(self) -> list[list[str]]:
        max_group_size = max(2, int(ceil(0.20 * max(1, len(self.task_ids)))))
        by_level: dict[int, list[str]] = {}
        for task_id in self.task_ids:
            level = self.evaluator.schedule.get(task_id).topo_level if task_id in self.evaluator.schedule else 0
            by_level.setdefault(level, []).append(task_id)

        groups: list[list[str]] = []
        for _level, level_tasks in sorted(by_level.items()):
            if len(level_tasks) <= max_group_size:
                groups.append(level_tasks)
                continue
            by_parent: dict[str, list[str]] = {}
            for task_id in level_tasks:
                parents = sorted(str(parent) for parent in self.evaluator.predecessors.get(task_id, []))
                key = parents[0] if parents else ""
                by_parent.setdefault(key, []).append(task_id)
            for _parent_key, parent_tasks in sorted(by_parent.items()):
                for index in range(0, len(parent_tasks), max_group_size):
                    groups.append(parent_tasks[index : index + max_group_size])
        return groups or [[task_id] for task_id in self.task_ids]

    def _linkage_diagnostics(self) -> dict[str, Any]:
        sizes = [len(group) for group in self.linkage_groups]
        topo_levels = {
            self.evaluator.schedule.get(task_id).topo_level
            for task_id in self.task_ids
            if task_id in self.evaluator.schedule
        }
        return {
            "numberOfLinkageGroups": len(self.linkage_groups),
            "meanLinkageGroupSize": round(mean(sizes), 6) if sizes else 0.0,
            "medianLinkageGroupSize": round(median(sizes), 6) if sizes else 0.0,
            "maxLinkageGroupSize": max(sizes, default=0),
            "minLinkageGroupSize": min(sizes, default=0),
            "numTopoLevels": len(topo_levels),
            "singletonGroups": sum(1 for size in sizes if size == 1),
        }

    def _record_crossover(
        self,
        candidate: AssignmentCandidate,
        parent_a: AssignmentCandidate,
        parent_b: AssignmentCandidate,
        crossover_applied: bool,
        groups_from_a: int,
        groups_from_b: int,
    ) -> None:
        best = self._best or candidate
        phase_start_evaluations = int(getattr(self, "phase_start_evaluations", 0))
        self.crossover_history.append(
            {
                "iteration": self._current_generation,
                "phase": "ga",
                "objectiveEvaluation": self.evaluator.objective_evaluations,
                "phaseEvaluation": self.evaluator.objective_evaluations - phase_start_evaluations,
                "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
                "bestScore": best.score.total_score,
                "currentScore": candidate.score.total_score,
                "crossoverType": "dag_linkage",
                "numberOfLinkageGroups": len(self.linkage_groups),
                "meanLinkageGroupSize": self.linkage_diagnostics["meanLinkageGroupSize"],
                "maxLinkageGroupSize": self.linkage_diagnostics["maxLinkageGroupSize"],
                "minLinkageGroupSize": self.linkage_diagnostics["minLinkageGroupSize"],
                "crossoverApplied": crossover_applied,
                "groupsFromParentA": groups_from_a,
                "groupsFromParentB": groups_from_b,
                "childParentAHamming": self._normalized_hamming_distance(candidate.assignment, parent_a.assignment),
                "childParentBHamming": self._normalized_hamming_distance(candidate.assignment, parent_b.assignment),
            }
        )


class HybridHarmonySearchLinkageGAAssignmentOptimizer(HybridHarmonySearchGAAssignmentOptimizer):
    """HS150 warm-start followed by GA850 with DAG/linkage-aware crossover."""

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
        optimizer = LinkageAwareSeededGeneticAlgorithmAssignmentOptimizer(
            tasks=self.tasks,
            resources=self.resources,
            evaluator=self.evaluator,
            options=ga_options,
            seed_candidates=seed_candidates,
        )
        optimizer.phase_start_evaluations = self.evaluator.objective_evaluations
        return optimizer.run()


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
                self._best = best

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
    """HS150 warm-start followed by GA850 with contribution-guided mutation."""

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


class ObjectiveGuidedMutationSeededGeneticAlgorithmAssignmentOptimizer(GuidedMutationSeededGeneticAlgorithmAssignmentOptimizer):
    """TA-4 guided mutation with weakness aligned to KPI, cost, and time objective terms."""

    def _history_row(
        self,
        iteration: int,
        population: list[AssignmentCandidate],
        best: AssignmentCandidate,
        phase_start_evaluations: int,
    ) -> dict[str, Any]:
        row = super()._history_row(iteration, population, best, phase_start_evaluations)
        weights = self.evaluator.strategy_weights()
        row.update(
            {
                "mutationType": "objective_aligned_guided",
                "objectiveKpiWeight": weights["kpi"],
                "objectiveCostWeight": weights["cost"],
                "objectiveTimeWeight": weights["makespan"],
            }
        )
        return row

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

        selected = [weaknesses[task_id] for task_id in selected_task_ids if task_id in weaknesses]
        unselected = [item for task_id, item in weaknesses.items() if task_id not in set(selected_task_ids)]
        all_items = list(weaknesses.values())
        stats = {
            "actualMutatedGeneCount": len(selected_task_ids),
            "KPIWeaknessMean": self._mean_component(all_items, "KPIWeakness"),
            "CostWeaknessMean": self._mean_component(all_items, "CostWeakness"),
            "TimeWeaknessMean": self._mean_component(all_items, "TimeWeakness"),
            "overallWeaknessMean": self._mean_component(all_items, "weakness"),
            "selectedKPIWeaknessMean": self._mean_component(selected, "KPIWeakness"),
            "selectedCostWeaknessMean": self._mean_component(selected, "CostWeakness"),
            "selectedTimeWeaknessMean": self._mean_component(selected, "TimeWeakness"),
            "selectedOverallWeaknessMean": self._mean_component(selected, "weakness"),
            "unselectedOverallWeaknessMean": self._mean_component(unselected, "weakness"),
            "overallWeaknessLift": self._lift(selected, all_items, "weakness"),
            "KPIWeaknessLift": self._lift(selected, all_items, "KPIWeakness"),
            "CostWeaknessLift": self._lift(selected, all_items, "CostWeakness"),
            "TimeWeaknessLift": self._lift(selected, all_items, "TimeWeakness"),
            "meanResourceWait": self._mean_component(all_items, "resourceWait"),
            "maxResourceWait": round(max((item["resourceWait"] for item in all_items), default=0.0), 6),
            "meanCriticality": self._mean_component(all_items, "criticality"),
            "numCriticalTasks": sum(1 for item in all_items if item["criticality"] >= 0.999999),
            "meanCriticalBlocking": self._mean_component(all_items, "criticalBlocking"),
            "numTasksWithResourceWait": sum(1 for item in all_items if item["resourceWait"] > 1e-9),
            "numTasksWithPositiveBlocking": sum(1 for item in all_items if item["criticalBlocking"] > 1e-9),
            "meanCostWeakness": self._mean_component(all_items, "CostWeakness"),
            "selectedMeanCostWeakness": self._mean_component(selected, "CostWeakness"),
            "costWeaknessLift": self._lift(selected, all_items, "CostWeakness"),
            "numTasksAtMinCost": sum(1 for item in all_items if item["CostWeakness"] <= 1e-9),
            "numTasksAboveMedianCost": self._num_above_median(all_items, "CostWeakness"),
            "numTasksAtMaxCost": sum(1 for item in all_items if item["CostWeakness"] >= 1.0 - 1e-9),
            "meanKPIWeakness": self._mean_component(all_items, "KPIWeakness"),
            "selectedMeanKPIWeakness": self._mean_component(selected, "KPIWeakness"),
            "kpiWeaknessLift": self._lift(selected, all_items, "KPIWeakness"),
            "meanTaskKPIContribution": self._mean_component(all_items, "taskKPIContribution"),
            "selectedMeanTaskKPIContribution": self._mean_component(selected, "taskKPIContribution"),
        }
        return mutated, stats

    def _weakness_scores(self, assignment: dict[str, str]) -> dict[str, dict[str, float]]:
        actual_schedule = self.evaluator.assignment_scheduler.build(assignment)
        time_components = self._time_components(assignment, actual_schedule)
        weights = self.evaluator.strategy_weights()
        output: dict[str, dict[str, float]] = {}
        for task_id in self.task_ids:
            resource_id = assignment.get(task_id)
            kpi_weakness, task_contribution = self._kpi_weakness(task_id, resource_id)
            cost_weakness = self._cost_weakness(task_id, resource_id)
            time_item = time_components.get(task_id, {})
            time_weakness = float(time_item.get("TimeWeakness", 0.0))
            weakness = (
                weights["kpi"] * kpi_weakness
                + weights["cost"] * cost_weakness
                + weights["makespan"] * time_weakness
            )
            output[task_id] = {
                "weakness": max(0.0, min(1.0, weakness)),
                "KPIWeakness": kpi_weakness,
                "CostWeakness": cost_weakness,
                "TimeWeakness": time_weakness,
                "taskKPIContribution": task_contribution,
                "resourceWait": float(time_item.get("resourceWait", 0.0)),
                "criticality": float(time_item.get("criticality", 0.0)),
                "criticalBlocking": float(time_item.get("criticalBlocking", 0.0)),
            }
        return output

    def _record_guided_mutation(self, candidate: AssignmentCandidate, stats: dict[str, Any]) -> None:
        best = self._best or candidate
        phase_start_evaluations = int(getattr(self, "phase_start_evaluations", 0))
        weights = self.evaluator.strategy_weights()
        self.guided_mutation_history.append(
            {
                "iteration": self._current_generation,
                "phase": "ga",
                "objectiveEvaluation": self.evaluator.objective_evaluations,
                "phaseEvaluation": self.evaluator.objective_evaluations - phase_start_evaluations,
                "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
                "bestScore": best.score.total_score,
                "currentScore": candidate.score.total_score,
                "mutationType": "objective_aligned_guided",
                "baselineMutationRate": self.options.mutation_rate,
                "expectedMutationCount": self._expected_mutation_count(),
                "objectiveKpiWeight": weights["kpi"],
                "objectiveCostWeight": weights["cost"],
                "objectiveTimeWeight": weights["makespan"],
                **stats,
            }
        )

    def _kpi_weakness(self, task_id: str, resource_id: str | None) -> tuple[float, float]:
        current = self._task_kpi_contribution(task_id, resource_id)
        options = self.candidate_pool.get(task_id, [])
        best = max((self._task_kpi_contribution(task_id, option) for option in options), default=0.0)
        if best <= 0:
            return 0.0, 1.0
        contribution = max(0.0, min(1.0, current / best))
        return 1.0 - contribution, contribution

    def _task_kpi_contribution(self, task_id: str, resource_id: str | None) -> float:
        if not resource_id:
            return 0.0
        task = self.evaluator.task_by_id.get(task_id, {})
        impacts = task.get("kpiImpacts") or []
        if not impacts:
            return 0.0
        skill_factor = self.evaluator.resource_skill_score(task_id, resource_id) / 100.0
        total = 0.0
        for impact in impacts:
            code = str(impact.get("kpiCode", "")).strip()
            if not code:
                continue
            target_doc = self.evaluator.target_by_code.get(code, {})
            target_weight = self._safe_float(target_doc.get("weight"), 1.0)
            target_value = self._safe_float(target_doc.get("targetValue"), 100.0)
            total += target_weight * self._safe_float(impact.get("weight")) * target_value * skill_factor
        return total

    def _cost_weakness(self, task_id: str, resource_id: str | None) -> float:
        options = self.candidate_pool.get(task_id, [])
        costs = [self._task_cost(task_id, option) for option in options]
        if not costs or resource_id is None:
            return 0.0
        min_cost = min(costs)
        max_cost = max(costs)
        if max_cost <= min_cost:
            return 0.0
        return max(0.0, min(1.0, (self._task_cost(task_id, resource_id) - min_cost) / (max_cost - min_cost)))

    def _task_cost(self, task_id: str, resource_id: str) -> float:
        schedule_item = self.evaluator.schedule.get(task_id)
        task = self.evaluator.task_by_id.get(task_id, {})
        duration = schedule_item.duration_hours if schedule_item else self._safe_float(task.get("estimatedHours"), 1.0)
        rate = self._safe_float(self.evaluator.resource_by_id.get(resource_id, {}).get("costPerHour"), 50.0)
        return duration * rate

    def _time_components(
        self,
        assignment: dict[str, str],
        actual_schedule: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        makespan = max((item.planned_end_hour for item in actual_schedule.values()), default=0.0)
        successors = self._actual_successors(assignment, actual_schedule)
        latest_start = self._latest_start_times(actual_schedule, successors, makespan)
        criticality: dict[str, float] = {}
        resource_wait: dict[str, float] = {}
        for task_id in self.task_ids:
            item = actual_schedule.get(task_id)
            if not item:
                criticality[task_id] = 0.0
                resource_wait[task_id] = 0.0
                continue
            duration = max(float(item.duration_hours), 1e-6)
            slack = max(0.0, latest_start.get(task_id, item.planned_start_hour) - item.planned_start_hour)
            criticality[task_id] = max(0.0, min(1.0, 1.0 / (1.0 + slack / duration)))
            pred_ready = max(
                (actual_schedule[pred].planned_end_hour for pred in self.evaluator.predecessors.get(task_id, []) if pred in actual_schedule),
                default=0.0,
            )
            resource_wait[task_id] = max(0.0, item.planned_start_hour - pred_ready)

        next_on_resource = self._next_task_on_resource(assignment, actual_schedule)
        output: dict[str, dict[str, float]] = {}
        for task_id in self.task_ids:
            item = actual_schedule.get(task_id)
            next_task = next_on_resource.get(task_id)
            critical_blocking = 0.0
            if item and next_task and next_task in actual_schedule:
                next_pred_ready = max(
                    (
                        actual_schedule[pred].planned_end_hour
                        for pred in self.evaluator.predecessors.get(next_task, [])
                        if pred in actual_schedule
                    ),
                    default=0.0,
                )
                critical_blocking = criticality.get(next_task, 0.0) * max(0.0, item.planned_end_hour - next_pred_ready)
            raw_time = criticality.get(task_id, 0.0) * resource_wait.get(task_id, 0.0) + critical_blocking
            time_weakness = max(0.0, min(1.0, raw_time / max(makespan, 1e-6))) if makespan > 0 else 0.0
            output[task_id] = {
                "TimeWeakness": time_weakness,
                "resourceWait": resource_wait.get(task_id, 0.0),
                "criticality": criticality.get(task_id, 0.0),
                "criticalBlocking": critical_blocking,
            }
        return output

    def _actual_successors(self, assignment: dict[str, str], actual_schedule: dict[str, Any]) -> dict[str, set[str]]:
        successors = {task_id: set() for task_id in self.task_ids}
        for task_id, preds in self.evaluator.predecessors.items():
            for pred in preds:
                if pred in successors and task_id in actual_schedule:
                    successors[pred].add(task_id)
        by_resource: dict[str, list[str]] = {}
        for task_id, resource_id in assignment.items():
            if task_id in actual_schedule and resource_id:
                by_resource.setdefault(resource_id, []).append(task_id)
        for task_ids in by_resource.values():
            ordered = sorted(task_ids, key=lambda item: (actual_schedule[item].planned_start_hour, actual_schedule[item].planned_end_hour, item))
            for first, second in zip(ordered, ordered[1:]):
                successors.setdefault(first, set()).add(second)
        return successors

    def _latest_start_times(
        self,
        actual_schedule: dict[str, Any],
        successors: dict[str, set[str]],
        makespan: float,
    ) -> dict[str, float]:
        ordered = sorted(
            [task_id for task_id in self.task_ids if task_id in actual_schedule],
            key=lambda task_id: (actual_schedule[task_id].planned_end_hour, actual_schedule[task_id].planned_start_hour, task_id),
            reverse=True,
        )
        latest_finish = {task_id: makespan for task_id in ordered}
        latest_start: dict[str, float] = {}
        for task_id in ordered:
            item = actual_schedule[task_id]
            succs = [succ for succ in successors.get(task_id, set()) if succ in latest_start]
            if succs:
                latest_finish[task_id] = min(latest_start[succ] for succ in succs)
            latest_start[task_id] = latest_finish[task_id] - item.duration_hours
        return latest_start

    def _next_task_on_resource(self, assignment: dict[str, str], actual_schedule: dict[str, Any]) -> dict[str, str]:
        output: dict[str, str] = {}
        by_resource: dict[str, list[str]] = {}
        for task_id, resource_id in assignment.items():
            if task_id in actual_schedule and resource_id:
                by_resource.setdefault(resource_id, []).append(task_id)
        for task_ids in by_resource.values():
            ordered = sorted(task_ids, key=lambda item: (actual_schedule[item].planned_start_hour, actual_schedule[item].planned_end_hour, item))
            for first, second in zip(ordered, ordered[1:]):
                output[first] = second
        return output

    def _mean_component(self, items: list[dict[str, float]], key: str) -> float | None:
        values = [float(item[key]) for item in items if key in item]
        return round(mean(values), 6) if values else None

    def _lift(self, selected: list[dict[str, float]], all_items: list[dict[str, float]], key: str) -> float | None:
        selected_mean = self._mean_component(selected, key)
        all_mean = self._mean_component(all_items, key)
        if selected_mean is None or all_mean is None:
            return None
        return round(selected_mean - all_mean, 6)

    def _num_above_median(self, items: list[dict[str, float]], key: str) -> int:
        values = [float(item[key]) for item in items if key in item]
        if not values:
            return 0
        midpoint = median(values)
        return sum(1 for value in values if value > midpoint)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default


class HybridHarmonySearchObjectiveGuidedMutationGAAssignmentOptimizer(HybridHarmonySearchGAAssignmentOptimizer):
    """HS150 warm-start followed by GA850 with objective-aligned guided mutation."""

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
        optimizer = ObjectiveGuidedMutationSeededGeneticAlgorithmAssignmentOptimizer(
            tasks=self.tasks,
            resources=self.resources,
            evaluator=self.evaluator,
            options=ga_options,
            seed_candidates=seed_candidates,
        )
        optimizer.phase_start_evaluations = self.evaluator.objective_evaluations
        return optimizer.run()

class ObjectiveGuidedMutationLiteSeededGeneticAlgorithmAssignmentOptimizer(
    ObjectiveGuidedMutationSeededGeneticAlgorithmAssignmentOptimizer
):
    """
    TA-4 Lite:
    Objective-aligned guided mutation with lightweight TimeWeakness.

    Keeps KPIWeakness and CostWeakness identical to TA-4 Full.

    Replaces the expensive TimeWeakness:
        criticality * resourceWait + criticalBlocking
    with:
        resourceWait / actualMakespan

    Does NOT use:
    - realized successor graph
    - slack / latest-start analysis
    - criticality
    - critical blocking
    """

    def _history_row(
        self,
        iteration: int,
        population: list[AssignmentCandidate],
        best: AssignmentCandidate,
        phase_start_evaluations: int,
    ) -> dict[str, Any]:
        row = super()._history_row(
            iteration,
            population,
            best,
            phase_start_evaluations,
        )
        row["mutationType"] = "objective_aligned_guided_lite"
        row["timeGuidanceType"] = "resource_wait_over_makespan"
        return row

    def _guided_mutate(
        self,
        assignment: dict[str, str],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        mutated = dict(assignment)

        weaknesses = self._weakness_scores(assignment)

        selected_task_ids = self._sample_mutation_tasks(
            weaknesses,
            self._mutation_attempt_count(),
        )

        for task_id in selected_task_ids:
            options = self.candidate_pool.get(task_id, [])
            if not options:
                continue

            current = mutated.get(task_id)
            alternatives = [
                item
                for item in options
                if item != current
            ]

            mutated[task_id] = self.random.choice(
                alternatives or options
            )

        selected_set = set(selected_task_ids)

        selected = [
            weaknesses[task_id]
            for task_id in selected_task_ids
            if task_id in weaknesses
        ]

        unselected = [
            item
            for task_id, item in weaknesses.items()
            if task_id not in selected_set
        ]

        all_items = list(weaknesses.values())

        normalized_waits = [
            float(item.get("TimeWeakness", 0.0))
            for item in all_items
        ]

        stats = {
            "actualMutatedGeneCount": len(selected_task_ids),

            "KPIWeaknessMean":
                self._mean_component(all_items, "KPIWeakness"),

            "CostWeaknessMean":
                self._mean_component(all_items, "CostWeakness"),

            "TimeWeaknessMean":
                self._mean_component(all_items, "TimeWeakness"),

            "overallWeaknessMean":
                self._mean_component(all_items, "weakness"),

            "selectedKPIWeaknessMean":
                self._mean_component(selected, "KPIWeakness"),

            "selectedCostWeaknessMean":
                self._mean_component(selected, "CostWeakness"),

            "selectedTimeWeaknessMean":
                self._mean_component(selected, "TimeWeakness"),

            "selectedOverallWeaknessMean":
                self._mean_component(selected, "weakness"),

            "unselectedOverallWeaknessMean":
                self._mean_component(unselected, "weakness"),

            "overallWeaknessLift":
                self._lift(selected, all_items, "weakness"),

            "KPIWeaknessLift":
                self._lift(selected, all_items, "KPIWeakness"),

            "CostWeaknessLift":
                self._lift(selected, all_items, "CostWeakness"),

            "TimeWeaknessLift":
                self._lift(selected, all_items, "TimeWeakness"),

            "meanResourceWait":
                self._mean_component(all_items, "resourceWait"),

            "maxResourceWait":
                round(
                    max(
                        (
                            float(item.get("resourceWait", 0.0))
                            for item in all_items
                        ),
                        default=0.0,
                    ),
                    6,
                ),

            "numTasksWithResourceWait":
                sum(
                    1
                    for item in all_items
                    if float(item.get("resourceWait", 0.0)) > 1e-9
                ),

            "meanNormalizedResourceWait":
                round(mean(normalized_waits), 6)
                if normalized_waits
                else 0.0,

            "meanCostWeakness":
                self._mean_component(all_items, "CostWeakness"),

            "selectedMeanCostWeakness":
                self._mean_component(selected, "CostWeakness"),

            "costWeaknessLift":
                self._lift(selected, all_items, "CostWeakness"),

            "numTasksAtMinCost":
                sum(
                    1
                    for item in all_items
                    if float(item.get("CostWeakness", 0.0)) <= 1e-9
                ),

            "numTasksAboveMedianCost":
                self._num_above_median(
                    all_items,
                    "CostWeakness",
                ),

            "numTasksAtMaxCost":
                sum(
                    1
                    for item in all_items
                    if float(item.get("CostWeakness", 0.0))
                    >= 1.0 - 1e-9
                ),

            "meanKPIWeakness":
                self._mean_component(all_items, "KPIWeakness"),

            "selectedMeanKPIWeakness":
                self._mean_component(selected, "KPIWeakness"),

            "kpiWeaknessLift":
                self._lift(selected, all_items, "KPIWeakness"),

            "meanTaskKPIContribution":
                self._mean_component(
                    all_items,
                    "taskKPIContribution",
                ),

            "selectedMeanTaskKPIContribution":
                self._mean_component(
                    selected,
                    "taskKPIContribution",
                ),
        }

        return mutated, stats

    def _weakness_scores(
        self,
        assignment: dict[str, str],
    ) -> dict[str, dict[str, float]]:
        """
        Same objective-aligned weakness as TA-4 Full, except that
        TimeWeakness is the lightweight resource-wait formulation.
        """
        actual_schedule = self.evaluator.assignment_scheduler.build(
            assignment
        )

        time_components = self._time_components(
            assignment,
            actual_schedule,
        )

        weights = self.evaluator.strategy_weights()

        output: dict[str, dict[str, float]] = {}

        for task_id in self.task_ids:
            resource_id = assignment.get(task_id)

            kpi_weakness, task_contribution = (
                self._kpi_weakness(
                    task_id,
                    resource_id,
                )
            )

            cost_weakness = self._cost_weakness(
                task_id,
                resource_id,
            )

            time_item = time_components.get(
                task_id,
                {},
            )

            time_weakness = float(
                time_item.get(
                    "TimeWeakness",
                    0.0,
                )
            )

            weakness = (
                weights["kpi"] * kpi_weakness
                + weights["cost"] * cost_weakness
                + weights["makespan"] * time_weakness
            )

            output[task_id] = {
                "weakness":
                    max(
                        0.0,
                        min(
                            1.0,
                            weakness,
                        ),
                    ),

                "KPIWeakness":
                    kpi_weakness,

                "CostWeakness":
                    cost_weakness,

                "TimeWeakness":
                    time_weakness,

                "taskKPIContribution":
                    task_contribution,

                "resourceWait":
                    float(
                        time_item.get(
                            "resourceWait",
                            0.0,
                        )
                    ),
            }

        return output

    def _time_components(
        self,
        assignment: dict[str, str],
        actual_schedule: dict[str, Any],
    ) -> dict[str, dict[str, float]]:
        """
        Lightweight time guidance:

            predReady_i
                = max(actual end of predecessors)

            resourceWait_i
                = max(0, actualStart_i - predReady_i)

            TimeWeakness_i
                = resourceWait_i / actualMakespan

        No deadline, workload, criticality or blocking is used.
        """

        del assignment  # kept only to match parent method signature

        makespan = max(
            (
                float(item.planned_end_hour)
                for item in actual_schedule.values()
            ),
            default=0.0,
        )

        output: dict[str, dict[str, float]] = {}

        for task_id in self.task_ids:
            item = actual_schedule.get(task_id)

            if item is None:
                output[task_id] = {
                    "TimeWeakness": 0.0,
                    "resourceWait": 0.0,
                }
                continue

            pred_ready = max(
                (
                    float(
                        actual_schedule[pred].planned_end_hour
                    )
                    for pred in self.evaluator.predecessors.get(
                        task_id,
                        [],
                    )
                    if pred in actual_schedule
                ),
                default=0.0,
            )

            resource_wait = max(
                0.0,
                float(item.planned_start_hour)
                - pred_ready,
            )

            if makespan > 0.0:
                time_weakness = (
                    resource_wait
                    / makespan
                )
            else:
                time_weakness = 0.0

            time_weakness = max(
                0.0,
                min(
                    1.0,
                    time_weakness,
                ),
            )

            output[task_id] = {
                "TimeWeakness":
                    time_weakness,

                "resourceWait":
                    resource_wait,
            }

        return output

    def _record_guided_mutation(
        self,
        candidate: AssignmentCandidate,
        stats: dict[str, Any],
    ) -> None:
        best = self._best or candidate

        phase_start_evaluations = int(
            getattr(
                self,
                "phase_start_evaluations",
                0,
            )
        )

        weights = self.evaluator.strategy_weights()

        self.guided_mutation_history.append(
            {
                "iteration":
                    self._current_generation,

                "phase":
                    "ga",

                "objectiveEvaluation":
                    self.evaluator.objective_evaluations,

                "phaseEvaluation":
                    self.evaluator.objective_evaluations
                    - phase_start_evaluations,

                "bestFoundAtEvaluation":
                    best.score.diagnostics.get(
                        "objectiveEvaluation"
                    ),

                "bestScore":
                    best.score.total_score,

                "currentScore":
                    candidate.score.total_score,

                "mutationType":
                    "objective_aligned_guided_lite",

                "timeGuidanceType":
                    "resource_wait_over_makespan",

                "baselineMutationRate":
                    self.options.mutation_rate,

                "expectedMutationCount":
                    self._expected_mutation_count(),

                "objectiveKpiWeight":
                    weights["kpi"],

                "objectiveCostWeight":
                    weights["cost"],

                "objectiveTimeWeight":
                    weights["makespan"],

                **stats,
            }
        )


class HybridHarmonySearchObjectiveGuidedMutationLiteGAAssignmentOptimizer(
    HybridHarmonySearchGAAssignmentOptimizer
):
    """
    HS150 warm-start followed by GA850 with
    lightweight objective-aligned guided mutation.
    """

    def _hs_ratio(self) -> float:
        return min(
            0.99,
            max(
                0.01,
                float(
                    getattr(
                        self.options,
                        "hybrid_hs_ratio",
                        0.15,
                    )
                ),
            ),
        )

    def _run_ga_phase(
        self,
        ga_iterations: int,
        seed_candidates: list[AssignmentCandidate],
        started_perf: float,
    ) -> OptimizationResult:
        ga_timeout = None

        if self.options.timeout:
            elapsed = (
                time.perf_counter()
                - started_perf
            )

            ga_timeout = max(
                0.1,
                self.options.timeout
                - elapsed,
            )

        ga_options = replace(
            self.options,
            max_iterations=ga_iterations,
            max_evaluations=self.options.max_evaluations,
        )

        if ga_timeout is not None:
            ga_options = replace(
                ga_options,
                timeout=ga_timeout,
            )

        optimizer = (
            ObjectiveGuidedMutationLiteSeededGeneticAlgorithmAssignmentOptimizer(
                tasks=self.tasks,
                resources=self.resources,
                evaluator=self.evaluator,
                options=ga_options,
                seed_candidates=seed_candidates,
            )
        )

        optimizer.phase_start_evaluations = (
            self.evaluator.objective_evaluations
        )

        return optimizer.run()


class ObjectiveGuidedMutationSelectiveSeededGeneticAlgorithmAssignmentOptimizer(
    ObjectiveGuidedMutationLiteSeededGeneticAlgorithmAssignmentOptimizer
):
    """
    TA-4 Selective:
    Multi-fidelity mutation guidance with selective activation.

    For each offspring after crossover:
    - With probability p: use TA-4 Lite objective-guided mutation
      (KPIWeakness + CostWeakness + TimeWeakness, builds actual schedule)
    - With probability 1-p: use Skill-only guided mutation
      (only SkillWeakness, NO schedule build)

    After mutation, all offspring receive exact evaluation via
    AssignmentEvaluator.evaluate() as usual.

    This reduces the computational overhead of TA-4 while retaining
    its temporal guidance for a fraction of offspring.
    """

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
        self.objective_guidance_probability = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.options,
                        "objective_guidance_probability",
                        0.5,
                    )
                ),
            ),
        )
        self.objective_guidance_count = 0
        self.skill_guidance_count = 0
        self.guidance_schedule_build_count = 0
        self._skill_weakness_cache: dict[str, dict[str, float]] = {}
        self._precompute_skill_weakness_cache()

    def _precompute_skill_weakness_cache(self) -> None:
        """Precompute skill weakness for all task-resource pairs (dataset-static)."""
        for task_id in self.task_ids:
            self._skill_weakness_cache[task_id] = {}
            for resource_id in self.candidate_pool.get(task_id, []):
                skill_match = (
                    self.evaluator.resource_skill_score(task_id, resource_id)
                    / 100.0
                )
                self._skill_weakness_cache[task_id][resource_id] = 1.0 - max(
                    0.0, min(1.0, skill_match)
                )

    def _history_row(
        self,
        iteration: int,
        population: list[AssignmentCandidate],
        best: AssignmentCandidate,
        phase_start_evaluations: int,
    ) -> dict[str, Any]:
        row = super()._history_row(
            iteration,
            population,
            best,
            phase_start_evaluations,
        )
        row["mutationType"] = "objective_aligned_guided_selective"
        row["selectiveGuidanceProbability"] = self.objective_guidance_probability
        row["objectiveGuidanceCount"] = self.objective_guidance_count
        row["skillGuidanceCount"] = self.skill_guidance_count
        row["guidanceScheduleBuildCount"] = self.guidance_schedule_build_count
        total = self.objective_guidance_count + self.skill_guidance_count
        row["observedObjectiveGuidanceRate"] = (
            round(self.objective_guidance_count / total, 6)
            if total > 0
            else None
        )
        return row

    def _guided_mutate(
        self,
        assignment: dict[str, str],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        mutated = dict(assignment)

        if self.random.random() < self.objective_guidance_probability:
            guidance_type = "objective"
            self.objective_guidance_count += 1
            self.guidance_schedule_build_count += 1
            weaknesses = self._weakness_scores(assignment)
        else:
            guidance_type = "skill"
            self.skill_guidance_count += 1
            weaknesses = self._skill_only_weakness_scores(assignment)

        selected_task_ids = self._sample_mutation_tasks(
            weaknesses,
            self._mutation_attempt_count(),
        )

        for task_id in selected_task_ids:
            options = self.candidate_pool.get(task_id, [])
            if not options:
                continue
            current = mutated.get(task_id)
            alternatives = [
                item
                for item in options
                if item != current
            ]
            mutated[task_id] = self.random.choice(
                alternatives or options
            )

        selected_set = set(selected_task_ids)

        selected = [
            weaknesses[task_id]
            for task_id in selected_task_ids
            if task_id in weaknesses
        ]

        unselected = [
            item
            for task_id, item in weaknesses.items()
            if task_id not in selected_set
        ]

        all_items = list(weaknesses.values())

        stats = {
            "guidanceType": guidance_type,
            "actualMutatedGeneCount": len(selected_task_ids),
            "overallWeaknessMean":
                self._mean_component(all_items, "weakness"),
            "selectedOverallWeaknessMean":
                self._mean_component(selected, "weakness"),
            "unselectedOverallWeaknessMean":
                self._mean_component(unselected, "weakness"),
            "overallWeaknessLift":
                self._lift(selected, all_items, "weakness"),
        }

        if guidance_type == "objective":
            stats.update({
                "KPIWeaknessMean":
                    self._mean_component(all_items, "KPIWeakness"),
                "CostWeaknessMean":
                    self._mean_component(all_items, "CostWeakness"),
                "TimeWeaknessMean":
                    self._mean_component(all_items, "TimeWeakness"),
                "selectedKPIWeaknessMean":
                    self._mean_component(selected, "KPIWeakness"),
                "selectedCostWeaknessMean":
                    self._mean_component(selected, "CostWeakness"),
                "selectedTimeWeaknessMean":
                    self._mean_component(selected, "TimeWeakness"),
                "meanResourceWait":
                    self._mean_component(all_items, "resourceWait"),
                "maxResourceWait":
                    round(
                        max(
                            (
                                float(item.get("resourceWait", 0.0))
                                for item in all_items
                            ),
                            default=0.0,
                        ),
                        6,
                    ),
                "numTasksWithResourceWait":
                    sum(
                        1
                        for item in all_items
                        if float(item.get("resourceWait", 0.0)) > 1e-9
                    ),
            })
        else:
            stats.update({
                "skillWeaknessMean":
                    self._mean_component(all_items, "skillWeakness"),
                "selectedSkillWeaknessMean":
                    self._mean_component(selected, "skillWeakness"),
            })

        return mutated, stats

    def _skill_only_weakness_scores(
        self,
        assignment: dict[str, str],
    ) -> dict[str, dict[str, float]]:
        """
        Skill-only weakness: does NOT build an actual schedule.

        weakness = skillWeakness = 1 - (resource_skill_score / 100)

        Uses precomputed cache for speed.
        """
        output: dict[str, dict[str, float]] = {}
        for task_id in self.task_ids:
            resource_id = assignment.get(task_id)
            if (
                resource_id
                and task_id in self._skill_weakness_cache
                and resource_id in self._skill_weakness_cache[task_id]
            ):
                skill_weakness = self._skill_weakness_cache[task_id][resource_id]
            elif resource_id:
                skill_match = (
                    self.evaluator.resource_skill_score(task_id, resource_id)
                    / 100.0
                )
                skill_weakness = 1.0 - max(0.0, min(1.0, skill_match))
            else:
                skill_weakness = 1.0

            output[task_id] = {
                "weakness": max(0.0, min(1.0, skill_weakness)),
                "skillWeakness": skill_weakness,
            }
        return output

    def _record_guided_mutation(
        self,
        candidate: AssignmentCandidate,
        stats: dict[str, Any],
    ) -> None:
        best = self._best or candidate

        phase_start_evaluations = int(
            getattr(
                self,
                "phase_start_evaluations",
                0,
            )
        )

        weights = self.evaluator.strategy_weights()
        total = self.objective_guidance_count + self.skill_guidance_count

        self.guided_mutation_history.append(
            {
                "iteration":
                    self._current_generation,

                "phase":
                    "ga",

                "objectiveEvaluation":
                    self.evaluator.objective_evaluations,

                "phaseEvaluation":
                    self.evaluator.objective_evaluations
                    - phase_start_evaluations,

                "bestFoundAtEvaluation":
                    best.score.diagnostics.get(
                        "objectiveEvaluation"
                    ),

                "bestScore":
                    best.score.total_score,

                "currentScore":
                    candidate.score.total_score,

                "mutationType":
                    "objective_aligned_guided_selective",

                "selectiveGuidanceProbability":
                    self.objective_guidance_probability,

                "objectiveGuidanceCount":
                    self.objective_guidance_count,

                "skillGuidanceCount":
                    self.skill_guidance_count,

                "guidanceScheduleBuildCount":
                    self.guidance_schedule_build_count,

                "observedObjectiveGuidanceRate":
                    round(self.objective_guidance_count / total, 6)
                    if total > 0
                    else None,

                "baselineMutationRate":
                    self.options.mutation_rate,

                "expectedMutationCount":
                    self._expected_mutation_count(),

                "objectiveKpiWeight":
                    weights["kpi"],

                "objectiveCostWeight":
                    weights["cost"],

                "objectiveTimeWeight":
                    weights["makespan"],

                **stats,
            }
        )


class HybridHarmonySearchObjectiveGuidedMutationSelectiveGAAssignmentOptimizer(
    HybridHarmonySearchGAAssignmentOptimizer
):
    """
    HS150 warm-start followed by GA850 with
    selective objective-aligned guided mutation (TA-4 Selective).
    """

    def _hs_ratio(self) -> float:
        return min(
            0.99,
            max(
                0.01,
                float(
                    getattr(
                        self.options,
                        "hybrid_hs_ratio",
                        0.15,
                    )
                ),
            ),
        )

    def _run_ga_phase(
        self,
        ga_iterations: int,
        seed_candidates: list[AssignmentCandidate],
        started_perf: float,
    ) -> OptimizationResult:
        ga_timeout = None

        if self.options.timeout:
            elapsed = (
                time.perf_counter()
                - started_perf
            )

            ga_timeout = max(
                0.1,
                self.options.timeout
                - elapsed,
            )

        ga_options = replace(
            self.options,
            max_iterations=ga_iterations,
            max_evaluations=self.options.max_evaluations,
        )

        if ga_timeout is not None:
            ga_options = replace(
                ga_options,
                timeout=ga_timeout,
            )

        optimizer = (
            ObjectiveGuidedMutationSelectiveSeededGeneticAlgorithmAssignmentOptimizer(
                tasks=self.tasks,
                resources=self.resources,
                evaluator=self.evaluator,
                options=ga_options,
                seed_candidates=seed_candidates,
            )
        )

        optimizer.phase_start_evaluations = (
            self.evaluator.objective_evaluations
        )

        return optimizer.run()


class DiverseSeededGeneticAlgorithmAssignmentOptimizer(SeededGeneticAlgorithmAssignmentOptimizer):
    """Seed GA with HS quality and diversity, then refresh collapsed populations."""

    elite_fraction = 0.2
    diverse_fraction = 0.2
    mutated_fraction = 0.2
    diversity_threshold = 0.15
    refresh_fraction = 0.1

    def _initial_population(self) -> list[AssignmentCandidate]:
        population_size = max(1, self.options.population_size)
        seed_pool = self._unique_sorted_candidates(self.seed_candidates)
        target_elite = min(len(seed_pool), max(1, int(round(population_size * self.elite_fraction))))
        target_diverse = min(
            max(0, len(seed_pool) - target_elite),
            max(1, int(round(population_size * self.diverse_fraction))),
        )
        target_mutated = max(1, int(round(population_size * self.mutated_fraction)))

        population: list[AssignmentCandidate] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        elites = seed_pool[:target_elite]
        self._append_cached(population, seen, elites, population_size)

        diverse = self._select_diverse_representatives(seed_pool[target_elite:], population, target_diverse)
        self._append_cached(population, seen, diverse, population_size)

        transferred_cached = len(population)
        mutated_new = self._append_mutated(population, seen, elites + diverse, target_mutated, population_size)
        random_new = self._append_random(population, seen, population_size)

        if not population:
            population.append(self._candidate(self._greedy_assignment()))
        population.sort(key=lambda item: item.score.rank_key(), reverse=True)
        self.transition_diagnostics = {
            "transitionEvaluation": int(getattr(self, "phase_start_evaluations", self.evaluator.objective_evaluations)),
            "gaInitialPopulationUniqueCount": self._unique_population_count(population),
            "gaInitialPopulationDiversity": self._population_diversity(population),
            "numTransferredCached": transferred_cached,
            "numMutatedNew": mutated_new,
            "numRandomNew": random_new,
            "populationSize": len(population),
        }
        return population

    def _next_generation(self, population: list[AssignmentCandidate]) -> list[AssignmentCandidate]:
        next_population = super()._next_generation(population)
        if self._population_diversity(next_population) >= self.diversity_threshold:
            return next_population
        if self._evaluation_budget_exhausted():
            return next_population

        population_size = len(next_population)
        elite_count = max(0, min(self.options.elitism_count, population_size))
        refresh_count = max(1, int(round(population_size * self.refresh_fraction)))
        seen = {self._assignment_key(candidate.assignment) for candidate in next_population[:elite_count]}
        refreshed = next_population[:elite_count]
        survivors = next_population[elite_count: max(elite_count, population_size - refresh_count)]
        for candidate in survivors:
            key = self._assignment_key(candidate.assignment)
            if key in seen:
                continue
            seen.add(key)
            refreshed.append(candidate)

        while len(refreshed) < population_size and not self._evaluation_budget_exhausted():
            candidate = self._candidate(self._random_assignment())
            key = self._assignment_key(candidate.assignment)
            if key in seen:
                continue
            seen.add(key)
            refreshed.append(candidate)
        refreshed.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return refreshed

    def _unique_sorted_candidates(self, candidates: list[AssignmentCandidate]) -> list[AssignmentCandidate]:
        unique: list[AssignmentCandidate] = []
        seen = set()
        for candidate in sorted(candidates, key=lambda item: item.score.rank_key(), reverse=True):
            key = self._assignment_key(candidate.assignment)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _append_cached(
        self,
        population: list[AssignmentCandidate],
        seen: set[tuple[tuple[str, str], ...]],
        candidates: list[AssignmentCandidate],
        population_size: int,
    ) -> None:
        for candidate in candidates:
            if len(population) >= population_size:
                break
            key = self._assignment_key(candidate.assignment)
            if key in seen:
                continue
            seen.add(key)
            population.append(candidate)

    def _select_diverse_representatives(
        self,
        candidates: list[AssignmentCandidate],
        selected: list[AssignmentCandidate],
        count: int,
    ) -> list[AssignmentCandidate]:
        output = []
        remaining = list(candidates)
        while remaining and len(output) < count:
            best_index = 0
            best_key = None
            anchors = selected + output
            for index, candidate in enumerate(remaining):
                if anchors:
                    min_distance = min(self._normalized_hamming_distance(candidate.assignment, item.assignment) for item in anchors)
                else:
                    min_distance = 1.0
                key = (min_distance, candidate.score.total_score)
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = index
            output.append(remaining.pop(best_index))
        return output

    def _append_mutated(
        self,
        population: list[AssignmentCandidate],
        seen: set[tuple[tuple[str, str], ...]],
        sources: list[AssignmentCandidate],
        count: int,
        population_size: int,
    ) -> int:
        if not sources:
            return 0
        added = 0
        attempts = 0
        while added < count and len(population) < population_size and not self._evaluation_budget_exhausted() and attempts < count * 20:
            attempts += 1
            source = self.random.choice(sources)
            assignment = self._mutate_seed_assignment(source.assignment)
            key = self._assignment_key(assignment)
            if key in seen:
                continue
            seen.add(key)
            population.append(self._candidate(assignment))
            added += 1
        return added

    def _append_random(
        self,
        population: list[AssignmentCandidate],
        seen: set[tuple[tuple[str, str], ...]],
        population_size: int,
    ) -> int:
        added = 0
        attempts = 0
        while len(population) < population_size and not self._evaluation_budget_exhausted() and attempts < population_size * 20:
            attempts += 1
            assignment = self._random_assignment()
            key = self._assignment_key(assignment)
            if key in seen:
                continue
            seen.add(key)
            population.append(self._candidate(assignment))
            added += 1
        return added

    def _mutate_seed_assignment(self, assignment: dict[str, str]) -> dict[str, str]:
        mutated = dict(assignment)
        changed = False
        rate = max(self.options.mutation_rate, 1.0 / max(1, len(self.task_ids)))
        for task_id in self.task_ids:
            if self.random.random() >= rate:
                continue
            options = self.candidate_pool.get(task_id, [])
            if not options:
                continue
            current = mutated.get(task_id)
            alternatives = [item for item in options if item != current]
            mutated[task_id] = self.random.choice(alternatives or options)
            changed = True
        if not changed and self.task_ids:
            task_id = self.random.choice(self.task_ids)
            options = self.candidate_pool.get(task_id, [])
            if options:
                current = mutated.get(task_id)
                alternatives = [item for item in options if item != current]
                mutated[task_id] = self.random.choice(alternatives or options)
        return mutated


class HybridHarmonySearchDiverseGAAssignmentOptimizer(HybridHarmonySearchGAAssignmentOptimizer):
    """Short HS warm-start followed by a diversity-preserving GA."""

    def _hs_ratio(self) -> float:
        value = float(getattr(self.options, "hybrid_hs_ratio", 0.15))
        return min(0.99, max(0.01, value))

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
        optimizer = DiverseSeededGeneticAlgorithmAssignmentOptimizer(
            tasks=self.tasks,
            resources=self.resources,
            evaluator=self.evaluator,
            options=ga_options,
            seed_candidates=seed_candidates,
        )
        optimizer.phase_start_evaluations = self.evaluator.objective_evaluations
        result = optimizer.run()
        self.transition_diagnostics = getattr(optimizer, "transition_diagnostics", {})
        return result

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
        diagnostics = getattr(self, "transition_diagnostics", {})
        if diagnostics:
            hs_best_score = hs_result.best.score.total_score
            transition_row = {
                "iteration": hs_result.iterations_run,
                "phase": "transition",
                "objectiveEvaluation": diagnostics.get("transitionEvaluation"),
                "phaseEvaluation": 0,
                "bestFoundAtEvaluation": hs_result.best.score.diagnostics.get("objectiveEvaluation"),
                "bestScore": hs_best_score,
                "currentScore": hs_best_score,
                "transitionEvaluation": diagnostics.get("transitionEvaluation"),
                "hsBestScore": hs_best_score,
                "hsMemoryUniqueCount": len(self._deduplicate_candidates(hs_result.memory + [hs_result.best])),
                "gaInitialPopulationUniqueCount": diagnostics.get("gaInitialPopulationUniqueCount"),
                "gaInitialPopulationDiversity": diagnostics.get("gaInitialPopulationDiversity"),
                "populationDiversity": diagnostics.get("gaInitialPopulationDiversity"),
                "uniquePopulationCount": diagnostics.get("gaInitialPopulationUniqueCount"),
                "populationSize": diagnostics.get("populationSize"),
                "numTransferredCached": diagnostics.get("numTransferredCached"),
                "numMutatedNew": diagnostics.get("numMutatedNew"),
                "numRandomNew": diagnostics.get("numRandomNew"),
            }
            history.append(transition_row)
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
        )


class TwoPhaseSeededGeneticAlgorithmAssignmentOptimizer(SeededGeneticAlgorithmAssignmentOptimizer):
    """Seeded GA that alternates exploitation and exploration without dropping global elite."""

    phase_budget = 100
    exploitation_mutation_rate = 0.12
    exploration_mutation_rate = 0.28
    exploitation_crossover_rate = 0.85
    exploration_crossover_rate = 0.8
    exploration_excluded_top_fraction = 0.1

    def run(self) -> OptimizationResult:
        started = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        phase_start_evaluations = int(getattr(self, "phase_start_evaluations", self.evaluator.objective_evaluations))
        population = self._initial_population()
        history = []
        best = population[0]
        history.append(
            self._history_row(
                iteration=0,
                population=population,
                best=best,
                phase_start_evaluations=phase_start_evaluations,
            )
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
                self._history_row(
                    iteration=generation,
                    population=population,
                    best=best,
                    phase_start_evaluations=phase_start_evaluations,
                )
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

    def _history_row(
        self,
        iteration: int,
        population: list[AssignmentCandidate],
        best: AssignmentCandidate,
        phase_start_evaluations: int,
    ) -> dict[str, Any]:
        phase_evaluation = self.evaluator.objective_evaluations - phase_start_evaluations
        current = population[0]
        return {
            "iteration": iteration,
            "phase": self._ga_phase_name(phase_evaluation),
            "objectiveEvaluation": self.evaluator.objective_evaluations,
            "phaseEvaluation": phase_evaluation,
            "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
            "bestScore": best.score.total_score,
            "bestFeasible": best.score.feasible,
            "currentScore": current.score.total_score,
            "currentFeasible": current.score.feasible,
            "violations": len(current.score.hard_violations),
            "populationDiversity": self._population_diversity(population),
            "uniquePopulationCount": self._unique_population_count(population),
            "populationSize": len(population),
        }

    def _next_generation(self, population: list[AssignmentCandidate]) -> list[AssignmentCandidate]:
        phase_start_evaluations = int(getattr(self, "phase_start_evaluations", self.evaluator.objective_evaluations))
        phase_evaluation = self.evaluator.objective_evaluations - phase_start_evaluations
        exploration = self._ga_phase_name(phase_evaluation) == "ga_explore"
        population_size = max(1, self.options.population_size)
        elite_count = max(0, min(self.options.elitism_count, population_size, len(population)))
        next_population = population[:elite_count]
        parent_pool = self._parent_pool(population, exploration)
        mutation_rate = self.exploration_mutation_rate if exploration else self.exploitation_mutation_rate
        crossover_rate = self.exploration_crossover_rate if exploration else self.exploitation_crossover_rate

        while len(next_population) < population_size and not self._evaluation_budget_exhausted():
            parent_a = self._select_parent(parent_pool)
            parent_b = self._select_parent(parent_pool)
            if exploration:
                parent_b = self._select_different_parent(parent_pool, parent_a)
            if self.random.random() < crossover_rate:
                child_assignment = self._crossover(parent_a.assignment, parent_b.assignment)
            else:
                child_assignment = dict(parent_a.assignment)
            child_assignment = self._mutate_with_rate(child_assignment, mutation_rate)
            next_population.append(self._candidate(child_assignment))

        next_population.sort(key=lambda item: item.score.rank_key(), reverse=True)
        return next_population

    def _ga_phase_name(self, phase_evaluation: int) -> str:
        phase_index = max(0, int(phase_evaluation) // self.phase_budget)
        return "ga_exploit" if phase_index % 2 == 0 else "ga_explore"

    def _parent_pool(self, population: list[AssignmentCandidate], exploration: bool) -> list[AssignmentCandidate]:
        if not exploration or len(population) <= 2:
            return population
        excluded = max(1, int(round(len(population) * self.exploration_excluded_top_fraction)))
        return population[min(excluded, len(population) - 1) :] or population

    def _select_different_parent(
        self,
        population: list[AssignmentCandidate],
        first: AssignmentCandidate,
    ) -> AssignmentCandidate:
        if len(population) < 2:
            return first
        for _ in range(5):
            candidate = self._select_parent(population)
            if self._assignment_key(candidate.assignment) != self._assignment_key(first.assignment):
                return candidate
        return self._select_parent(population)

    def _mutate_with_rate(self, assignment: dict[str, str], mutation_rate: float) -> dict[str, str]:
        mutated = dict(assignment)
        for task_id in self.task_ids:
            if self.random.random() >= mutation_rate:
                continue
            options = self.candidate_pool.get(task_id, [])
            if not options:
                continue
            current = mutated.get(task_id)
            alternatives = [item for item in options if item != current]
            mutated[task_id] = self.random.choice(alternatives or options)
        return mutated


class HybridHarmonySearchTwoPhaseGAAssignmentOptimizer(HybridHarmonySearchDiverseGAAssignmentOptimizer):
    """HS150 warm-start followed by alternating exploitation/exploration GA."""

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
        optimizer = TwoPhaseSeededGeneticAlgorithmAssignmentOptimizer(
            tasks=self.tasks,
            resources=self.resources,
            evaluator=self.evaluator,
            options=ga_options,
            seed_candidates=seed_candidates,
        )
        optimizer.phase_start_evaluations = self.evaluator.objective_evaluations
        result = optimizer.run()
        self.transition_diagnostics = getattr(optimizer, "transition_diagnostics", {})
        return result


class HybridHarmonySearchAdaptiveSwitchGAAssignmentOptimizer(HybridHarmonySearchGAAssignmentOptimizer):
    """Standalone HS kernel with adaptive switch to seeded GA."""

    min_hs_evaluations = 100
    max_hs_evaluations = 250
    window_evaluations = 50
    improvement_threshold = 0.15

    def run(self) -> OptimizationResult:
        started = datetime.now(timezone.utc)
        started_perf = time.perf_counter()

        hs_result, switch_reason = self._run_adaptive_hs_phase(started_perf)
        switch_evaluation = self.evaluator.objective_evaluations
        seed_candidates = self._seed_candidates_from_hs(hs_result.memory + [hs_result.best])
        ga_iterations = max(1, self.options.max_iterations - hs_result.iterations_run)
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
                "switchEvaluation": switch_evaluation,
                "switchReason": switch_reason,
                "hsBestAtSwitch": hs_result.best.score.total_score,
            }
            for item in hs_result.history
        ]
        history.append(
            {
                "iteration": hs_result.iterations_run,
                "phase": "transition",
                "objectiveEvaluation": switch_evaluation,
                "phaseEvaluation": 0,
                "bestFoundAtEvaluation": hs_result.best.score.diagnostics.get("objectiveEvaluation"),
                "bestScore": hs_result.best.score.total_score,
                "currentScore": hs_result.best.score.total_score,
                "switchEvaluation": switch_evaluation,
                "switchReason": switch_reason,
                "hsBestAtSwitch": hs_result.best.score.total_score,
            }
        )
        history.extend(
            {
                **item,
                "iteration": hs_result.iterations_run + int(item.get("iteration") or 0),
                "phase": "ga",
                "switchEvaluation": switch_evaluation,
                "switchReason": switch_reason,
                "hsBestAtSwitch": hs_result.best.score.total_score,
            }
            for item in ga_result.history
        )
        return OptimizationResult(
            best=best,
            memory=memory[: max(1, self.options.top_candidates)],
            history=history,
            iterations_run=hs_result.iterations_run + ga_result.iterations_run,
            converged=ga_result.converged,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )

    def _run_adaptive_hs_phase(self, started_perf: float) -> tuple[OptimizationResult, str]:
        started = datetime.now(timezone.utc)
        phase_start_evaluations = self.evaluator.objective_evaluations
        phase_budget = phase_start_evaluations + min(self.max_hs_evaluations, int(self.options.max_evaluations or self.max_hs_evaluations))
        memory = self._initial_memory()
        best = memory[0]
        best_timeline: list[tuple[int, float]] = [(self.evaluator.objective_evaluations, best.score.total_score)]
        history = [
            {
                "iteration": 0,
                "phase": "hs",
                "objectiveEvaluation": self.evaluator.objective_evaluations,
                "phaseEvaluation": self.evaluator.objective_evaluations - phase_start_evaluations,
                "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
                "bestScore": best.score.total_score,
                "currentScore": best.score.total_score,
            }
        ]
        iterations_run = 0
        switch_reason = "max_hs_budget"

        for iteration in range(1, self.options.max_iterations + 1):
            if self.options.timeout and (time.perf_counter() - started_perf) >= self.options.timeout:
                switch_reason = "max_hs_budget"
                break
            if self.evaluator.objective_evaluations >= phase_budget:
                switch_reason = "max_hs_budget"
                break

            new_candidate = self._candidate(self._improvise(memory))
            worst = memory[-1]
            if self._better(new_candidate, worst):
                memory[-1] = new_candidate
                memory.sort(key=lambda item: item.score.rank_key(), reverse=True)
            if self._better(new_candidate, best):
                best = new_candidate

            current_evaluation = self.evaluator.objective_evaluations
            best_timeline.append((current_evaluation, best.score.total_score))
            history.append(
                {
                    "iteration": iteration,
                    "phase": "hs",
                    "objectiveEvaluation": current_evaluation,
                    "phaseEvaluation": current_evaluation - phase_start_evaluations,
                    "bestFoundAtEvaluation": best.score.diagnostics.get("objectiveEvaluation"),
                    "bestScore": best.score.total_score,
                    "currentScore": new_candidate.score.total_score,
                }
            )
            iterations_run = iteration

            phase_evaluation = current_evaluation - phase_start_evaluations
            if phase_evaluation >= self.max_hs_evaluations:
                switch_reason = "max_hs_budget"
                break
            if phase_evaluation >= self.min_hs_evaluations:
                previous_score = self._best_score_at_or_before(
                    best_timeline,
                    current_evaluation - self.window_evaluations,
                )
                if best.score.total_score - previous_score < self.improvement_threshold:
                    switch_reason = "stagnation"
                    break

        memory.sort(key=lambda item: item.score.rank_key(), reverse=True)
        if self._better(memory[0], best):
            best = memory[0]
        return (
            OptimizationResult(
                best=best,
                memory=memory,
                history=history,
                iterations_run=iterations_run,
                converged=switch_reason == "stagnation",
                started_at=started,
                finished_at=datetime.now(timezone.utc),
            ),
            switch_reason,
        )

    def _best_score_at_or_before(self, timeline: list[tuple[int, float]], target_evaluation: int) -> float:
        score = timeline[0][1]
        for evaluation, best_score in timeline:
            if evaluation > target_evaluation:
                break
            score = best_score
        return score
