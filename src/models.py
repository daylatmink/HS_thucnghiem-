from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class OptimizerOptions:
    harmony_memory_size: int = 20
    hmcr: float = 0.9
    par: float = 0.25
    max_iterations: int = 500
    top_candidates: int = 3
    seed: int = 42
    simulation_iterations: int = 20
    timeout: float | None = None
    max_evaluations: int | None = None
    fixed_budget: bool = False
    convergence_threshold: float = 0.001
    population_size: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.05
    elitism_count: int = 2
    tournament_size: int = 3
    hybrid_hs_ratio: float = 0.2
    objective_guidance_probability: float = 0.5


@dataclass(frozen=True)
class TaskSchedule:
    task_id: str
    planned_start_hour: float
    planned_end_hour: float
    duration_hours: float
    topo_level: int
    critical: bool = False


@dataclass
class CandidateScore:
    total_score: float
    kpi_score: float
    skill_score: float
    workload_score: float
    cost_score: float
    schedule_score: float
    hard_violations: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return not self.hard_violations

    def rank_key(self) -> tuple[float, float, float]:
        return (
            1.0 if self.feasible else 0.0,
            self.total_score,
            -float(len(self.hard_violations)),
        )


@dataclass
class AssignmentCandidate:
    assignment: dict[str, str]
    score: CandidateScore
    simulation: dict[str, Any] | None = None
    final_score: float | None = None
    assignment_id: str | None = None

    def final_rank_key(self) -> tuple[float, float, float, float]:
        simulated = self.simulation or {}
        feasible = 1.0 if simulated.get("isFeasible", self.score.feasible) else 0.0
        total_score = float(self.final_score if self.final_score is not None else self.score.total_score)
        bottlenecks = float(len(simulated.get("bottlenecks", [])))
        risks = float(len(simulated.get("risks", [])))
        return (feasible, total_score, -bottlenecks, -risks)


@dataclass
class OptimizationResult:
    best: AssignmentCandidate
    memory: list[AssignmentCandidate]
    history: list[dict[str, Any]]
    iterations_run: int
    converged: bool
    started_at: datetime
    finished_at: datetime
