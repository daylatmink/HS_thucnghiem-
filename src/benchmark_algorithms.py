from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from csv_runner import run_from_csv


DEFAULT_ALGORITHMS = "hs,ga,hybrid_hs_ga,hybrid_hs_ga_guided_mutation"
STOCHASTIC_ALGORITHMS = {
    "hs",
    "ga",
    "hybrid_hs_ga",
    "hybrid_hs_ga_guided_mutation",
    "random",
}


def discover_datasets(examples_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in examples_dir.iterdir()
        if path.is_dir()
        and (path / "tasks.csv").exists()
        and (path / "resources.csv").exists()
    )


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seed_list(value: str) -> list[int]:
    text = value.strip()
    if ".." in text:
        start, end = text.split("..", 1)
        return list(range(int(start), int(end) + 1))
    return [int(seed) for seed in parse_csv_list(value)]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in parse_csv_list(value)]


def algorithm_label(algorithm: str, max_evaluations: int | None, hybrid_hs_ratio: float) -> str:
    if algorithm not in {"hybrid_hs_ga", "hybrid_hs_ga_guided_mutation"}:
        return algorithm
    if max_evaluations:
        hs_budget = max(1, int(max_evaluations * hybrid_hs_ratio))
        ga_budget = max_evaluations - hs_budget
        suffix = f"hs{hs_budget}_ga{ga_budget}"
    else:
        suffix = f"hs{hybrid_hs_ratio:.2f}"
    if algorithm == "hybrid_hs_ga_guided_mutation":
        return f"hybrid_hs_ga_guided_mutation_{suffix}"
    return f"hybrid_hs_ga_{suffix}"


def build_run_args(
    dataset_dir: Path,
    algorithm: str,
    seed: int,
    args: argparse.Namespace,
    hybrid_hs_ratio: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        algorithm=algorithm,
        tasks=str(dataset_dir / "tasks.csv"),
        resources=str(dataset_dir / "resources.csv"),
        kpi_definitions=str(dataset_dir / "kpi-definitions.csv"),
        kpi_targets=str(dataset_dir / "kpi-targets.csv"),
        cycle=str(dataset_dir / "cycle.csv"),
        strategy=args.strategy,
        harmony_memory_size=args.harmony_memory_size,
        hmcr=args.hmcr,
        par=args.par,
        max_iterations=args.max_iterations,
        max_evaluations=args.max_evaluations,
        fixed_budget=args.fixed_budget,
        top_candidates=args.top_candidates,
        seed=seed,
        timeout=args.timeout,
        population_size=args.population_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism_count=args.elitism_count,
        tournament_size=args.tournament_size,
        hybrid_hs_ratio=hybrid_hs_ratio,
        output_json=None,
        output_csv=None,
    )


def run_to_raw_row(dataset: str, algorithm: str, seed: int, output: dict[str, Any]) -> dict[str, Any]:
    score = output["best"]["score"]
    diagnostics = score.get("diagnostics") or {}
    runtime = output.get("runtime") or {}
    return {
        "dataset": dataset,
        "algorithm": algorithm,
        "seed": seed,
        "totalScore": score["totalScore"],
        "kpiScore": score["kpiScore"],
        "costScore": score["costScore"],
        "timeScore": score["scheduleScore"],
        "estimatedKpis": json.dumps(diagnostics.get("estimatedKpis") or {}, ensure_ascii=False, sort_keys=True),
        "totalCost": diagnostics.get("totalCost"),
        "baseMakespan": diagnostics.get("baseMakespan"),
        "actualMakespan": diagnostics.get("makespan"),
        "feasible": score["feasible"],
        "hardViolations": len(score.get("hardViolations") or []),
        "objectiveEvaluations": output.get("objectiveEvaluations"),
        "evaluationAtBest": output.get("evaluationAtBest"),
        "iterations": output.get("iterationsRun"),
        "runtimeSeconds": runtime.get("durationSeconds"),
        "converged": output.get("converged"),
        "objectiveGuidanceCount": diagnostics.get("objectiveGuidanceCount"),
        "skillGuidanceCount": diagnostics.get("skillGuidanceCount"),
        "observedObjectiveGuidanceRate": diagnostics.get("observedObjectiveGuidanceRate"),
        "guidanceScheduleBuildCount": diagnostics.get("guidanceScheduleBuildCount"),
    }


def convergence_rows(dataset: str, algorithm: str, seed: int, output: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in output.get("history") or []:
        rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "seed": seed,
                "objectiveEvaluation": item.get("objectiveEvaluation"),
                "phaseEvaluation": item.get("phaseEvaluation"),
                "bestFoundAtEvaluation": item.get("bestFoundAtEvaluation"),
                "bestScore": item.get("bestScore"),
                "currentScore": item.get("currentScore"),
                "iteration": item.get("iteration"),
                "phase": item.get("phase"),
                "island": item.get("island"),
                "islandEvaluation": item.get("islandEvaluation"),
                "globalBestScore": item.get("globalBestScore"),
                "islandBestScore": item.get("islandBestScore"),
                "populationDiversity": item.get("populationDiversity"),
                "uniquePopulationCount": item.get("uniquePopulationCount"),
                "populationSize": item.get("populationSize"),
                "migrationCount": item.get("migrationCount"),
                "migrationEvaluation": item.get("migrationEvaluation"),
                "sourceIsland": item.get("sourceIsland"),
                "targetIsland": item.get("targetIsland"),
                "migrantType": item.get("migrantType"),
                "migrantScore": item.get("migrantScore"),
                "cached": item.get("cached"),
                "inserted": item.get("inserted"),
                "transitionEvaluation": item.get("transitionEvaluation"),
                "switchEvaluation": item.get("switchEvaluation"),
                "switchReason": item.get("switchReason"),
                "hsBestAtSwitch": item.get("hsBestAtSwitch"),
                "hsBestScore": item.get("hsBestScore"),
                "hsMemoryUniqueCount": item.get("hsMemoryUniqueCount"),
                "gaInitialPopulationUniqueCount": item.get("gaInitialPopulationUniqueCount"),
                "gaInitialPopulationDiversity": item.get("gaInitialPopulationDiversity"),
                "numTransferredCached": item.get("numTransferredCached"),
                "numMutatedNew": item.get("numMutatedNew"),
                "numRandomNew": item.get("numRandomNew"),
                "mutationRate": item.get("mutationRate"),
                "baselineMutationRate": item.get("baselineMutationRate"),
                "mutationMin": item.get("mutationMin"),
                "mutationMax": item.get("mutationMax"),
                "offspringSuccess": item.get("offspringSuccess"),
                "referenceParentScore": item.get("referenceParentScore"),
                "offspringScore": item.get("offspringScore"),
                "successWindowCount": item.get("successWindowCount"),
                "successWindowSuccessful": item.get("successWindowSuccessful"),
                "offspringSuccessRate": item.get("offspringSuccessRate"),
                "adaptiveUpdate": item.get("adaptiveUpdate"),
                "adaptiveState": item.get("adaptiveState"),
                "previousMutationRate": item.get("previousMutationRate"),
                "newMutationRate": item.get("newMutationRate"),
                "crossoverType": item.get("crossoverType"),
                "numberOfLinkageGroups": item.get("numberOfLinkageGroups"),
                "meanLinkageGroupSize": item.get("meanLinkageGroupSize"),
                "medianLinkageGroupSize": item.get("medianLinkageGroupSize"),
                "maxLinkageGroupSize": item.get("maxLinkageGroupSize"),
                "minLinkageGroupSize": item.get("minLinkageGroupSize"),
                "numTopoLevels": item.get("numTopoLevels"),
                "singletonGroups": item.get("singletonGroups"),
                "crossoverApplied": item.get("crossoverApplied"),
                "groupsFromParentA": item.get("groupsFromParentA"),
                "groupsFromParentB": item.get("groupsFromParentB"),
                "childParentAHamming": item.get("childParentAHamming"),
                "childParentBHamming": item.get("childParentBHamming"),
                "mutationType": item.get("mutationType"),
                "expectedMutationCount": item.get("expectedMutationCount"),
                "actualMutatedGeneCount": item.get("actualMutatedGeneCount"),
                "meanWeakness": item.get("meanWeakness"),
                "maxWeakness": item.get("maxWeakness"),
                "minWeakness": item.get("minWeakness"),
                "meanSelectedWeakness": item.get("meanSelectedWeakness"),
                "meanUnselectedWeakness": item.get("meanUnselectedWeakness"),
                "selectedSkillWeaknessMean": item.get("selectedSkillWeaknessMean"),
                "selectedWorkloadPressureMean": item.get("selectedWorkloadPressureMean"),
                "selectedTimelinePressureMean": item.get("selectedTimelinePressureMean"),
                "objectiveKpiWeight": item.get("objectiveKpiWeight"),
                "objectiveCostWeight": item.get("objectiveCostWeight"),
                "objectiveTimeWeight": item.get("objectiveTimeWeight"),
                "KPIWeaknessMean": item.get("KPIWeaknessMean"),
                "CostWeaknessMean": item.get("CostWeaknessMean"),
                "TimeWeaknessMean": item.get("TimeWeaknessMean"),
                "overallWeaknessMean": item.get("overallWeaknessMean"),
                "selectedKPIWeaknessMean": item.get("selectedKPIWeaknessMean"),
                "selectedCostWeaknessMean": item.get("selectedCostWeaknessMean"),
                "selectedTimeWeaknessMean": item.get("selectedTimeWeaknessMean"),
                "selectedOverallWeaknessMean": item.get("selectedOverallWeaknessMean"),
                "unselectedOverallWeaknessMean": item.get("unselectedOverallWeaknessMean"),
                "overallWeaknessLift": item.get("overallWeaknessLift"),
                "KPIWeaknessLift": item.get("KPIWeaknessLift"),
                "CostWeaknessLift": item.get("CostWeaknessLift"),
                "TimeWeaknessLift": item.get("TimeWeaknessLift"),
                "meanResourceWait": item.get("meanResourceWait"),
                "maxResourceWait": item.get("maxResourceWait"),
                "meanNormalizedResourceWait": item.get("meanNormalizedResourceWait"),
                "timeGuidanceType": item.get("timeGuidanceType"),
                "meanCriticality": item.get("meanCriticality"),
                "numCriticalTasks": item.get("numCriticalTasks"),
                "meanCriticalBlocking": item.get("meanCriticalBlocking"),
                "numTasksWithResourceWait": item.get("numTasksWithResourceWait"),
                "numTasksWithPositiveBlocking": item.get("numTasksWithPositiveBlocking"),
                "meanCostWeakness": item.get("meanCostWeakness"),
                "selectedMeanCostWeakness": item.get("selectedMeanCostWeakness"),
                "costWeaknessLift": item.get("costWeaknessLift"),
                "numTasksAtMinCost": item.get("numTasksAtMinCost"),
                "numTasksAboveMedianCost": item.get("numTasksAboveMedianCost"),
                "numTasksAtMaxCost": item.get("numTasksAtMaxCost"),
                "meanKPIWeakness": item.get("meanKPIWeakness"),
                "selectedMeanKPIWeakness": item.get("selectedMeanKPIWeakness"),
                "kpiWeaknessLift": item.get("kpiWeaknessLift"),
                "meanTaskKPIContribution": item.get("meanTaskKPIContribution"),
                "selectedMeanTaskKPIContribution": item.get("selectedMeanTaskKPIContribution"),
                "selectiveGuidanceProbability": item.get("selectiveGuidanceProbability"),
                "objectiveGuidanceCount": item.get("objectiveGuidanceCount"),
                "skillGuidanceCount": item.get("skillGuidanceCount"),
                "guidanceScheduleBuildCount": item.get("guidanceScheduleBuildCount"),
                "observedObjectiveGuidanceRate": item.get("observedObjectiveGuidanceRate"),
                "guidanceType": item.get("guidanceType"),
                "skillWeaknessMean": item.get("skillWeaknessMean"),
                "selectedSkillWeaknessMean": item.get("selectedSkillWeaknessMean"),
            }
        )
    score = output["best"]["score"]
    final_row = {
        "dataset": dataset,
        "algorithm": algorithm,
        "seed": seed,
        "objectiveEvaluation": output.get("objectiveEvaluations"),
        "phaseEvaluation": None,
        "bestFoundAtEvaluation": output.get("evaluationAtBest"),
        "bestScore": score.get("totalScore"),
        "currentScore": score.get("totalScore"),
        "iteration": output.get("iterationsRun"),
        "phase": rows[-1].get("phase") if rows else None,
        "island": None,
        "islandEvaluation": None,
        "globalBestScore": score.get("totalScore"),
        "islandBestScore": None,
        "populationDiversity": None,
        "uniquePopulationCount": None,
        "populationSize": None,
        "migrationCount": None,
        "migrationEvaluation": None,
        "sourceIsland": None,
        "targetIsland": None,
        "migrantType": None,
        "migrantScore": None,
        "cached": None,
        "inserted": None,
        "transitionEvaluation": None,
        "switchEvaluation": None,
        "switchReason": None,
        "hsBestAtSwitch": None,
        "hsBestScore": None,
        "hsMemoryUniqueCount": None,
        "gaInitialPopulationUniqueCount": None,
        "gaInitialPopulationDiversity": None,
        "numTransferredCached": None,
        "numMutatedNew": None,
        "numRandomNew": None,
        "mutationRate": None,
        "baselineMutationRate": None,
        "mutationMin": None,
        "mutationMax": None,
        "offspringSuccess": None,
        "referenceParentScore": None,
        "offspringScore": None,
        "successWindowCount": None,
        "successWindowSuccessful": None,
        "offspringSuccessRate": None,
        "adaptiveUpdate": None,
        "adaptiveState": None,
        "previousMutationRate": None,
        "newMutationRate": None,
        "crossoverType": None,
        "numberOfLinkageGroups": None,
        "meanLinkageGroupSize": None,
        "medianLinkageGroupSize": None,
        "maxLinkageGroupSize": None,
        "minLinkageGroupSize": None,
        "numTopoLevels": None,
        "singletonGroups": None,
        "crossoverApplied": None,
        "groupsFromParentA": None,
        "groupsFromParentB": None,
        "childParentAHamming": None,
        "childParentBHamming": None,
        "mutationType": None,
        "expectedMutationCount": None,
        "actualMutatedGeneCount": None,
        "meanWeakness": None,
        "maxWeakness": None,
        "minWeakness": None,
        "meanSelectedWeakness": None,
        "meanUnselectedWeakness": None,
        "selectedSkillWeaknessMean": None,
        "selectedWorkloadPressureMean": None,
        "selectedTimelinePressureMean": None,
        "objectiveKpiWeight": None,
        "objectiveCostWeight": None,
        "objectiveTimeWeight": None,
        "KPIWeaknessMean": None,
        "CostWeaknessMean": None,
        "TimeWeaknessMean": None,
        "overallWeaknessMean": None,
        "selectedKPIWeaknessMean": None,
        "selectedCostWeaknessMean": None,
        "selectedTimeWeaknessMean": None,
        "selectedOverallWeaknessMean": None,
        "unselectedOverallWeaknessMean": None,
        "overallWeaknessLift": None,
        "KPIWeaknessLift": None,
        "CostWeaknessLift": None,
        "TimeWeaknessLift": None,
        "meanResourceWait": None,
        "maxResourceWait": None,
        "meanNormalizedResourceWait": None,
        "timeGuidanceType": None,
        "meanCriticality": None,
        "numCriticalTasks": None,
        "meanCriticalBlocking": None,
        "numTasksWithResourceWait": None,
        "numTasksWithPositiveBlocking": None,
        "meanCostWeakness": None,
        "selectedMeanCostWeakness": None,
        "costWeaknessLift": None,
        "numTasksAtMinCost": None,
        "numTasksAboveMedianCost": None,
        "numTasksAtMaxCost": None,
        "meanKPIWeakness": None,
        "selectedMeanKPIWeakness": None,
        "kpiWeaknessLift": None,
        "meanTaskKPIContribution": None,
        "selectedMeanTaskKPIContribution": None,
        "selectiveGuidanceProbability": None,
        "objectiveGuidanceCount": None,
        "skillGuidanceCount": None,
        "guidanceScheduleBuildCount": None,
        "observedObjectiveGuidanceRate": None,
        "guidanceType": None,
        "skillWeaknessMean": None,
        "selectedSkillWeaknessMean": None,
    }
    if not rows or any(str(rows[-1].get(key)) != str(final_row.get(key)) for key in ("objectiveEvaluation", "bestFoundAtEvaluation", "bestScore")):
        rows.append(final_row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "totalScore",
        "kpiScore",
        "costScore",
        "timeScore",
        "totalCost",
        "actualMakespan",
        "runtimeSeconds",
        "evaluationAtBest",
    ]
    higher_is_better = {"totalScore", "kpiScore", "costScore", "timeScore"}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        groups.setdefault((row["dataset"], row["algorithm"]), []).append(row)

    summary = []
    for (dataset, algorithm), rows in sorted(groups.items()):
        output: dict[str, Any] = {
            "dataset": dataset,
            "algorithm": algorithm,
            "runs": len(rows),
            "feasibleRate": round(
                sum(1 for row in rows if str(row.get("feasible")).lower() == "true") / len(rows),
                6,
            ),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric) not in ("", None)]
            output[f"{metric}_mean"] = round(statistics.mean(values), 6) if values else None
            output[f"{metric}_std"] = round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0
            if metric in higher_is_better:
                best = max(values) if values else None
                worst = min(values) if values else None
            else:
                best = min(values) if values else None
                worst = max(values) if values else None
            output[f"{metric}_best"] = round(best, 6) if best is not None else None
            output[f"{metric}_worst"] = round(worst, 6) if worst is not None else None
        summary.append(output)
    return summary


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    number = _float_or_none(value)
    return int(number) if number is not None else None


def is_stochastic_algorithm(algorithm: str) -> bool:
    return (
        algorithm in STOCHASTIC_ALGORITHMS
        or algorithm.startswith("hybrid_hs_ga_")
        or algorithm.startswith("hybrid_hs_ga_guided_mutation_")
    )


def is_hybrid_algorithm(algorithm: str) -> bool:
    return (
        algorithm in {"hybrid_hs_ga", "hybrid_hs_ga_guided_mutation"}
        or algorithm.startswith("hybrid_hs_ga_")
        or algorithm.startswith("hybrid_hs_ga_guided_mutation_")
    )


def validate_benchmark_rows(
    raw_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    max_evaluations: int | None,
    fixed_budget: bool,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    histories: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in history_rows:
        key = (str(row.get("dataset")), str(row.get("algorithm")), str(row.get("seed")))
        histories.setdefault(key, []).append(row)

    for row in raw_rows:
        key = (str(row.get("dataset")), str(row.get("algorithm")), str(row.get("seed")))
        algorithm = str(row.get("algorithm"))
        objective_evaluations = _int_or_none(row.get("objectiveEvaluations"))
        evaluation_at_best = _int_or_none(row.get("evaluationAtBest"))
        total_score = _float_or_none(row.get("totalScore"))
        if objective_evaluations is None or evaluation_at_best is None:
            issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "missing_raw_evaluation"})
            continue
        if evaluation_at_best > objective_evaluations:
            issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "evaluationAtBest_gt_objectiveEvaluations"})
        if max_evaluations is not None and objective_evaluations > max_evaluations:
            issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "objectiveEvaluations_gt_maxEvaluations"})
        if fixed_budget and max_evaluations is not None and is_stochastic_algorithm(algorithm) and objective_evaluations != max_evaluations:
            issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "fixed_budget_not_exhausted"})

        run_history = histories.get(key, [])
        if not run_history:
            issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "missing_history"})
            continue
        previous_evaluation = -1
        previous_best = None
        for item in run_history:
            objective_evaluation = _int_or_none(item.get("objectiveEvaluation"))
            best_score = _float_or_none(item.get("bestScore"))
            if objective_evaluation is None:
                issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "missing_history_evaluation"})
                break
            if objective_evaluation < previous_evaluation:
                issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "history_evaluation_decreased"})
                break
            if max_evaluations is not None and objective_evaluation > max_evaluations:
                issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "history_evaluation_gt_maxEvaluations"})
                break
            if best_score is not None and previous_best is not None and best_score + 1e-9 < previous_best:
                issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "history_bestScore_decreased"})
                break
            previous_evaluation = objective_evaluation
            if best_score is not None:
                previous_best = best_score
        final = run_history[-1]
        final_evaluation = _int_or_none(final.get("objectiveEvaluation"))
        final_best = _float_or_none(final.get("bestScore"))
        if final_evaluation != objective_evaluations:
            issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "final_history_evaluation_mismatch"})
        if total_score is not None and final_best is not None and abs(final_best - total_score) > 1e-6:
            issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "final_history_bestScore_mismatch"})
        if is_hybrid_algorithm(algorithm):
            phases = {str(item.get("phase")) for item in run_history}
            has_ga_phase = any(phase == "ga" or phase.startswith("ga_") for phase in phases)
            if not has_ga_phase or "hs" not in phases:
                issues.append({"dataset": key[0], "algorithm": key[1], "seed": key[2], "issue": "hybrid_missing_ga_or_hs_phase"})
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark assignment algorithms with a common NFE budget.")
    parser.add_argument("--examples-dir", default=str(Path(__file__).resolve().parents[1] / "datasets"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "benchmark_runs"))
    parser.add_argument("--datasets", help="Comma-separated dataset folder names. Defaults to all.")
    parser.add_argument("--algorithms", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds or inclusive range like 1..30.")
    parser.add_argument("--strategy", default="BALANCED")
    parser.add_argument("--max-evaluations", type=int, default=1000)
    parser.add_argument(
        "--fixed-budget",
        action="store_true",
        help="Do not stop stochastic optimizers early on convergence before max evaluations.",
    )
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--harmony-memory-size", type=int, default=30)
    parser.add_argument("--hmcr", type=float, default=0.9)
    parser.add_argument("--par", type=float, default=0.25)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--elitism-count", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument(
        "--hybrid-hs-ratio",
        type=float,
        default=0.15,
        help="NFE share for HS phase in HS->GA hybrids. Example: 0.15 means HS 15%%, GA 85%%.",
    )
    parser.add_argument(
        "--hybrid-hs-ratios",
        default="0.15",
        help="Comma-separated NFE shares for HS->GA hybrids. Example: 0.1,0.15,0.2.",
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip benchmark CSV invariant validation.")
    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    output_dir = Path(args.output_dir)
    algorithms = parse_csv_list(args.algorithms)
    hybrid_hs_ratios = parse_float_list(args.hybrid_hs_ratios)
    seeds = parse_seed_list(args.seeds)
    datasets = discover_datasets(examples_dir)
    if args.datasets:
        selected = set(parse_csv_list(args.datasets))
        datasets = [dataset for dataset in datasets if dataset.name in selected]

    raw_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for dataset_dir in datasets:
        for algorithm in algorithms:
            ratios = hybrid_hs_ratios if algorithm in {"hybrid_hs_ga", "hybrid_hs_ga_guided_mutation"} else [args.hybrid_hs_ratio]
            for hybrid_hs_ratio in ratios:
                label = algorithm_label(algorithm, args.max_evaluations, hybrid_hs_ratio)
                for seed in seeds:
                    print(f"run dataset={dataset_dir.name} algorithm={label} seed={seed}", flush=True)
                    output = run_from_csv(build_run_args(dataset_dir, algorithm, seed, args, hybrid_hs_ratio))
                    raw_rows.append(run_to_raw_row(dataset_dir.name, label, seed, output))
                    history_rows.extend(convergence_rows(dataset_dir.name, label, seed, output))
                    write_csv(output_dir / "raw_results.csv", raw_rows)
                    write_csv(output_dir / "convergence_history.csv", history_rows)
                    write_csv(output_dir / "summary_results.csv", summarize(raw_rows))

    if not args.skip_validation:
        issues = validate_benchmark_rows(raw_rows, history_rows, args.max_evaluations, args.fixed_budget)
        report_fields = ["dataset", "algorithm", "seed", "issue"]
        write_csv(output_dir / "validation_report.csv", issues, report_fields)
        validation_summary = {
            "rawRows": len(raw_rows),
            "historyRows": len(history_rows),
            "issues": len(issues),
            "fixedBudget": args.fixed_budget,
            "maxEvaluations": args.max_evaluations,
        }
        (output_dir / "validation_summary.json").write_text(
            json.dumps(validation_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if issues:
            print(f"validation failed: {len(issues)} issue(s). See {output_dir / 'validation_report.csv'}", flush=True)
            raise SystemExit(1)
        print("validation passed", flush=True)

    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
