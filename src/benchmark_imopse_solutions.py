from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from csv_runner import (
    load_cycle,
    load_kpi_definitions,
    load_kpi_targets,
    load_resources,
    load_tasks,
)
from evaluator import AssignmentEvaluator, safe_float
from schedule import SchedulePreprocessor


@dataclass(frozen=True)
class NativeSolution:
    solution_id: int
    evaluation: list[float]
    float_genotype: list[float]


def parse_vector(value: str | None) -> list[float]:
    return [float(item) for item in (value or "").split() if item]


def read_native_solutions(path: Path) -> list[NativeSolution]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file, delimiter=";")
        return [
            NativeSolution(
                solution_id=int(row["solutionId"]),
                evaluation=parse_vector(row.get("evaluation")),
                float_genotype=parse_vector(row.get("floatGenotype")),
            )
            for row in rows
        ]


def _skill_level_map(resource: dict[str, Any]) -> dict[str, float]:
    return {
        str(skill.get("skillName", "")).strip(): safe_float(skill.get("level"), 0.0)
        for skill in resource.get("skills") or []
        if str(skill.get("skillName", "")).strip()
    }


def build_candidate_pools(
    tasks: list[dict[str, Any]], resources: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Rebuild MSRCPSP_TA2's per-task capable-resource list in CSV order."""
    pools: dict[str, list[str]] = {}
    skill_maps = [(str(resource["resourceId"]), _skill_level_map(resource)) for resource in resources]
    for task in tasks:
        required = task.get("requiredSkills") or []
        pool = []
        for resource_id, skill_levels in skill_maps:
            if all(
                skill_levels.get(str(item.get("skillName", "")).strip(), 0.0)
                >= safe_float(item.get("level"), 0.0)
                for item in required
            ):
                pool.append(resource_id)
        pools[str(task["taskId"])] = pool
    return pools


def decode_assignment(
    genotype: list[float],
    tasks: list[dict[str, Any]],
    candidate_pools: dict[str, list[str]],
) -> dict[str, str]:
    if len(genotype) != len(tasks):
        raise ValueError(f"genotype has {len(genotype)} genes; expected {len(tasks)}")
    assignment: dict[str, str] = {}
    for task, gene in zip(tasks, genotype):
        task_id = str(task["taskId"])
        pool = candidate_pools[task_id]
        index = int(gene)
        if not pool:
            raise ValueError(f"task {task_id} has no skill-feasible resource")
        if index < 0 or index >= len(pool):
            raise ValueError(f"task {task_id} gene index {index} is outside [0, {len(pool) - 1}]")
        assignment[task_id] = pool[index]
    return assignment


def choose_blind(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose without KPI access using equal normalized native objectives."""
    makespans = [float(row["originalMakespan"]) for row in rows]
    costs = [float(row["originalCost"]) for row in rows]
    makespan_range = max(makespans) - min(makespans)
    cost_range = max(costs) - min(costs)
    for row in rows:
        normalized_makespan = (
            0.0 if makespan_range == 0 else (float(row["originalMakespan"]) - min(makespans)) / makespan_range
        )
        normalized_cost = 0.0 if cost_range == 0 else (float(row["originalCost"]) - min(costs)) / cost_range
        row["blindSelectionScore"] = round(0.5 * normalized_makespan + 0.5 * normalized_cost, 12)
    return min(
        rows,
        key=lambda row: (
            row["blindSelectionScore"],
            float(row["originalMakespan"]),
            float(row["originalCost"]),
            int(row["solutionId"]),
        ),
    )


def choose_reranked(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            str(row["feasible"]).lower() == "true",
            float(row["totalScore"]),
            -int(row["hardViolations"]),
            -int(row["solutionId"]),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _score_row(score: Any) -> dict[str, Any]:
    diagnostics = score.diagnostics or {}
    return {
        "totalScore": score.total_score,
        "kpiScore": score.kpi_score,
        "costScore": score.cost_score,
        "timeScore": score.schedule_score,
        "estimatedKpis": json.dumps(diagnostics.get("estimatedKpis") or {}, ensure_ascii=False, sort_keys=True),
        "totalCost": diagnostics.get("totalCost"),
        "baseMakespan": diagnostics.get("baseMakespan"),
        "actualMakespan": diagnostics.get("makespan"),
        "feasible": score.feasible,
        "hardViolations": len(score.hard_violations),
        "hardViolationDetails": json.dumps(score.hard_violations, ensure_ascii=False, sort_keys=True),
    }


def _selected_row(row: dict[str, Any], policy: str, pareto_size: int) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "sourceDataset": row["sourceDataset"],
        "datasetSource": row["datasetSource"],
        "instanceType": row["instanceType"],
        "algorithm": f"imopse_{row['nativeAlgorithm']}_{policy}",
        "nativeAlgorithm": row["nativeAlgorithm"],
        "selectionPolicy": policy,
        "seed": row["seed"],
        **{key: row[key] for key in (
            "totalScore", "kpiScore", "costScore", "timeScore", "estimatedKpis",
            "totalCost", "baseMakespan", "actualMakespan", "feasible",
            "hardViolations", "hardViolationDetails",
        )},
        "nativeObjectiveEvaluations": row["nativeObjectiveEvaluations"],
        "runtimeSeconds": row["runtimeSeconds"],
        "paretoSize": pareto_size,
        "selectedSolutionId": row["solutionId"],
        "originalMakespan": row["originalMakespan"],
        "originalCost": row["originalCost"],
        "blindSelectionScore": row["blindSelectionScore"],
    }


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "totalScore", "kpiScore", "costScore", "timeScore", "totalCost",
        "actualMakespan", "originalMakespan", "originalCost", "paretoSize", "runtimeSeconds",
    )
    maximize = {"totalScore", "kpiScore", "costScore", "timeScore"}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        groups.setdefault((str(row["dataset"]), str(row["algorithm"])), []).append(row)
    output_rows = []
    for (dataset, algorithm), rows in sorted(groups.items()):
        output: dict[str, Any] = {
            "dataset": dataset,
            "algorithm": algorithm,
            "nativeAlgorithm": rows[0]["nativeAlgorithm"],
            "selectionPolicy": rows[0]["selectionPolicy"],
            "runs": len(rows),
            "feasibleRate": round(sum(bool(row["feasible"]) for row in rows) / len(rows), 6),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric) not in (None, "")]
            output[f"{metric}_mean"] = round(statistics.mean(values), 6) if values else None
            output[f"{metric}_std"] = round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0
            best_fn, worst_fn = (max, min) if metric in maximize else (min, max)
            output[f"{metric}_best"] = round(best_fn(values), 6) if values else None
            output[f"{metric}_worst"] = round(worst_fn(values), 6) if values else None
        output_rows.append(output)
    return output_rows


def _dataset_context(dataset_dir: Path, strategy: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], AssignmentEvaluator]:
    tasks = load_tasks(dataset_dir / "tasks.csv")
    resources = load_resources(dataset_dir / "resources.csv")
    definitions = load_kpi_definitions(dataset_dir / "kpi-definitions.csv")
    targets = load_kpi_targets(dataset_dir / "kpi-targets.csv")
    cycle = load_cycle(dataset_dir / "cycle.csv")
    _, schedule, predecessors, _ = SchedulePreprocessor(tasks, cycle).build()
    evaluator = AssignmentEvaluator(tasks, resources, targets, definitions, schedule, predecessors, strategy=strategy)
    return tasks, resources, evaluator


def benchmark(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    solutions_root = Path(args.solutions_root)
    datasets_dir = Path(args.datasets_dir)
    rescored: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()

    for path in sorted(solutions_root.glob("*/*/seed_*/run_0/pareto_solutions.csv")):
        relative = path.relative_to(solutions_root)
        native_algorithm, source_dataset, seed_part = relative.parts[:3]
        seed = int(seed_part.removeprefix("seed_"))
        run_key = (native_algorithm, source_dataset, seed)
        if run_key in seen:
            issues.append({"algorithm": native_algorithm, "sourceDataset": source_dataset, "seed": seed, "issue": "duplicate run"})
            continue
        seen.add(run_key)
        dataset_dir = datasets_dir / f"msrcpsp_{source_dataset}"
        if not dataset_dir.exists():
            issues.append({"algorithm": native_algorithm, "sourceDataset": source_dataset, "seed": seed, "issue": f"missing dataset {dataset_dir}"})
            continue
        try:
            tasks, resources, evaluator = _dataset_context(dataset_dir, args.strategy)
            pools = build_candidate_pools(tasks, resources)
            native_solutions = read_native_solutions(path)
            metadata_path = path.parent.parent / "native_run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
            if not native_solutions:
                raise ValueError("empty Pareto front")
            run_rows = []
            for native in native_solutions:
                if len(native.evaluation) < 2:
                    raise ValueError(f"solution {native.solution_id} has fewer than two native objectives")
                assignment = decode_assignment(native.float_genotype, tasks, pools)
                score = evaluator.evaluate(assignment)
                row = {
                    "dataset": dataset_dir.name,
                    "sourceDataset": source_dataset,
                    "nativeAlgorithm": native_algorithm,
                    "datasetSource": metadata.get("datasetSource", ""),
                    "instanceType": metadata.get("instanceType", "official_def"),
                    "seed": seed,
                    "solutionId": native.solution_id,
                    "originalMakespan": native.evaluation[0],
                    "originalCost": native.evaluation[1],
                    **_score_row(score),
                    "rescoreEvaluation": evaluator.objective_evaluations,
                    "blindSelectionScore": None,
                    "selectedBlind": False,
                    "selectedParetoRerank": False,
                    "nativeObjectiveEvaluations": int(metadata.get("objectiveEvaluations", args.native_nfe)),
                    "runtimeSeconds": metadata.get("runtimeSeconds"),
                }
                run_rows.append(row)
            blind = choose_blind(run_rows)
            reranked = choose_reranked(run_rows)
            blind["selectedBlind"] = True
            reranked["selectedParetoRerank"] = True
            selected.append(_selected_row(blind, "blind", len(run_rows)))
            selected.append(_selected_row(reranked, "pareto_rerank", len(run_rows)))
            rescored.extend(run_rows)
        except (OSError, ValueError, KeyError) as exc:
            issues.append({"algorithm": native_algorithm, "sourceDataset": source_dataset, "seed": seed, "issue": str(exc)})

    if not seen:
        issues.append({"algorithm": "", "sourceDataset": "", "seed": "", "issue": "no pareto_solutions.csv files found"})
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "pareto_rescored.csv", rescored)
    write_csv(output_dir / "raw_results.csv", selected)
    write_csv(output_dir / "summary_results.csv", summarize(selected))
    write_csv(
        output_dir / "validation_report.csv",
        issues,
        ["algorithm", "sourceDataset", "seed", "issue"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_summary.json").write_text(
        json.dumps({"runsFound": len(seen), "runsScored": len(selected) // 2, "issues": len(issues)}, indent=2),
        encoding="utf-8",
    )
    return selected, issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore exported iMOPSE Pareto assignments with the KPI-Cost-Time objective.")
    parser.add_argument("--solutions-root", default="results/imopse_native")
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--output-dir", default="results/imopse_kpi_benchmark")
    parser.add_argument("--strategy", default="BALANCED")
    parser.add_argument("--native-nfe", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    result_rows, validation_issues = benchmark(parse_args())
    print(f"scored {len(result_rows) // 2} native runs; validation issues: {len(validation_issues)}")
