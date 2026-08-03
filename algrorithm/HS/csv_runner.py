from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from baselines import GreedyAssignmentOptimizer, RandomMultiStartAssignmentOptimizer
from evaluator import AssignmentEvaluator
from harmony_search import HarmonySearchAssignmentOptimizer, coerce_hs_options
from schedule import SchedulePreprocessor


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        text = _clean(value)
        return default if text == "" else float(text)
    except ValueError:
        return default


def _bool(value: Any, default: bool = False) -> bool:
    text = _clean(value).lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def _split_csv_cell(value: Any) -> list[str]:
    text = _clean(value)
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def _parse_skills(value: Any) -> list[dict[str, Any]]:
    skills = []
    for item in _split_csv_cell(value):
        if ":" in item:
            name, level = item.split(":", 1)
            skills.append({"skillName": name.strip(), "level": level.strip()})
        else:
            skills.append({"skillName": item, "level": 3})
    return skills


def _parse_required_skills(value: Any) -> list[dict[str, Any]]:
    skills = []
    for item in _split_csv_cell(value):
        if ":" in item:
            name, level = item.split(":", 1)
            skills.append({"skillName": name.strip(), "level": level.strip()})
        else:
            skills.append({"skillName": item, "level": 3})
    return skills


def _parse_kpi_impacts(value: Any) -> list[dict[str, Any]]:
    impacts = []
    for item in _split_csv_cell(value):
        if ":" in item:
            code, weight = item.split(":", 1)
            impacts.append({"kpiCode": code.strip(), "weight": _float(weight, 0.0)})
        else:
            impacts.append({"kpiCode": item, "weight": 1.0})
    return impacts


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = []
    for idx, row in enumerate(read_csv(path), start=1):
        task_code = _clean(row.get("taskCode"))
        task_id = _clean(row.get("taskId")) or task_code or f"TASK_{idx}"
        tasks.append(
            {
                **row,
                "taskId": task_id,
                "taskCode": task_code or task_id,
                "estimatedHours": _float(row.get("estimatedHours"), 1.0),
                "dependencies": _split_csv_cell(row.get("dependencies")),
                "requiredSkills": _parse_required_skills(row.get("requiredSkills")),
                "kpiImpacts": _parse_kpi_impacts(row.get("kpiImpacts")),
            }
        )
    return tasks


def load_resources(path: Path) -> list[dict[str, Any]]:
    resources = []
    for row in read_csv(path):
        resource_id = _clean(row.get("resourceId"))
        if not resource_id:
            continue
        resources.append(
            {
                **row,
                "resourceId": resource_id,
                "capacity": _float(row.get("capacity"), 1.0),
                "currentLoad": _float(row.get("currentLoad"), 0.0),
                "availableHours": _float(row.get("availableHours"), 0.0),
                "costPerHour": _float(row.get("costPerHour"), 50.0),
                "skills": _parse_skills(row.get("skills")),
            }
        )
    return resources


def load_kpi_definitions(path: Path) -> list[dict[str, Any]]:
    definitions = []
    for row in read_csv(path):
        definitions.append(
            {
                **row,
                "minValue": _float(row.get("minValue"), 0.0),
                "maxValue": _float(row.get("maxValue"), 100.0),
                "higherIsBetter": _bool(row.get("higherIsBetter"), True),
            }
        )
    return definitions


def load_kpi_targets(path: Path) -> list[dict[str, Any]]:
    targets = []
    for row in read_csv(path):
        targets.append(
            {
                **row,
                "targetValue": _float(row.get("targetValue"), 100.0),
                "weight": _float(row.get("weight"), 1.0),
                "warningThreshold": _float(row.get("warningThreshold"), 0.0),
                "criticalThreshold": _float(row.get("criticalThreshold"), 0.0),
            }
        )
    return targets


def load_cycle(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    rows = read_csv(path)
    return rows[0] if rows else {}


def candidate_to_dict(candidate: Any) -> dict[str, Any]:
    return {
        "assignment": candidate.assignment,
        "score": {
            "totalScore": candidate.score.total_score,
            "kpiScore": candidate.score.kpi_score,
            "skillScore": candidate.score.skill_score,
            "workloadScore": candidate.score.workload_score,
            "costScore": candidate.score.cost_score,
            "scheduleScore": candidate.score.schedule_score,
            "feasible": candidate.score.feasible,
            "hardViolations": candidate.score.hard_violations,
            "diagnostics": candidate.score.diagnostics,
        },
    }


def write_assignment_csv(path: Path, result: Any, schedule: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "taskId",
                "resourceId",
                "plannedStartHour",
                "plannedEndHour",
                "durationHours",
                "totalScore",
                "feasible",
            ],
        )
        writer.writeheader()
        actual_schedule = (result.best.score.diagnostics or {}).get("actualSchedule", {})
        for task_id, resource_id in result.best.assignment.items():
            item = schedule.get(task_id)
            actual_item = actual_schedule.get(task_id, {})
            writer.writerow(
                {
                    "taskId": task_id,
                    "resourceId": resource_id,
                    "plannedStartHour": actual_item.get("plannedStartHour", item.planned_start_hour if item else ""),
                    "plannedEndHour": actual_item.get("plannedEndHour", item.planned_end_hour if item else ""),
                    "durationHours": actual_item.get("durationHours", item.duration_hours if item else ""),
                    "totalScore": result.best.score.total_score,
                    "feasible": result.best.score.feasible,
                }
            )


def run_from_csv(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_tasks(Path(args.tasks))
    resources = load_resources(Path(args.resources))
    definitions = load_kpi_definitions(Path(args.kpi_definitions)) if args.kpi_definitions else []
    targets = load_kpi_targets(Path(args.kpi_targets)) if args.kpi_targets else []
    cycle = load_cycle(Path(args.cycle)) if args.cycle else {}

    if not tasks:
        raise ValueError("tasks CSV is empty or missing task rows")
    if not resources:
        raise ValueError("resources CSV is empty or missing resource rows")

    _, schedule, predecessors, schedule_metadata = SchedulePreprocessor(tasks, cycle).build()
    evaluator = AssignmentEvaluator(
        tasks=tasks,
        resources=resources,
        targets=targets,
        definitions=definitions,
        schedule=schedule,
        predecessors=predecessors,
        strategy=args.strategy,
    )
    optimizer_cls = {
        "hs": HarmonySearchAssignmentOptimizer,
        "random": RandomMultiStartAssignmentOptimizer,
        "greedy": GreedyAssignmentOptimizer,
    }[args.algorithm]
    result = optimizer_cls(
        tasks=tasks,
        resources=resources,
        evaluator=evaluator,
        options=coerce_hs_options(
            {
                "harmonyMemorySize": args.harmony_memory_size,
                "hmcr": args.hmcr,
                "par": args.par,
                "maxIterations": args.max_iterations,
                "topCandidates": args.top_candidates,
                "seed": args.seed,
                "timeout": args.timeout,
            }
        ),
    ).run()

    output = {
        "algorithm": args.algorithm,
        "best": candidate_to_dict(result.best),
        "topCandidates": [candidate_to_dict(item) for item in result.memory[: args.top_candidates]],
        "history": result.history,
        "iterationsRun": result.iterations_run,
        "converged": result.converged,
        "startedAt": result.started_at.isoformat(),
        "finishedAt": result.finished_at.isoformat(),
        "scheduleMetadata": {**schedule_metadata, "predecessors": predecessors},
    }

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        write_assignment_csv(Path(args.output_csv), result, schedule)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run assignment optimizer from CSV files.")
    parser.add_argument("--algorithm", default="hs", choices=["hs", "random", "greedy"], help="Optimizer to run.")
    parser.add_argument("--tasks", required=True, help="Path to task CSV.")
    parser.add_argument("--resources", required=True, help="Path to resource CSV.")
    parser.add_argument("--kpi-definitions", help="Path to KPI definition CSV.")
    parser.add_argument("--kpi-targets", help="Path to KPI target CSV.")
    parser.add_argument("--cycle", help="Path to cycle CSV.")
    parser.add_argument("--strategy", default="BALANCED", choices=["BALANCED", "KPI_FOCUSED", "COST_FOCUSED", "SPEED_FOCUSED"])
    parser.add_argument("--harmony-memory-size", type=int, default=20)
    parser.add_argument("--hmcr", type=float, default=0.9)
    parser.add_argument("--par", type=float, default=0.25)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--output-json", help="Optional output JSON path.")
    parser.add_argument("--output-csv", help="Optional best assignment CSV path.")
    return parser


def main() -> None:
    output = run_from_csv(build_parser().parse_args())
    print(json.dumps(output["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
