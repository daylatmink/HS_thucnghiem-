from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any


SKILL_RE = re.compile(r"(Q\d+)\s*:\s*(\d+)")


def parse_def(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    resources = []
    tasks = []
    section = None
    characteristics: dict[str, int] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line or set(line) <= {"="}:
            continue
        if line.startswith("Tasks:"):
            characteristics["tasks"] = int(line.split(":", 1)[1].strip())
            continue
        if line.startswith("Resources:"):
            characteristics["resources"] = int(line.split(":", 1)[1].strip())
            continue
        if line.startswith("Precedence relations:"):
            characteristics["precedenceRelations"] = int(line.split(":", 1)[1].strip())
            continue
        if line.startswith("Number of skill types:"):
            characteristics["skillTypes"] = int(line.split(":", 1)[1].strip())
            continue
        if line.startswith("ResourceID"):
            section = "resources"
            continue
        if line.startswith("TaskID"):
            section = "tasks"
            continue
        if section == "resources":
            parsed = parse_resource_line(line)
            if parsed:
                resources.append(parsed)
        elif section == "tasks":
            parsed = parse_task_line(line)
            if parsed:
                tasks.append(parsed)

    return {
        "source": path,
        "name": path.stem,
        "characteristics": characteristics,
        "resources": resources,
        "tasks": tasks,
    }


def parse_resource_line(line: str) -> dict[str, Any] | None:
    numbers = re.findall(r"^\s*(\d+)\s+([0-9]+(?:\.[0-9]+)?)", line)
    if not numbers:
        return None
    resource_id, salary = numbers[0]
    skills = [(skill, int(level) + 1) for skill, level in SKILL_RE.findall(line)]
    return {
        "id": int(resource_id),
        "salary": float(salary),
        "skills": skills,
    }


def parse_task_line(line: str) -> dict[str, Any] | None:
    match = re.match(r"^\s*(\d+)\s+(\d+)\s+(Q\d+)\s*:\s*(\d+)(.*)$", line)
    if not match:
        return None
    task_id, duration, skill, level, tail = match.groups()
    predecessors = [int(item) for item in re.findall(r"\d+", tail)]
    return {
        "id": int(task_id),
        "duration": int(duration),
        "skill": skill,
        "level": int(level) + 1,
        "predecessors": predecessors,
    }


def write_dataset(parsed: dict[str, Any], output_root: Path, source_label: str | None = None) -> Path:
    dataset_name = f"msrcpsp_{parsed['name']}"
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    tasks = parsed["tasks"]
    resources = parsed["resources"]
    skills = sorted({task["skill"] for task in tasks})
    total_duration = sum(task["duration"] for task in tasks)
    resource_count = max(1, len(resources))
    capacity = max(80, math.ceil(total_duration / resource_count))
    if resource_count <= 3:
        capacity = max(120, capacity)

    task_counts_by_skill = {skill: 0 for skill in skills}
    for task in tasks:
        task_counts_by_skill[task["skill"]] += 1

    source_text = source_label or str(parsed["source"])
    team_id = f"MSRCPSP_{parsed['name']}"
    team_name = f"MSRCPSP {parsed['name']}"

    write_tasks(dataset_dir / "tasks.csv", tasks, task_counts_by_skill, source_text)
    write_resources(dataset_dir / "resources.csv", resources, capacity)
    write_kpi_definitions(dataset_dir / "kpi-definitions.csv", skills)
    write_kpi_targets(dataset_dir / "kpi-targets.csv", skills, team_id, team_name)
    write_cycle(dataset_dir / "cycle.csv", parsed["name"], source_text)
    return dataset_dir


def write_tasks(path: Path, tasks: list[dict[str, Any]], task_counts_by_skill: dict[str, int], source_text: str) -> None:
    fieldnames = [
        "taskCode",
        "taskName",
        "description",
        "estimatedHours",
        "deadline",
        "priority",
        "dependencies",
        "requiredSkills",
        "kpiImpacts",
    ]
    project_weight = 1.0 / max(1, len(tasks))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for task in tasks:
            skill_weight = 1.0 / max(1, task_counts_by_skill[task["skill"]])
            dependencies = ",".join(f"TASK_{item}" for item in task["predecessors"])
            kpi_impacts = (
                f"PROJECT_COMPLETION:{project_weight:.8f},"
                f"{task['skill']}_DELIVERY_COMPLETION:{skill_weight:.8f}"
            )
            writer.writerow(
                {
                    "taskCode": f"TASK_{task['id']}",
                    "taskName": f"MSRCPSP task {task['id']}",
                    "description": f"Source {source_text} task {task['id']}",
                    "estimatedHours": task["duration"],
                    "deadline": "",
                    "priority": "MEDIUM",
                    "dependencies": dependencies,
                    "requiredSkills": f"{task['skill']}:{task['level']}",
                    "kpiImpacts": kpi_impacts,
                }
            )


def write_resources(path: Path, resources: list[dict[str, Any]], capacity: int) -> None:
    fieldnames = [
        "resourceId",
        "resourceName",
        "resourceType",
        "capacity",
        "skills",
        "currentLoad",
        "availabilityFrom",
        "availabilityTo",
        "availableHours",
        "costPerHour",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for resource in resources:
            skills = ", ".join(f"{skill}:{level}" for skill, level in resource["skills"])
            writer.writerow(
                {
                    "resourceId": f"RES_{resource['id']}",
                    "resourceName": f"MSRCPSP Resource {resource['id']}",
                    "resourceType": "HUMAN",
                    "capacity": capacity,
                    "skills": skills,
                    "currentLoad": 0,
                    "availabilityFrom": "2026-04-01",
                    "availabilityTo": "2026-04-30",
                    "availableHours": capacity,
                    "costPerHour": resource["salary"],
                }
            )


def write_kpi_definitions(path: Path, skills: list[str]) -> None:
    fieldnames = [
        "kpiCode",
        "kpiName",
        "description",
        "unit",
        "formula",
        "category",
        "dataType",
        "minValue",
        "maxValue",
        "higherIsBetter",
    ]
    rows = [
        {
            "kpiCode": "PROJECT_COMPLETION",
            "kpiName": "Project completion",
            "description": "Weighted delivery completion for all tasks adjusted by skill fit",
            "unit": "PERCENT",
            "formula": "sum(task impact * target * skill_factor)",
            "category": "DELIVERY",
            "dataType": "NUMERIC",
            "minValue": 0,
            "maxValue": 100,
            "higherIsBetter": "true",
        }
    ]
    for skill in skills:
        rows.append(
            {
                "kpiCode": f"{skill}_DELIVERY_COMPLETION",
                "kpiName": f"{skill} delivery completion",
                "description": f"Weighted delivery completion for tasks requiring {skill} adjusted by skill fit",
                "unit": "PERCENT",
                "formula": "sum(task impact * target * skill_factor)",
                "category": "DELIVERY",
                "dataType": "NUMERIC",
                "minValue": 0,
                "maxValue": 100,
                "higherIsBetter": "true",
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_kpi_targets(path: Path, skills: list[str], team_id: str, team_name: str) -> None:
    fieldnames = [
        "kpiCode",
        "targetValue",
        "priority",
        "weight",
        "warningThreshold",
        "criticalThreshold",
        "appliesToType",
        "appliesToTeamId",
        "appliesToTeamName",
    ]
    skill_weight = 0.7 / max(1, len(skills))
    rows = [
        {
            "kpiCode": "PROJECT_COMPLETION",
            "targetValue": 100,
            "priority": "HIGH",
            "weight": 0.3,
            "warningThreshold": 80,
            "criticalThreshold": 60,
            "appliesToType": "PROJECT",
            "appliesToTeamId": team_id,
            "appliesToTeamName": team_name,
        }
    ]
    for skill in skills:
        rows.append(
            {
                "kpiCode": f"{skill}_DELIVERY_COMPLETION",
                "targetValue": 100,
                "priority": "MEDIUM",
                "weight": round(skill_weight, 8),
                "warningThreshold": 80,
                "criticalThreshold": 60,
                "appliesToType": "PROJECT",
                "appliesToTeamId": team_id,
                "appliesToTeamName": team_name,
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_cycle(path: Path, dataset_name: str, source_text: str) -> None:
    fieldnames = ["cycleName", "startDate", "endDate", "cycleType", "previousCycleId", "notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "cycleName": f"MSRCPSP {dataset_name}",
                "startDate": "2026-04-01",
                "endDate": "2026-04-30",
                "cycleType": "PROJECT",
                "previousCycleId": "",
                "notes": f"Converted from {source_text}",
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MSRCPSP .def files to HS CSV dataset folders.")
    parser.add_argument("inputs", nargs="+", help="Input .def files.")
    parser.add_argument("--output-root", default=str(Path(__file__).parent / "examples"))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    for item in args.inputs:
        source = Path(item)
        parsed = parse_def(source)
        dataset_dir = write_dataset(parsed, output_root, source_label=str(source).replace("\\", "/"))
        print(f"converted {source} -> {dataset_dir}")


if __name__ == "__main__":
    main()
