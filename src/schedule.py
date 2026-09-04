from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from app.modules.assignment.algorithms.models import TaskSchedule
except ModuleNotFoundError:
    from models import TaskSchedule


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


class SchedulePreprocessor:
    def __init__(self, tasks: list[dict[str, Any]], cycle: dict[str, Any] | None = None) -> None:
        self.tasks = tasks
        self.cycle = cycle or {}
        self.task_by_id = {str(task.get("taskId")): task for task in tasks if task.get("taskId")}
        self.task_id_by_code = {
            str(task.get("taskCode")): str(task.get("taskId"))
            for task in tasks
            if task.get("taskId") and task.get("taskCode")
        }

    def build(self) -> tuple[list[str], dict[str, TaskSchedule], dict[str, list[str]], dict[str, Any]]:
        predecessors = self._build_predecessors()
        order = self._topological_order(predecessors)
        topo_level = self._topo_levels(order, predecessors)
        schedule = self._build_schedule(order, predecessors, topo_level)
        metadata = {
            "hasCycle": len(order) != len(self.task_by_id),
            "criticalPath": self._critical_path(order, schedule),
            "cycleStart": self._cycle_start().isoformat(),
        }
        return order, schedule, predecessors, metadata

    def _build_predecessors(self) -> dict[str, list[str]]:
        predecessors: dict[str, list[str]] = {}
        for task_id, task in self.task_by_id.items():
            resolved = []
            for raw_dependency in task.get("dependencies") or []:
                dependency = str(raw_dependency)
                resolved_id = self.task_id_by_code.get(dependency, dependency)
                if resolved_id in self.task_by_id and resolved_id != task_id:
                    resolved.append(resolved_id)
            predecessors[task_id] = sorted(set(resolved))
        return predecessors

    def _topological_order(self, predecessors: dict[str, list[str]]) -> list[str]:
        children: dict[str, list[str]] = defaultdict(list)
        indegree = {task_id: len(preds) for task_id, preds in predecessors.items()}
        for task_id, preds in predecessors.items():
            for pred in preds:
                children[pred].append(task_id)

        ready = deque(sorted(task_id for task_id, degree in indegree.items() if degree == 0))
        order = []
        while ready:
            task_id = ready.popleft()
            order.append(task_id)
            for child in sorted(children.get(task_id, [])):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)

        if len(order) < len(indegree):
            remaining = sorted(task_id for task_id in indegree if task_id not in order)
            order.extend(remaining)
        return order

    def _topo_levels(self, order: list[str], predecessors: dict[str, list[str]]) -> dict[str, int]:
        levels: dict[str, int] = {}
        for task_id in order:
            preds = predecessors.get(task_id, [])
            levels[task_id] = 0 if not preds else 1 + max(levels.get(pred, 0) for pred in preds)
        return levels

    def _build_schedule(
        self,
        order: list[str],
        predecessors: dict[str, list[str]],
        topo_level: dict[str, int],
    ) -> dict[str, TaskSchedule]:
        task_end: dict[str, float] = {}
        schedule: dict[str, TaskSchedule] = {}
        for task_id in order:
            task = self.task_by_id[task_id]
            duration = max(safe_float(task.get("estimatedHours")), 1.0)
            deps_ready = max((task_end.get(pred, 0.0) for pred in predecessors.get(task_id, [])), default=0.0)
            start = deps_ready
            end = start + duration
            task_end[task_id] = end
            schedule[task_id] = TaskSchedule(
                task_id=task_id,
                planned_start_hour=round(start, 2),
                planned_end_hour=round(end, 2),
                duration_hours=round(duration, 2),
                topo_level=topo_level.get(task_id, 0),
            )
        return schedule

    def _critical_path(self, order: list[str], schedule: dict[str, TaskSchedule]) -> list[str]:
        if not order:
            return []
        max_end = max((item.planned_end_hour for item in schedule.values()), default=0.0)
        critical = [
            task_id
            for task_id in order
            if schedule.get(task_id) and schedule[task_id].planned_end_hour >= max_end
        ]
        return critical[-5:]

    def _cycle_start(self) -> datetime:
        return parse_datetime(self.cycle.get("startDate")) or datetime.now(timezone.utc)

    def hour_to_datetime(self, hour: float) -> str:
        return (self._cycle_start() + timedelta(hours=hour)).isoformat()


class ResourceAwareScheduler:
    def __init__(
        self,
        tasks: list[dict[str, Any]],
        base_schedule: dict[str, TaskSchedule],
        predecessors: dict[str, list[str]],
    ) -> None:
        self.tasks = tasks
        self.base_schedule = base_schedule
        self.predecessors = predecessors
        self.task_by_id = {str(task.get("taskId")): task for task in tasks if task.get("taskId")}

    def build(self, assignment: dict[str, str]) -> dict[str, TaskSchedule]:
        task_end: dict[str, float] = {}
        resource_available: dict[str, float] = {}
        schedule: dict[str, TaskSchedule] = {}
        for task_id in self._ordered_task_ids(assignment):
            resource_id = assignment.get(task_id)
            task = self.task_by_id.get(task_id, {})
            base_item = self.base_schedule.get(task_id)
            duration = base_item.duration_hours if base_item else max(safe_float(task.get("estimatedHours")), 1.0)
            deps_ready = max((task_end.get(pred, 0.0) for pred in self.predecessors.get(task_id, [])), default=0.0)
            resource_ready = resource_available.get(resource_id, 0.0) if resource_id else 0.0
            start = max(deps_ready, resource_ready)
            end = start + duration
            task_end[task_id] = end
            if resource_id:
                resource_available[resource_id] = end
            schedule[task_id] = TaskSchedule(
                task_id=task_id,
                planned_start_hour=round(start, 2),
                planned_end_hour=round(end, 2),
                duration_hours=round(duration, 2),
                topo_level=base_item.topo_level if base_item else 0,
                critical=base_item.critical if base_item else False,
            )
        return schedule

    def _ordered_task_ids(self, assignment: dict[str, str]) -> list[str]:
        return sorted(
            [task_id for task_id in assignment if task_id in self.task_by_id],
            key=lambda task_id: (
                self.base_schedule.get(task_id).topo_level if task_id in self.base_schedule else 0,
                self.base_schedule.get(task_id).planned_start_hour if task_id in self.base_schedule else 0.0,
                task_id,
            ),
        )

