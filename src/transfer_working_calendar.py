from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from csv_runner import load_cycle, load_tasks
from schedule import ResourceAwareScheduler, SchedulePreprocessor


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_clock(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


@dataclass(frozen=True)
class WorkingCalendar:
    morning_start: time
    morning_end: time
    afternoon_start: time
    afternoon_end: time
    working_days: frozenset[int]

    def _at(self, day: date, clock: time, tz: timezone) -> datetime:
        return datetime.combine(day, clock, tzinfo=tz)

    def next_working_start(self, value: datetime) -> datetime:
        current = value
        while True:
            if current.weekday() not in self.working_days:
                current = self._at(current.date() + timedelta(days=1), self.morning_start, current.tzinfo or timezone.utc)
                continue
            morning_start = self._at(current.date(), self.morning_start, current.tzinfo or timezone.utc)
            morning_end = self._at(current.date(), self.morning_end, current.tzinfo or timezone.utc)
            afternoon_start = self._at(current.date(), self.afternoon_start, current.tzinfo or timezone.utc)
            afternoon_end = self._at(current.date(), self.afternoon_end, current.tzinfo or timezone.utc)
            if current < morning_start:
                return morning_start
            if current < morning_end:
                return current
            if current < afternoon_start:
                return afternoon_start
            if current < afternoon_end:
                return current
            current = self._at(current.date() + timedelta(days=1), self.morning_start, current.tzinfo or timezone.utc)

    def add_working_hours(self, start: datetime, hours: float) -> datetime:
        current = self.next_working_start(start)
        remaining = max(0.0, float(hours))
        if remaining == 0:
            return current
        while remaining > 1e-9:
            morning_end = self._at(current.date(), self.morning_end, current.tzinfo or timezone.utc)
            afternoon_end = self._at(current.date(), self.afternoon_end, current.tzinfo or timezone.utc)
            interval_end = morning_end if current < morning_end else afternoon_end
            available = max(0.0, (interval_end - current).total_seconds() / 3600.0)
            consumed = min(available, remaining)
            current += timedelta(hours=consumed)
            remaining -= consumed
            if remaining > 1e-9:
                current = self.next_working_start(current)
        return current

    def work_segments(self, start: datetime, hours: float) -> list[dict[str, Any]]:
        current = self.next_working_start(start)
        remaining = max(0.0, float(hours))
        segments = []
        while remaining > 1e-9:
            morning_end = self._at(current.date(), self.morning_end, current.tzinfo or timezone.utc)
            afternoon_end = self._at(current.date(), self.afternoon_end, current.tzinfo or timezone.utc)
            interval_end = morning_end if current < morning_end else afternoon_end
            consumed = min((interval_end - current).total_seconds() / 3600.0, remaining)
            end = current + timedelta(hours=consumed)
            segments.append(
                {
                    "start": current.isoformat(),
                    "end": end.isoformat(),
                    "hours": round(consumed, 6),
                }
            )
            remaining -= consumed
            current = end if remaining <= 1e-9 else self.next_working_start(end)
        return segments


def project_start(cycle: dict[str, Any], override: str | None, tz: timezone, calendar: WorkingCalendar) -> datetime:
    raw = override or str(cycle.get("startDate") or "").strip()
    if not raw:
        raise ValueError("project start is missing; provide --project-start or cycle.startDate")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return calendar.next_working_start(parsed.astimezone(tz))


def cycle_horizon(cycle: dict[str, Any], tz: timezone, calendar: WorkingCalendar) -> datetime:
    raw = str(cycle.get("endDate") or "").strip()
    if not raw:
        raise ValueError("cycle.endDate is required for --horizon-mode cycle_end")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    return datetime.combine(parsed.astimezone(tz).date(), calendar.afternoon_end, tzinfo=tz)


def transfer_assignment(
    assignment_path: Path,
    assignments_root: Path,
    datasets_dir: Path,
    output_root: Path,
    calendar: WorkingCalendar,
    args: argparse.Namespace,
) -> dict[str, Any]:
    relative = assignment_path.relative_to(assignments_root)
    if len(relative.parts) != 3:
        raise ValueError(f"unexpected assignment path: {relative}")
    algorithm, dataset, seed_file = relative.parts
    assignment_rows = read_csv(assignment_path)
    assignment = {row["taskId"]: row["resourceId"] for row in assignment_rows}
    dataset_dir = datasets_dir / dataset
    tasks = load_tasks(dataset_dir / "tasks.csv")
    cycle = load_cycle(dataset_dir / "cycle.csv")
    order, base_schedule, predecessors, _ = SchedulePreprocessor(tasks, cycle).build()
    if set(assignment) != {str(task["taskId"]) for task in tasks}:
        raise ValueError(f"{relative}: assignment does not cover every task exactly once")
    raw_schedule = ResourceAwareScheduler(tasks, base_schedule, predecessors).build(assignment)
    tz = timezone(timedelta(hours=args.utc_offset_hours))
    start = project_start(cycle, args.project_start, tz, calendar)
    base_makespan = max((item.planned_end_hour for item in base_schedule.values()), default=0.0)
    actual_makespan = max((item.planned_end_hour for item in raw_schedule.values()), default=0.0)
    if args.horizon_mode == "cpm_factor":
        horizon_hours = base_makespan * args.horizon_factor
        horizon_end = calendar.add_working_hours(start, horizon_hours)
    elif args.horizon_mode == "cycle_end":
        horizon_hours = None
        horizon_end = cycle_horizon(cycle, tz, calendar)
    else:
        horizon_hours = None
        horizon_end = None

    output_rows = []
    for task_id in order:
        item = raw_schedule[task_id]
        task_start = calendar.next_working_start(
            calendar.add_working_hours(start, item.planned_start_hour)
        )
        segments = calendar.work_segments(task_start, item.duration_hours)
        task_end = datetime.fromisoformat(segments[-1]["end"]) if segments else task_start
        within_horizon = horizon_end is None or task_end <= horizon_end
        output_rows.append(
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "seed": seed_file.removeprefix("seed_").removesuffix(".csv"),
                "taskId": task_id,
                "resourceId": assignment[task_id],
                "durationWorkingHours": item.duration_hours,
                "rawStartWorkingHour": item.planned_start_hour,
                "rawEndWorkingHour": item.planned_end_hour,
                "calendarStart": task_start.isoformat(),
                "calendarEnd": task_end.isoformat(),
                "workSegments": json.dumps(segments, ensure_ascii=False, separators=(",", ":")),
                "withinWorkingCalendar": True,
                "withinProjectHorizon": within_horizon if horizon_end else "",
                "projectHorizonEnd": horizon_end.isoformat() if horizon_end else "",
            }
        )
    output_path = output_root / algorithm / dataset / seed_file
    write_csv(output_path, output_rows)
    final_end = max(datetime.fromisoformat(row["calendarEnd"]) for row in output_rows)
    return {
        "dataset": dataset,
        "algorithm": algorithm,
        "seed": seed_file.removeprefix("seed_").removesuffix(".csv"),
        "tasks": len(output_rows),
        "baseMakespanWorkingHours": base_makespan,
        "actualMakespanWorkingHours": actual_makespan,
        "calendarStart": start.isoformat(),
        "calendarEnd": final_end.isoformat(),
        "horizonMode": args.horizon_mode,
        "horizonFactor": args.horizon_factor if args.horizon_mode == "cpm_factor" else "",
        "horizonWorkingHours": horizon_hours if horizon_hours is not None else "",
        "projectHorizonEnd": horizon_end.isoformat() if horizon_end else "",
        "workingCalendarFeasible": horizon_end is None or final_end <= horizon_end,
        "outputFile": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map fixed benchmark assignments onto a weekday working calendar.")
    parser.add_argument("--assignments-root", required=True)
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--project-start", help="Optional ISO date/datetime override; defaults to cycle.startDate.")
    parser.add_argument("--workday-start", default="08:00")
    parser.add_argument("--lunch-start", default="12:00")
    parser.add_argument("--lunch-end", default="13:00")
    parser.add_argument("--workday-end", default="17:00")
    parser.add_argument("--working-days", default="0,1,2,3,4", help="Python weekday numbers; Monday=0.")
    parser.add_argument("--utc-offset-hours", type=float, default=7.0)
    parser.add_argument("--horizon-mode", choices=["none", "cycle_end", "cpm_factor"], default="none")
    parser.add_argument("--horizon-factor", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignments_root = Path(args.assignments_root)
    output_root = Path(args.output_dir)
    calendar = WorkingCalendar(
        parse_clock(args.workday_start),
        parse_clock(args.lunch_start),
        parse_clock(args.lunch_end),
        parse_clock(args.workday_end),
        frozenset(int(value) for value in args.working_days.split(",") if value.strip()),
    )
    if not (calendar.morning_start < calendar.morning_end <= calendar.afternoon_start < calendar.afternoon_end):
        raise ValueError("working-hour intervals must be ordered and non-overlapping")
    assignment_files = sorted(assignments_root.glob("*/*/seed_*.csv"))
    if not assignment_files:
        raise ValueError(f"no assignment files found under {assignments_root}")
    summaries = [
        transfer_assignment(
            path,
            assignments_root,
            Path(args.datasets_dir),
            output_root,
            calendar,
            args,
        )
        for path in assignment_files
    ]
    write_csv(output_root / "calendar_summary.csv", summaries)
    print(f"transferred {len(summaries)} assignments to {output_root}")


if __name__ == "__main__":
    main()
