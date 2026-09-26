from datetime import datetime, time, timezone
from pathlib import Path
import csv
import sys
import tempfile
import unittest


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transfer_working_calendar import WorkingCalendar  # noqa: E402
from benchmark_algorithms import write_run_assignment  # noqa: E402


class WorkingCalendarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calendar = WorkingCalendar(
            time(8), time(12), time(13), time(17), frozenset(range(5))
        )

    def test_ten_hours_cross_lunch_and_day(self) -> None:
        start = datetime(2026, 4, 6, 10, tzinfo=timezone.utc)  # Monday
        self.assertEqual(
            datetime(2026, 4, 7, 12, tzinfo=timezone.utc),
            self.calendar.add_working_hours(start, 10),
        )

    def test_work_crosses_weekend(self) -> None:
        start = datetime(2026, 4, 3, 16, tzinfo=timezone.utc)  # Friday
        self.assertEqual(
            datetime(2026, 4, 6, 9, tzinfo=timezone.utc),
            self.calendar.add_working_hours(start, 2),
        )

    def test_segments_never_include_lunch_or_night(self) -> None:
        start = datetime(2026, 4, 6, 10, tzinfo=timezone.utc)
        segments = self.calendar.work_segments(start, 10)
        self.assertEqual([2.0, 4.0, 4.0], [item["hours"] for item in segments])
        self.assertEqual("2026-04-07T12:00:00+00:00", segments[-1]["end"])

    def test_benchmark_assignment_is_written_per_run(self) -> None:
        output = {
            "best": {
                "assignment": {"TASK_1": "RES_2"},
                "score": {
                    "totalScore": 88.5,
                    "feasible": True,
                    "diagnostics": {
                        "actualSchedule": {
                            "TASK_1": {
                                "plannedStartHour": 0.0,
                                "plannedEndHour": 3.0,
                                "durationHours": 3.0,
                            }
                        }
                    },
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = write_run_assignment(
                Path(temporary), "msrcpsp_test", "ga", 7, output
            )
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual("TASK_1", rows[0]["taskId"])
            self.assertEqual("RES_2", rows[0]["resourceId"])
            self.assertEqual("3.0", rows[0]["rawEndHour"])


if __name__ == "__main__":
    unittest.main()
