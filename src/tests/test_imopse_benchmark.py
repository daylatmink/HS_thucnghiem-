from pathlib import Path
import sys
import unittest


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_imopse_solutions import (  # noqa: E402
    build_candidate_pools,
    choose_blind,
    decode_assignment,
    parse_vector,
)
from csv_runner import load_resources, load_tasks  # noqa: E402


class ImopseBenchmarkTests(unittest.TestCase):
    def test_parse_vector(self) -> None:
        self.assertEqual(parse_vector("0 2.5 3"), [0.0, 2.5, 3.0])
        self.assertEqual(parse_vector(""), [])

    def test_decoded_assignment_is_skill_feasible(self) -> None:
        dataset = SRC.parent / "datasets" / "msrcpsp_10_3_5_3"
        tasks = load_tasks(dataset / "tasks.csv")
        resources = load_resources(dataset / "resources.csv")
        pools = build_candidate_pools(tasks, resources)
        assignment = decode_assignment([0.0] * len(tasks), tasks, pools)
        self.assertEqual(len(assignment), len(tasks))
        self.assertTrue(all(assignment[str(task["taskId"])] in pools[str(task["taskId"])] for task in tasks))

    def test_blind_selection_uses_only_native_objectives(self) -> None:
        rows = [
            {"solutionId": 0, "originalMakespan": 10, "originalCost": 30, "totalScore": 1},
            {"solutionId": 1, "originalMakespan": 20, "originalCost": 10, "totalScore": 99},
        ]
        first = choose_blind(rows)
        rows[0]["totalScore"], rows[1]["totalScore"] = 100, 0
        second = choose_blind(rows)
        self.assertEqual(first["solutionId"], second["solutionId"])

    def test_decode_rejects_out_of_range_gene(self) -> None:
        tasks = [{"taskId": "TASK_1"}]
        with self.assertRaisesRegex(ValueError, "outside"):
            decode_assignment([2.0], tasks, {"TASK_1": ["RES_1"]})


if __name__ == "__main__":
    unittest.main()
