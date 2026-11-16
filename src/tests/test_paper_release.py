from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from check_publication import HYBRID_CLASSES, paper_files  # noqa: E402
from benchmark_algorithms import DEFAULT_ALGORITHMS  # noqa: E402
from csv_runner import build_parser  # noqa: E402
from hybrid_hs_ga import GuidedMutationSeededGeneticAlgorithmAssignmentOptimizer  # noqa: E402


class PaperReleaseTests(unittest.TestCase):
    def test_manifest_inputs_are_complete(self) -> None:
        for name in paper_files():
            with self.subTest(path=name):
                self.assertTrue((ROOT / name).is_file())

    def test_only_paper_hybrid_classes_remain(self) -> None:
        tree = ast.parse((ROOT / "src/hybrid_hs_ga.py").read_text(encoding="utf-8"))
        self.assertEqual(HYBRID_CLASSES, {node.name for node in tree.body if isinstance(node, ast.ClassDef)})

    def test_guidance_is_skill_only(self) -> None:
        cls = GuidedMutationSeededGeneticAlgorithmAssignmentOptimizer
        self.assertEqual((1.0, 0.0, 0.0), (cls.skill_weight, cls.workload_weight, cls.timeline_weight))
        self.assertEqual(1e-6, cls.weakness_epsilon)

    def test_csv_default_is_a_controlled_configuration(self) -> None:
        args = build_parser().parse_args(["--tasks", "tasks.csv", "--resources", "resources.csv"])
        self.assertEqual("ga", args.algorithm)

    def test_benchmark_defaults_to_the_four_controlled_configurations(self) -> None:
        self.assertEqual(
            {"ga", "skill_guided_ga", "hybrid_hs_ga", "hybrid_hs_ga_guided_mutation"},
            set(DEFAULT_ALGORITHMS.split(",")),
        )

    def test_ignore_excludes_local_artifacts_but_not_paper_inputs(self) -> None:
        if not (ROOT / ".git").exists():
            self.skipTest("Git metadata is absent in a source archive")
        excluded = [
            ".codex/session.json", ".agents/skills/local.md", ".claude/settings.json",
            ".cursor/rules/local.md", ".chart-data-test/chart.json",
            "implementation_plan.md", "slides/deck.pptx", "reports/local.md",
            "results/private/raw_results.csv", "checkpoints/local.zip",
            "rawdata/PTSP_Task_202604(Worklogs).csv", "datasets/MSLIB_v2_2/readme.txt",
            "datasets_real_v2/manifest.csv", "src/convert_ptsp_real.py",
        ]
        for name in excluded:
            with self.subTest(path=name):
                result = subprocess.run(["git", "check-ignore", "--no-index", "-q", name], cwd=ROOT)
                self.assertEqual(0, result.returncode)
        for name in paper_files():
            with self.subTest(path=name):
                result = subprocess.run(["git", "check-ignore", "--no-index", "-q", name], cwd=ROOT)
                self.assertEqual(1, result.returncode)


if __name__ == "__main__":
    unittest.main()
