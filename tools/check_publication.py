"""Check the Git publication surface without deleting local research files."""

from __future__ import annotations

import ast
import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = {
    "Small": "rawdata/MSRCPSP/Small",
    "iMOPSE": "rawdata/imopse",
    "GenRegular": "rawdata/MSRCPSP/GenRegular",
    "GenBig": "rawdata/MSRCPSP/GenBig",
}
HYBRID_CLASSES = {
    "SeededGeneticAlgorithmAssignmentOptimizer",
    "HybridHarmonySearchGAAssignmentOptimizer",
    "GuidedMutationSeededGeneticAlgorithmAssignmentOptimizer",
    "HybridHarmonySearchGuidedMutationGAAssignmentOptimizer",
}
SOURCE_FILES = {
    "__init__.py", "analyze_paper_update.py", "benchmark_algorithms.py",
    "benchmark_imopse_solutions.py", "convert_msrcpsp_def.py", "csv_runner.py",
    "evaluator.py", "genetic_algorithm.py", "harmony_search.py", "hybrid_hs_ga.py",
    "models.py", "schedule.py", "skill_guided_ga.py",
}
TEST_FILES = {"test_smoke.py", "test_imopse_benchmark.py", "test_paper_release.py"}


def paper_files(root: Path = ROOT) -> set[str]:
    with (root / "datasets/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        datasets = list(csv.DictReader(stream))
    names = [row["Dataset"] for row in datasets]
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError("The paper manifest must contain 16 unique dataset names")
    allowed = {
        ".gitignore", "README.md", "requirements.txt", "datasets/manifest.csv",
        "scripts/run_paper_experiments.ps1", "tools/check_publication.py",
        "tools/imopse/README.md", "tools/imopse/run_imopse_screening.ps1",
        "tools/imopse/export_pareto_genotypes.patch",
    }
    allowed.update(f"src/{name}" for name in SOURCE_FILES)
    allowed.update(f"src/tests/{name}" for name in TEST_FILES)
    allowed.update(f"template/{name}" for name in (
        "task-template (1).csv", "resource-template.csv", "kpi-definition-template.csv",
        "kpi-target-template.csv", "cycle-template.csv",
    ))
    allowed.update(f"tools/imopse/configs/{name}_MSRCPSP_1000NFE.cfg" for name in (
        "BNTGA", "MOEAD", "NSGAII",
    ))
    for row in datasets:
        name = row["Dataset"]
        allowed.add(f"{SOURCE_ROOTS[row['Source']]}/{name}.def")
        allowed.update(f"datasets/msrcpsp_{name}/{file}.csv" for file in (
            "tasks", "resources", "kpi-definitions", "kpi-targets", "cycle",
        ))
    return allowed


def git_paths(*arguments: str) -> set[str]:
    output = subprocess.check_output(["git", *arguments, "-z"], cwd=ROOT)
    return {item.decode("utf-8") for item in output.split(b"\0") if item}


def check_publication() -> list[str]:
    errors = []
    allowed = paper_files()
    candidate_paths = git_paths("ls-files", "--cached", "--others", "--exclude-standard")
    unexpected = candidate_paths - allowed
    if unexpected:
        errors.extend(f"Outside publication scope: {name}" for name in sorted(unexpected))
    missing = allowed - candidate_paths
    if missing:
        errors.extend(f"Missing or ignored publication input: {name}" for name in sorted(missing))
    tracked_ignored = git_paths("ls-files", "--cached", "--ignored", "--exclude-standard")
    if tracked_ignored:
        errors.extend(f"Still tracked despite ignore rule: {name}" for name in sorted(tracked_ignored))
    tree = ast.parse((ROOT / "src/hybrid_hs_ga.py").read_text(encoding="utf-8"))
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    if classes != HYBRID_CLASSES:
        errors.append(f"Unexpected hybrid classes: {sorted(classes ^ HYBRID_CLASSES)}")
    return errors


def main() -> None:
    errors = check_publication()
    if errors:
        raise SystemExit("Publication check FAILED:\n" + "\n".join(errors))
    print(f"Publication check PASSED: {len(paper_files())} code, config and input files; 16 datasets")
    print("Local ignored artifacts and Git history are not deleted by this check.")


if __name__ == "__main__":
    main()
