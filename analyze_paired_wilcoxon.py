from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


SKILL_GUIDED_ALGORITHM = "hybrid_hs_ga_guided_mutation_hs150_ga850"
COMPARISONS = {
    "Skill-Guided HS-GA vs Basic HS-GA": "HS150-GA850",
    "Skill-Guided HS-GA vs GA": "GA",
}
ALPHA = 0.05


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values while preserving the original order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, index in enumerate(order):
        corrected = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, corrected)
        adjusted[index] = running_max
    return adjusted


def validate_unique_runs(frame: pd.DataFrame, name: str) -> None:
    required = {"dataset", "algorithm", "seed", "totalScore"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    duplicates = frame.duplicated(["dataset", "algorithm", "seed"], keep=False)
    if duplicates.any():
        keys = frame.loc[duplicates, ["dataset", "algorithm", "seed"]]
        raise ValueError(f"{name} contains duplicate run keys:\n{keys.to_string(index=False)}")


def paired_rows(
    guided: pd.DataFrame,
    baseline: pd.DataFrame,
    comparison: str,
) -> list[dict[str, object]]:
    merged = guided.merge(
        baseline,
        on=["dataset", "seed"],
        how="outer",
        suffixes=("_A", "_B"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(
            f"Unmatched dataset/seed keys for {comparison}:\n"
            f"{unmatched[['dataset', 'seed', '_merge']].to_string(index=False)}"
        )

    output: list[dict[str, object]] = []
    for dataset, group in merged.groupby("dataset", sort=True):
        group = group.sort_values("seed")
        score_a = pd.to_numeric(group["totalScore_A"], errors="raise").to_numpy(dtype=float)
        score_b = pd.to_numeric(group["totalScore_B"], errors="raise").to_numpy(dtype=float)
        differences = score_a - score_b
        if np.allclose(differences, 0.0):
            raw_p = 1.0
        else:
            raw_p = float(
                wilcoxon(
                    score_a,
                    score_b,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                ).pvalue
            )
        output.append(
            {
                "dataset": dataset,
                "comparison": comparison,
                "n_pairs": int(len(group)),
                "mean_A": float(np.mean(score_a)),
                "mean_B": float(np.mean(score_b)),
                "median_paired_difference": float(np.median(differences)),
                "raw_p": raw_p,
            }
        )
    return output


def add_holm_conclusions(rows: list[dict[str, object]]) -> None:
    adjusted = holm_adjust([float(row["raw_p"]) for row in rows])
    for row, holm_p in zip(rows, adjusted, strict=True):
        row["holm_p"] = holm_p
        median_difference = float(row["median_paired_difference"])
        mean_difference = float(row["mean_A"]) - float(row["mean_B"])
        direction = median_difference if median_difference != 0 else mean_difference
        if holm_p >= ALPHA:
            conclusion = "non_significant"
        elif direction > 0:
            conclusion = "significant_win"
        elif direction < 0:
            conclusion = "significant_loss"
        else:
            conclusion = "non_significant"
        row["conclusion"] = conclusion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run per-dataset paired Wilcoxon tests from raw per-seed benchmark scores."
    )
    parser.add_argument(
        "--baseline-raw",
        type=Path,
        default=Path("results/final_benchmark/final_benchmark_raw.csv"),
    )
    parser.add_argument(
        "--guided-raw",
        type=Path,
        default=Path("results/final_benchmark/skill_guided_ta3_raw_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final_benchmark"),
    )
    args = parser.parse_args()

    baseline_raw = pd.read_csv(args.baseline_raw)
    guided_raw = pd.read_csv(args.guided_raw)
    validate_unique_runs(baseline_raw, str(args.baseline_raw))
    validate_unique_runs(guided_raw, str(args.guided_raw))

    guided = guided_raw.loc[
        guided_raw["algorithm"] == SKILL_GUIDED_ALGORITHM,
        ["dataset", "seed", "totalScore"],
    ].copy()
    if guided.empty:
        raise ValueError(f"No rows found for {SKILL_GUIDED_ALGORITHM}")

    all_rows: list[dict[str, object]] = []
    for comparison, baseline_algorithm in COMPARISONS.items():
        baseline = baseline_raw.loc[
            baseline_raw["algorithm"] == baseline_algorithm,
            ["dataset", "seed", "totalScore"],
        ].copy()
        if baseline.empty:
            raise ValueError(f"No rows found for {baseline_algorithm}")
        rows = paired_rows(guided, baseline, comparison)
        if len(rows) != 16:
            raise ValueError(f"Expected 16 dataset comparisons for {comparison}, got {len(rows)}")
        if any(int(row["n_pairs"]) != 30 for row in rows):
            raise ValueError(f"Expected 30 paired seeds per dataset for {comparison}")
        add_holm_conclusions(rows)
        all_rows.extend(rows)

    result = pd.DataFrame(
        all_rows,
        columns=[
            "dataset",
            "comparison",
            "n_pairs",
            "mean_A",
            "mean_B",
            "median_paired_difference",
            "raw_p",
            "holm_p",
            "conclusion",
        ],
    )
    summary = (
        result.groupby(["comparison", "conclusion"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=["significant_win", "non_significant", "significant_loss"],
            fill_value=0,
        )
        .reset_index()
    )
    summary["total_instances"] = summary[
        ["significant_win", "non_significant", "significant_loss"]
    ].sum(axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "paired_wilcoxon_results.csv"
    summary_path = args.output_dir / "paired_wilcoxon_summary.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig", float_format="%.12g")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(result.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print(f"\nWrote {result_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
