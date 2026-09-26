from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "imopse_kpi_benchmark" / "skill_guided_comparison"

SOURCES = {
    "Skill-Guided GA": (
        ROOT / "results" / "ga_skill_guided_comparison" / "raw_results.csv",
        "skill_guided_ga",
        30,
    ),
    "Skill-Guided HS150-GA850": (
        ROOT / "results" / "final_benchmark" / "skill_guided_ta3_raw_results.csv",
        "hybrid_hs_ga_guided_mutation_hs150_ga850",
        30,
    ),
    "BNTGA Blind": (
        ROOT / "results" / "imopse_kpi_benchmark" / "raw_results.csv",
        "imopse_BNTGA_blind",
        10,
    ),
    "BNTGA Pareto-Rerank": (
        ROOT / "results" / "imopse_kpi_benchmark" / "raw_results.csv",
        "imopse_BNTGA_pareto_rerank",
        10,
    ),
    "MOEA/D Blind": (
        ROOT / "results" / "imopse_kpi_benchmark" / "raw_results.csv",
        "imopse_MOEAD_blind",
        10,
    ),
    "MOEA/D Pareto-Rerank": (
        ROOT / "results" / "imopse_kpi_benchmark" / "raw_results.csv",
        "imopse_MOEAD_pareto_rerank",
        10,
    ),
    "NSGA-II Blind": (
        ROOT / "results" / "imopse_kpi_benchmark" / "raw_results.csv",
        "imopse_NSGAII_blind",
        10,
    ),
    "NSGA-II Pareto-Rerank": (
        ROOT / "results" / "imopse_kpi_benchmark" / "raw_results.csv",
        "imopse_NSGAII_pareto_rerank",
        10,
    ),
}

SKILL_GUIDED = ("Skill-Guided GA", "Skill-Guided HS150-GA850")
IMOPSE = tuple(name for name in SOURCES if name not in SKILL_GUIDED)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def load_scores() -> dict[str, dict[str, list[float]]]:
    cache: dict[Path, list[dict[str, str]]] = {}
    scores: dict[str, dict[str, list[float]]] = {}
    for label, (path, algorithm, expected_runs) in SOURCES.items():
        cache.setdefault(path, read_rows(path))
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in cache[path]:
            if row.get("algorithm") == algorithm:
                grouped[row["dataset"]].append(float(row["totalScore"]))
        if len(grouped) != 16:
            raise ValueError(f"{label}: expected 16 datasets, found {len(grouped)}")
        invalid = {dataset: len(values) for dataset, values in grouped.items() if len(values) != expected_runs}
        if invalid:
            raise ValueError(f"{label}: unexpected run counts: {invalid}")
        scores[label] = dict(grouped)
    datasets = set(scores[next(iter(scores))])
    for label, grouped in scores.items():
        if set(grouped) != datasets:
            raise ValueError(f"{label}: dataset set differs from other algorithms")
    return scores


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def dataset_table(scores: dict[str, dict[str, list[float]]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in sorted(scores[next(iter(scores))]):
        row: dict[str, Any] = {"dataset": dataset}
        means = {}
        for label in SOURCES:
            mean_value, std_value = mean_std(scores[label][dataset])
            means[label] = mean_value
            row[f"{label} mean"] = round(mean_value, 6)
            row[f"{label} std"] = round(std_value, 6)
            row[f"{label} mean ± std"] = f"{mean_value:.3f} ± {std_value:.3f}"
        row["bestAlgorithm"] = max(means, key=means.get)
        row["bestMeanTotalScore"] = round(max(means.values()), 6)
        rows.append(row)
    return rows


def pairwise_tables(
    scores: dict[str, dict[str, list[float]]], tolerance: float = 1e-9
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    details = []
    summary = []
    aggregate: dict[str, dict[str, float]] = {
        label: {"wins": 0, "ties": 0, "losses": 0, "difference": 0.0}
        for label in SKILL_GUIDED
    }
    datasets = sorted(scores[next(iter(scores))])
    for skill_label in SKILL_GUIDED:
        for comparator in IMOPSE:
            wins = ties = losses = 0
            differences = []
            for dataset in datasets:
                skill_mean = statistics.mean(scores[skill_label][dataset])
                comparator_mean = statistics.mean(scores[comparator][dataset])
                difference = skill_mean - comparator_mean
                if difference > tolerance:
                    outcome = "win"
                    wins += 1
                elif difference < -tolerance:
                    outcome = "loss"
                    losses += 1
                else:
                    outcome = "tie"
                    ties += 1
                differences.append(difference)
                details.append(
                    {
                        "dataset": dataset,
                        "skillGuidedAlgorithm": skill_label,
                        "imopseAlgorithm": comparator,
                        "skillGuidedRuns": len(scores[skill_label][dataset]),
                        "imopseRuns": len(scores[comparator][dataset]),
                        "skillGuidedMeanTotalScore": round(skill_mean, 6),
                        "imopseMeanTotalScore": round(comparator_mean, 6),
                        "meanDifference": round(difference, 6),
                        "outcome": outcome,
                    }
                )
            summary.append(
                {
                    "skillGuidedAlgorithm": skill_label,
                    "imopseAlgorithm": comparator,
                    "datasets": len(datasets),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "winRate": round(wins / len(datasets), 6),
                    "nonLossRate": round((wins + ties) / len(datasets), 6),
                    "meanScoreDifference": round(statistics.mean(differences), 6),
                    "medianScoreDifference": round(statistics.median(differences), 6),
                }
            )
            aggregate[skill_label]["wins"] += wins
            aggregate[skill_label]["ties"] += ties
            aggregate[skill_label]["losses"] += losses
            aggregate[skill_label]["difference"] += sum(differences)
    aggregate_rows = []
    comparisons_per_skill = len(IMOPSE) * len(datasets)
    for skill_label, values in aggregate.items():
        aggregate_rows.append(
            {
                "skillGuidedAlgorithm": skill_label,
                "imopseComparisons": comparisons_per_skill,
                "wins": int(values["wins"]),
                "ties": int(values["ties"]),
                "losses": int(values["losses"]),
                "winRate": round(values["wins"] / comparisons_per_skill, 6),
                "nonLossRate": round((values["wins"] + values["ties"]) / comparisons_per_skill, 6),
                "meanScoreDifference": round(values["difference"] / comparisons_per_skill, 6),
            }
        )
    return details, summary, aggregate_rows


def compare_with_best_imopse(
    scores: dict[str, dict[str, list[float]]], tolerance: float = 1e-9
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    details = []
    counts = {label: {"wins": 0, "ties": 0, "losses": 0, "differences": []} for label in SKILL_GUIDED}
    for dataset in sorted(scores[next(iter(scores))]):
        imopse_means = {label: statistics.mean(scores[label][dataset]) for label in IMOPSE}
        best_imopse = max(imopse_means, key=imopse_means.get)
        best_mean = imopse_means[best_imopse]
        for skill_label in SKILL_GUIDED:
            skill_mean = statistics.mean(scores[skill_label][dataset])
            difference = skill_mean - best_mean
            if difference > tolerance:
                outcome = "win"
            elif difference < -tolerance:
                outcome = "loss"
            else:
                outcome = "tie"
            count_key = {"win": "wins", "tie": "ties", "loss": "losses"}[outcome]
            counts[skill_label][count_key] += 1
            counts[skill_label]["differences"].append(difference)
            details.append(
                {
                    "dataset": dataset,
                    "skillGuidedAlgorithm": skill_label,
                    "skillGuidedMeanTotalScore": round(skill_mean, 6),
                    "bestImopseAlgorithm": best_imopse,
                    "bestImopseMeanTotalScore": round(best_mean, 6),
                    "meanDifference": round(difference, 6),
                    "outcome": outcome,
                }
            )
    summary = []
    for skill_label, values in counts.items():
        total = len(values["differences"])
        summary.append(
            {
                "skillGuidedAlgorithm": skill_label,
                "datasets": total,
                "wins": values["wins"],
                "ties": values["ties"],
                "losses": values["losses"],
                "winRate": round(values["wins"] / total, 6),
                "meanScoreDifference": round(statistics.mean(values["differences"]), 6),
                "medianScoreDifference": round(statistics.median(values["differences"]), 6),
            }
        )
    return details, summary


def write_report(
    path: Path,
    pairwise_summary: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    best_summary: list[dict[str, Any]],
) -> None:
    lines = [
        "# Skill-Guided vs iMOPSE",
        "",
        "Comparison unit: mean TotalScore per dataset. Skill-Guided uses 30 runs per dataset; each iMOPSE configuration uses 10 runs. Seeds are not paired across the two frameworks.",
        "",
        "## Pairwise win/tie/loss",
        "",
        "| Skill-Guided | iMOPSE | W/T/L | Win rate | Mean difference |",
        "|---|---|---:|---:|---:|",
    ]
    for row in pairwise_summary:
        lines.append(
            f"| {row['skillGuidedAlgorithm']} | {row['imopseAlgorithm']} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {100 * row['winRate']:.2f}% | "
            f"{row['meanScoreDifference']:+.3f} |"
        )
    lines.extend(["", "## Aggregate over six iMOPSE configurations", ""])
    for row in aggregate:
        lines.append(
            f"- **{row['skillGuidedAlgorithm']}**: {row['wins']}/{row['imopseComparisons']} wins "
            f"({100 * row['winRate']:.2f}%), mean difference {row['meanScoreDifference']:+.3f}."
        )
    lines.extend(["", "## Against the best iMOPSE mean on each dataset", ""])
    for row in best_summary:
        lines.append(
            f"- **{row['skillGuidedAlgorithm']}**: {row['wins']}/{row['datasets']} wins, "
            f"{row['ties']} ties, {row['losses']} losses ({100 * row['winRate']:.2f}% win rate); "
            f"mean difference {row['meanScoreDifference']:+.3f}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    scores = load_scores()
    table = dataset_table(scores)
    details, summary, aggregate = pairwise_tables(scores)
    best_details, best_summary = compare_with_best_imopse(scores)
    write_rows(OUTPUT_DIR / "eight_algorithms_total_score.csv", table)
    write_rows(OUTPUT_DIR / "skill_guided_vs_imopse_details.csv", details)
    write_rows(OUTPUT_DIR / "skill_guided_vs_imopse_win_rates.csv", summary)
    write_rows(OUTPUT_DIR / "skill_guided_vs_imopse_overall.csv", aggregate)
    write_rows(OUTPUT_DIR / "skill_guided_vs_best_imopse_details.csv", best_details)
    write_rows(OUTPUT_DIR / "skill_guided_vs_best_imopse_summary.csv", best_summary)
    write_report(OUTPUT_DIR / "comparison_report.md", summary, aggregate, best_summary)
    print(f"wrote comparison for {len(table)} datasets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
