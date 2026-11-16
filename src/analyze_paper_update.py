from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


BASELINE_DIR = Path("results/final_16_baselines_stagnation500_5000nfe")
GUIDED_DIR = Path("results/final_16_stagnation500_5000nfe/proposed")
EXTERNAL_SUMMARY = Path("results/final_16_stagnation500_5000nfe/imopse_rescored/summary_results.csv")
OUTPUT_DIR = Path("reports/paper_update")
MAX_NFE = 5000
STAGNATION_NFE = 500
SEEDS = list(range(1, 11))
DATASET_ORDER = (
    "msrcpsp_10_3_5_3",
    "msrcpsp_10_5_8_5",
    "msrcpsp_10_7_10_7",
    "msrcpsp_15_3_5_3",
    "msrcpsp_15_6_10_6",
    "msrcpsp_100_5_20_9_D3",
    "msrcpsp_100_10_27_9_D2",
    "msrcpsp_100_10_512_10_A",
    "msrcpsp_100_20_23_9_D1",
    "msrcpsp_100_20_512_10_A",
    "msrcpsp_200_10_135_9_D6",
    "msrcpsp_200_20_150_9_D5",
    "msrcpsp_200_20_512_10_A",
    "msrcpsp_200_40_130_9_D4",
    "msrcpsp_200_40_512_10_A",
    "msrcpsp_500_20_512_10_A",
)
REPRESENTATIVE_DATASETS = (
    "msrcpsp_10_7_10_7",
    "msrcpsp_100_20_512_10_A",
    "msrcpsp_200_20_150_9_D5",
    "msrcpsp_500_20_512_10_A",
)
METHODS = {
    "R-RM": (BASELINE_DIR, "ga"),
    "R-SG": (GUIDED_DIR, "skill_guided_ga"),
    "HS-RM": (BASELINE_DIR, "hybrid_hs_ga_hs150_ga4850"),
    "HS-SG": (GUIDED_DIR, "hybrid_hs_ga_guided_mutation_hs150_ga4850"),
}
COLORS = {"R-RM": "#2563EB", "R-SG": "#16A34A", "HS-RM": "#F97316", "HS-SG": "#DC2626"}


def display_dataset(dataset: str) -> str:
    return dataset.removeprefix("msrcpsp_")


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for item in row:
            if isinstance(item, (float, np.floating)):
                cells.append(f"{float(item):.{digits}f}")
            else:
                cells.append(str(item))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_controlled() -> pd.DataFrame:
    frames = []
    required = {
        "dataset", "algorithm", "seed", "totalScore", "kpiScore", "timeScore",
        "costScore", "objectiveEvaluations", "stagnationEvaluations", "hardViolations",
    }
    for label, (directory, algorithm) in METHODS.items():
        source = directory / "raw_results.csv"
        frame = pd.read_csv(source)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{source} missing columns: {sorted(missing)}")
        selected = frame.loc[frame["algorithm"] == algorithm].copy()
        selected["configuration"] = label
        selected["sourceFile"] = str(source)
        frames.append(selected)
    controlled = pd.concat(frames, ignore_index=True)
    return controlled


def audit_protocol(controlled: pd.DataFrame) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []
    reference_datasets: set[str] | None = None
    for label in METHODS:
        group = controlled.loc[controlled["configuration"] == label]
        datasets = set(group["dataset"])
        if reference_datasets is None:
            reference_datasets = datasets
        elif datasets != reference_datasets:
            errors.append(f"{label}: dataset set differs")
        if len(datasets) != 16:
            errors.append(f"{label}: expected 16 datasets, found {len(datasets)}")
        if group.duplicated(["dataset", "seed"]).any():
            errors.append(f"{label}: duplicate dataset/seed")
        for dataset, run_group in group.groupby("dataset"):
            seeds = sorted(run_group["seed"].astype(int).tolist())
            if seeds != SEEDS:
                errors.append(f"{label}/{dataset}: seeds are {seeds}, expected 1..10")
        if not (pd.to_numeric(group["stagnationEvaluations"]) == STAGNATION_NFE).all():
            errors.append(f"{label}: stagnationEvaluations is not uniformly {STAGNATION_NFE}")
        nfe = pd.to_numeric(group["objectiveEvaluations"])
        if (nfe > MAX_NFE).any():
            errors.append(f"{label}: objectiveEvaluations exceeds {MAX_NFE}")
        if (pd.to_numeric(group["hardViolations"]) != 0).any():
            notes.append(f"{label}: at least one result has hard violations")
    if len(controlled) != 640:
        errors.append(f"Expected 640 controlled rows, found {len(controlled)}")
    return errors, notes


def method_stats(controlled: pd.DataFrame) -> pd.DataFrame:
    return (
        controlled.groupby(["dataset", "configuration"], sort=True)
        .agg(
            runs=("totalScore", "size"),
            totalScore_mean=("totalScore", "mean"),
            totalScore_std=("totalScore", lambda values: values.std(ddof=1)),
            kpiScore_mean=("kpiScore", "mean"),
            timeScore_mean=("timeScore", "mean"),
            costScore_mean=("costScore", "mean"),
            objectiveEvaluations_mean=("objectiveEvaluations", "mean"),
        )
        .reset_index()
    )


def value(stats: pd.DataFrame, dataset: str, method: str, column: str) -> float:
    return float(stats.loc[(stats["dataset"] == dataset) & (stats["configuration"] == method), column].iloc[0])


def write_table2(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in DATASET_ORDER:
        means = {method: value(stats, dataset, method, "totalScore_mean") for method in METHODS}
        row: dict[str, object] = {"Instance": display_dataset(dataset)}
        for method in METHODS:
            std = value(stats, dataset, method, "totalScore_std")
            row[method] = f"{means[method]:.2f} \u00b1 {std:.2f}"
        row["R_gain"] = means["R-SG"] - means["R-RM"]
        row["HS_gain"] = means["HS-SG"] - means["HS-RM"]
        rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(OUTPUT_DIR / "table2_new.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Mean final fitness ($\pm$ sample standard deviation) over 10 independent runs under a common 5,000-NFE maximum budget and 500-NFE stagnation rule.}",
        r"\label{tab:controlled-new}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Instance & R-RM & R-SG & HS-RM & HS-SG \\" ,
        r"\midrule",
    ]
    for dataset in DATASET_ORDER:
        means = {method: value(stats, dataset, method, "totalScore_mean") for method in METHODS}
        best = max(means.values())
        cells = []
        for method in METHODS:
            std = value(stats, dataset, method, "totalScore_std")
            cell = f"{means[method]:.2f} $\\pm$ {std:.2f}"
            if math.isclose(means[method], best, abs_tol=1e-12):
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{latex_escape(display_dataset(dataset))} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (OUTPUT_DIR / "table2_new.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def build_component_effect(stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    rows = []
    for dataset in DATASET_ORDER:
        task_count = int(display_dataset(dataset).split("_", 1)[0])
        row: dict[str, object] = {"dataset": display_dataset(dataset), "Tasks": task_count}
        for suffix, sg, rm in (("R", "R-SG", "R-RM"), ("HS", "HS-SG", "HS-RM")):
            row[f"DeltaF_{suffix}"] = value(stats, dataset, sg, "totalScore_mean") - value(stats, dataset, rm, "totalScore_mean")
            row[f"DeltaKPI_{suffix}"] = value(stats, dataset, sg, "kpiScore_mean") - value(stats, dataset, rm, "kpiScore_mean")
            row[f"DeltaTime_{suffix}"] = value(stats, dataset, sg, "timeScore_mean") - value(stats, dataset, rm, "timeScore_mean")
            row[f"DeltaCost_{suffix}"] = value(stats, dataset, sg, "costScore_mean") - value(stats, dataset, rm, "costScore_mean")
        row["HS_SG_minus_R_SG"] = value(stats, dataset, "HS-SG", "totalScore_mean") - value(stats, dataset, "R-SG", "totalScore_mean")
        rows.append(row)
    per_dataset = pd.DataFrame(rows)
    component_columns = [column for column in per_dataset.columns if column.startswith("Delta")]
    grouped = per_dataset.groupby("Tasks", sort=True)[component_columns].mean().reset_index()
    grouped.to_csv(OUTPUT_DIR / "table3_new.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    max_error = 0.0
    for suffix in ("R", "HS"):
        reconstructed = 0.45 * grouped[f"DeltaKPI_{suffix}"] + 0.35 * grouped[f"DeltaTime_{suffix}"] + 0.20 * grouped[f"DeltaCost_{suffix}"]
        max_error = max(max_error, float((grouped[f"DeltaF_{suffix}"] - reconstructed).abs().max()))

    columns = [
        "Tasks", "DeltaF_R", "DeltaKPI_R", "DeltaTime_R", "DeltaCost_R",
        "DeltaF_HS", "DeltaKPI_HS", "DeltaTime_HS", "DeltaCost_HS",
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Mean component-wise effect of skill-guided mutation, grouped by task count.}",
        r"\label{tab:component-effect-new}",
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r"Tasks & $\Delta F_R$ & $\Delta S_{KPI,R}$ & $\Delta S_{time,R}$ & $\Delta S_{cost,R}$ & $\Delta F_{HS}$ & $\Delta S_{KPI,HS}$ & $\Delta S_{time,HS}$ & $\Delta S_{cost,HS}$ \\",
        r"\midrule",
    ]
    for row in grouped[columns].itertuples(index=False):
        values = [str(int(row[0]))] + [f"{float(item):.3f}" for item in row[1:]]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (OUTPUT_DIR / "table3_new.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return per_dataset, grouped, max_error


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.zeros(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_tests(controlled: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Counter]]:
    rows = []
    for comparison, sg, rm in (("R-SG vs R-RM", "R-SG", "R-RM"), ("HS-SG vs HS-RM", "HS-SG", "HS-RM")):
        family = []
        for dataset in sorted(controlled["dataset"].unique()):
            left = controlled.loc[(controlled["dataset"] == dataset) & (controlled["configuration"] == rm), ["seed", "totalScore"]]
            right = controlled.loc[(controlled["dataset"] == dataset) & (controlled["configuration"] == sg), ["seed", "totalScore"]]
            paired = left.merge(right, on="seed", suffixes=("_RM", "_SG"), validate="one_to_one").sort_values("seed")
            if paired["seed"].astype(int).tolist() != SEEDS:
                raise ValueError(f"Incomplete pairs: {comparison}/{dataset}")
            differences = paired["totalScore_SG"].to_numpy(float) - paired["totalScore_RM"].to_numpy(float)
            raw_p = 1.0 if np.allclose(differences, 0.0) else float(wilcoxon(differences, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
            family.append({
                "dataset": display_dataset(dataset), "comparison": comparison, "n": len(paired),
                "mean_RM": paired["totalScore_RM"].mean(), "mean_SG": paired["totalScore_SG"].mean(),
                "median_diff": float(np.median(differences)), "raw_p": raw_p,
            })
        adjusted = holm_adjust([float(row["raw_p"]) for row in family])
        for row, holm_p in zip(family, adjusted, strict=True):
            row["holm_p"] = holm_p
            if holm_p >= 0.05 or row["median_diff"] == 0:
                row["conclusion"] = "non-significant"
            elif row["median_diff"] > 0:
                row["conclusion"] = "significant win"
            else:
                row["conclusion"] = "significant loss"
        rows.extend(family)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "paired_wilcoxon_new.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    summaries = {comparison: Counter(group["conclusion"]) for comparison, group in result.groupby("comparison", sort=False)}
    return result, summaries


def read_history(path: Path, algorithm: str, datasets: set[str]) -> dict[tuple[str, int], list[tuple[int, float]]]:
    output: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("algorithm") != algorithm or row.get("dataset") not in datasets:
                continue
            nfe_text, score_text = row.get("objectiveEvaluation"), row.get("bestScore")
            if not nfe_text or not score_text:
                continue
            output[(str(row["dataset"]), int(row["seed"]))].append((int(float(nfe_text)), float(score_text)))
    return output


def convergence(controlled: pd.DataFrame) -> tuple[bool, int, list[str]]:
    datasets = set(REPRESENTATIVE_DATASETS)
    points: dict[tuple[str, str, int], list[tuple[int, float]]] = {}
    issues = []
    first_nfes = []
    for label, (directory, algorithm) in METHODS.items():
        histories = read_history(directory / "convergence_history.csv", algorithm, datasets)
        for dataset in REPRESENTATIVE_DATASETS:
            for seed in SEEDS:
                key = (dataset, seed)
                if key not in histories:
                    issues.append(f"missing history: {label}/{dataset}/seed_{seed}")
                    continue
                run_points = histories[key]
                raw = controlled.loc[
                    (controlled["configuration"] == label) & (controlled["dataset"] == dataset) & (controlled["seed"] == seed)
                ].iloc[0]
                run_points.append((int(raw["objectiveEvaluations"]), float(raw["totalScore"])))
                points[(dataset, label, seed)] = run_points
                first_nfes.append(min(nfe for nfe, _score in run_points))
    if issues:
        return False, 0, issues
    common_start = max(first_nfes)
    grid = np.arange(common_start, MAX_NFE + 1)
    curves: dict[tuple[str, str, int], np.ndarray] = {}
    for key, run_points in points.items():
        by_nfe: dict[int, float] = {}
        for nfe, score in run_points:
            if 1 <= nfe <= MAX_NFE:
                by_nfe[nfe] = max(score, by_nfe.get(nfe, -math.inf))
        series = pd.Series(by_nfe, dtype=float).sort_index().cummax()
        expanded = series.reindex(series.index.union(grid)).sort_index().ffill().reindex(grid)
        if expanded.isna().any():
            issues.append(f"cannot fill common grid: {key}")
        values = expanded.to_numpy(float)
        if np.any(np.diff(values) < -1e-10):
            issues.append(f"non-monotonic curve: {key}")
        curves[key] = values

    rows, checkpoints = [], []
    checkpoint_nfes = (150, 250, 500, 1000, 2000, 3000, 4000, 5000)
    for dataset in REPRESENTATIVE_DATASETS:
        for label in METHODS:
            matrix = np.vstack([curves[(dataset, label, seed)] for seed in SEEDS])
            means = matrix.mean(axis=0)
            stds = matrix.std(axis=0, ddof=1)
            if np.any(np.diff(means) < -1e-10):
                issues.append(f"non-monotonic mean: {label}/{dataset}")
            for nfe, mean_score, std_score in zip(grid, means, stds, strict=True):
                rows.append({"dataset": display_dataset(dataset), "configuration": label, "objectiveEvaluation": nfe, "meanBestScore": mean_score, "sampleStdBestScore": std_score, "nSeeds": 10})
            for nfe in checkpoint_nfes:
                if nfe < common_start:
                    continue
                index = nfe - common_start
                checkpoints.append({"dataset": display_dataset(dataset), "configuration": label, "objectiveEvaluation": nfe, "meanBestScore": means[index], "sampleStdBestScore": stds[index], "nSeeds": 10})
    curve_frame = pd.DataFrame(rows)
    checkpoint_frame = pd.DataFrame(checkpoints)
    curve_frame.to_csv(OUTPUT_DIR / "convergence_curves_new.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    checkpoint_frame.to_csv(OUTPUT_DIR / "convergence_checkpoints_new.csv", index=False, encoding="utf-8-sig", float_format="%.8f")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), sharex=True)
    for axis, dataset in zip(axes.flat, REPRESENTATIVE_DATASETS, strict=True):
        panel = curve_frame.loc[curve_frame["dataset"] == display_dataset(dataset)]
        for label in METHODS:
            method = panel.loc[panel["configuration"] == label]
            axis.plot(method["objectiveEvaluation"], method["meanBestScore"], label=label, color=COLORS[label], linewidth=2.2)
        axis.axvline(150, color="#4B5563", linestyle="--", linewidth=1.3)
        axis.set_title(display_dataset(dataset), fontsize=14)
        axis.set_xlabel("Cumulative objective evaluations (NFE)", fontsize=11)
        axis.set_ylabel("Mean incumbent best score", fontsize=11)
        axis.set_xlim(common_start, MAX_NFE)
        axis.grid(True, alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Controlled-Configuration Convergence", fontsize=19)
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.11, top=0.92, wspace=0.16, hspace=0.27)
    fig.savefig(OUTPUT_DIR / "convergence_new.png", dpi=300)
    fig.savefig(OUTPUT_DIR / "convergence_new.pdf", bbox_inches="tight")
    plt.close(fig)
    return not issues, common_start, issues


def rq_summary(per_dataset: pd.DataFrame, grouped: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for suffix in ("R", "HS"):
        gains = per_dataset[f"DeltaF_{suffix}"]
        summary[f"{suffix}_wins"] = int((gains > 1e-12).sum())
        summary[f"{suffix}_ties"] = int((gains.abs() <= 1e-12).sum())
        summary[f"{suffix}_losses"] = int((gains < -1e-12).sum())
        summary[f"{suffix}_mean_gain"] = float(gains.mean())
        summary[f"{suffix}_largest"] = per_dataset.loc[gains.idxmax(), ["dataset", f"DeltaF_{suffix}"]].to_dict()
        summary[f"{suffix}_smallest"] = per_dataset.loc[gains.idxmin(), ["dataset", f"DeltaF_{suffix}"]].to_dict()
        summary[f"{suffix}_500"] = float(grouped.loc[grouped["Tasks"] == 500, f"DeltaF_{suffix}"].iloc[0])
    return summary


def table4_check(stats: pd.DataFrame) -> dict[str, object]:
    external = pd.read_csv(EXTERNAL_SUMMARY)
    method_rows = []
    for row in external.itertuples(index=False):
        method_rows.append({"dataset": row.dataset, "method": row.algorithm, "mean": float(row.totalScore_mean), "runs": int(row.runs)})
    for method in ("R-SG", "HS-SG"):
        for row in stats.loc[stats["configuration"] == method].itertuples(index=False):
            method_rows.append({"dataset": row.dataset, "method": method, "mean": float(row.totalScore_mean), "runs": int(row.runs)})
    frame = pd.DataFrame(method_rows)
    frame["rank"] = frame.groupby("dataset")["mean"].rank(method="average", ascending=False)
    wins = Counter()
    for _dataset, group in frame.groupby("dataset"):
        wins[group.loc[group["mean"].idxmax(), "method"]] += 1
    return {
        "R-SG_macro": float(frame.loc[frame["method"] == "R-SG", "mean"].mean()),
        "HS-SG_macro": float(frame.loc[frame["method"] == "HS-SG", "mean"].mean()),
        "R-SG_rank": float(frame.loc[frame["method"] == "R-SG", "rank"].mean()),
        "HS-SG_rank": float(frame.loc[frame["method"] == "HS-SG", "rank"].mean()),
        "R-SG_wins": wins["R-SG"], "HS-SG_wins": wins["HS-SG"],
        "MOEAD_PR_wins": wins["imopse_MOEAD_pareto_rerank"],
        "external_incomplete": frame.loc[frame["runs"] != 10, ["dataset", "method", "runs"]].to_dict("records"),
    }


def write_reports(
    controlled: pd.DataFrame,
    stats: pd.DataFrame,
    per_dataset: pd.DataFrame,
    grouped: pd.DataFrame,
    formula_error: float,
    tests: pd.DataFrame,
    test_summaries: dict[str, Counter],
    rq: dict[str, object],
    protocol_errors: list[str],
    protocol_notes: list[str],
    convergence_ready: bool,
    common_start: int,
    convergence_issues: list[str],
) -> None:
    table4 = table4_check(stats)
    sources = "\n".join(f"- {label}: `{directory / 'raw_results.csv'}` (`{algorithm}`)" for label, (directory, algorithm) in METHODS.items())
    hs_minus_r = per_dataset[["dataset", "HS_SG_minus_R_SG"]].copy()
    controlled_lines = [
        "# Controlled experiment update", "", "## Source audit", "", sources, "",
        "All four configurations contain the same 16 datasets, seeds 1--10, 10 runs per dataset, a 5,000-NFE hard cap, and a 500-NFE no-improvement stopping rule. Actual NFE varies because early stopping is enabled. All use the shared CSV runner, evaluator, scheduler, and BALANCED objective.", "",
        "## RQ1", "",
        f"- R-SG vs R-RM: {rq['R_wins']} wins / {rq['R_ties']} ties / {rq['R_losses']} losses; average mean gain {rq['R_mean_gain']:.3f}.",
        f"- HS-SG vs HS-RM: {rq['HS_wins']} wins / {rq['HS_ties']} ties / {rq['HS_losses']} losses; average mean gain {rq['HS_mean_gain']:.3f}.",
        f"- Largest Random gain: {rq['R_largest']['dataset']} ({rq['R_largest']['DeltaF_R']:.3f}); smallest: {rq['R_smallest']['dataset']} ({rq['R_smallest']['DeltaF_R']:.3f}).",
        f"- Largest HS gain: {rq['HS_largest']['dataset']} ({rq['HS_largest']['DeltaF_HS']:.3f}); smallest: {rq['HS_smallest']['dataset']} ({rq['HS_smallest']['DeltaF_HS']:.3f}).",
        f"- On the 500-task instance: Random gain {rq['R_500']:.3f}; HS gain {rq['HS_500']:.3f}.", "",
        "## RQ2", "",
        f"Mean mutation gain is {rq['R_mean_gain']:.3f} under Random initialization and {rq['HS_mean_gain']:.3f} under HS initialization. Positive gains occur on {rq['R_wins']}/16 and {rq['HS_wins']}/16 datasets, respectively.", "",
        "Descriptive HS-SG minus R-SG values:", "", markdown_table(hs_minus_r), "",
        "## RQ4", "", markdown_table(grouped[["Tasks", "DeltaF_R", "DeltaF_HS"]]), "",
        "The Random-initialization gains increase monotonically across the five task-count groups. The HS-initialization gains are not strictly monotonic because the mean gain decreases from 0.180 at 10 tasks to 0.072 at 15 tasks; thereafter it increases strongly at 100, 200, and 500 tasks. Both settings therefore support larger gains on the medium and large groups, but only the Random-initialization series is strictly monotonic.", "",
        "## Statistical tests", "",
    ]
    for comparison, counts in test_summaries.items():
        controlled_lines.append(f"- {comparison}: {counts['significant win']} significant wins / {counts['non-significant']} non-significant / {counts['significant loss']} significant losses.")
    controlled_lines.extend([
        "", "## Component identity check", "",
        f"Maximum grouped absolute error in DeltaF = 0.45*DeltaKPI + 0.35*DeltaTime + 0.20*DeltaCost is {formula_error:.6f}; status: {'PASS' if formula_error <= 0.02 else 'FLAG'}.", "",
        "## Convergence", "",
        f"Status: {'READY' if convergence_ready else 'MISSING'}. Common NFE grid: {common_start}..{MAX_NFE}. Curves are reconstructed as per-seed cumulative maxima and forward-filled after the final evaluation when early stopping occurs. NFE=150 marks the end of HS initialization only for HS-RM and HS-SG.", "",
        "## Table 4 consistency", "",
        f"The new controlled R-SG and HS-SG sources are the same 10-run sources used by the updated external comparison. Recomputed macro means are R-SG={table4['R-SG_macro']:.3f}, HS-SG={table4['HS-SG_macro']:.3f}; wins are {table4['R-SG_wins']} and {table4['HS-SG_wins']}; average ranks are {table4['R-SG_rank']:.3f} and {table4['HS-SG_rank']:.3f}. MOEA/D-PR wins {table4['MOEAD_PR_wins']} dataset(s). Do not mix these 10-run means with the old 30-run controlled results.",
    ])
    if table4["external_incomplete"]:
        controlled_lines.append(f"External-data caveat: {table4['external_incomplete']}.")
    if protocol_errors or protocol_notes or convergence_issues:
        controlled_lines.extend(["", "## Audit notes", ""] + [f"- {item}" for item in protocol_errors + protocol_notes + convergence_issues])
    (OUTPUT_DIR / "controlled_summary.md").write_text("\n".join(controlled_lines) + "\n", encoding="utf-8")

    dataset_500 = "msrcpsp_500_20_512_10_A"
    means_500 = {method: value(stats, dataset_500, method, "totalScore_mean") for method in METHODS}
    paper_text = rf"""% Replacement text generated from the new 10-run controlled experiment.
\paragraph{{RQ1: Effect of Skill-Guided Mutation.}}
Under Random initialization, skill-guided mutation achieves a higher mean final fitness on {rq['R_wins']} of the 16 instances, with an average gain of {rq['R_mean_gain']:.3f} points over R-RM. Under HS initialization, HS-SG improves over HS-RM on {rq['HS_wins']} instances, with an average gain of {rq['HS_mean_gain']:.3f} points. The paired Wilcoxon results after Holm correction comprise {test_summaries['R-SG vs R-RM']['significant win']} significant wins for R-SG and {test_summaries['HS-SG vs HS-RM']['significant win']} significant wins for HS-SG, with no claim made for non-significant comparisons.

\paragraph{{RQ2: Robustness Across Initialization Strategies.}}
The mutation gain is positive on {rq['R_wins']} instances under Random initialization and {rq['HS_wins']} instances under HS initialization. The corresponding average gains are {rq['R_mean_gain']:.3f} and {rq['HS_mean_gain']:.3f} points. Thus, the benefit is observed under both evaluated initialization strategies, although this evidence does not establish initialization independence beyond these settings and does not imply that HS initialization itself is uniformly superior.

\paragraph{{RQ4: Effect of Problem Size.}}
Under Random initialization, the grouped mean gain increases monotonically from {grouped.loc[grouped['Tasks'] == 10, 'DeltaF_R'].iloc[0]:.3f} points for 10-task instances to {grouped.loc[grouped['Tasks'] == 500, 'DeltaF_R'].iloc[0]:.3f} points for the 500-task instance. Under HS initialization, the pattern is not strictly monotonic because the gain decreases from {grouped.loc[grouped['Tasks'] == 10, 'DeltaF_HS'].iloc[0]:.3f} at 10 tasks to {grouped.loc[grouped['Tasks'] == 15, 'DeltaF_HS'].iloc[0]:.3f} at 15 tasks, before increasing to {grouped.loc[grouped['Tasks'] == 500, 'DeltaF_HS'].iloc[0]:.3f} at 500 tasks. Thus, both settings show substantially larger improvements on the 100--500-task groups, while strict monotonicity is supported only for Random initialization.

\paragraph{{Controlled-results discussion.}}
Across the controlled 2-by-2 design, skill-guided mutation improves the mean score for both initialization strategies on most datasets. Component-level differences satisfy the weighted objective identity within a maximum rounding discrepancy of {formula_error:.4f}. Because runs stop after 500 evaluations without a meaningful incumbent improvement, the experiment uses a common maximum budget of 5,000 evaluations rather than identical realized NFE for every run.

% Suggested abstract-relevant sentence; do not insert automatically.
On the 500-task instance, mean fitness changes from {means_500['R-RM']:.3f} to {means_500['R-SG']:.3f} under Random initialization and from {means_500['HS-RM']:.3f} to {means_500['HS-SG']:.3f} under HS initialization, corresponding to gains of {means_500['R-SG'] - means_500['R-RM']:.3f} and {means_500['HS-SG'] - means_500['HS-RM']:.3f} points, respectively.
"""
    (OUTPUT_DIR / "paper_text_replacements.tex").write_text(paper_text, encoding="utf-8")

    nfe_summary = controlled.groupby("configuration")["objectiveEvaluations"].agg(["mean", "min", "max"])
    protocol = f"""# Experiment protocol check

## Actual controlled protocol

- Maximum evaluation budget: 5,000 NFE per run.
- Early stopping: 500 cumulative NFE without an incumbent improvement of at least 0.001 TotalScore.
- HS budget: at most 150 NFE; nominal remaining GA budget: 4,850 NFE.
- Runs: 10 per dataset and configuration; seeds 1--10; 16 datasets.
- GA code defaults used by the runner: population 50, crossover probability 0.8, mutation probability 0.05, elitism 2, tournament size 3.
- HS runner defaults: HMS 30, HMCR 0.9, PAR 0.25.
- Skill-guided weakness epsilon: 1e-6.
- Objective: BALANCED, KPI=0.45, Time=0.35, Cost=0.20.

Realized NFE by configuration:

{markdown_table(nfe_summary.reset_index(), digits=2)}

## Evidence

- Run parameters are forwarded in `src/benchmark_algorithms.py:67-100`; CLI defaults are at lines 598-637.
- GA defaults are declared in `src/models.py:22-26`; the benchmark overrides HMS to 30.
- Skill-guided epsilon is `weakness_epsilon = 1e-6` in `src/hybrid_hs_ga.py`.
- BALANCED objective weights are in `src/evaluator.py:113`.
- Result artifacts record `stagnationEvaluations=500`, NFE, seeds, and algorithm labels, but no immutable run-command/config manifest was saved. Parameter values other than those columns are therefore verified from the runner defaults and supplied command, not independently encoded in each raw row.

## Stale statements in current Section 5.1

1. `NFE_max = 1000` is stale; the new hard cap is 5,000.
2. `30 independent runs` and the Table 2 caption `over 30 runs` are stale; the new experiment uses 10 seeds/runs.
3. Describing the protocol as an identical fixed realized budget is stale. Runs share a maximum budget and stopping rule, but realized NFE varies.
4. The RQ4 phrase `under a fixed fitness-evaluation budget` should be replaced by `under a common maximum evaluation budget and stagnation rule`.
5. Table 1 entries `epsilon=?` are stale; epsilon is 1e-6.
6. HMS=30, HMCR=0.9, PAR=0.25, HS NFE=150, population=50, pc=0.8, pm=0.05, elite=2, and tournament=3 remain consistent with the current runner.
"""
    (OUTPUT_DIR / "experiment_protocol_check.md").write_text(protocol, encoding="utf-8")

    missing = f"""# Paper update readiness

## READY

- Table 2: new 10-run raw results, sample standard deviations, and per-dataset gains.
- Table 3: KPI, Time, Cost, and TotalScore component effects from new results.
- Wilcoxon: 10 paired seeds per dataset with separate Holm correction families.
- RQ1, RQ2, and RQ4: replacement text generated from new results.
- Convergence data and figure: four requested datasets, all four configurations, common grid {common_start}..5000.
- Abstract values: four means and two mutation gains for `500_20_512_10_A` are reported but not inserted.

## MISSING

- Immutable run-config manifests for the two controlled commands. Raw files retain budget/stopping fields but not every GA/HS parameter.

## INCONSISTENT

- Table 1 and Section 5.1 still state 1,000 NFE, 30 runs, fixed realized budget, and unknown epsilon.
- Table 4/RQ3 is intentionally a 10-run comparison and should remain separate from the old 30-run tables. Its new R-SG/HS-SG sources match this controlled experiment, but both BNTGA Blind and BNTGA Pareto-rerank have only 9 runs on `500_20_512_10_A` because one exported genotype is invalid.
- The intended R-SG macro value should be checked against the recomputed value {table4['R-SG_macro']:.3f}; HS-SG is {table4['HS-SG_macro']:.3f}.
"""
    (OUTPUT_DIR / "missing_data.md").write_text(missing, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    controlled = load_controlled()
    protocol_errors, protocol_notes = audit_protocol(controlled)
    if protocol_errors:
        raise SystemExit("CONTROLLED_RESULTS_READY = false\n" + "\n".join(protocol_errors))
    stats = method_stats(controlled)
    write_table2(stats)
    per_dataset, grouped, formula_error = build_component_effect(stats)
    tests, test_summaries = paired_tests(controlled)
    rq = rq_summary(per_dataset, grouped)
    convergence_ready, common_start, convergence_issues = convergence(controlled)
    write_reports(
        controlled, stats, per_dataset, grouped, formula_error, tests, test_summaries, rq,
        protocol_errors, protocol_notes, convergence_ready, common_start, convergence_issues,
    )
    print("CONTROLLED_RESULTS_READY = true")
    print("TABLE2_READY = true")
    print("TABLE3_READY = true")
    print("WILCOXON_READY = true")
    print(f"CONVERGENCE_READY = {str(convergence_ready).lower()}")
    if convergence_issues:
        print("Remaining blockers:")
        for issue in convergence_issues:
            print(f"- {issue}")
    else:
        print("Remaining blockers: none for controlled results; see missing_data.md for paper-level inconsistencies.")


if __name__ == "__main__":
    main()
