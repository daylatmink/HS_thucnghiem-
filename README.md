# MS-RCPSP Assignment Optimization

This repository contains a Colab-ready implementation of resource assignment algorithms for MS-RCPSP-style datasets converted to CSV.

## Main Algorithms

The report focuses on four algorithms:

| CLI name | Description |
|---|---|
| `hs` | Harmony Search baseline |
| `ga` | Genetic Algorithm baseline |
| `hybrid_hs_ga` | Basic Hybrid HS150 -> GA850 |
| `hybrid_hs_ga_guided_mutation` | Skill-Guided HS150 -> GA850, TA-3 skill-only |

All algorithms use the same evaluator, scheduler, constraints, objective function, seed handling, and fixed objective-evaluation budget.

## Repository Layout

```text
src/                         Python source code
datasets/                    16 formatted MS-RCPSP datasets
results/final_benchmark/     Precomputed benchmark results
results/convergence_analysis/ Convergence tables and figures
report_assets/               Tables/figures used in the report
experiment_report.ipynb      Colab notebook
```

The legacy development copy remains under `algrorithm/HS/`.

## Objective

The current objective has three components:

```text
totalScore = w_kpi * kpiScore + w_time * timeScore + w_cost * costScore - penalty
```

Default `BALANCED` weights:

```text
kpi = 0.45
time = 0.35
cost = 0.20
```

`timeScore` is based on actual makespan after scheduling the candidate assignment.

## Run a Single Dataset

```bash
python src/csv_runner.py \
  --algorithm hybrid_hs_ga_guided_mutation \
  --tasks datasets/msrcpsp_10_3_5_3/tasks.csv \
  --resources datasets/msrcpsp_10_3_5_3/resources.csv \
  --kpi-definitions datasets/msrcpsp_10_3_5_3/kpi-definitions.csv \
  --kpi-targets datasets/msrcpsp_10_3_5_3/kpi-targets.csv \
  --cycle datasets/msrcpsp_10_3_5_3/cycle.csv \
  --max-evaluations 1000 \
  --fixed-budget \
  --harmony-memory-size 30 \
  --population-size 50 \
  --hybrid-hs-ratio 0.15 \
  --seed 1
```

## Run Full Benchmark

```bash
python src/benchmark_algorithms.py \
  --algorithms hs,ga,hybrid_hs_ga,hybrid_hs_ga_guided_mutation \
  --seeds 1..30 \
  --max-evaluations 1000 \
  --fixed-budget \
  --harmony-memory-size 30 \
  --population-size 50 \
  --hybrid-hs-ratios 0.15 \
  --output-dir results/benchmark_runs
```

Outputs:

- `raw_results.csv`
- `summary_results.csv`
- `convergence_history.csv`
- `validation_report.csv`
- `validation_summary.json`

## Precomputed Results

Main tables are available in:

```text
results/final_benchmark/main_4_algorithms_total_score_runtime_table.csv
results/final_benchmark/final_benchmark_summary.csv
results/final_benchmark/skill_guided_ta3_summary_results.csv
```

The Skill-Guided results must come from the TA-3 skill-only files:

```text
results/final_benchmark/skill_guided_ta3_*.csv
```

Do not use `results/combined_benchmark_with_guided_mutation/` for the final Skill-Guided comparison.

## Tests

```bash
python -m unittest discover -s src/tests
```

