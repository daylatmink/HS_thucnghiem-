# KPI-Aware MS-RCPSP Assignment Optimization

Reproduction code for the paper's controlled initialization/mutation experiment and external iMOPSE comparison.

## Compared Configurations

| Paper name | Algorithm identifier | Initialization | Mutation |
|---|---|---|---|
| R-RM | `ga` | Baseline GA initialization | Random |
| R-SG | `skill_guided_ga` | Same baseline GA initialization | Skill-guided |
| HS-RM | `hybrid_hs_ga` | HS warm-start | Random |
| HS-SG | `hybrid_hs_ga_guided_mutation` | Same HS warm-start | Skill-guided (skill-only) |

The HS kernel is retained as a dependency of both hybrids. Standalone HS is not one of the four controlled configurations. "Random initialization" is the paper label for the existing baseline GA initializer; no initialization operators are changed for publication.

All four configurations share the evaluator, scheduler, representation and constraint handling. The objective is unchanged:

```text
totalScore = 0.45 * kpiScore + 0.35 * timeScore + 0.20 * costScore - penalty
```

`timeScore` uses the scheduled assignment's actual makespan relative to the pre-optimization base makespan. Resource assignment may have a skill mismatch; the external native iMOPSE search uses stricter skill-feasible pools. Their comparison is therefore not an identical-search-space experiment.

## Paper Protocol

- Inputs: the 16 folders listed in `datasets/manifest.csv`, with fixed, pre-generated KPI definitions and task impacts shared by every method/seed.
- Independent seeds: 1 through 10 for each dataset/configuration.
- Maximum budget: 5,000 objective evaluations per run.
- Stopping: 500 NFE without an incumbent improvement of at least 0.001; realized NFE can differ between runs.
- Hybrid initialization: at most 150 HS NFE, using `--hybrid-hs-ratios 0.03`; GA uses the remaining global budget, without re-evaluating transferred HS candidates.
- GA: population 50, crossover 0.8, mutation 0.05, elitism 2, tournament size 3.
- HS: HMS 30, HMCR 0.9, PAR 0.25.
- Skill-guided mutation: epsilon `1e-6`, weakness weights skill/workload/timeline = `1/0/0`.

`--fixed-budget` disables the legacy generation/iteration stagnation stop. An explicit `--stagnation-evaluations 500` still applies. This is a common maximum-budget protocol, not 5,000 realized evaluations in every run. Old 1,000-NFE / 30-seed outputs are not the current paper experiment.

## Repository Contents

```text
src/                         Optimizers, shared evaluation/scheduling, analysis
src/tests/                   Tests for the published components
datasets/                    16 formatted inputs and their manifest
rawdata/                     The 16 corresponding original DEF inputs
template/                    CSV schema examples
scripts/run_paper_experiments.ps1  Explicit paper protocol
tools/imopse/                Native configs, Pareto export patch and adapter guide
```

Generated results, old variants, PTSP/MSLIB adaptations, presentation files, local plans, editor/session state and backups are excluded from version control. They can remain on the development machine. This source release does not bundle the historical raw experiment results.

## Install And Test

Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s src/tests -p "test_paper*.py"
python -m unittest discover -s src/tests -p "test_smoke.py"
python -m unittest discover -s src/tests -p "test_imopse_benchmark.py"
python tools/check_publication.py
```

The focused test commands also work in a development checkout containing ignored tests for unrelated adaptations. A clean clone includes only publication tests.

## Run The Controlled Experiment

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_paper_experiments.ps1
```

This runs the four configurations on exactly the 16 manifest datasets and seeds 1..10. Baseline and guided runs use the same explicit settings, and are saved separately to match the analysis script:

```text
results/final_16_baselines_stagnation500_5000nfe/
results/final_16_stagnation500_5000nfe/proposed/
```

Each directory contains raw per-seed results, summaries, convergence history and validation reports. To retain best assignments, pass `-SaveAssignments` to the script. The script launches experiments only when explicitly invoked; repository cleanup does not run the full benchmark.

For a cross-platform terminal or Colab, the same four configurations can be run in one command:

```bash
python src/benchmark_algorithms.py --algorithms ga,skill_guided_ga,hybrid_hs_ga,hybrid_hs_ga_guided_mutation --datasets msrcpsp_10_3_5_3,msrcpsp_10_5_8_5,msrcpsp_10_7_10_7,msrcpsp_15_3_5_3,msrcpsp_15_6_10_6,msrcpsp_100_5_20_9_D3,msrcpsp_100_10_27_9_D2,msrcpsp_100_10_512_10_A,msrcpsp_100_20_23_9_D1,msrcpsp_100_20_512_10_A,msrcpsp_200_10_135_9_D6,msrcpsp_200_20_150_9_D5,msrcpsp_200_20_512_10_A,msrcpsp_200_40_130_9_D4,msrcpsp_200_40_512_10_A,msrcpsp_500_20_512_10_A --seeds 1..10 --strategy BALANCED --max-evaluations 5000 --fixed-budget --stagnation-evaluations 500 --max-iterations 100000 --harmony-memory-size 30 --hmcr 0.9 --par 0.25 --population-size 50 --crossover-rate 0.8 --mutation-rate 0.05 --elitism-count 2 --tournament-size 3 --hybrid-hs-ratios 0.03 --output-dir results/paper_controlled
```

The cross-platform command puts all four methods in one output directory. The analysis script currently expects the two-directory layout produced by the PowerShell script; its input paths must be adjusted before analyzing a combined output directory.

## External Comparison And Analysis

See `tools/imopse/README.md` to build the native executable and run BNTGA, MOEA/D and NSGA-II with 5,000 native NFE and seeds 1..10. Blind and Pareto-rerank select from the same exported archive; native search and post-hoc KPI rescoring are accounted for separately. Do not describe this comparison as an identical objective or identical search space.

After both controlled result directories and the external rescoring results are available:

```powershell
python src/analyze_paper_update.py
```

This regenerates tables, paired Wilcoxon tests with Holm correction, component differences and convergence figures under `reports/paper_update/`. It requires real raw per-seed inputs; excluded historical files are not silently substituted. Parameter values in the script are explicit reproduction settings; historical raw rows did not encode every GA/HS setting in an immutable config manifest.
