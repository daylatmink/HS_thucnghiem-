# External iMOPSE Comparison

The adapter runs native BNTGA, MOEA/D and NSGA-II on the 16 original DEF inputs in the paper manifest. Native search optimizes makespan and cost, not the added KPI objective. Exported Pareto assignments are evaluated afterwards with this project's unchanged KPI-Cost-Time evaluator and scheduler.

## Build The Native Executable

Clone the official implementation outside this repository and apply the export patch before building:

```powershell
git clone https://github.com/imopse/iMOPSE.git C:/work/iMOPSE
git -C C:/work/iMOPSE apply "${PWD}/tools/imopse/export_pareto_genotypes.patch"
cmake -S C:/work/iMOPSE/optimizer -B C:/work/iMOPSE/build -G Ninja
cmake --build C:/work/iMOPSE/build --config Release
```

The patch exports `pareto_solutions.csv`; it does not change the search operators, native objective or archive handling. The external repository/build is not bundled here. Existing historical results do not record an immutable upstream commit hash; exact source-version equivalence should not be claimed without the corresponding build provenance.

## Paper Run

Run from this repository's root after building:

```powershell
powershell -ExecutionPolicy Bypass -File tools/imopse/run_imopse_screening.ps1 -OptimizerPath C:/work/iMOPSE/build/imopse.exe -ManifestPath datasets/manifest.csv -OutputRoot results/final_16_stagnation500_5000nfe/imopse_native -FirstSeed 1 -SeedCount 10 -MaxEvaluations 5000
python src/benchmark_imopse_solutions.py --solutions-root results/final_16_stagnation500_5000nfe/imopse_native --datasets-dir datasets --output-dir results/final_16_stagnation500_5000nfe/imopse_rescored --strategy BALANCED --native-nfe 5000
```

There are 16 datasets x 3 native methods x 10 seeds = 480 requested native runs. An invalid/missing exported genotype must be reported in validation, not silently replaced with another seed. In the historical results, one BNTGA export on the 500-task dataset was invalid, leaving nine valid runs for that dataset/method.

The template config filenames retain `1000NFE` for compatibility. The runner derives the actual `GenerationLimit` from `-MaxEvaluations`, so the paper command uses population 50 and 99 generations: `50 + 99 * 50 = 5000`. This is a native evaluation budget; it does not impose the Python optimizers' 500-NFE KPI-score stagnation rule.

The runner records the configured native NFE and native runtime in `native_run_metadata.json`. Post-hoc rescoring evaluations are separate and are not added to native search NFE.

## Selection Policies

- `blind`: select from the exported archive using only the equal-weight min-max normalized native makespan/cost criterion.
- `pareto_rerank`: rescore that same archive using KPI-Cost-Time and select the highest TotalScore.

Both policies share a native archive; they are not independent native runs. Native iMOPSE maintains assignments satisfying every required skill/level, whereas the Python optimizers may assign skill-mismatched resources. Therefore this is a post-hoc external comparison, not evidence from identical objectives, search spaces or stopping criteria.

Rescoring outputs `pareto_rescored.csv`, raw per-run policy results, summaries and validation reports. All generated output stays in the ignored `results/` tree. PTSP/MSLIB adaptation pipelines and extra screening instances are outside this paper release.
