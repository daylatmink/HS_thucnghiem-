param(
    [switch]$SaveAssignments,
    [switch]$IncludeExternal,
    [string]$OptimizerPath,
    [switch]$Analyze,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($IncludeExternal -and (-not $OptimizerPath -or -not (Test-Path -LiteralPath $OptimizerPath -PathType Leaf))) {
    throw "IncludeExternal requires OptimizerPath pointing to the patched native iMOPSE executable."
}
if ($OptimizerPath) {
    $OptimizerPath = (Resolve-Path -LiteralPath $OptimizerPath).Path
}

Push-Location $repoRoot
try {
    $manifest = @(Import-Csv "datasets/manifest.csv")
    if ($manifest.Count -ne 16) {
        throw "Expected exactly 16 paper datasets."
    }
    $datasetNames = @($manifest | ForEach-Object { "msrcpsp_$($_.Dataset)" })
    foreach ($dataset in $datasetNames) {
        foreach ($file in @("tasks.csv", "resources.csv", "kpi-definitions.csv", "kpi-targets.csv", "cycle.csv")) {
            if (-not (Test-Path -LiteralPath "datasets/$dataset/$file")) {
                throw "Missing input: datasets/$dataset/$file"
            }
        }
    }
    $common = @(
        "src/benchmark_algorithms.py",
        "--datasets", ($datasetNames -join ','),
        "--seeds", "1..10", "--strategy", "BALANCED",
        "--max-evaluations", "5000", "--fixed-budget",
        "--stagnation-evaluations", "500", "--max-iterations", "100000",
        "--harmony-memory-size", "30", "--hmcr", "0.9", "--par", "0.25",
        "--population-size", "50", "--crossover-rate", "0.8",
        "--mutation-rate", "0.05", "--elitism-count", "2", "--tournament-size", "3",
        "--hybrid-hs-ratios", "0.03"
    )
    if ($SaveAssignments) {
        $common += "--save-assignments"
    }
    $runs = @(
        @{
            Algorithms = "ga,hybrid_hs_ga"
            Output = "results/final_16_baselines_stagnation500_5000nfe"
        },
        @{
            Algorithms = "skill_guided_ga,hybrid_hs_ga_guided_mutation"
            Output = "results/final_16_stagnation500_5000nfe/proposed"
        }
    )
    foreach ($run in $runs) {
        $commandArgs = $common + @("--algorithms", $run.Algorithms, "--output-dir", $run.Output)
        if ($DryRun) {
            Write-Output ("python " + ($commandArgs -join ' '))
        } else {
            & python @commandArgs
            if ($LASTEXITCODE -ne 0) {
                throw "Controlled benchmark failed for $($run.Algorithms)"
            }
        }
    }
    if ($IncludeExternal) {
        $nativeRoot = "results/final_16_stagnation500_5000nfe/imopse_native"
        $rescoredRoot = "results/final_16_stagnation500_5000nfe/imopse_rescored"
        if ($DryRun) {
            Write-Output "Native BNTGA/MOEAD/NSGAII: seeds 1..10, MaxEvaluations 5000, output $nativeRoot"
            Write-Output "Rescore native archives with BALANCED objective into $rescoredRoot"
        } else {
            & "tools/imopse/run_imopse_screening.ps1" -OptimizerPath $OptimizerPath -OutputRoot $nativeRoot -FirstSeed 1 -SeedCount 10 -MaxEvaluations 5000
            & python src/benchmark_imopse_solutions.py --solutions-root $nativeRoot --datasets-dir datasets --output-dir $rescoredRoot --strategy BALANCED --native-nfe 5000
            if ($LASTEXITCODE -ne 0) {
                throw "External KPI rescoring failed."
            }
        }
    }
    if ($Analyze) {
        if ($DryRun) {
            Write-Output "python src/analyze_paper_update.py"
        } else {
            & python src/analyze_paper_update.py
            if ($LASTEXITCODE -ne 0) {
                throw "Paper analysis failed."
            }
        }
    }
} finally {
    Pop-Location
}
