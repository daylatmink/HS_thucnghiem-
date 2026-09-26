param(
    [Parameter(Mandatory = $true)]
    [string]$OptimizerPath,
    [string]$OutputRoot = "results/imopse_native",
    [string]$ManifestPath = "datasets/manifest.csv",
    [string[]]$DatasetNames = @(),
    [int]$FirstSeed = 1,
    [int]$SeedCount = 10
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$optimizer = (Resolve-Path $OptimizerPath).Path
$outputBase = Join-Path $repoRoot $OutputRoot
$manifest = (Resolve-Path (Join-Path $repoRoot $ManifestPath)).Path

$algorithms = [ordered]@{
    BNTGA = Join-Path $PSScriptRoot "configs/BNTGA_MSRCPSP_1000NFE.cfg"
    MOEAD = Join-Path $PSScriptRoot "configs/MOEAD_MSRCPSP_1000NFE.cfg"
    NSGAII = Join-Path $PSScriptRoot "configs/NSGAII_MSRCPSP_1000NFE.cfg"
}
$sourceRoots = @{
    "Small" = "rawdata/MSRCPSP/Small"
    "iMOPSE" = "rawdata/imopse"
    "GenRegular" = "rawdata/MSRCPSP/GenRegular"
    "GenBig" = "rawdata/MSRCPSP/GenBig"
}
$datasets = @(Import-Csv $manifest)
if ($DatasetNames.Count -gt 0) {
    $datasets = @($datasets | Where-Object { $DatasetNames -contains $_.Dataset })
}
if ($datasets.Count -eq 0) {
    throw "No datasets selected from manifest: $manifest"
}

foreach ($algorithm in $algorithms.Keys) {
    $config = (Resolve-Path $algorithms[$algorithm]).Path
    foreach ($datasetRow in $datasets) {
        $dataset = $datasetRow.Dataset
        $datasetSource = $datasetRow.Source
        $sourceRoot = $sourceRoots[$datasetSource]
        if (-not $sourceRoot) {
            throw "Unsupported dataset source '$datasetSource' for dataset '$dataset'"
        }
        $instance = (Resolve-Path (Join-Path $repoRoot "$sourceRoot/$dataset.def")).Path
        for ($seed = $FirstSeed; $seed -lt $FirstSeed + $SeedCount; $seed++) {
            $output = Join-Path $outputBase "$algorithm/$dataset/seed_$seed"
            New-Item -ItemType Directory -Force -Path $output | Out-Null
            Write-Host "run algorithm=$algorithm dataset=$dataset seed=$seed"
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            & $optimizer $config MSRCPSP_TA2 $instance $output 1 $seed
            $exitCode = $LASTEXITCODE
            $stopwatch.Stop()
            if ($exitCode -ne 0) {
                throw "iMOPSE failed: algorithm=$algorithm dataset=$dataset seed=$seed exit=$exitCode"
            }
            [ordered]@{
                algorithm = $algorithm
                dataset = $dataset
                datasetSource = $datasetSource
                instanceType = "official_def"
                instancePath = $instance
                seed = $seed
                objectiveEvaluations = 1000
                runtimeSeconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 6)
            } | ConvertTo-Json | Set-Content -Path (Join-Path $output "native_run_metadata.json") -Encoding utf8
            $pareto = Join-Path $output "run_0/pareto_solutions.csv"
            if (-not (Test-Path $pareto)) {
                throw "Missing Pareto genotype output: $pareto. Apply export_pareto_genotypes.patch before building iMOPSE."
            }
        }
    }
}
