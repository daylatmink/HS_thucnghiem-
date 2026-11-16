param(
    [Parameter(Mandatory = $true)]
    [string]$OptimizerPath,
    [string]$OutputRoot = "results/imopse_native",
    [string]$ManifestPath = "datasets/manifest.csv",
    [string[]]$DatasetNames = @(),
    [string[]]$Algorithms = @(),
    [int]$FirstSeed = 1,
    [int]$SeedCount = 10,
    [int]$MaxEvaluations = 1000,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$optimizer = (Resolve-Path $OptimizerPath).Path
$outputBase = Join-Path $repoRoot $OutputRoot
$manifest = (Resolve-Path (Join-Path $repoRoot $ManifestPath)).Path

$populationSize = 50
if ($MaxEvaluations -lt $populationSize -or $MaxEvaluations % $populationSize -ne 0) {
    throw "MaxEvaluations must be a positive multiple of $populationSize."
}
$generationLimit = [int]($MaxEvaluations / $populationSize) - 1

$algorithmConfigs = [ordered]@{
    BNTGA = Join-Path $PSScriptRoot "configs/BNTGA_MSRCPSP_1000NFE.cfg"
    MOEAD = Join-Path $PSScriptRoot "configs/MOEAD_MSRCPSP_1000NFE.cfg"
    NSGAII = Join-Path $PSScriptRoot "configs/NSGAII_MSRCPSP_1000NFE.cfg"
}
$requestedAlgorithms = @(
    $Algorithms |
        ForEach-Object { $_ -split ',' } |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
if ($requestedAlgorithms.Count -gt 0) {
    $unknownAlgorithms = @($requestedAlgorithms | Where-Object { -not $algorithmConfigs.Contains($_) })
    if ($unknownAlgorithms.Count -gt 0) {
        throw "Unsupported algorithms: $($unknownAlgorithms -join ', ')"
    }
    $selectedAlgorithmConfigs = [ordered]@{ }
    foreach ($name in $requestedAlgorithms) {
        $selectedAlgorithmConfigs[$name] = $algorithmConfigs[$name]
    }
    $algorithmConfigs = $selectedAlgorithmConfigs
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

foreach ($algorithm in $algorithmConfigs.Keys) {
    $baseConfig = (Resolve-Path $algorithmConfigs[$algorithm]).Path
    $generatedConfigDirectory = Join-Path $outputBase "_configs"
    New-Item -ItemType Directory -Force -Path $generatedConfigDirectory | Out-Null
    $config = Join-Path $generatedConfigDirectory "${algorithm}_${MaxEvaluations}NFE.cfg"
    $configText = Get-Content $baseConfig -Raw
    $configText = $configText -replace '(?m)^GenerationLimit\s+\d+\s*$', "GenerationLimit $generationLimit"
    Set-Content -Path $config -Value $configText -Encoding ascii
    foreach ($datasetRow in $datasets) {
        $dataset = $datasetRow.Dataset
        $datasetSource = $datasetRow.Source
        if ($datasetRow.DefPath) {
            $instance = (Resolve-Path (Join-Path $repoRoot $datasetRow.DefPath)).Path
            $instanceType = if ($datasetRow.InstanceType) { $datasetRow.InstanceType } else { "generated_def" }
        } else {
            $sourceRoot = $sourceRoots[$datasetSource]
            if (-not $sourceRoot) {
                throw "Unsupported dataset source '$datasetSource' for dataset '$dataset'"
            }
            $instance = (Resolve-Path (Join-Path $repoRoot "$sourceRoot/$dataset.def")).Path
            $instanceType = "official_def"
        }
        for ($seed = $FirstSeed; $seed -lt $FirstSeed + $SeedCount; $seed++) {
            $output = Join-Path $outputBase "$algorithm/$dataset/seed_$seed"
            New-Item -ItemType Directory -Force -Path $output | Out-Null
            $metadataPath = Join-Path $output "native_run_metadata.json"
            $pareto = Join-Path $output "run_0/pareto_solutions.csv"
            if (-not $Force -and (Test-Path $metadataPath) -and (Test-Path $pareto)) {
                $metadata = Get-Content $metadataPath -Raw | ConvertFrom-Json
                if ($metadata.objectiveEvaluations -eq $MaxEvaluations -and (Get-Item $pareto).Length -gt 0) {
                    Write-Host "skip completed algorithm=$algorithm dataset=$dataset seed=$seed"
                    continue
                }
            }
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
                datasetDirectory = $datasetRow.DatasetDirectory
                instanceType = $instanceType
                instancePath = $instance
                seed = $seed
                objectiveEvaluations = $MaxEvaluations
                runtimeSeconds = [Math]::Round($stopwatch.Elapsed.TotalSeconds, 6)
            } | ConvertTo-Json | Set-Content -Path $metadataPath -Encoding utf8
            if (-not (Test-Path $pareto)) {
                throw "Missing Pareto genotype output: $pareto. Apply export_pareto_genotypes.patch before building iMOPSE."
            }
        }
    }
}
