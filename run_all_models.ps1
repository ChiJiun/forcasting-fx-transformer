$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "outputs\full_model_runs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$models = @(
    "transformer.py",
    "informer.py",
    "autoformer.py",
    "PatchTST.py"
)

$summaryPath = Join-Path $logDir "run_summary.csv"
"model,status,start_time,end_time,seconds,log_file" | Set-Content -Path $summaryPath -Encoding UTF8

foreach ($model in $models) {
    $start = Get-Date
    $modelName = [System.IO.Path]::GetFileNameWithoutExtension($model)
    $logFile = Join-Path $logDir "$modelName.log"
    Write-Host "[$($start.ToString('s'))] Starting $model"

    Push-Location $root
    try {
        & $python $model *> $logFile
        $status = "success"
    }
    catch {
        $status = "failed"
        Add-Content -Path $logFile -Value $_.Exception.ToString()
    }
    finally {
        Pop-Location
    }

    $end = Get-Date
    $seconds = [int]($end - $start).TotalSeconds
    Write-Host "[$($end.ToString('s'))] Finished $model with status $status in ${seconds}s"
    "$model,$status,$($start.ToString('s')),$($end.ToString('s')),$seconds,$logFile" | Add-Content -Path $summaryPath -Encoding UTF8

    if ($status -ne "success") {
        throw "$model failed. See $logFile"
    }
}
