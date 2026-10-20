param(
    [string]$Python = 'D:\App\Anaconda3\python.exe',
    [switch]$SkipArima
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$invokeArgs = @((Join-Path $root 'run_delivery.py'), '--python', $Python)
if ($SkipArima) { $invokeArgs += '--skip-arima' }
& $Python @invokeArgs
if ($LASTEXITCODE -ne 0) { throw "delivery failed: $LASTEXITCODE" }
