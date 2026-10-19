param(
    [string]$Python = 'D:\App\Anaconda3\python.exe',
    [switch]$SkipArima
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root
try {
    & $Python '05_扩展行业\check_environment.py'
    if ($LASTEXITCODE -ne 0) { throw "环境或输入检查失败，退出码: $LASTEXITCODE" }
    if ($SkipArima) {
        & $Python '05_扩展行业\run_original_extensions.py' --skip-arima
    } else {
        & $Python '05_扩展行业\run_original_extensions.py'
    }
    if ($LASTEXITCODE -ne 0) { throw "复跑失败，退出码: $LASTEXITCODE" }
} finally {
    Pop-Location
}
