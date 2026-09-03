# scripts/run_sweep.ps1 [-Dir configs/ade20k] [-Filter substring] [-OutDir results] [-Force]
param(
  [string]$Dir = "configs/ade20k",
  [string]$Filter = "",
  [string]$OutDir = "results",
  [switch]$Force
)
$ErrorActionPreference = "Continue"
$failed = @()
Get-ChildItem $Dir -Filter *.yaml | Where-Object { $_.Name -like "*$Filter*" } | ForEach-Object {
  $id = $_.BaseName
  if ((Test-Path "$OutDir/$id/metrics.json") -and -not $Force) { Write-Host "skip $id"; return }
  Write-Host "=== $id"
  $pyArgs = @("-m", "ttr.run", "--config", $_.FullName, "out_dir=$OutDir")
  if ($Force) { $pyArgs += "--force" }
  & .venv\Scripts\python.exe @pyArgs
  if ($LASTEXITCODE -ne 0) { $failed += $id }
}
if ($failed.Count -gt 0) {
  Write-Host "FAILED ($($failed.Count)): $($failed -join ', ')"
} else {
  Write-Host "all runs completed"
}
