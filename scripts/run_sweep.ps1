# scripts/run_sweep.ps1 <configs-dir> [filter-substring]
param([string]$Dir = "configs/ade20k", [string]$Filter = "")
$ErrorActionPreference = "Continue"
Get-ChildItem $Dir -Filter *.yaml | Where-Object { $_.Name -like "*$Filter*" } | ForEach-Object {
  $id = $_.BaseName
  if (Test-Path "results/$id/metrics.json") { Write-Host "skip $id"; return }
  Write-Host "=== $id"
  python -m ttr.run --config $_.FullName
}
