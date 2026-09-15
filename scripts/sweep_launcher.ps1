# ADE20K factorial launcher: waits until no other Python process holds the GPU for three
# consecutive minutes, then runs the linear-head cells (configs/ade20k without __mask) filter by
# filter, skipping cells whose results/<id>/metrics.json exists. Logs under results/sweep_logs/.
param(
  [string]$Repo = "D:/JHU-Transformers-Project",
  [string[]]$Filters = @("frozen", "lora", "full")
)
Set-Location $Repo
$logDir = "results/sweep_logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$launcherLog = "$logDir/launcher.log"
function Get-GpuPythonPids {
  $apps = & nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
  if (-not $apps) { return @() }
  return @($apps | Where-Object { $_ -match 'python' } | ForEach-Object { ($_ -split ',')[0].Trim() })
}
"$(Get-Date -Format s) launcher started (pid $PID), filters: $($Filters -join ', ')" | Out-File -Append $launcherLog
$quiet = 0
while ($quiet -lt 3) {
  $gpuPids = Get-GpuPythonPids
  if ($gpuPids.Count -eq 0) { $quiet++ } else {
    $quiet = 0
    "$(Get-Date -Format s) waiting: GPU held by python pid(s) $($gpuPids -join ',')" | Out-File -Append $launcherLog
  }
  Start-Sleep -Seconds 60
}
"$(Get-Date -Format s) GPU free for 3 min; starting sweep" | Out-File -Append $launcherLog
foreach ($f in $Filters) {
  $log = "$logDir/ade20k_$f.log"
  "=== filter $f start $(Get-Date -Format s)" | Out-File -Append $log
  $failed = @()
  $cfgs = Get-ChildItem configs/ade20k -Filter *.yaml | Where-Object { $_.Name -like "*__${f}__*" -and $_.Name -notlike "*__mask*" } | Sort-Object Name
  foreach ($c in $cfgs) {
    $id = $c.BaseName
    if (Test-Path "results/$id/metrics.json") { "skip $id" | Out-File -Append $log; continue }
    "=== $id start $(Get-Date -Format s)" | Out-File -Append $log
    $p = Start-Process -FilePath "$Repo/.venv/Scripts/python.exe" -ArgumentList @("-m", "ttr.run", "--config", $c.FullName, "out_dir=results") -WorkingDirectory $Repo -RedirectStandardOutput "$logDir/_out.tmp" -RedirectStandardError "$logDir/_err.tmp" -Wait -NoNewWindow -PassThru
    Get-Content "$logDir/_out.tmp" | Out-File -Append $log
    Get-Content "$logDir/_err.tmp" | Where-Object { $_ -notmatch 'unauthenticated requests' } | Out-File -Append $log
    if ($p.ExitCode -ne 0) { $failed += $id; "FAILED $id exit $($p.ExitCode)" | Out-File -Append $log }
    "=== $id end $(Get-Date -Format s)" | Out-File -Append $log
  }
  $summary = "$f done $(Get-Date -Format s) cells=$($cfgs.Count) failed=$($failed.Count) $($failed -join ',')"
  $summary | Out-File -Append $log
  $summary | Out-File "$logDir/DONE_$f.txt"
  $summary | Out-File -Append $launcherLog
}
Remove-Item "$logDir/_out.tmp", "$logDir/_err.tmp" -ErrorAction SilentlyContinue
"$(Get-Date -Format s) launcher finished" | Out-File -Append $launcherLog
