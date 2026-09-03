# scripts/download_ade20k.ps1  -- ~1 GB
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force data/ade20k | Out-Null
$zip = "data/ade20k/ADEChallengeData2016.zip"
if (-not (Test-Path $zip)) {
  Invoke-WebRequest -Uri "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip" -OutFile $zip
}
Expand-Archive -Path $zip -DestinationPath data/ade20k -Force
Get-ChildItem data/ade20k/ADEChallengeData2016/images/training | Measure-Object | Select-Object Count
