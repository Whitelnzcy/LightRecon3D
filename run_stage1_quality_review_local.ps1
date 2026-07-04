param(
  [string]$Python = "python",
  [string]$Checkpoint = "local_outputs/stage1_large_80m_train2048_sharded_v1/best.pt",
  [string]$FeatureCachePath = "",
  [string]$FeatureCacheGlob = "",
  [string]$Split = "train",
  [string]$OutputDir = "local_outputs/stage1_quality_split_4096_v1",
  [int]$BatchSize = 1,
  [int]$MaxSamples = 128,
  [int]$ReviewSamplesPerBucket = 20,
  [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $FeatureCachePath -and -not $FeatureCacheGlob) {
  throw "Provide -FeatureCachePath or -FeatureCacheGlob. The quality review requires cached DUSt3R features."
}

$argsList = @(
  "build_stage1_prediction_quality_splits.py",
  "--checkpoint", $Checkpoint,
  "--split", $Split,
  "--output_dir", $OutputDir,
  "--batch_size", "$BatchSize",
  "--max_samples", "$MaxSamples",
  "--review_samples_per_bucket", "$ReviewSamplesPerBucket",
  "--device", $Device
)

if ($FeatureCachePath) {
  $argsList += @("--feature_cache_path", $FeatureCachePath)
}
if ($FeatureCacheGlob) {
  $argsList += @("--feature_cache_glob", $FeatureCacheGlob)
}

Write-Host "[quality-review] python=$Python"
Write-Host "[quality-review] checkpoint=$Checkpoint"
Write-Host "[quality-review] output=$OutputDir"
& $Python @argsList

Write-Host "[quality-review] done"
Write-Host "Review images: $OutputDir\review"
Write-Host "Summary:       $OutputDir\summary.json"
Write-Host "CSV:           $OutputDir\all_samples.csv"
