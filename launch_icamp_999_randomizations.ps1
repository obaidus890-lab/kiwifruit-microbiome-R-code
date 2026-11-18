# ============================================================================
# launch_icamp_999_randomizations.ps1
#
# ============================================================================
param(
  [int]$Rand = 999,
  [int]$Bsl  = 12,
  [switch]$Dry
)

$ErrorActionPreference = 'Stop'
$Root = if ($env:KIWI_ROOT) { $env:KIWI_ROOT } else { $PSScriptRoot }
$Rscript = if ($env:RSCRIPT) { $env:RSCRIPT } else { 'Rscript' }
$Script = Join-Path $PSScriptRoot 'run_icamp_999_randomizations.R'
$LogDir = Join-Path $Root 'result\r_out\icamp_999_logs'
$OutRoot = Join-Path $Root 'result\r_out'
$env:KIWI_ROOT = $Root
if ($Rand -ne 999 -or $Bsl -ne 12) { throw 'For the final analysis, use Rand=999 and Bsl=12.' }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Get-Command $Rscript -ErrorAction SilentlyContinue)) { throw "Rscript not found: $Rscript" }
if (-not (Test-Path $Script))  { throw "R script not found: $Script" }

$groups = @(
  @{ ds = 'ITS'; comp = 'F';  nw = 12; taxa =  364 },
  @{ ds = 'ITS'; comp = 'R';  nw = 12; taxa =  433 },
  @{ ds = 'ITS'; comp = 'L';  nw = 12; taxa =  519 },
  @{ ds = 'ITS'; comp = 'S';  nw = 12; taxa =  537 },
  @{ ds = 'ITS'; comp = 'RS'; nw = 12; taxa =  610 },
  @{ ds = 'ITS'; comp = 'BS'; nw = 12; taxa =  651 },
  @{ ds = '16S'; comp = 'F';  nw = 12; taxa =  764 },
  @{ ds = '16S'; comp = 'L';  nw = 10; taxa = 1559 },
  @{ ds = '16S'; comp = 'R';  nw = 10; taxa = 1598 },
  @{ ds = '16S'; comp = 'S';  nw = 10; taxa = 1745 },
  @{ ds = '16S'; comp = 'BS'; nw =  6; taxa = 2439 },
  @{ ds = '16S'; comp = 'RS'; nw =  6; taxa = 2596 }
)

$effRand = if ($Dry) { 0 } else { $Rand }
$summary = @()
$stamp   = Get-Date -Format 'yyyyMMdd_HHmmss'

Write-Host ("==== iCAMP uniform rerun: rand={0} bin.size.limit={1} dry={2} start={3}" -f `
            $effRand, $Bsl, [bool]$Dry, $stamp)

foreach ($g in $groups) {
  $ds = $g.ds; $comp = $g.comp
  $tag = "${ds}_${comp}"
  $runinfo = Join-Path $OutRoot ("icamp_{0}\_runinfo_{0}_{1}.tsv" -f $ds, $comp)

  if (-not $Dry -and (Test-Path $runinfo)) {
    $ri = Import-Csv $runinfo -Delimiter "`t"
    if ([int]$ri.rand -eq $Rand -and [int]$ri.'bin.size.limit' -eq $Bsl) {
      Write-Host ("  [SKIP] {0} already done at rand={1} bsl={2}" -f $tag, $Rand, $Bsl)
      $summary += [pscustomobject]@{ group = $tag; status = 'SKIP'; minutes = 0 }
      continue
    }
  }

  $log = Join-Path $LogDir ("{0}_{1}_{2}.log" -f $tag, $effRand, $stamp)

  $attempts = @($g.nw)
  if ($g.nw -gt 2) { $attempts += [math]::Max(2, [int]($g.nw / 2)) }
  if ($g.nw -gt 1) { $attempts += 1 }
  if ($Dry) { $attempts = @($g.nw) }

  $status = 'FAIL'; $mins = 0; $usedNw = $null
  foreach ($nw in $attempts) {
    Write-Host ("  [RUN ] {0} (taxa~{1}, nworker={2}) -> {3}" -f $tag, $g.taxa, $nw, (Split-Path $log -Leaf))
    $t0 = Get-Date
    & $Rscript $Script $ds $comp $effRand $Bsl $nw *>&1 |
      Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    if ($code -eq 0) { $status = 'OK'; $usedNw = $nw; break }
    Write-Host ("  [RETRY] {0} failed at nworker={1} (exit={2}) after {3} min" -f $tag, $nw, $code, $mins)
    $status = "FAIL(exit=$code)"
  }
  if ($null -eq $usedNw -and $status -eq 'OK') { $usedNw = $attempts[0] }
  Write-Host ("  [{0}] {1} in {2} min (nworker={3})" -f $status, $tag, $mins, $usedNw)
  $summary += [pscustomobject]@{ group = $tag; status = $status; minutes = $mins; nworker = $usedNw }
}

Write-Host "`n==== SUMMARY ===="
$summary | Format-Table -AutoSize | Out-String | Write-Host
$summary | Export-Csv -Path (Join-Path $LogDir ("_summary_{0}.csv" -f $stamp)) `
                      -NoTypeInformation -Encoding UTF8
Write-Host ("==== finished {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
