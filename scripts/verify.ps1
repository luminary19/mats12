<#
.SYNOPSIS
  Re-runnable end-to-end health check of the live pod (no Claude needed).

.DESCRIPTION
  SSH connectivity, GPU (skipped gracefully on CPU pods), /workspace mount +
  writability, venv + trackio, the baked TRACKIO_DIR, and an scp round-trip.
  Each check validates SPECIFIC expected content (not mere non-emptiness), so an
  error string can't masquerade as a pass. Exits non-zero if anything fails.

.EXAMPLE
  .\verify.ps1
#>
[CmdletBinding()]
param()

. "$PSScriptRoot\lib.ps1"

$pod = Resolve-RunpodPodOrThrow
Write-RunpodBanner -Pod $pod -Action "verify"
$script:fail = 0

function Check {
    param([string]$Name, [bool]$Ok, [string]$Detail = "")
    if ($Ok) { Write-Host ("  PASS  {0}  {1}" -f $Name, $Detail) -ForegroundColor Green }
    else     { Write-Host ("  FAIL  {0}  {1}" -f $Name, $Detail) -ForegroundColor Red; $script:fail++ }
}
function Info {
    param([string]$Name, [string]$Detail)
    Write-Host ("  INFO  {0}  {1}" -f $Name, $Detail) -ForegroundColor DarkGray
}

$probe = "probe-$((Get-Date).Ticks)"

# 1. SSH connectivity - require the exact echoed token
$r = Invoke-PodSsh -Pod $pod -Command "echo __ssh_ok__" -AllowFail
Check "ssh connectivity" ($r.Ok -and ($r.StdOut.Trim() -eq "__ssh_ok__")) $r.StdErr

# 2. GPU - skip gracefully on CPU pods, validate a real GPU name otherwise
$r = Invoke-PodSsh -Pod $pod -Command "command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader || echo __no_gpu__" -AllowFail
$gpu = $r.StdOut.Trim()
if (-not $r.Ok) { Check "gpu" $false $r.StdErr }
elseif ($gpu -eq "__no_gpu__") { Info "gpu" "CPU pod - no nvidia-smi (expected for -Cpu pods)" }
else { Check "gpu" ($gpu -match '(?i)nvidia|rtx|tesla|h100|h200|a100|a40|l4|l40') $gpu }

# 3. /workspace mounted + writable
$r = Invoke-PodSsh -Pod $pod -Command "echo $probe > $($script:Sprint.WorkspacePath)/.vprobe && cat $($script:Sprint.WorkspacePath)/.vprobe && rm -f $($script:Sprint.WorkspacePath)/.vprobe" -AllowFail
Check "/workspace writable" ($r.Ok -and ($r.StdOut -match $probe))

# 4. venv + trackio - require a real version string
$r = Invoke-PodSsh -Pod $pod -Command ". $($script:Sprint.RemoteVenv)/bin/activate && python -c 'import trackio; print(trackio.__version__)'" -AllowFail
$tk = $r.StdOut.Trim()
Check "venv + trackio" ($r.Ok -and ($tk -eq $script:Sprint.TrackioVersion)) ("trackio " + $tk)

# 5. TRACKIO_DIR baked into the venv activate
$r = Invoke-PodSsh -Pod $pod -Command ". $($script:Sprint.RemoteVenv)/bin/activate && printenv TRACKIO_DIR" -AllowFail
Check "TRACKIO_DIR pinned" ($r.Ok -and ($r.StdOut.Trim() -eq "$($script:Sprint.RemoteTrackio)")) $r.StdOut.Trim()

# 6. scp round-trip (up then down, compare content)
$tmp = [System.IO.Path]::GetTempFileName()
Set-Content -Path $tmp -Value "rt-$probe" -NoNewline
$up = Copy-FileToPod -Pod $pod -LocalPath $tmp -RemotePath "$($script:Sprint.WorkspacePath)/.vprobe_up"
$back = [System.IO.Path]::GetTempFileName()
$down = Copy-FileFromPod -Pod $pod -RemotePath "$($script:Sprint.WorkspacePath)/.vprobe_up" -LocalPath $back
$match = $false
if (Test-Path $back) { $match = ((Get-Content $back -Raw).Trim() -eq "rt-$probe") }
Invoke-PodSsh -Pod $pod -Command "rm -f $($script:Sprint.WorkspacePath)/.vprobe_up" -AllowFail | Out-Null
Remove-Item $tmp, $back -ErrorAction SilentlyContinue
Check "scp round-trip" ($up -and $down -and $match)

Write-Host ""
if ($script:fail -eq 0) {
    Write-Host "ALL CHECKS PASSED - the pod is ready." -ForegroundColor Green
} else {
    Write-Host ("{0} check(s) FAILED." -f $script:fail) -ForegroundColor Red
    exit 1
}
