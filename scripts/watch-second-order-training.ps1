#requires -Version 5.1
<# Detached fail-open watcher: mirror, cryptographically validate, then stop only the verified pod. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')][string]$RunId,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9_-]{1,128}$')][string]$ExpectedPodId,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedCommit,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')][string]$AcceptedSmokeRunId,
    [ValidateSet('StartDetached','Watch')][string]$Action = 'StartDetached',
    [string]$ValidatorPython = '',
    [int]$PollSeconds = 60
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($PollSeconds -lt 10) { throw 'PollSeconds must be at least 10.' }
if ([string]::IsNullOrWhiteSpace($ValidatorPython)) { $ValidatorPython = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $ValidatorPython)) { throw 'ValidatorPython must be an existing project-controlled interpreter.' }
$originalLocation = Get-Location
$originalPath = $env:PATH
$logRoot = Join-Path (Join-Path $PSScriptRoot '..\runs') ('watcher-' + $RunId)
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$log = Join-Path $logRoot 'watcher.log'
function Write-WatcherLog([string]$Message) { Add-Content -LiteralPath $log -Value ('{0:o} {1}' -f [DateTime]::UtcNow, $Message) -Encoding ASCII }
function Invoke-ExactRunMirror($Pod,[string[]]$RunIds) {
    $stage = Join-Path $script:Sprint.LocalRuns ('.second-order-mirror-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    try {
        foreach($id in $RunIds) {
            if($id -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$'){throw 'Unsafe mirror run ID.'}
            $remote=$script:Sprint.RemoteRuns+'/'+$id
            $command="cd $(ConvertTo-PosixSingleQuoted $remote) && find . -type f -print0 | sort -z | xargs -0 sha256sum"
            $inventory=Invoke-PodSsh -Pod $Pod -Command $command
            if(-not $inventory.Ok -or -not $inventory.Lines){throw 'Exact remote inventory failed or was empty.'}
            $target=Join-Path $stage $id; New-Item -ItemType Directory -Path $target -Force | Out-Null
            foreach($line in $inventory.Lines){if($line -notmatch '^([0-9a-f]{64})  \./(.+)$'){throw 'Invalid exact inventory row.'};$hash=$Matches[1];$relative=$Matches[2];if(-not(Test-SafeRelPath $relative)){throw 'Unsafe exact inventory path.'};$local=Join-Path $target ($relative-replace '/','\');$parent=Split-Path $local -Parent;New-Item -ItemType Directory -Path $parent -Force|Out-Null;$temp=Get-LocalAtomicTempPath $local;if(-not(Copy-FileFromPod $Pod ($remote+'/'+$relative) $temp)){throw 'Exact mirror download failed.'};if((Get-FileHash $temp -Algorithm SHA256).Hash.ToLowerInvariant()-ne$hash){throw 'Exact mirror hash mismatch.'};Move-Item -LiteralPath $temp -Destination $local}
            Set-Content -LiteralPath (Join-Path $logRoot ($id + '.inventory.tsv')) -Value ($inventory.Lines -join "`n") -Encoding ASCII
        }
        foreach($id in $RunIds){$final=Join-Path $script:Sprint.LocalRuns $id;$candidate=Join-Path $stage $id;$backup=$final+'.mirror-backup-'+[Guid]::NewGuid().ToString('N');if(Test-Path $final){Move-Item -LiteralPath $final -Destination $backup};Move-Item -LiteralPath $candidate -Destination $final;if(Test-Path $backup){Remove-Item -LiteralPath $backup -Recurse -Force}}
    } catch {Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue;throw} finally {if(Test-Path $stage){Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue}}
}
if ($Action -eq 'StartDetached') {
    $scriptPath = $MyInvocation.MyCommand.Path
    & $ValidatorPython -c "import experiment.train_llama32_lora_second_order"; if ($LASTEXITCODE -ne 0) { throw 'ValidatorPython cannot import the validator.' }
    Write-WatcherLog ('ValidatorPython=' + $ValidatorPython + '; version=' + (& $ValidatorPython --version 2>&1))
    $arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Action Watch -RunId "{1}" -ExpectedPodId "{2}" -ExpectedCommit "{3}" -AcceptedSmokeRunId "{4}" -ValidatorPython "{5}" -PollSeconds {6}' -f $scriptPath, $RunId, $ExpectedPodId, $ExpectedCommit, $AcceptedSmokeRunId, $ValidatorPython, $PollSeconds
    Start-Process -FilePath powershell.exe -ArgumentList $arguments -WorkingDirectory (Split-Path -Parent $PSScriptRoot) -WindowStyle Hidden | Out-Null
    Write-WatcherLog 'Detached watcher started.'
    return
}
try {
    . (Join-Path $PSScriptRoot 'lib.ps1')
    $runDir = $script:Sprint.RemoteRuns + '/' + $RunId
    while ($true) {
        try {
            $pods = @(Get-RunpodExactNamePods)
            if ($pods.Count -ne 1 -or [string]$pods[0].id -ne $ExpectedPodId) { throw 'Pod identity is absent or ambiguous; leaving pod untouched.' }
            $pod = Resolve-RunpodPodOrThrow
            $commit = Invoke-PodSsh -Pod $pod -Command "cd '$($script:Sprint.RemoteCode)/mats12' && git rev-parse HEAD"
            if (-not $commit.Ok -or $commit.StdOut.Trim() -ne $ExpectedCommit) { throw 'Remote commit differs; leaving pod untouched.' }
            $state = Invoke-PodSsh -Pod $pod -Command "if test -f '$runDir/DONE' && test -f '$runDir/CRASHED'; then echo AMBIGUOUS; elif test -f '$runDir/DONE'; then echo DONE; elif test -f '$runDir/CRASHED'; then echo CRASHED; else { test -r '$runDir/launch.json' || { echo AMBIGUOUS; exit 0; }; pid=`$(tr ',' '\n' < '$runDir/launch.json' | grep pid | tr -cd '0-9'); start=`$(tr ',' '\n' < '$runDir/launch.json' | grep start_identity | tr -cd '0-9'); test -n `$pid && test -n `$start && test -r /proc/`$pid/stat && test `$(awk '{print `$22}' /proc/`$pid/stat) = `$start && echo RUNNING || echo STALE; }; fi"
            $value = $state.StdOut.Trim()
            if ($value -eq 'RUNNING') { Write-WatcherLog 'Remote run is running.'; Start-Sleep -Seconds $PollSeconds; continue }
            if ($value -ne 'DONE') { throw ('Remote terminal/process state is ' + $value + '; leaving pod untouched.') }
            Invoke-ExactRunMirror $pod @($AcceptedSmokeRunId,$RunId)
            $localRun = Join-Path $script:Sprint.LocalRuns $RunId
            & $ValidatorPython -m experiment.train_llama32_lora_second_order --validate-completed --validation-mode static --run-kind full --run-dir $localRun
            if ($LASTEXITCODE -ne 0) { throw 'Local cryptographic full-run validation failed; leaving pod untouched.' }
            & (Join-Path $PSScriptRoot 'pod-down.ps1') -ExpectedPodId $ExpectedPodId -Confirm:$false
            $remaining = @(Get-RunpodExactNamePods)
            if ($remaining.Count -ne 0) { throw 'pod-down did not verify deletion; investigate billing state.' }
            Write-WatcherLog 'Validated local mirror and verified pod deletion. Volume was not touched.'
            break
        } catch {
            Write-WatcherLog ('WATCHER ERROR: ' + $_.Exception.Message)
            break
        }
    }
} finally {
    Set-Location $originalLocation
    $env:PATH = $originalPath
}
