#Requires -Version 5.1
<# Plan, smoke, or formally evaluate one frozen Qwen3.5-4B arm. Never provisions or deletes a pod. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('Plan','Smoke','Formal')][string]$Action,
    [Parameter(Mandatory=$true)][ValidateSet('qwen35_4b_base','qwen35_4b_abliterated_sft')][string]$Arm,
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')][string]$RunId,
    [Parameter(Mandatory=$true)][ValidatePattern('^/workspace/runs/[A-Za-z0-9][A-Za-z0-9._/-]{0,255}/model-manifest\.json$')][string]$StagingManifest,
    [ValidateSet(2,90)][int]$QuestionLimit = 2,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')][string]$SmokeRunId,
    [switch]$Execute
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib.ps1')
$AuthorizedCheckpoint = '/workspace/runs/qwen35-4b-abliterated-seed42-1ep-20260902T014813Z/checkpoints/step-000157'
function Quote-Remote([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -match '[\x00-\x1f]') { throw 'Unsafe remote argument.' }
    ConvertTo-PosixSingleQuoted $Value
}
function Invoke-QwenEvaluation([string[]]$Mode) {
    $pod = Resolve-RunpodPodOrThrow
    $project = $script:Sprint.RemoteCode + '/mats12'
    $python = '/tmp/mats12-qwen35-4b-train-venv/bin/python'
    $run = $script:Sprint.RemoteRuns + '/' + $RunId
    $arguments = @('-m','experiment.evaluate_qwen35_4b') + $Mode + @('--arm',$Arm,'--run-dir',$run,'--runs-root',$script:Sprint.RemoteRuns,'--staging-manifest',$StagingManifest,'--checkpoint',$AuthorizedCheckpoint)
    if ($SmokeRunId) { $arguments += @('--smoke-run',($script:Sprint.RemoteRuns + '/' + $SmokeRunId)) }
    $quoted = @($arguments | ForEach-Object { Quote-Remote $_ }) -join ' '
    $command = "test -d $(Quote-Remote $project) && test -d $(Quote-Remote $AuthorizedCheckpoint) && if test ! -x $(Quote-Remote $python); then echo 'Missing Qwen disposable venv; run scripts/stage-qwen35-4b.ps1 -Action Prepare first.' >&2; exit 1; fi && cd $(Quote-Remote $project) && $(Quote-Remote $python) $quoted"
    Invoke-PodSsh $pod $command
}
switch ($Action) {
    'Plan' {
        Invoke-QwenEvaluation @('--plan','--question-limit',[string]$QuestionLimit) | Out-Host
        if (-not $Execute) { Write-Host 'Plan validation passed. Re-run with -Action Smoke or Formal -Execute to allocate the model.' }
    }
    'Smoke' {
        if ($QuestionLimit -ne 2 -or $SmokeRunId) { throw 'Smoke is exactly two questions and cannot consume a smoke run.' }
        Invoke-QwenEvaluation @('--plan','--question-limit','2') | Out-Host
        if ($Execute) { Invoke-QwenEvaluation @('--execute','--question-limit','2') | Out-Host }
        else { Write-Host 'Smoke plan passed. Add -Execute to start GPU inference.' }
    }
    'Formal' {
        if (-not $SmokeRunId) { throw 'Formal requires -SmokeRunId for the same arm.' }
        Invoke-QwenEvaluation @('--plan','--question-limit','90') | Out-Host
        if ($Execute) { Invoke-QwenEvaluation @('--execute','--question-limit','90') | Out-Host }
        else { Write-Host 'Formal plan and matching-arm smoke validation passed. Add -Execute to start GPU inference.' }
    }
}
