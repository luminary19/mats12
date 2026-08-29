#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$Execute,
    [switch]$MigrateCurrentRun,
    [ValidateRange(1, 64)]
    [int]$Concurrency = 16
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repoRoot 'runs\behavioral-probe-judge'
$keyName = 'OPENROUTER_API_KEY'

function Invoke-JudgeCommand {
    param([string[]]$Arguments)

    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Judge command failed with exit code $LASTEXITCODE."
    }
}

Push-Location $repoRoot
try {
    if ($MigrateCurrentRun) {
        if ($Execute) {
            throw 'Use -MigrateCurrentRun by itself; it performs no paid calls.'
        }
        Invoke-JudgeCommand @(
            '-m', 'experiment.judge_probe',
            '--migrate-current-run',
            '--run-dir', $runDir
        )
        Write-Host 'Exact historical manifest migration completed. Re-run without switches to validate the plan.'
        return
    }

    Invoke-JudgeCommand @(
        '-m', 'experiment.judge_probe',
        '--plan',
        '--run-dir', $runDir
    )

    if (-not $Execute) {
        Write-Host 'Plan validation passed. Re-run with -Execute to start paid OpenRouter judging.'
        return
    }

    $userKey = [Environment]::GetEnvironmentVariable($keyName, 'User')
    if ([string]::IsNullOrWhiteSpace($userKey)) {
        throw "$keyName is not set in the global HKCU user environment. Set it with the global keys.ps1 helper, then rerun this script."
    }

    $previousProcessKey = [Environment]::GetEnvironmentVariable($keyName, 'Process')
    try {
        [Environment]::SetEnvironmentVariable($keyName, $userKey, 'Process')
        Invoke-JudgeCommand @(
            '-m', 'experiment.judge_probe',
            '--execute',
            '--run-dir', $runDir,
            '--concurrency', [string]$Concurrency
        )
    }
    finally {
        [Environment]::SetEnvironmentVariable($keyName, $previousProcessKey, 'Process')
        $userKey = $null
    }
}
finally {
    Pop-Location
}
