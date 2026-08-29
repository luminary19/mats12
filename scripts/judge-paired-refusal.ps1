#Requires -Version 5.1
[CmdletBinding()]
param(
    [string]$RunDir,
    [switch]$Execute,
    [ValidateRange(1, 64)]
    [int]$Concurrency = 16
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $RunDir = Join-Path $repoRoot 'runs\paired-refusal-judge'
}
$keyName = 'OPENROUTER_API_KEY'

function Invoke-PairedRefusalCommand {
    param([string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Paired refusal command failed with exit code $LASTEXITCODE."
    }
}

Push-Location $repoRoot
try {
    Invoke-PairedRefusalCommand @('-m', 'experiment.judge_paired_refusal', '--prepare', '--run-dir', $RunDir)
    Invoke-PairedRefusalCommand @('-m', 'experiment.judge_paired_refusal', '--plan', '--run-dir', $RunDir)
    if (-not $Execute) {
        Write-Host 'Prepare and plan validation passed. Re-run with -Execute to start paid OpenRouter judging.'
        return
    }

    $userKey = [Environment]::GetEnvironmentVariable($keyName, 'User')
    if ([string]::IsNullOrWhiteSpace($userKey)) {
        throw "$keyName is not set in the HKCU user environment."
    }
    $previousProcessKey = [Environment]::GetEnvironmentVariable($keyName, 'Process')
    try {
        [Environment]::SetEnvironmentVariable($keyName, $userKey, 'Process')
        Invoke-PairedRefusalCommand @(
            '-m', 'experiment.judge_paired_refusal', '--execute', '--run-dir', $RunDir,
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
