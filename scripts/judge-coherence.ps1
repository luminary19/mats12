#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$RunDir,
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$keyName = 'OPENROUTER_API_KEY'

function Invoke-CoherenceCommand {
    param([string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Coherence command failed with exit code $LASTEXITCODE."
    }
}

Push-Location $repoRoot
try {
    Invoke-CoherenceCommand @('-m', 'experiment.prepare_coherence_study', '--prepare', '--run-dir', $RunDir)
    Invoke-CoherenceCommand @('-m', 'experiment.judge_coherence', '--plan', '--run-dir', $RunDir)
    if (-not $Execute) {
        Write-Host 'Prepare and plan validation passed. Re-run with -Execute to start paid OpenRouter judging.'
        return
    }
    $userKey = [Environment]::GetEnvironmentVariable($keyName, 'User')
    if ([string]::IsNullOrWhiteSpace($userKey)) {
        throw "$keyName is not set in HKCU User environment."
    }
    $previousProcessKey = [Environment]::GetEnvironmentVariable($keyName, 'Process')
    try {
        [Environment]::SetEnvironmentVariable($keyName, $userKey, 'Process')
        Invoke-CoherenceCommand @('-m', 'experiment.judge_coherence', '--execute', '--run-dir', $RunDir)
    }
    finally {
        [Environment]::SetEnvironmentVariable($keyName, $previousProcessKey, 'Process')
        $userKey = $null
    }
}
finally {
    Pop-Location
}
