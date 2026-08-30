#requires -Version 5.1
[CmdletBinding()]
param(
 [Parameter(Mandatory=$true)][string]$RunRoot,
 [Parameter(Mandatory=$true)][string]$Input,
 [Parameter(Mandatory=$true)][string]$Checkpoint,
 [Parameter(Mandatory=$true)][string]$StagingManifest,
 [int]$BatchSize=256,
 [string]$RemotePython='/root/mats12-second-order-venv/bin/python',
 [switch]$Prepare,[switch]$Smoke,[switch]$Start,[switch]$Monitor,[switch]$Finalize
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'lib.ps1')
$actions=@($Prepare,$Smoke,$Start,$Monitor,$Finalize | Where-Object { $_ })
if ($actions.Count -ne 1) { throw 'Specify exactly one action per invocation.' }
function Quote-Remote([string]$Value) {
 if ($null -eq $Value) { throw 'Remote argument is null.' }
 foreach ($character in $Value.ToCharArray()) { if ([int][char]$character -lt 32) { throw 'Remote argument contains a control character.' } }
 return ConvertTo-PosixSingleQuoted $Value
}
function Invoke-SecondOrderRemote([string[]]$Mode) {
 $pod=Resolve-RunpodPodOrThrow
 $args=@('-m','experiment.generate_second_order_20k') + $Mode + @('--run-root',$RunRoot,'--runs-root',$script:Sprint.RemoteRuns,'--input',$Input,'--checkpoint',$Checkpoint,'--staging-manifest',$StagingManifest,'--batch-size',[string]$BatchSize)
 $quoted=@($args | ForEach-Object { Quote-Remote $_ }) -join ' '
 $project=$script:Sprint.RemoteCode + '/mats12'
 $command="test -x $(Quote-Remote $RemotePython) && test -d $(Quote-Remote $project) && cd $(Quote-Remote $project) && $(Quote-Remote $RemotePython) $quoted"
 return Invoke-PodSsh -Pod $pod -Command $command
}
if ($Prepare) { (Invoke-SecondOrderRemote @('--prepare')).StdOut }
if ($Smoke) { (Invoke-SecondOrderRemote @('--smoke')).StdOut }
if ($Start) { (Invoke-SecondOrderRemote @('--coordinator-start')).StdOut }
if ($Monitor) { (Invoke-SecondOrderRemote @('--monitor')).StdOut }
if ($Finalize) { (Invoke-SecondOrderRemote @('--finalize')).StdOut }
