#requires -Version 5.1
<# Install the pinned disposable runtime and stage the exact Qwen3.5-4B-Base snapshot. Never provisions or deletes. #>
[CmdletBinding()]
param(
 [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')][string]$RunId,
 [Parameter(Mandatory=$true)][ValidateSet('Plan','Prepare','Stage','Monitor')][string]$Action
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'lib.ps1')
function Q([string]$Value){ConvertTo-PosixSingleQuoted $Value}
$pod=Resolve-RunpodPodOrThrow
$project=$script:Sprint.RemoteCode+'/mats12'
$python='/tmp/mats12-qwen35-4b-train-venv/bin/python'
$run=$script:Sprint.RemoteRuns+'/'+$RunId
switch($Action){
 'Plan' {(Invoke-PodSsh $pod "cd $(Q $project) && python3 -m experiment.stage_qwen35_4b_base --plan").StdOut}
 'Prepare' {
  $command="set -eu; cd $(Q $project); rm -rf /tmp/mats12-qwen35-4b-train-venv; python3 -m venv /tmp/mats12-qwen35-4b-train-venv; $python -m pip install --upgrade pip -q > /tmp/qwen35-install.log 2>&1; $python -m pip install -q -r experiment/requirements-qwen35-4b-runpod.txt >> /tmp/qwen35-install.log 2>&1; $python -m pip freeze | grep -E '^(torch|transformers|peft|accelerate|safetensors|huggingface-hub|flash-linear-attention|causal-conv1d)=='"
  (Invoke-PodSsh $pod $command).StdOut
 }
 'Stage' {
  $inner="cd $(Q $project) && exec $(Q $python) -m experiment.stage_qwen35_4b_base --execute --run-dir $(Q $run)"
  $template="set -eu`ntest -x $(Q $python)`ntest ! -e $(Q $run)`nsetsid sh -c $(Q $inner) > /tmp/qwen35-stage.stdout 2> /tmp/qwen35-stage.stderr < /dev/null &`npid=`$!`nprintf '%s``n' `"`$pid`"`n"
  $local=[IO.Path]::GetTempFileName();$remote='/tmp/qwen35-stage-launch-'+[Guid]::NewGuid().ToString('N')+'.sh'
  try {
   [IO.File]::WriteAllText($local,($template-replace "`r",''),(New-Object Text.UTF8Encoding($false)))
   if(-not(Copy-FileToPod -Pod $pod -LocalPath $local -RemotePath $remote)){throw 'Stage launch transfer failed.'}
   (Invoke-PodSsh $pod ("bash "+(Q $remote)+"; code=`$?; rm -f "+(Q $remote)+"; exit `$code")).StdOut
  } finally {Remove-Item -LiteralPath $local -Force -ErrorAction SilentlyContinue;try{Invoke-PodSsh -Pod $pod -Command ("rm -f "+(Q $remote)) -AllowFail|Out-Null}catch{}}
 }
 'Monitor' {(Invoke-PodSsh $pod "if test -f $(Q ($run+'/DONE')); then echo DONE; elif test -f $(Q ($run+'/CRASHED')); then echo CRASHED; else echo RUNNING; fi").StdOut}
}
