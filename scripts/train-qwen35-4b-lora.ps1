#requires -Version 5.1
<# Plan, smoke, launch, or monitor the separate Qwen3.5-4B LoRA trainer. Never provisions or deletes. #>
[CmdletBinding()]
param(
 [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')][string]$RunId,
 [Parameter(Mandatory=$true)][ValidateSet('Plan','Smoke','ValidateSmoke','StartFull','ResumeFull','Monitor')][string]$Action,
 [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')][string]$ParentAcceptedSmokeRunId,
 [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')][string]$ParentResumeSmokeRunId,
 [Parameter(Mandatory=$true)][ValidatePattern('^/workspace/runs/[A-Za-z0-9][A-Za-z0-9._/-]{0,255}/model-manifest\.json$')][string]$StagingManifest,
 [ValidatePattern('^/workspace/runs/[A-Za-z0-9][A-Za-z0-9._/-]{0,255}/checkpoints/step-[0-9]{6}$')][string]$ResumeFrom
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'lib.ps1')
function Quote-Remote([string]$Value) { if ([string]::IsNullOrWhiteSpace($Value) -or $Value -match '[\x00-\x1f]') { throw 'Unsafe remote argument.' }; ConvertTo-PosixSingleQuoted $Value }
function Get-Qwen35Arguments([string[]]$Mode,[string]$RunDir,[string]$Project) {
 $corpusRoot=$script:Sprint.RemoteRuns+'/abliterated-20000-20260829T022737Z'
 $corpusPath=$corpusRoot+'/output/rollouts.jsonl'
 $manifestPath=$corpusRoot+'/output/manifest.json'
 $evaluationPath=$script:Sprint.RemoteCode+'/external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json'
 @('-m','experiment.train_qwen35_4b_lora_local')+$Mode+@('--corpus',$corpusPath,'--corpus-manifest',$manifestPath,'--finalizer-manifest',($corpusRoot+'/manifest.json'),'--staging-manifest',$StagingManifest,'--evaluation-questions',$evaluationPath,'--run-dir',$RunDir)
}
function Invoke-Qwen35Remote([string[]]$Mode,[switch]$Detached) {
 $pod=Resolve-RunpodPodOrThrow; $project=$script:Sprint.RemoteCode+'/mats12'; $python='/tmp/mats12-qwen35-4b-train-venv/bin/python'; $run=$script:Sprint.RemoteRuns+'/'+$RunId
 $remoteArgs=Get-Qwen35Arguments -Mode $Mode -RunDir $run -Project $project; $quoted=@($remoteArgs|ForEach-Object{Quote-Remote $_}) -join ' '
 if(-not $Detached) { return Invoke-PodSsh $pod "test -x $(Quote-Remote $python) && test -d $(Quote-Remote $project) && test ! -e $(Quote-Remote $run) && cd $(Quote-Remote $project) && $(Quote-Remote $python) $quoted" }
 $gate="while test ! -f $(Quote-Remote ($run+'/launch.ready')); do sleep 1; done; rm -f $(Quote-Remote ($run+'/launch.ready')); exec $(Quote-Remote $python) $quoted"
 $template=@'
set -eu
cd __PROJECT__
test -x __PYTHON__
test ! -e __RUN__
commit=$(git rev-parse HEAD)
test -z "$(git status --porcelain)"
mkdir __RUN__
: > __STDOUT__
: > __STDERR__
setsid sh -c __GATE__ > __STDOUT__ 2> __STDERR__ < /dev/null &
pid=$!
start=$(awk '{print $22}' /proc/$pid/stat)
test -n "$start"
tmp=__TEMP__
printf '{"format":"qwen35-4b-trainer-launch-v1","run_id":"__RUNID__","commit":"%s","pid":%s,"start_identity":"%s"}\n' "$commit" "$pid" "$start" > "$tmp"
sync -f "$tmp"
mv "$tmp" __LAUNCH__
sync -f __LAUNCH__
sync -f __RUN__
: > __READY__
sync -f __READY__
i=0
while test ! -f __ACK__; do
 test -f __CRASHED__ && exit 1
 test -d /proc/$pid || exit 1
 i=$((i+1)); test $i -le 120 || exit 1
 sleep 1
done
printf '%s\n' "$pid"
'@
 $values=@{'__PROJECT__'=Quote-Remote $project;'__PYTHON__'=Quote-Remote $python;'__RUN__'=Quote-Remote $run;'__STDOUT__'=Quote-Remote ($run+'/stdout.log');'__STDERR__'=Quote-Remote ($run+'/stderr.log');'__GATE__'=Quote-Remote $gate;'__TEMP__'=Quote-Remote ($run+'/.launch.json.tmp');'__LAUNCH__'=Quote-Remote ($run+'/launch.json');'__READY__'=Quote-Remote ($run+'/launch.ready');'__ACK__'=Quote-Remote ($run+'/launcher-adopted.json');'__CRASHED__'=Quote-Remote ($run+'/CRASHED');'__RUNID__'=$RunId}
 foreach($key in $values.Keys){$template=$template.Replace($key,[string]$values[$key])}
 $localScript=[IO.Path]::GetTempFileName()
 $remoteScript='/tmp/mats12-qwen35-4b-launch-'+[Guid]::NewGuid().ToString('N')+'.sh'
 try {
  [IO.File]::WriteAllText($localScript,($template -replace "`r",'')+"`n",(New-Object Text.UTF8Encoding($false)))
  if(-not(Copy-FileToPod -Pod $pod -LocalPath $localScript -RemotePath $remoteScript)){throw 'Detached launch script transfer failed.'}
  $remoteQuoted=Quote-Remote $remoteScript
  return Invoke-PodSsh $pod "bash $remoteQuoted; code=`$?; rm -f $remoteQuoted; exit `$code"
 } finally {
  Remove-Item -LiteralPath $localScript -Force -ErrorAction SilentlyContinue
  try { Invoke-PodSsh -Pod $pod -Command ("rm -f "+(Quote-Remote $remoteScript)) -AllowFail | Out-Null } catch {}
 }
}
function Invoke-Qwen35SmokeValidation([string]$SmokeRun) {
 $pod=Resolve-RunpodPodOrThrow; $project=$script:Sprint.RemoteCode+'/mats12'; $python='/tmp/mats12-qwen35-4b-train-venv/bin/python'
 $remoteArgs=Get-Qwen35Arguments -Mode @('--validate-completed','--validation-mode','runtime','--run-kind','smoke') -RunDir $SmokeRun -Project $project
 $quoted=@($remoteArgs|ForEach-Object{Quote-Remote $_}) -join ' '
 return Invoke-PodSsh $pod "test -x $(Quote-Remote $python) && cd $(Quote-Remote $project) && $(Quote-Remote $python) $quoted 2> /tmp/qwen35-smoke-validation.stderr"
}
switch($Action){
 'Plan' {(Invoke-Qwen35Remote -Mode @('--plan')).StdOut}
 'Smoke' { $mode=@('--execute','--run-kind','smoke','--max-steps','1'); if($ResumeFrom){$mode+=@('--resume-from',$ResumeFrom)}; (Invoke-Qwen35Remote -Mode $mode -Detached).StdOut }
 'ValidateSmoke' { $smoke=$script:Sprint.RemoteRuns+'/'+$RunId; (Invoke-Qwen35SmokeValidation -SmokeRun $smoke).StdOut }
 'StartFull' { if(!$ParentAcceptedSmokeRunId -or !$ParentResumeSmokeRunId){throw 'StartFull requires fresh-smoke and resume-smoke runs.'}; $smoke=$script:Sprint.RemoteRuns+'/'+$ParentAcceptedSmokeRunId; $resumeSmoke=$script:Sprint.RemoteRuns+'/'+$ParentResumeSmokeRunId; Invoke-Qwen35SmokeValidation -SmokeRun $smoke | Out-Null; Invoke-Qwen35SmokeValidation -SmokeRun $resumeSmoke | Out-Null; (Invoke-Qwen35Remote -Mode @('--execute','--run-kind','full','--accepted-smoke-run',$smoke,'--accepted-resume-smoke-run',$resumeSmoke) -Detached).StdOut }
 'ResumeFull' { if(!$ResumeFrom){throw 'ResumeFull requires -ResumeFrom.'}; (Invoke-Qwen35Remote -Mode @('--execute','--run-kind','full','--resume-from',$ResumeFrom) -Detached).StdOut }
 'Monitor' { $pod=Resolve-RunpodPodOrThrow; $run=$script:Sprint.RemoteRuns+'/'+$RunId; (Invoke-PodSsh $pod "if test -f $(Quote-Remote ($run+'/DONE')) && test -f $(Quote-Remote ($run+'/CRASHED')); then echo AMBIGUOUS; elif test -f $(Quote-Remote ($run+'/DONE')); then echo DONE; elif test -f $(Quote-Remote ($run+'/CRASHED')); then echo CRASHED; else echo RUNNING; fi").StdOut }
}
