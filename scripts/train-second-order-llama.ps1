#requires -Version 5.1
<# Plan, smoke, launch, or monitor the separate second-order trainer. Never provisions or deletes. #>
[CmdletBinding()]
param(
 [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')][string]$RunId,
 [Parameter(Mandatory=$true)][ValidateSet('Plan','Smoke','StartFull','Monitor')][string]$Action,
 [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')][string]$ParentAcceptedSmokeRunId,
 [string]$StagingManifest='/workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json'
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'lib.ps1')
function Quote-Remote([string]$Value) { if ([string]::IsNullOrWhiteSpace($Value) -or $Value -match '[\x00-\x1f]') { throw 'Unsafe remote argument.' }; ConvertTo-PosixSingleQuoted $Value }
function Get-SecondOrderArguments([string[]]$Mode,[string]$RunDir,[string]$Project) {
 $corpusRoot=$script:Sprint.RemoteRuns+'/second-order-llama20k-hf128-continuation-seed42-20260830T080010Z'
 $corpusPath=$corpusRoot+'/final/output/rollouts.jsonl'
 $manifestPath=$corpusRoot+'/final/output/manifest.json'
 $evaluationPath=$script:Sprint.RemoteCode+'/external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json'
 @('-m','experiment.train_llama32_lora_second_order')+$Mode+@('--corpus',$corpusPath,'--corpus-manifest',$manifestPath,'--staging-manifest',$StagingManifest,'--evaluation-questions',$evaluationPath,'--run-dir',$RunDir)
}
function Invoke-SecondOrderRemote([string[]]$Mode,[switch]$Detached) {
 $pod=Resolve-RunpodPodOrThrow; $project=$script:Sprint.RemoteCode+'/mats12'; $python='/tmp/mats12-second-order-train-venv/bin/python'; $run=$script:Sprint.RemoteRuns+'/'+$RunId
 $remoteArgs=Get-SecondOrderArguments -Mode $Mode -RunDir $run -Project $project; $quoted=@($remoteArgs|ForEach-Object{Quote-Remote $_}) -join ' '
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
printf '{"format":"second-order-trainer-launch-v1","run_id":"__RUNID__","commit":"%s","pid":%s,"start_identity":"%s"}\n' "$commit" "$pid" "$start" > "$tmp"
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
 return Invoke-PodSsh $pod $template
}
switch($Action){
 'Plan' {(Invoke-SecondOrderRemote -Mode @('--plan')).StdOut}
 'Smoke' {(Invoke-SecondOrderRemote -Mode @('--execute','--run-kind','smoke','--max-steps','1') -Detached).StdOut}
 'StartFull' { if(!$ParentAcceptedSmokeRunId){throw 'StartFull requires parent-accepted smoke.'}; $smoke=$script:Sprint.RemoteRuns+'/'+$ParentAcceptedSmokeRunId; (Invoke-SecondOrderRemote -Mode @('--execute','--run-kind','full','--accepted-smoke-run',$smoke) -Detached).StdOut }
 'Monitor' { $pod=Resolve-RunpodPodOrThrow; $run=$script:Sprint.RemoteRuns+'/'+$RunId; (Invoke-PodSsh $pod "test -f $(Quote-Remote ($run+'/DONE')) && echo DONE || test -f $(Quote-Remote ($run+'/CRASHED')) && echo CRASHED || echo RUNNING").StdOut }
}
