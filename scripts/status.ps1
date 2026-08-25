<# Report exact configured pod state without conflating SSH readiness with billing. #>
[CmdletBinding()] param()
. "$PSScriptRoot\lib.ps1"
Write-Host "=== $($script:Sprint.ProjectName) RunPod status ===" -ForegroundColor Cyan
try { $volume=Get-RunpodVolumeByName; if($volume){Write-Host "Volume : $($volume.name) $($volume.size) GB $($volume.dataCenterId) ($($volume.id))"}else{Write-Host "Volume : not found" -ForegroundColor Yellow} } catch { Write-Host "Volume : error - $($_.Exception.Message)" -ForegroundColor Red }
try {
    $pods=@(Get-RunpodExactNamePods)
    if($pods.Count -eq 0){Write-Host "Pod    : absent (no pod compute billing)" -ForegroundColor Green}
    elseif($pods.Count -ne 1){Write-Host "Pod    : multiple exact-name pods; ownership ambiguous" -ForegroundColor Red}
    else { $pod=$pods[0]; Write-Host ("Pod    : {0} {1} {2} ~`${3}/hr" -f $pod.name,$pod.status,$pod.dataCenterId,$pod.cost) -ForegroundColor Yellow; if(-not(ConvertTo-RunpodActivePod $pod)){Write-Host "         Direct SSH unavailable; this does not establish that billing stopped." -ForegroundColor DarkYellow} }
} catch { Write-Host "Pod    : API error - $($_.Exception.Message)" -ForegroundColor Red }
$files=@(Get-ChildItem $script:Sprint.LocalRuns -Recurse -File -ErrorAction SilentlyContinue);$size=($files|Measure-Object Length -Sum).Sum;Write-Host ("Mirror : {0} file(s), {1:N1} MB" -f $files.Count,($size/1MB))
