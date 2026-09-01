<# Safely delete the one exact configured pod through REST v2; preserve volume. #>
[CmdletBinding(SupportsShouldProcess=$true)]
param([string]$ExpectedPodId)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\lib.ps1"

function Test-RunpodDeleted {
    param([string]$PodId)
    try {
        $pods = @(Get-RunpodV2Pods)
        return -not ($pods | Where-Object { [string]$_.id -eq $PodId })
    } catch {
        if ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 404) { return $true }
        throw
    }
}

$volume = Get-RunpodVolumeByName
if (-not $volume) { throw "Configured volume was not found; refusing deletion without ownership proof." }
$pods = @(Get-RunpodExactNamePods)
if ($pods.Count -eq 0) { Write-Host "No configured pod exists. Nothing to stop." -ForegroundColor Green; return }
if ($pods.Count -ne 1) { throw "Multiple exact-name pods exist; refusing deletion." }
$pod = $pods[0]
if ($ExpectedPodId -and [string]$pod.id -ne $ExpectedPodId) { throw "Configured pod ID differs from ExpectedPodId; refusing deletion." }
$mount = Get-RunpodNetworkMount $pod
if (-not $mount -or [string]$mount.volumeId -ne [string]$volume.id -or [string]$mount.path -ne $script:Sprint.WorkspacePath -or [string]$pod.dataCenterId -ne $script:Sprint.DataCenterId) {
    throw "Exact-name pod does not prove configured volume/path/datacenter ownership; refusing deletion."
}
$label = "$($pod.name) ($($pod.id), $($pod.status))"
if ($PSCmdlet.ShouldProcess($label, "delete v2 pod to stop compute billing")) {
    $beforeDelete = @(Get-RunpodExactNamePods)
    if ($beforeDelete.Count -ne 1 -or [string]$beforeDelete[0].id -ne [string]$pod.id -or ($ExpectedPodId -and [string]$beforeDelete[0].id -ne $ExpectedPodId)) { throw "Pod identity changed before deletion; refusing deletion." }
    $beforeMount = Get-RunpodNetworkMount $beforeDelete[0]
    if (-not $beforeMount -or [string]$beforeMount.volumeId -ne [string]$volume.id -or [string]$beforeDelete[0].dataCenterId -ne $script:Sprint.DataCenterId) { throw "Pod ownership changed before deletion; refusing deletion." }
    try { Invoke-RunpodV2Api -Path "/pods/$($pod.id)" -Method DELETE | Out-Null } catch { if (-not ($_.Exception.Response -and [int]$_.Exception.Response.StatusCode -eq 404)) { throw } }
    Start-Sleep -Milliseconds 800
    if (-not (Test-RunpodDeleted -PodId $pod.id)) { throw "Deletion was not confirmed by explicit absence or 404; the pod may still be billing." }
    Write-Host "Deleted $label. Volume '$($script:Sprint.VolumeName)' was preserved." -ForegroundColor Green
}
