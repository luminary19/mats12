<# Start the exact configured pod session. -ListGpu is read-only. #>
[CmdletBinding()]
param(
    [Alias("GpuId")][string[]]$Gpu,
    [switch]$ListGpu,
    [switch]$AvailableOnly,
    [int]$DiskGb,
    [string]$Image,
    [string[]]$ExtraPort = @(),
    [string[]]$AllowedCudaVersions = @(),
    [string]$TemplateId,
    [switch]$PublicDashboard
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\lib.ps1"
if ($ListGpu) { & "$PSScriptRoot\pod-up.ps1" -ListGpu -AvailableOnly:$AvailableOnly; exit $LASTEXITCODE }
& "$PSScriptRoot\volume-ensure.ps1"
Write-Host "Volume creation, if needed, may start storage billing." -ForegroundColor Yellow
if (-not $Gpu) {
    $pods = @(Get-RunpodExactNamePods)
    if ($pods.Count -eq 0) { throw "No configured pod exists. Inspect -ListGpu and pass one explicit -Gpu." }
    if ($pods.Count -ne 1) { throw "Multiple exact-name pods exist; refusing ambiguity." }
    $active = ConvertTo-RunpodActivePod $pods[0]
    if (-not $active) { throw "Configured pod is $($pods[0].status), not direct-SSH ready. Check status before creating or deleting anything." }
    & "$PSScriptRoot\runpod-sync.ps1"
} else {
    & "$PSScriptRoot\pod-up.ps1" -Gpu $Gpu -DiskGb $DiskGb -Image $Image -ExtraPort $ExtraPort -AllowedCudaVersions $AllowedCudaVersions -TemplateId $TemplateId -PublicDashboard:$PublicDashboard
}
& "$PSScriptRoot\pod-bootstrap.ps1"
Write-Host "Session ready. Start pull-loop.ps1 separately and run pod-down.ps1 when finished." -ForegroundColor Green
