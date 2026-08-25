<#
.SYNOPSIS
  Ensure the persistent network volume exists (idempotent).

.DESCRIPTION
  Creates the $($script:Sprint.VolumeName) network volume if it doesn't already exist.
  Safe to run repeatedly - does nothing if the volume is already there.
  The existing volume's datacenter controls Pod placement. DataCenter is used
  only when the volume does not yet exist.

.EXAMPLE
  .\volume-ensure.ps1
  .\volume-ensure.ps1 -SizeGb 100
#>
[CmdletBinding()]
param(
    [int]$SizeGb,
    [string]$DataCenter
)

. "$PSScriptRoot\lib.ps1"

if (-not $SizeGb) {
    $SizeGb = $script:Sprint.DefaultVolumeSizeGb
}
if (-not $DataCenter) {
    $DataCenter = $script:Sprint.DataCenterId
}
if ($SizeGb -lt 1) {
    throw "SizeGb must be at least 1."
}

$name = $script:Sprint.VolumeName
# Get-RunpodVolumeByName returns $null only for genuine absence; an API/network
# error propagates as an exception (so we never "create" because a check failed).
$vol = Get-RunpodVolumeByName -Name $name
if ($vol) {
    Write-Host ("Volume '{0}' already exists: {1} ({2} GB, {3})" -f $name, $vol.id, $vol.size, $vol.dataCenterId) -ForegroundColor Green
    return
}

Write-Host ("Creating volume '{0}' ({1} GB) in {2}..." -f $name, $SizeGb, $DataCenter) -ForegroundColor Cyan
$body = [ordered]@{ name = $name; size = $SizeGb; dataCenterId = $DataCenter }
try {
    $vol = Invoke-RunpodApi -Path "/networkvolumes" -Method "POST" -Body $body
    Write-Host ("Created volume {0} ({1} GB, {2})" -f $vol.id, $vol.size, $vol.dataCenterId) -ForegroundColor Green
} catch {
    Write-Warning "Volume creation failed."
    if ($_.ErrorDetails.Message) { Write-Host $_.ErrorDetails.Message -ForegroundColor Red }
    exit 1
}
