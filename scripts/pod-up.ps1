<# Create or adopt the single configured pod through RunPod REST v2. #>
[CmdletBinding()]
param(
    [Alias("GpuId")][string[]]$Gpu,
    [switch]$ListGpu,
    [switch]$AvailableOnly,
    [int]$DiskGb,
    [string]$Image,
    [string[]]$ExtraPort = @(),
    [ValidateSet("11.8","12.0","12.1","12.2","12.3","12.4","12.5","12.6","12.7","12.8","12.9","13.0")][string[]]$AllowedCudaVersions = @(),
    [string]$TemplateId,
    [switch]$PublicDashboard,
    [switch]$NoSync,
    [switch]$DryRun,
    [string]$EvidencePath,
    [ValidateRange(1,5)][int]$CreateAttempts = 1,
    [ValidateRange(1,60)][int]$RetryBaseSeconds = 2
)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\lib.ps1"

function Get-UtcStamp { return (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ") }
function Get-Sha256Text {
    param([string]$Text)
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Text)))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}
function ConvertTo-SafePodUpText {
    param($Value)
    if ($null -eq $Value) { return $null }
    $text = [string]$Value
    $text = $text -replace '(?i)("PUBLIC_KEY"\s*:\s*)"[^"]*"', '$1"<redacted>"'
    $text = $text -replace '(?i)bearer\s+[^\s,;]+', 'Bearer <redacted>'
    $text = $text -replace '(?i)RUNPOD_API_KEY\s*[:=]\s*[^\s,;}]+', 'RUNPOD_API_KEY=<redacted>'
    $text = $text -replace 'ssh-(rsa|ed25519)\s+[^\s"]+', 'ssh-$1 <redacted>'
    return $text
}
function Write-AtomicJson {
    param([string]$Path, $Value)
    if (-not $Path) { return }
    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $temporary = Get-LocalAtomicTempPath -FinalPath $Path
    [IO.File]::WriteAllText($temporary, ($Value | ConvertTo-Json -Depth 20) + [Environment]::NewLine, [Text.Encoding]::UTF8)
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}
function Get-RunpodHttpFailure {
    param($ErrorRecord)
    $status = $null; $detail = $null; $retryAfter = $null
    if ($ErrorRecord.Exception -and $ErrorRecord.Exception.Response) {
        $response = $ErrorRecord.Exception.Response
        try { $status = [int]$response.StatusCode } catch { }
        try { $seconds = 0; $header = $response.Headers["Retry-After"]; if ($header -and [int]::TryParse([string]$header,[ref]$seconds) -and $seconds -gt 0) { $retryAfter = $seconds } } catch { }
    }
    if ($ErrorRecord.ErrorDetails.Message) { $detail = $ErrorRecord.ErrorDetails.Message }
    if (-not $detail) { $detail = $ErrorRecord.Exception.Message }
    return [pscustomobject]@{ Status=$status; Detail=(ConvertTo-SafePodUpText $detail); RetryAfterSeconds=$retryAfter }
}
function Get-RunpodCreateClassification {
    param([Nullable[int]]$Status)
    if ($null -eq $Status) { return "ambiguous_transport" }
    if ($Status -eq 429 -or ($Status -ge 500 -and $Status -le 599)) { return "transient" }
    if ($Status -eq 400) { return "placement_or_cross_field" }
    if ($Status -eq 401) { return "authentication_failure" }
    if ($Status -eq 402) { return "insufficient_balance" }
    if ($Status -eq 403) { return "access_denied" }
    if ($Status -eq 422) { return "validation" }
    return "unknown_failure"
}
function New-RunpodV2CreateBody {
    param([string]$GpuId,[string]$VolumeId,[string]$RequestId,[string]$PublicKey)
    $ports = @($script:RunpodConfig.DefaultPorts) + $ExtraPort
    if ($PublicDashboard) { $ports += "7860/http" }
    if (-not $PublicKey) { $PublicKey = (Get-Content "$($script:Sprint.SshKey).pub" -Raw).Trim() }
    if (-not $PublicKey) { throw "Configured SSH public key is empty." }
    $body = [ordered]@{
        name = $script:Sprint.PodName
        image = $Image
        disk = $DiskGb
        cloud = $script:RunpodConfig.DefaultCloud
        gpu = [ordered]@{ id=$GpuId; count=1 }
        mounts = [ordered]@{ network=@([ordered]@{ volumeId=$VolumeId; path=$script:Sprint.WorkspacePath }) }
        ports = @($ports | Select-Object -Unique)
        env = [ordered]@{ PUBLIC_KEY=$PublicKey; MATS12_REQUEST_ID=$RequestId }
        startSsh = $true
    }
    if ($AllowedCudaVersions.Count) { $body.gpu.allowedCudaVersions = @($AllowedCudaVersions) }
    if ($TemplateId) { $body.templateId = $TemplateId }
    return $body
}
function New-SanitizedCreateRequest {
    param([System.Collections.IDictionary]$Body)
    $safe = [ordered]@{}
    foreach ($key in $Body.Keys) {
        if ($key -eq "env") { $safe.env = @{ PUBLIC_KEY="<redacted>"; MATS12_REQUEST_ID=$Body.env.MATS12_REQUEST_ID } }
        else { $safe[$key] = $Body[$key] }
    }
    return $safe
}
function Get-PodEvidence {
    param($Pod)
    if (-not $Pod) { return $null }
    return [ordered]@{ id=$Pod.id; name=$Pod.name; status=$Pod.status; gpu=if($Pod.gpu){$Pod.gpu.id}else{$null}; data_center_id=$Pod.dataCenterId; volume_id=(Get-RunpodNetworkMount $Pod).volumeId; cost_per_hr=$Pod.cost }
}
function Invoke-CreateReconciliation {
    param([System.Collections.IDictionary]$Body,[int]$TimeoutSec,[int]$PollSec)
    $polls = @(); $deadline = (Get-Date).AddSeconds($TimeoutSec)
    do {
        try {
            $pods = @(Get-RunpodExactNamePods)
            $outcome = if($pods.Count -eq 0){"zero"}elseif($pods.Count -gt 1){"multiple"}elseif(Test-RunpodPodBaseIdentity -Pod $pods[0] -Body $Body -RequireCorrelation){"correlated_match"}elseif(Test-RunpodPodBaseIdentity -Pod $pods[0] -Body $Body){"uncorrelated_match"}else{"mismatch"}
            $polls += [ordered]@{ utc=Get-UtcStamp; exact_name_count=$pods.Count; outcome=$outcome; pod_ids=@($pods|ForEach-Object{$_.id}) }
            if ($outcome -ne "zero") { return [pscustomobject]@{ Outcome=$outcome; Pod=if($pods.Count -eq 1){$pods[0]}else{$null}; Polls=$polls } }
        } catch { $polls += [ordered]@{ utc=Get-UtcStamp; outcome="reconciliation_error"; detail=(ConvertTo-SafePodUpText $_.Exception.Message) }; return [pscustomobject]@{ Outcome="reconciliation_error"; Pod=$null; Polls=$polls } }
        if ((Get-Date) -lt $deadline) { Start-Sleep -Seconds $PollSec }
    } while ((Get-Date) -lt $deadline)
    return [pscustomobject]@{ Outcome="zero"; Pod=$null; Polls=$polls }
}

if ($MyInvocation.InvocationName -eq ".") { return }
if (-not $Image) { $Image = $script:RunpodConfig.DefaultImage }
if (-not $DiskGb) { $DiskGb = [int]$script:RunpodConfig.DefaultDiskGb }
if (-not $Gpu -and -not $ListGpu) { throw "Pass one explicit -Gpu, or use -ListGpu." }
if ($ExtraPort | Where-Object { $_ -notmatch '^\d+/(tcp|http)$' }) { throw "Each ExtraPort must be PORT/tcp or PORT/http." }

if ($ListGpu) {
    $catalog = @(Get-RunpodGpuCatalog -Cloud $script:RunpodConfig.DefaultCloud)
    $rows = foreach($item in $catalog) { $dc=@($item.dataCenters|Where-Object{$_.id -eq $script:Sprint.DataCenterId}|Select-Object -First 1); [pscustomobject]@{ID=$item.id;DisplayName=$item.name;Availability=if($dc.Count){$dc[0].availability}else{"NONE"}} }
    if($AvailableOnly){$rows=@($rows|Where-Object{$_.Availability-ne"NONE"})}; $rows|Sort-Object ID|Format-Table -AutoSize; return
}
$volume = Get-RunpodVolumeByName
if (-not $volume) { throw "Volume '$($script:Sprint.VolumeName)' not found. Run volume-ensure.ps1 first." }
if ([string]$volume.dataCenterId -ne $script:Sprint.DataCenterId) { throw "Resolved volume datacenter does not match configuration." }
$catalog = @(Get-RunpodGpuCatalog -Cloud $script:RunpodConfig.DefaultCloud)
$gpuId = @(Resolve-RunpodGpuIds -Requested $Gpu -Catalog $catalog)[0]
$requestId = [Guid]::NewGuid().ToString("N")
$body = New-RunpodV2CreateBody -GpuId $gpuId -VolumeId $volume.id -RequestId $requestId
$safe = New-SanitizedCreateRequest $body
$evidence = [ordered]@{ schema=3; api_version="v2"; created_utc=Get-UtcStamp; request_hash=(Get-Sha256Text ($safe|ConvertTo-Json -Compress -Depth 20)); request=$safe; attempts=@(); catalog_cuda_versions=@($catalog|Where-Object{$_.id -eq $gpuId}|ForEach-Object{@{gpu_id=$gpuId;cuda_versions=@($_.cudaVersions)}}) }
$existing = Invoke-CreateReconciliation -Body $body -TimeoutSec 1 -PollSec 1
$created = $null
if ($existing.Outcome -in @("correlated_match", "uncorrelated_match")) {
    $evidence.preflight = $existing.Polls
    $status = [string]$existing.Pod.status
    if ($DryRun) {
        $evidence.dry_run = $true
        $evidence.would_adopt = Get-PodEvidence $existing.Pod
        Write-AtomicJson $EvidencePath $evidence
        $safe | ConvertTo-Json -Depth 20
        return
    }
    if ($status -in @("EXITED", "ERROR", "TERMINATED")) {
        Write-AtomicJson $EvidencePath $evidence
        throw "Configured pod exists in $status state. Terminate or explicitly recover it before starting a new session."
    }
    $created = $existing.Pod
    Write-Host "Adopting configured pod in $status state and waiting for direct SSH." -ForegroundColor Cyan
} elseif ($existing.Outcome -ne "zero") {
    $evidence.preflight = $existing.Polls
    Write-AtomicJson $EvidencePath $evidence
    throw "Preflight found an ambiguous or mismatched exact-name pod; do not create another."
} elseif ($DryRun) {
    $evidence.dry_run = $true
    Write-AtomicJson $EvidencePath $evidence
    $safe | ConvertTo-Json -Depth 20
    return
}
for($attempt=1;$attempt -le $CreateAttempts -and -not $created;$attempt++) {
    $record=[ordered]@{utc=Get-UtcStamp;attempt=$attempt;request_hash=$evidence.request_hash;classification=$null;reconciliation=@()}; $evidence.attempts += $record; Write-AtomicJson $EvidencePath $evidence
    try { $created=Invoke-RunpodV2Api -Path "/pods" -Method POST -Body $body; $record.classification="created"; $record.created_pod=Get-PodEvidence $created; break }
    catch {
        $failure=Get-RunpodHttpFailure $_; $record.http_status=$failure.Status; $record.detail=$failure.Detail; $record.retry_after_seconds=$failure.RetryAfterSeconds; $record.classification=Get-RunpodCreateClassification $failure.Status
        if($record.classification -notin @("transient","ambiguous_transport")){break}
        $reconciled=Invoke-CreateReconciliation -Body $body -TimeoutSec ([int]$script:RunpodConfig.ReconcileTimeoutSec) -PollSec ([int]$script:RunpodConfig.ReconcilePollSec); $record.reconciliation=$reconciled.Polls
        if($reconciled.Outcome -eq "correlated_match"){$created=$reconciled.Pod;$record.classification="created_after_reconciliation";break}
        if($reconciled.Outcome -ne "zero"){$record.retry_suppressed=$reconciled.Outcome;break}
        if($attempt -lt $CreateAttempts){$backoff=[int][Math]::Min(30,$RetryBaseSeconds*[Math]::Pow(2,$attempt-1));$delay=if($failure.RetryAfterSeconds){[Math]::Max($backoff,[int]$failure.RetryAfterSeconds)}else{$backoff};$record.retry_delay_seconds=$delay;Write-AtomicJson $EvidencePath $evidence;Start-Sleep -Seconds $delay}
    }
}
Write-AtomicJson $EvidencePath $evidence
if(-not $created){throw "Pod creation was not confirmed. Inspect sanitized evidence and the provider console before retrying."}
$deadline = (Get-Date).AddMinutes(3)
$ready = if (ConvertTo-RunpodActivePod $created) { $created } else { $null }
while (-not $ready -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 8
    $current = Invoke-RunpodV2Api -Path "/pods/$($created.id)"
    if (ConvertTo-RunpodActivePod $current) { $ready = $current; break }
    if ([string]$current.status -in @("EXITED", "ERROR", "TERMINATED")) { break }
}
if(-not $ready){throw "Pod was created and may be billing, but direct SSH is not ready. Check status and terminate it if unused."}
if(-not $NoSync){& "$PSScriptRoot\runpod-sync.ps1"}
Write-Host "Pod is RUNNING and may bill until pod-down.ps1 confirms deletion." -ForegroundColor Yellow
