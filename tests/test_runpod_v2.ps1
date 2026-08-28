$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $root "scripts\lib.ps1")
. (Join-Path $root "scripts\pod-up.ps1")

$script:count = 0
function Assert-True {
    param([bool]$Value, [string]$Message)
    $script:count++
    if (-not $Value) { throw "FAIL: $Message" }
}
function Assert-Equal {
    param($Actual, $Expected, [string]$Message)
    $script:count++
    if ($Actual -ne $Expected) { throw "FAIL: $Message; expected '$Expected', got '$Actual'" }
}
function Assert-ThrowsLike {
    param([scriptblock]$Action, [string]$Pattern, [string]$Message)
    $script:count++
    try {
        & $Action
        throw "FAIL: $Message; expected an exception"
    } catch {
        if ($_.Exception.Message -notmatch $Pattern) { throw "FAIL: $Message; unexpected exception: $($_.Exception.Message)" }
    }
}

Assert-Equal $script:Sprint.PodName "mats12-pod" "config exact pod identity"
Assert-Equal $script:Sprint.VolumeName "mats12" "config volume identity"
Assert-Equal $script:Sprint.TrackioVersion $script:RunpodConfig.TrackioVersion "config Trackio version projection"

$expectedVolumeSizeGb = (Import-PowerShellDataFile -LiteralPath (Join-Path $root "config\runpod.psd1")).DefaultVolumeSizeGb
$global:TestRunpodVolumeEnsureRequest = $null
$priorApiKey = $env:RUNPOD_API_KEY
try {
    $env:RUNPOD_API_KEY = "unit-test-runpod-key"
    function Invoke-RestMethod {
        param($Method, $Uri, $Headers, $ContentType, $Body)
        if ($Method -eq "GET") { return [pscustomobject]@{ items = @() } }
        if ($Method -eq "POST") {
            $global:TestRunpodVolumeEnsureRequest = $Body | ConvertFrom-Json
            return [pscustomobject]@{ id = "test-volume"; size = $global:TestRunpodVolumeEnsureRequest.size; dataCenterId = $global:TestRunpodVolumeEnsureRequest.dataCenterId }
        }
        throw "Unexpected mocked RunPod method: $Method"
    }
    & (Join-Path $root "scripts\volume-ensure.ps1")
    Assert-Equal ([int]$global:TestRunpodVolumeEnsureRequest.size) ([int]$expectedVolumeSizeGb) "volume ensure default size comes from runpod config"
} finally {
    Remove-Item -Path Function:\Invoke-RestMethod -ErrorAction SilentlyContinue
    Remove-Variable -Name TestRunpodVolumeEnsureRequest -Scope Global -ErrorAction SilentlyContinue
    $env:RUNPOD_API_KEY = $priorApiKey
}

$priorApiKey = $env:RUNPOD_API_KEY
$priorConfigPath = $env:RUNPOD_CONFIG_PATH
try {
    $env:RUNPOD_API_KEY = "unit-test-runpod-key"
    $env:RUNPOD_CONFIG_PATH = $null
    Assert-Equal (Get-RunpodApiKey) "unit-test-runpod-key" "environment API key resolves"
} finally {
    $env:RUNPOD_API_KEY = $priorApiKey
    $env:RUNPOD_CONFIG_PATH = $priorConfigPath
}

Assert-Equal (Get-RunpodCreateClassification $null) "ambiguous_transport" "statusless classification"
Assert-Equal (Get-RunpodCreateClassification 429) "transient" "429 classification"
Assert-Equal (Get-RunpodCreateClassification 503) "transient" "5xx classification"
Assert-Equal (Get-RunpodCreateClassification 400) "placement_or_cross_field" "400 classification"
Assert-Equal (Get-RunpodCreateClassification 401) "authentication_failure" "401 classification"
Assert-Equal (Get-RunpodCreateClassification 402) "insufficient_balance" "402 classification"
Assert-Equal (Get-RunpodCreateClassification 403) "access_denied" "403 classification"
Assert-Equal (Get-RunpodCreateClassification 422) "validation" "422 classification"

$AllowedCudaVersions = @("12.8")
$ExtraPort = @()
$PublicDashboard = $false
$Image = "image"
$DiskGb = 20
$body = New-RunpodV2CreateBody -GpuId "gpu" -VolumeId "volume" -RequestId "request" -PublicKey "ssh-ed25519 value"
Assert-Equal ([int]$body.gpu.count) ([int]$script:RunpodConfig.DefaultGpuCount) "default GPU count comes from runpod config"
$fourGpuBody = New-RunpodV2CreateBody -GpuId "gpu" -VolumeId "volume" -RequestId "request" -PublicKey "ssh-ed25519 value" -GpuCount 4
Assert-Equal ([int]$fourGpuBody.gpu.count) 4 "explicit GPU count is encoded"
Assert-Equal $body.env.MATS12_REQUEST_ID "request" "correlation field"
Assert-Equal $body.gpu.allowedCudaVersions[0] "12.8" "optional CUDA"
$safe = New-SanitizedCreateRequest $body
Assert-Equal $safe.env.PUBLIC_KEY "<redacted>" "redaction"
Assert-Equal (Get-Sha256Text ($safe | ConvertTo-Json -Compress -Depth 20)) (Get-Sha256Text ($safe | ConvertTo-Json -Compress -Depth 20)) "hash stability"

$AllowedCudaVersions = @()
$noCuda = New-RunpodV2CreateBody -GpuId "gpu" -VolumeId "volume" -RequestId "request" -PublicKey "ssh-ed25519 value"
Assert-True (-not $noCuda.gpu.Contains("allowedCudaVersions")) "CUDA omission"

$script:TestCatalogPath = $null
function Invoke-RunpodCatalogApi {
    param([string]$Path)
    $script:TestCatalogPath = $Path
    return [pscustomobject]@{ gpus = @() }
}
@(Get-RunpodGpuCatalog -GpuCount 4) | Out-Null
Assert-True ($script:TestCatalogPath -match "count=4") "catalog availability uses requested GPU count"
Remove-Item -Path Function:\Invoke-RunpodCatalogApi
Remove-Variable -Name TestCatalogPath -Scope Script

$pod = [pscustomobject]@{
    id = "id"; name = $script:Sprint.PodName; status = "RUNNING"
    gpu = [pscustomobject]@{ id = "gpu"; count = 1 }
    image = "image"; disk = 20; cloud = "SECURE"; dataCenterId = "EU-RO-1"
    ports = @("22/tcp"); cudaVersion = $null
    env = [pscustomobject]@{ MATS12_REQUEST_ID = "request" }
    mounts = [pscustomobject]@{ network = @([pscustomobject]@{ volumeId = "volume"; path = "/workspace" }) }
    ssh = [pscustomobject]@{ direct = [pscustomobject]@{ host = "203.0.113.1"; port = 22; username = "root" } }
}
Assert-True (Test-RunpodPodBaseIdentity $pod $noCuda -RequireCorrelation) "unrestricted CUDA accepts unknown host CUDA"
Assert-True (-not (Test-RunpodPodBaseIdentity $pod $body -RequireCorrelation)) "pinned CUDA rejects unknown host CUDA"
$pod.cudaVersion = "12.8"
Assert-True (Test-RunpodPodBaseIdentity $pod $body -RequireCorrelation) "pinned CUDA accepts exact host CUDA"
$pod.gpu.count = 4
Assert-True (Test-RunpodPodBaseIdentity $pod $fourGpuBody -RequireCorrelation) "explicit GPU count matches identity"
$pod.gpu.count = 1
Assert-True (-not (Test-RunpodPodBaseIdentity $pod $fourGpuBody -RequireCorrelation)) "mismatched GPU count rejects identity"
Assert-Equal ([int](Get-PodEvidence $pod).gpu_count) 1 "pod evidence records delivered GPU count"
$pod.image = "other"
Assert-True (-not (Test-RunpodPodBaseIdentity $pod $body)) "mismatched image rejected"
$pod.image = "image"
$active = ConvertTo-RunpodActivePod $pod
Assert-Equal $active.Ip "203.0.113.1" "direct SSH mapping"
Assert-Equal ([int]$active.GpuCount) 1 "active pod mapping records delivered GPU count"
$pod.status = "PROVISIONING"
Assert-True (-not (ConvertTo-RunpodActivePod $pod)) "provisioning is not SSH ready"
$pod.status = "RUNNING"

Assert-True (Test-SafeRelPath "run/file.txt") "safe relative path"
Assert-True (-not (Test-SafeRelPath "../escape")) "traversal rejected"
Assert-Equal (Get-RunpodV1ListItems ([pscustomobject]@{ items = @(1, 2) }) "network-volume").Count 2 "v1 items envelope"
Assert-ThrowsLike { Get-RunpodV1ListItems ([pscustomobject]@{ bad = @() }) "network-volume" } "items envelope" "unknown v1 envelope rejected"

function Invoke-RunpodApi {
    param($Path, $Method, $Body)
    return [pscustomobject]@{ items = @(
        [pscustomobject]@{ name = "mats12"; id = "one" },
        [pscustomobject]@{ name = "mats12"; id = "two" }
    ) }
}
Assert-ThrowsLike { Get-RunpodVolumeByName } "Multiple network volumes" "duplicate volume names fail closed"

function Invoke-RunpodV2Api {
    param($Path, $Method, $Body)
    return [pscustomobject]@{ pods = @($pod, [pscustomobject]@{ name = "other" }) }
}
Assert-Equal @(Get-RunpodExactNamePods).Count 1 "exact response filtering"
$reconciled = Invoke-CreateReconciliation -Body $body -TimeoutSec 1 -PollSec 1
Assert-Equal $reconciled.Outcome "correlated_match" "reconciliation adopts only correlated full identity"

. (Join-Path $root "scripts\runpod-sync.ps1")
. (Join-Path $root "scripts\push.ps1")
. (Join-Path $root "scripts\pull-loop.ps1")
Assert-True (-not (Test-SshAlias "good`nHost evil")) "newline alias rejected"
Assert-True (-not (Test-SshHostValue "host`nProxyCommand evil")) "newline host rejected"
Assert-True (-not (Test-SshUserToken "root`nHost evil")) "newline user rejected"
Assert-True (-not (Test-SshIdentityPath "key`nHost evil")) "newline identity rejected"
Assert-ThrowsLike { & (Join-Path $root "scripts\pod-up.ps1") -GpuCount 0 } "ValidateRange|range" "pod-up rejects GPU count below range"
Assert-ThrowsLike { & (Join-Path $root "scripts\session-up.ps1") -GpuCount 9 } "ValidateRange|range" "session-up rejects GPU count above range"

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("mats12-test-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory $tempRoot | Out-Null
try {
    $externalConfig = Join-Path $tempRoot "runpod-secret.psd1"
    [IO.File]::WriteAllText($externalConfig, "@{ RUNPOD_API_KEY = 'external-unit-test-key' }")
    $priorApiKey = $env:RUNPOD_API_KEY
    $priorConfigPath = $env:RUNPOD_CONFIG_PATH
    try {
        $env:RUNPOD_API_KEY = $null
        $env:RUNPOD_CONFIG_PATH = $externalConfig
        Assert-Equal (Get-RunpodApiKey) "external-unit-test-key" "external PSD1 API key resolves"
    } finally {
        $env:RUNPOD_API_KEY = $priorApiKey
        $env:RUNPOD_CONFIG_PATH = $priorConfigPath
    }

    $config = Join-Path $tempRoot "config"
    [IO.File]::WriteAllText($config, "# === safe (managed by old/scripts/runpod-sync.ps1 - do not edit by hand) ===`nHost safe`n    HostName stale.example`n# === end safe ===`n`nHost unrelated`n    HostName example`n")
    $block = New-ManagedSshBlock "safe" "example.com" "root" 22 "C:\key"
    Update-ManagedSshConfig $config "safe" $block
    $updatedConfig = [IO.File]::ReadAllText($config)
    Assert-True ($updatedConfig -match "Host unrelated") "unrelated SSH entry preserved"
    Assert-True ($updatedConfig -notmatch "stale.example") "stale managed alias removed"
    Assert-Equal ([regex]::Matches($updatedConfig, "(?m)^Host safe$").Count) 1 "only one managed alias remains"
    Update-ManagedSshConfig $config "safe" $block
    $secondUpdate = [IO.File]::ReadAllText($config)
    Assert-Equal $secondUpdate $updatedConfig "SSH update is byte-idempotent"
    Assert-True (Test-Path "$config.bak") "SSH backup exists"
    Assert-Equal ([IO.File]::ReadAllText("$config.bak")) $updatedConfig "SSH backup preserves previous config"

    $multiConfig = Join-Path $tempRoot "config-multiple"
    $multiOriginal = "# === safe (managed by old/scripts/runpod-sync.ps1 - do not edit by hand) ===`r`nHost safe`r`n    HostName first-stale.example`r`n# === end safe ===`r`n`r`nHost manual-critical`r`n    HostName manual.example`r`n`r`n# === safe (managed by newer/scripts/runpod-sync.ps1) ===`r`nHost safe`r`n    HostName second-stale.example`r`n# === end safe ===`r`n`r`nHost manual-tail`r`n    HostName tail.example`r`n"
    [IO.File]::WriteAllText($multiConfig, $multiOriginal)
    Update-ManagedSshConfig $multiConfig "safe" $block
    $multiUpdated = [IO.File]::ReadAllText($multiConfig)
    Assert-True ($multiUpdated -match "Host manual-critical`n    HostName manual.example") "manual host between managed blocks preserved"
    Assert-True ($multiUpdated -match "Host manual-tail`n    HostName tail.example") "manual host after managed blocks preserved"
    Assert-True ($multiUpdated -notmatch "first-stale|second-stale") "multiple stale managed aliases removed"
    Assert-Equal ([regex]::Matches($multiUpdated, "(?m)^Host safe$").Count) 1 "multiple managed aliases collapse to one"

    $brokenConfig = Join-Path $tempRoot "config-broken"
    $brokenOriginal = "# === safe (managed by old/scripts/runpod-sync.ps1 - do not edit by hand) ===`nHost safe`n    HostName truncated.example`nHost manual-critical`n    HostName preserve.example`n# === safe (managed by newer/scripts/runpod-sync.ps1) ===`nHost safe`n    HostName complete.example`n# === end safe ===`n"
    [IO.File]::WriteAllText($brokenConfig, $brokenOriginal)
    Assert-ThrowsLike { Update-ManagedSshConfig $brokenConfig "safe" $block } "Malformed managed SSH block" "truncated managed block fails closed"
    Assert-Equal ([IO.File]::ReadAllText($brokenConfig)) $brokenOriginal "failed SSH update preserves the original file exactly"

    $remoteTemp = "/tmp/a b'c;`$x"
    $remoteFinal = "/in/final a'c;`$x"
    $command = New-RemoteNoClobberPublishCommand $remoteTemp $remoteFinal
    Assert-True ($command -match "if \[ -e") "single no-clobber decision"
    Assert-True ($command -match "'\\''") "apostrophes are POSIX quoted"
    Assert-True ($command -match '\$x') "shell metacharacter remains inside quotes"

    $final = Join-Path $tempRoot "final"
    $temp = Get-LocalAtomicTempPath $final
    [IO.File]::WriteAllText($temp, "bytes")
    Publish-DownloadedFile $temp $final 5 0
    Assert-Equal ([IO.File]::ReadAllText($final)) "bytes" "atomic initial publish preserves bytes"

    $readOnlyFinal = Join-Path $tempRoot "read-only-final"
    $readOnlyTemp = Get-LocalAtomicTempPath $readOnlyFinal
    [IO.File]::WriteAllText($readOnlyTemp, "fixed")
    (Get-Item -LiteralPath $readOnlyTemp).IsReadOnly = $true
    Publish-DownloadedFile $readOnlyTemp $readOnlyFinal 5 0
    Assert-Equal ([IO.File]::ReadAllText($readOnlyFinal)) "fixed" "read-only SCP temp publishes atomically"
    Assert-True (-not (Get-Item -LiteralPath $readOnlyFinal).IsReadOnly) "published mirror is locally writable"

    [IO.File]::WriteAllText($final, "old")
    $replacement = Get-LocalAtomicTempPath $final
    [IO.File]::WriteAllText($replacement, "newer")
    Publish-DownloadedFile $replacement $final 5 0
    Assert-Equal ([IO.File]::ReadAllText($final)) "newer" "atomic replacement preserves new bytes"

    [IO.File]::WriteAllText($final, "stable")
    $mismatch = Get-LocalAtomicTempPath $final
    [IO.File]::WriteAllText($mismatch, "bad")
    Assert-ThrowsLike { Publish-DownloadedFile $mismatch $final 99 0 } "size mismatch" "size mismatch refuses replacement"
    Assert-Equal ([IO.File]::ReadAllText($final)) "stable" "size mismatch preserves final file"
    Assert-True (-not (Test-Path $mismatch)) "size mismatch removes temp file"
} finally {
    Remove-Item $tempRoot -Recurse -Force
}

Write-Host "PASS: $script:count focused RunPod tests"
