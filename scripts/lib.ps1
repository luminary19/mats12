# Shared RunPod helpers. All settings come from config/runpod.psd1.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Get-RunpodConfig {
    $root = Split-Path -Parent $PSScriptRoot
    $path = Join-Path $root "config\runpod.psd1"
    if (-not (Test-Path -LiteralPath $path)) { throw "RunPod config not found: $path" }
    $config = Import-PowerShellDataFile -LiteralPath $path
    $required = @(
        "ProjectName", "DefaultPodName", "VolumeName", "DefaultVolumeSizeGb",
        "DataCenterId", "WorkspacePath", "RemoteRuns", "RemoteInbox", "RemoteCode",
        "RemoteTrackio", "RemoteVenv", "DefaultImage", "DefaultDiskGb", "DefaultCloud",
        "DefaultGpuCount", "DefaultPorts", "TrackioVersion", "SshAlias", "SshUser",
        "SshKeyRelativePath", "LocalRunsDirectory", "LocalUploadDirectory", "HeartbeatName",
        "HeartbeatMaxAgeSec", "ReconcileTimeoutSec", "ReconcilePollSec", "ApiBase",
        "V2ApiBase", "CatalogApiBase"
    )
    foreach ($name in $required) {
        if (-not $config.ContainsKey($name) -or [string]::IsNullOrWhiteSpace([string]$config[$name])) {
            throw "RunPod config is missing '$name'."
        }
    }
    if ([int]$config.DefaultGpuCount -ne 1) { throw "This controller supports exactly one GPU." }
    if ([int]$config.ReconcileTimeoutSec -lt 1 -or [int]$config.ReconcilePollSec -lt 1) { throw "Reconciliation timing must be positive." }
    if ([string]$config.DataCenterId -notmatch "^[A-Z0-9-]+$") { throw "Configured datacenter is invalid." }
    return $config
}

$script:RunpodConfig = Get-RunpodConfig
$script:ProjectRoot = Split-Path -Parent $PSScriptRoot
$keyRelative = $script:RunpodConfig.SshKeyRelativePath.Replace("/", [string][IO.Path]::DirectorySeparatorChar)
$script:Sprint = [ordered]@{
    ProjectName = $script:RunpodConfig.ProjectName
    PodName = $script:RunpodConfig.DefaultPodName
    VolumeName = $script:RunpodConfig.VolumeName
    DataCenterId = $script:RunpodConfig.DataCenterId
    WorkspacePath = $script:RunpodConfig.WorkspacePath
    RemoteRuns = $script:RunpodConfig.RemoteRuns
    RemoteInbox = $script:RunpodConfig.RemoteInbox
    RemoteCode = $script:RunpodConfig.RemoteCode
    RemoteTrackio = $script:RunpodConfig.RemoteTrackio
    RemoteVenv = $script:RunpodConfig.RemoteVenv
    LocalRuns = Join-Path $script:ProjectRoot $script:RunpodConfig.LocalRunsDirectory
    LocalUpload = Join-Path $script:ProjectRoot $script:RunpodConfig.LocalUploadDirectory
    SshKey = Join-Path $HOME $keyRelative
    SshUser = $script:RunpodConfig.SshUser
    SshAlias = $script:RunpodConfig.SshAlias
    HeartbeatName = $script:RunpodConfig.HeartbeatName
    HeartbeatMaxAgeSec = [int]$script:RunpodConfig.HeartbeatMaxAgeSec
    ApiBase = $script:RunpodConfig.ApiBase
    V2ApiBase = $script:RunpodConfig.V2ApiBase
    CatalogApiBase = $script:RunpodConfig.CatalogApiBase
}

function Get-HeartbeatMmin { return [Math]::Max(1, [Math]::Ceiling($script:Sprint.HeartbeatMaxAgeSec / 60.0)) }
function Test-RunpodSecretValue { param([string]$Value) return -not [string]::IsNullOrWhiteSpace($Value) -and $Value -notmatch "\s" }
function Get-RunpodApiKey {
    if (Test-RunpodSecretValue $env:RUNPOD_API_KEY) { return $env:RUNPOD_API_KEY }
    if (-not $env:RUNPOD_CONFIG_PATH) {
        throw "RUNPOD_API_KEY is unset. Set a nonblank key in the process/user environment or set RUNPOD_CONFIG_PATH to an uncommitted external PowerShell data file."
    }
    $externalPath = [IO.Path]::GetFullPath($env:RUNPOD_CONFIG_PATH)
    $rootPath = [IO.Path]::GetFullPath($script:ProjectRoot).TrimEnd("\", "/") + [IO.Path]::DirectorySeparatorChar
    if (-not (Test-Path -LiteralPath $externalPath)) { throw "RUNPOD_CONFIG_PATH does not exist." }
    if ($externalPath.StartsWith($rootPath, [StringComparison]::OrdinalIgnoreCase)) { throw "RUNPOD_CONFIG_PATH must be outside this repository." }
    $external = Import-PowerShellDataFile -LiteralPath $externalPath
    if (Test-RunpodSecretValue ([string]$external.RUNPOD_API_KEY)) { return [string]$external.RUNPOD_API_KEY }
    throw "RUNPOD_CONFIG_PATH does not contain a valid nonblank RUNPOD_API_KEY."
}

function Invoke-RunpodApi {
    param([string]$Path, [string]$Method = "GET", $Body = $null)
    $headers = @{ Authorization = "Bearer $(Get-RunpodApiKey)" }
    $uri = $script:Sprint.ApiBase + $Path
    if ($null -ne $Body) { return Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers -ContentType "application/json" -Body ($Body | ConvertTo-Json -Depth 20) }
    return Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers
}
function Invoke-RunpodV2Api {
    param([string]$Path, [string]$Method = "GET", $Body = $null)
    $headers = @{ Authorization = "Bearer $(Get-RunpodApiKey)" }
    $uri = $script:Sprint.V2ApiBase + $Path
    if ($null -ne $Body) { return Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers -ContentType "application/json" -Body ($Body | ConvertTo-Json -Depth 20) }
    return Invoke-RestMethod -Method $Method -Uri $uri -Headers $headers
}
function Invoke-RunpodCatalogApi { param([string]$Path) return Invoke-RestMethod -Method GET -Uri ($script:Sprint.CatalogApiBase + $Path) -Headers @{ Authorization = "Bearer $(Get-RunpodApiKey)" } }

function Get-RunpodV1ListItems {
    param($Response, [string]$ResourceName)
    if ($Response -is [array]) { return @($Response) }
    if ($null -eq $Response -or -not $Response.PSObject.Properties["items"] -or $Response.items -isnot [array]) {
        throw "Unexpected v1 $ResourceName list response; expected an items envelope."
    }
    return @($Response.items)
}
function Get-RunpodVolumeByName {
    param([string]$Name = $script:Sprint.VolumeName)
    $matches = @(Get-RunpodV1ListItems -Response (Invoke-RunpodApi -Path "/networkvolumes") -ResourceName "network-volume" | Where-Object { [string]$_.name -eq $Name })
    if ($matches.Count -gt 1) { throw "Multiple network volumes are named '$Name'. Resolve the duplicate before continuing." }
    if ($matches.Count -eq 0) { return $null }
    return $matches[0]
}
function Get-RunpodGpuCatalog {
    param([ValidateSet("SECURE", "COMMUNITY")][string]$Cloud = "SECURE")
    $response = Invoke-RunpodCatalogApi -Path "/v2/catalog/gpus?include=AVAILABILITY&product=POD&count=1&cloud=$Cloud"
    if ($response.gpus) { return @($response.gpus) }
    return @($response)
}
function Resolve-RunpodGpuIds {
    param([string[]]$Requested, [object[]]$Catalog)
    $resolved = @()
    foreach ($raw in @($Requested | ForEach-Object { ([string]$_).Split(",") })) {
        $value = $raw.Trim()
        if (-not $value) { continue }
        $matches = @($Catalog | Where-Object { ([string]$_.id).Equals($value, [StringComparison]::OrdinalIgnoreCase) -or ([string]$_.name).Equals($value, [StringComparison]::OrdinalIgnoreCase) })
        if ($matches.Count -eq 0) { $matches = @($Catalog | Where-Object { ([string]$_.id).IndexOf($value, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or ([string]$_.name).IndexOf($value, [StringComparison]::OrdinalIgnoreCase) -ge 0 }) }
        if ($matches.Count -gt 1) { throw "GPU '$value' is ambiguous; use an exact catalog ID." }
        if ($matches.Count -eq 1) { $resolved += [string]$matches[0].id } else { $resolved += $value }
    }
    if ($resolved.Count -ne 1) { throw "Pass exactly one explicit GPU ID." }
    return [string[]]$resolved
}
function ConvertTo-PosixSingleQuoted { param([string]$Text) return "'" + ($Text -replace "'", "'\''") + "'" }
function Test-SafeRelPath {
    param([string]$Rel)
    return [bool]($Rel -and $Rel -notmatch "[\x00-\x1f]" -and $Rel -notmatch "(^|[\\/])\.\.([\\/]|$)" -and $Rel -notmatch "^[\\/]" -and $Rel -notmatch "^[A-Za-z]:" -and -not $Rel.Contains("\"))
}
function Get-RunpodV2Pods {
    $response = Invoke-RunpodV2Api -Path "/pods"
    if ($null -eq $response -or -not $response.PSObject.Properties["pods"] -or $response.pods -isnot [array]) { throw "Unexpected v2 pod list response; expected a pods envelope." }
    return @($response.pods)
}
function Get-RunpodExactNamePods { return @(Get-RunpodV2Pods | Where-Object { [string]$_.name -eq $script:Sprint.PodName }) }
function Get-RunpodNetworkMount {
    param($Pod)
    $mounts = @($Pod.mounts.network)
    if ($mounts.Count -ne 1) { return $null }
    return $mounts[0]
}
function Test-RunpodKnownStatus { param([string]$Status) return $Status -in @("PROVISIONING", "STARTING", "RUNNING", "EXITED", "ERROR", "TERMINATED") }
function Test-RunpodPodBaseIdentity {
    param($Pod, [System.Collections.IDictionary]$Body, [switch]$RequireCorrelation)
    if (-not $Pod -or -not (Test-RunpodKnownStatus ([string]$Pod.status))) { return $false }
    if ($Pod.PSObject.Properties["cluster"] -and $Pod.cluster) { return $false }
    $mount = Get-RunpodNetworkMount $Pod
    if (-not $mount) { return $false }
    $podPorts = (@($Pod.ports | ForEach-Object { [string]$_ } | Sort-Object) -join ",")
    $requestPorts = (@($Body.ports | ForEach-Object { [string]$_ } | Sort-Object) -join ",")
    $portMatch = $podPorts -eq $requestPorts
    $cuda = @($Body.gpu.allowedCudaVersions | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) })
    $cudaMatch = $cuda.Count -eq 0 -or ($Pod.cudaVersion -and ([string]$Pod.cudaVersion -in $cuda))
    $matches = ([string]$Pod.name -eq [string]$Body.name -and [string]$Pod.gpu.id -eq [string]$Body.gpu.id -and [int]$Pod.gpu.count -eq [int]$Body.gpu.count -and [string]$Pod.image -eq [string]$Body.image -and [int]$Pod.disk -eq [int]$Body.disk -and [string]$Pod.cloud -eq [string]$Body.cloud -and [string]$Pod.dataCenterId -eq $script:Sprint.DataCenterId -and [string]$mount.volumeId -eq [string]$Body.mounts.network[0].volumeId -and [string]$mount.path -eq $script:Sprint.WorkspacePath -and $portMatch -and $cudaMatch)
    if (-not $matches -or -not $RequireCorrelation) { return $matches }
    return [string]$Pod.env.MATS12_REQUEST_ID -eq [string]$Body.env.MATS12_REQUEST_ID
}
function ConvertTo-RunpodActivePod {
    param($Pod)
    if (-not $Pod -or [string]$Pod.status -ne "RUNNING" -or -not $Pod.ssh.direct -or -not $Pod.ssh.direct.host -or -not $Pod.ssh.direct.port) { return $null }
    return [pscustomobject]@{ Id=$Pod.id; Name=$Pod.name; Ip=[string]$Pod.ssh.direct.host; Port=[int]$Pod.ssh.direct.port; User=if($Pod.ssh.direct.username){[string]$Pod.ssh.direct.username}else{$script:Sprint.SshUser}; Gpu=[string]$Pod.gpu.id; DataCenter=[string]$Pod.dataCenterId; CostPerHr=$Pod.cost }
}
function Resolve-RunpodOwnedPod {
    param([System.Collections.IDictionary]$Body, [switch]$RequireCorrelation)
    $pods = @(Get-RunpodExactNamePods)
    if ($pods.Count -eq 0) { return $null }
    if ($pods.Count -ne 1) { throw "Multiple pods share configured pod name '$($script:Sprint.PodName)'; refusing ambiguity." }
    if (-not (Test-RunpodPodBaseIdentity -Pod $pods[0] -Body $Body -RequireCorrelation:$RequireCorrelation)) { throw "Exact-name pod does not match configured identity; inspect or terminate it manually." }
    return $pods[0]
}
function Get-SshBaseArgs { param($Pod) return @("-i", $script:Sprint.SshKey, "-p", "$($Pod.Port)", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30") }
function Invoke-PodSsh {
    param($Pod, [string]$Command, [switch]$AllowFail)
    $target = "$(if($Pod.User){$Pod.User}else{$script:Sprint.SshUser})@$($Pod.Ip)"
    $raw = & ssh @(Get-SshBaseArgs $Pod) $target $Command 2>&1
    $exitCode = $LASTEXITCODE; $out = @(); $err = @()
    foreach ($item in $raw) { if ($item -is [Management.Automation.ErrorRecord]) { $err += $item.ToString() } else { $out += [string]$item } }
    $result = [pscustomobject]@{ ExitCode=$exitCode; Ok=($exitCode -eq 0); StdOut=($out -join "`n"); StdErr=($err -join "`n"); Lines=$out }
    if (-not $result.Ok -and -not $AllowFail) { throw "ssh failed (exit $exitCode): $($result.StdErr) $($result.StdOut)" }
    return $result
}
function Copy-FileToPod { param($Pod,[string]$LocalPath,[string]$RemotePath) & scp -p -P "$($Pod.Port)" -i $script:Sprint.SshKey -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15 $LocalPath "$(if($Pod.User){$Pod.User}else{$script:Sprint.SshUser})@$($Pod.Ip):$RemotePath"; return ($LASTEXITCODE -eq 0) }
function Copy-FileFromPod { param($Pod,[string]$RemotePath,[string]$LocalPath) & scp -p -P "$($Pod.Port)" -i $script:Sprint.SshKey -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15 "$(if($Pod.User){$Pod.User}else{$script:Sprint.SshUser})@$($Pod.Ip):$RemotePath" $LocalPath; return ($LASTEXITCODE -eq 0) }
function Get-LocalAtomicTempPath { param([string]$FinalPath) $full=[IO.Path]::GetFullPath($FinalPath); return (Join-Path (Split-Path -Parent $full) ("." + [IO.Path]::GetFileName($full) + "." + [Guid]::NewGuid().ToString("N") + ".tmp")) }
function Write-RunpodBanner { param($Pod,[string]$Action) Write-Host ("[{0}] {1} ({2}, {3}) {4}:{5} ~`${6}/hr" -f $Action,$Pod.Name,$Pod.Gpu,$Pod.DataCenter,$Pod.Ip,$Pod.Port,$Pod.CostPerHr) -ForegroundColor Cyan }
function Resolve-RunpodPodOrThrow {
    $pods = @(Get-RunpodExactNamePods)
    if ($pods.Count -ne 1) { throw "Expected one exact configured pod; found $($pods.Count)." }
    $active = ConvertTo-RunpodActivePod $pods[0]
    if (-not $active) { throw "Configured pod is not RUNNING with direct SSH." }
    return $active
}
