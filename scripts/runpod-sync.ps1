<# Safely update one managed SSH alias block for the exact configured pod. #>
[CmdletBinding()]
param([string]$SshConfig = (Join-Path $HOME ".ssh\config"), [string]$Alias)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\lib.ps1"

function Test-SshAlias { param([string]$Value) return [bool]($Value -match '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') }
function Test-SshUserToken { param([string]$Value) return [bool]($Value -and $Value -match '^[A-Za-z_][A-Za-z0-9._-]{0,63}$') }
function Test-SshHostValue {
    param([string]$Value)
    if (-not $Value -or $Value -match '[\x00-\x20#;{}\\]' -or $Value -match '^-') { return $false }
    $ip = [Net.IPAddress]::None
    if ([Net.IPAddress]::TryParse($Value, [ref]$ip)) { return $true }
    return [bool]($Value.Length -le 253 -and $Value -match '^(?=.{1,253}$)([A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$')
}
function Test-SshIdentityPath { param([string]$Value) return [bool]($Value -and $Value -notmatch '[\x00-\x1f]') }
function New-ManagedSshBlock {
    param([string]$Alias,[string]$HostName,[string]$User,[int]$Port,[string]$IdentityPath)
    if (-not(Test-SshAlias $Alias)) { throw "Unsafe SSH alias." }
    if (-not(Test-SshHostValue $HostName)) { throw "Unsafe SSH host." }
    if (-not(Test-SshUserToken $User)) { throw "Unsafe SSH username." }
    if ($Port -lt 1 -or $Port -gt 65535) { throw "Unsafe SSH port." }
    if (-not(Test-SshIdentityPath $IdentityPath)) { throw "Unsafe SSH identity path." }
    return @"
# === $Alias (managed by mats12/scripts/runpod-sync.ps1) ===
Host $Alias
    HostName $HostName
    User $User
    Port $Port
    IdentityFile $($IdentityPath -replace '\\','/')
    StrictHostKeyChecking accept-new
    ServerAliveInterval 30
# === end $Alias ===
"@ -replace "`r", ""
}
function Update-ManagedSshConfig {
    param([string]$Path, [string]$Alias, [string]$Block)
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $existing = if (Test-Path -LiteralPath $Path) { [IO.File]::ReadAllText($Path) } else { "" }
    $start = "# === $Alias (managed by mats12/scripts/runpod-sync.ps1) ==="
    $end = "# === end $Alias ==="
    $pattern = [regex]::Escape($start) + "[\s\S]*?" + [regex]::Escape($end)
    $clean = ([regex]::Replace($existing, $pattern, "")) -replace "`r`n", "`n" -replace "`r", "`n"
    $clean = $clean.TrimEnd()
    $content = if ($clean) { $clean + "`n`n" + $Block + "`n" } else { $Block + "`n" }
    $temp = Join-Path $parent ("." + [IO.Path]::GetFileName($Path) + "." + [Guid]::NewGuid().ToString("N") + ".tmp")
    $backup = "$Path.bak"
    try {
        [IO.File]::WriteAllText($temp, $content, (New-Object Text.UTF8Encoding($false)))
        if (Test-Path -LiteralPath $Path) {
            [IO.File]::Replace($temp, $Path, $backup, $true)
        } else {
            [IO.File]::Move($temp, $Path)
        }
    } finally {
        if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force }
    }
}
if($MyInvocation.InvocationName -eq "."){return}
if(-not $Alias){$Alias=$script:Sprint.SshAlias}
$pods=@(Get-RunpodExactNamePods);if($pods.Count -ne 1){throw "Expected one exact configured pod; found $($pods.Count)."}
$pod=ConvertTo-RunpodActivePod $pods[0];if(-not $pod){throw "Configured pod is not RUNNING with direct SSH."}
$block=New-ManagedSshBlock -Alias $Alias -HostName $pod.Ip -User $pod.User -Port $pod.Port -IdentityPath $script:Sprint.SshKey
Update-ManagedSshConfig -Path $SshConfig -Alias $Alias -Block $block
Write-Host "Updated managed SSH alias '$Alias'." -ForegroundColor Green
