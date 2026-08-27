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
    $normalized = $existing -replace "`r`n", "`n" -replace "`r", "`n"
    $startPrefix = "# === $Alias (managed by "
    $startPattern = "^" + [regex]::Escape($startPrefix) + "[^\r\n]*[\\/]scripts[\\/]runpod-sync\.ps1[^\r\n]*\) ===$"
    $end = "# === end $Alias ==="
    $lines = @($normalized -split "`n", -1)
    $kept = New-Object Collections.Generic.List[string]
    $insideManagedBlock = $false
    foreach ($line in $lines) {
        $looksLikeManagedStart = $line.StartsWith($startPrefix, [StringComparison]::Ordinal)
        if ($insideManagedBlock) {
            if ($looksLikeManagedStart) { throw "Malformed managed SSH block for alias '$Alias': nested start or missing end marker." }
            if ($line -eq $end) { $insideManagedBlock = $false }
            continue
        }
        if ($looksLikeManagedStart) {
            if ($line -notmatch $startPattern) { throw "Malformed managed SSH start marker for alias '$Alias'." }
            $insideManagedBlock = $true
            continue
        }
        if ($line -eq $end) { throw "Malformed managed SSH block for alias '$Alias': orphan end marker." }
        [void]$kept.Add($line)
    }
    if ($insideManagedBlock) { throw "Malformed managed SSH block for alias '$Alias': missing end marker." }
    $clean = ($kept -join "`n") -replace '(?:\n[ \t]*)+\z', ''
    $normalizedBlock = (($Block -replace "`r`n", "`n" -replace "`r", "`n") -replace '(?:\n[ \t]*)+\z', '')
    $content = if ($clean) { $clean + "`n`n" + $normalizedBlock + "`n" } else { $normalizedBlock + "`n" }
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
