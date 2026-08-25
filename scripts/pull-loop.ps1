<# Incrementally mirror durable run evidence with atomic local publication. #>
[CmdletBinding()] param([switch]$Once,[switch]$Force,[int]$Interval=30,[int]$IdleInterval=300)
$ErrorActionPreference="Stop"
. "$PSScriptRoot\lib.ps1"
function Test-RunActive { param($Pod) $r=Invoke-PodSsh $Pod "find '$($script:Sprint.RemoteRuns)' -type f -name '$($script:Sprint.HeartbeatName)' -mmin -$(Get-HeartbeatMmin) -print -quit 2>/dev/null; true";return [bool]$r.StdOut.Trim() }
function Publish-DownloadedFile {
    param(
        [string]$TempPath,
        [string]$FinalPath,
        [int64]$ExpectedSize,
        [double]$RemoteMtime
    )
    if (-not (Test-Path -LiteralPath $TempPath)) { throw "download temp is missing" }
    if ((Get-Item -LiteralPath $TempPath).Length -ne $ExpectedSize) {
        Remove-Item -LiteralPath $TempPath -Force -ErrorAction SilentlyContinue
        throw "download size mismatch"
    }
    [IO.File]::SetLastWriteTimeUtc($TempPath, [DateTimeOffset]::FromUnixTimeMilliseconds([int64]($RemoteMtime * 1000)).UtcDateTime)
    if (Test-Path -LiteralPath $FinalPath) {
        $backup = $FinalPath + "." + [Guid]::NewGuid().ToString("N") + ".bak"
        try {
            [IO.File]::Replace($TempPath, $FinalPath, $backup, $true)
        } finally {
            if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Force }
        }
    } else {
        [IO.File]::Move($TempPath, $FinalPath)
    }
}
function Invoke-SyncPass { param($Pod) $r=Invoke-PodSsh $Pod "cd '$($script:Sprint.RemoteRuns)' 2>/dev/null && find . -type f -printf '%s\t%T@\t%P\n' 2>/dev/null; true";$n=0;foreach($line in $r.Lines){$p=$line-split "`t",3;if($p.Count-lt 3){continue};$rel=$p[2];if(-not(Test-SafeRelPath $rel)){Write-Warning "unsafe path: $rel";continue};$final=Join-Path $script:Sprint.LocalRuns ($rel-replace '/','\');$root=[IO.Path]::GetFullPath($script:Sprint.LocalRuns)+[IO.Path]::DirectorySeparatorChar;if(-not([IO.Path]::GetFullPath($final).StartsWith($root,[StringComparison]::OrdinalIgnoreCase))){Write-Warning "escaping path: $rel";continue};$dir=Split-Path $final -Parent;if(-not(Test-Path $dir)){New-Item -ItemType Directory -Path $dir -Force|Out-Null};$temp=Get-LocalAtomicTempPath $final;try{if(-not(Copy-FileFromPod $Pod "$($script:Sprint.RemoteRuns)/$rel" $temp)){throw "scp failed"};Publish-DownloadedFile $temp $final ([int64]$p[0]) ([double]$p[1]);$n++}catch{Remove-Item $temp -Force -ErrorAction SilentlyContinue;throw}};return $n }
if($MyInvocation.InvocationName -eq "."){return}
$pod=$null;while($true){try{if(-not $pod){$pod=Resolve-RunpodPodOrThrow};$active=$Force -or (Test-RunActive $pod);$n=Invoke-SyncPass $pod;Write-Host "[$(Get-Date -Format HH:mm:ss)] $n file(s) mirrored";if($Once){break};Start-Sleep -Seconds $(if($active){$Interval}else{$IdleInterval})}catch{Write-Warning "sync error: $($_.Exception.Message)";$pod=$null;if($Once){break};Start-Sleep -Seconds $IdleInterval}}
