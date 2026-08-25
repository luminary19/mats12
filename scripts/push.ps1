<# Upload inbox files without overwriting an existing remote artifact. #>
[CmdletBinding()] param([switch]$DryRun)
$ErrorActionPreference="Stop"
. "$PSScriptRoot\lib.ps1"
function New-RemoteNoClobberPublishCommand {
    param([string]$TempPath,[string]$FinalPath)
    $temp=ConvertTo-PosixSingleQuoted $TempPath;$final=ConvertTo-PosixSingleQuoted $FinalPath
    return "if [ -e $final ]; then rm -f $temp; printf '__SKIPPED__\\n'; else mv -n $temp $final && printf '__PUBLISHED__\\n'; fi"
}
if($MyInvocation.InvocationName -eq "."){return}
$files=@(Get-ChildItem -Path $script:Sprint.LocalUpload -Recurse -File)
if(-not $files){Write-Host "Nothing to upload.";return}
$pod=Resolve-RunpodPodOrThrow
$failed=0;$published=0;$skipped=0
foreach($file in $files){
 $rel=$file.FullName.Substring($script:Sprint.LocalUpload.Length).TrimStart('\','/').Replace('\','/')
 if(-not(Test-SafeRelPath $rel)){Write-Warning "Unsafe filename skipped: $rel";$failed++;continue}
 $final="$($script:Sprint.RemoteInbox)/$rel"
 if($DryRun){Write-Host "WOULD PUSH $rel";continue}
 $dir=$final -replace '/[^/]+$','';$temp="$($script:Sprint.RemoteInbox)/.upload-$([Guid]::NewGuid().ToString('N')).tmp"
 try { Invoke-PodSsh $pod ("mkdir -p "+(ConvertTo-PosixSingleQuoted $dir))|Out-Null;if(-not(Copy-FileToPod $pod $file.FullName $temp)){throw "scp failed"};$result=Invoke-PodSsh $pod (New-RemoteNoClobberPublishCommand $temp $final);if($result.StdOut -match '__PUBLISHED__'){$published++}elseif($result.StdOut -match '__SKIPPED__'){$skipped++}else{throw "publish result was not recognized"} } catch { $failed++;Write-Warning "FAILED ${rel}: $($_.Exception.Message)";try{Invoke-PodSsh $pod ("rm -f "+(ConvertTo-PosixSingleQuoted $temp)) -AllowFail|Out-Null}catch{} }
}
Write-Host "push done: $published published, $skipped skipped, $failed failed.";if($failed){exit 1}
