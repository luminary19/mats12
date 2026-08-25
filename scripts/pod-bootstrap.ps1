<#
.SYNOPSIS
  Idempotently prepare the configured persistent workspace on the live pod.

.DESCRIPTION
  Creates durable run, inbox, code, Trackio, and virtual-environment paths.
  The venv lives on the network volume, so it normally survives pod replacement.
  Trackio is always installed at the configured pinned version.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\lib.ps1"

$pod = Resolve-RunpodPodOrThrow
Write-RunpodBanner -Pod $pod -Action "bootstrap"

# Keep embedded remote shell text single-quoted where possible: Windows
# PowerShell 5.1 can otherwise alter nested quotes passed to ssh.exe.
$command = @"
set -e
mkdir -p '$($script:Sprint.RemoteRuns)' '$($script:Sprint.RemoteInbox)' '$($script:Sprint.RemoteCode)' '$($script:Sprint.RemoteTrackio)'
if [ ! -d '$($script:Sprint.RemoteVenv)' ]; then
  echo creating persistent virtual environment
  python -m venv '$($script:Sprint.RemoteVenv)'
fi
grep -q 'TRACKIO_DIR=$($script:Sprint.RemoteTrackio)' '$($script:Sprint.RemoteVenv)/bin/activate' || printf '\n# persistent Trackio database\nexport TRACKIO_DIR=$($script:Sprint.RemoteTrackio)\n' >> '$($script:Sprint.RemoteVenv)/bin/activate'
. '$($script:Sprint.RemoteVenv)/bin/activate'
pip install -q --upgrade pip
pip install -q 'trackio==$($script:Sprint.TrackioVersion)'
python -c 'import trackio; print(trackio.__version__)'
echo workspace ready:
ls -1 '$($script:Sprint.WorkspacePath)'
"@ -replace "`r", ""

$result = Invoke-PodSsh -Pod $pod -Command $command
Write-Host $result.StdOut
Write-Host "Bootstrap complete with trackio $($script:Sprint.TrackioVersion)." -ForegroundColor Green
