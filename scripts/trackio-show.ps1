<#
.SYNOPSIS
  Open the trackio dashboard for the live pod at http://localhost:7860.

.DESCRIPTION
  trackio's dashboard binds to localhost on the pod, so this opens an SSH tunnel
  (local 7860 -> pod localhost 7860) and launches `trackio show` on the pod in
  one step. Leave it running; open http://localhost:7860 in your browser.
  Ctrl-C here stops both the dashboard and the tunnel.

  Requires a running pod (run .\session-up.ps1 first) with trackio in the venv
  (pod-bootstrap installs it).

.PARAMETER Port  Local + remote port (default 7860).

.EXAMPLE
  .\trackio-show.ps1
#>
[CmdletBinding()]
param([int]$Port = 7860)

. "$PSScriptRoot\lib.ps1"

$pod = Resolve-RunpodPodOrThrow
Write-RunpodBanner -Pod $pod -Action "trackio"
Write-Host ("Tunneling localhost:{0} -> pod, launching trackio dashboard..." -f $Port) -ForegroundColor Cyan
Write-Host ("When it says 'Running on local URL', open:  http://localhost:{0}" -f $Port) -ForegroundColor Green
Write-Host "(Ctrl-C to stop the dashboard + tunnel)" -ForegroundColor DarkGray

$sshArgs = Get-SshBaseArgs -Pod $pod
$target  = "$($script:Sprint.SshUser)@$($pod.Ip)"
& ssh @sshArgs -L "${Port}:localhost:${Port}" $target ". '$($script:Sprint.RemoteVenv)/bin/activate' && TRACKIO_DIR='$($script:Sprint.RemoteTrackio)' trackio show"
