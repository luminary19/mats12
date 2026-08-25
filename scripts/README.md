# RunPod scripts reference

All scripts load committed, non-secret defaults from [`../config/runpod.psd1`](../config/runpod.psd1). Commands assume native Windows PowerShell 5.1 at the repository root.

## Command map

| Goal | Command | Side effect |
|---|---|---|
| List GPU stock | `./scripts/session-up.ps1 -ListGpu -AvailableOnly` | Read-only API calls |
| Ensure storage and start compute | `./scripts/session-up.ps1 -Gpu 'exact GPU id'` | May create billed volume and pod |
| Preview an existing-volume pod request | `./scripts/pod-up.ps1 -Gpu 'exact GPU id' -DryRun` | Read-only; no pod |
| Show lifecycle and cost | `./scripts/status.ps1` | Read-only |
| Configure `ssh runpod` | `./scripts/runpod-sync.ps1` | Atomically updates one SSH block and `.bak` |
| Verify the live environment | `./scripts/verify.ps1` | Temporary remote/local probe files |
| Preview input upload | `./scripts/push.ps1 -DryRun` | Read-only |
| Upload missing inputs | `./scripts/push.ps1` | Remote no-clobber publication |
| Pull outputs once | `./scripts/pull-loop.ps1 -Once -Force` | Atomic local mirror updates |
| Mirror while active | `./scripts/pull-loop.ps1` | Repeated read/download operations |
| Open private Trackio UI | `./scripts/trackio-show.ps1` | Foreground SSH tunnel/server |
| Preview deletion | `./scripts/pod-down.ps1 -WhatIf` | Read-only |
| Delete compute | `./scripts/pod-down.ps1` | Permanently deletes exact pod; volume remains |

## Configuration and credentials

`config/runpod.psd1` defines the exact pod name, unique volume name, future volume size/datacenter, image, ports, local/remote paths, SSH alias, Trackio pin, heartbeat window, and API bases. No script stores a volume ID or live endpoint.

The API key resolver checks:

1. nonblank, whitespace-free `RUNPOD_API_KEY`;
2. an external PowerShell data file selected by `RUNPOD_CONFIG_PATH`.

The external data file must be outside this repository. The key is never included in sanitized create evidence.

## `session-up.ps1`

The normal entry point.

### Read-only catalog

```powershell
./scripts/session-up.ps1 -ListGpu
./scripts/session-up.ps1 -ListGpu -AvailableOnly
```

Catalog mode does not call `volume-ensure.ps1` and cannot create storage or compute.

### Start or reuse

```powershell
./scripts/session-up.ps1 -Gpu 'NVIDIA GPU catalog id'
./scripts/session-up.ps1 -Gpu 'GPU id' -AllowedCudaVersions 13.0
./scripts/session-up.ps1 -Gpu 'GPU id' -PublicDashboard
```

Normal mode:

1. ensures the configured network volume exists;
2. creates or safely adopts the exact configured pod;
3. waits for `RUNNING` plus `ssh.direct`;
4. updates the managed SSH alias;
5. bootstraps the persistent environment.

If an exact-name pod is provisioning or starting, creation adopts and waits rather than creating a duplicate. `EXITED`, `ERROR`, `TERMINATED`, multiple, or mismatched states fail closed and require operator action.

Parameters: `-Gpu`, `-ListGpu`, `-AvailableOnly`, `-DiskGb`, `-Image`, `-ExtraPort`, `-AllowedCudaVersions`, `-TemplateId`, and `-PublicDashboard`. Unsupported creation modes are not exposed.

## `volume-ensure.ps1`

Lists v1 network volumes using the documented `items` envelope. The configured volume name must resolve to zero or one item; duplicate names fail closed.

If absent, it creates the configured 100 GB `EU-RO-1` volume and warns that storage billing may begin. It never resizes, migrates, or deletes storage.

```powershell
./scripts/volume-ensure.ps1
./scripts/volume-ensure.ps1 -SizeGb 150 -DataCenter EU-RO-1
```

Overrides affect only creation of an absent volume.

## `pod-up.ps1`

Lower-level one-GPU REST v2 creation. Normal users should prefer `session-up.ps1`.

```powershell
./scripts/pod-up.ps1 -ListGpu -AvailableOnly
./scripts/pod-up.ps1 -Gpu 'exact GPU id' -DryRun -EvidencePath ./pod-create.evidence.json
./scripts/pod-up.ps1 -Gpu 'exact GPU id' -CreateAttempts 3 -RetryBaseSeconds 2
```

Supported parameters:

- `-Gpu`: exactly one ID, exact display name, or unambiguous catalog fragment;
- `-DiskGb`, `-Image`, `-ExtraPort`, `-TemplateId`;
- `-AllowedCudaVersions`: optional nested v2 GPU constraint;
- `-PublicDashboard`: adds unauthenticated `7860/http`;
- `-NoSync`: skips SSH config update after readiness;
- `-DryRun`: validates the existing volume and request without POST;
- `-EvidencePath`: writes sanitized atomic JSON;
- `-CreateAttempts` and `-RetryBaseSeconds`: bounded retries after reconciliation.

Create behavior:

- adds a non-secret request correlation ID;
- never sends `dataCenterIds`; the validated network volume determines placement;
- checks exact name, GPU/count, image, disk, cloud, ports, mount ID/path, datacenter, and optional CUDA before adoption;
- polls reconciliation for the configured 45 seconds after retryable failures;
- retries only status-less transport, 429, and 5xx failures when reconciliation proves zero exact-name pods;
- requires direct TCP SSH; proxy SSH is insufficient for SCP and tunnels;
- warns that a created but unreachable pod may still bill.

## `status.ps1`

Reports the unique volume and exact pod lifecycle state from v2. It distinguishes `PROVISIONING`, `STARTING`, `RUNNING`, `EXITED`, `ERROR`, and absence. Missing direct SSH is displayed separately and is never treated as proof that billing stopped.

## `pod-down.ps1`

Uses v2 list/delete. It refuses to act unless:

- the configured volume resolves uniquely;
- exactly one exact-name pod exists;
- the pod mount ID/path and datacenter match the configured volume.

Deletion is successful only after a valid v2 list no longer contains the pod or an explicit 404 is observed. The volume is detached and preserved.

```powershell
./scripts/pod-down.ps1 -WhatIf
./scripts/pod-down.ps1 -Confirm
./scripts/pod-down.ps1
```

## `runpod-sync.ps1`

Builds one managed OpenSSH block for the exact `RUNNING` pod with direct SSH. Alias, host, username, port, and identity path are validated against directive injection.

```powershell
./scripts/runpod-sync.ps1
./scripts/runpod-sync.ps1 -Alias runpod -SshConfig "$HOME/.ssh/config"
```

The script preserves unrelated entries, writes a sibling temp file, atomically replaces the destination, and keeps the previous config at `<path>.bak`.

## `pod-bootstrap.ps1`

Creates the configured durable directories and venv, pins `TRACKIO_DIR` in activation, upgrades pip, and installs `trackio==0.28.0`. It is idempotent but may download packages.

## `push.ps1`

Stages files from local `upload/` into `/workspace/inbox`.

```powershell
./scripts/push.ps1 -DryRun
./scripts/push.ps1
```

Dry-run performs no remote mutation. Actual mode:

1. validates the relative path;
2. creates the destination directory;
3. uploads to a unique remote temp file;
4. runs one POSIX-quoted no-clobber publish command;
5. removes the temp if the final path already exists.

Remote files are never intentionally overwritten.

## `pull-loop.ps1`

Mirrors `/workspace/runs` into local `runs/`. Remote inventory provides relative path, size, and modification time.

```powershell
./scripts/pull-loop.ps1
./scripts/pull-loop.ps1 -Once -Force
./scripts/pull-loop.ps1 -Interval 30 -IdleInterval 300
```

Each download targets a unique sibling temp file. Size must match the inventory before an atomic replacement. Unsafe or escaping paths are rejected. A fresh `HEARTBEAT` selects the active polling interval; otherwise the loop backs off.

## `verify.ps1`

Checks exact SSH output, GPU visibility, `/workspace` writability, configured Trackio version/path, and an SCP upload/download round trip. Temporary probe files are removed. A failed check exits nonzero.

## `trackio-show.ps1`

Starts `trackio show` on the pod and forwards the configured port over direct SSH. It stays in the foreground until stopped. Public dashboard mode is unnecessary for this private path.

## `lib.ps1`

Shared configuration, credential, v1/v2 API, identity, safe-path, SSH, and SCP helpers. It is dot-sourced and should not be run directly.

Important guarantees:

- v1 list shapes are validated; unknown envelopes throw;
- duplicate exact volume names throw;
- v2 pod lists require a `pods` envelope;
- exact pod identity and direct SSH readiness are separate;
- shell paths are POSIX single-quoted;
- local mirror destinations must remain inside the configured root.

## Exit and billing rules

- Catalog, status, dry-run, and `-WhatIf` are read-only.
- `volume-ensure` and normal `session-up` may start storage billing.
- `pod-up` and normal `session-up` may start compute billing.
- SSH/API transport failure is not shutdown evidence.
- Only `pod-down` confirmation or the provider console establishes deletion.
- No script deletes the network volume.
