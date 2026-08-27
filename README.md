# mats12 RunPod control plane

A Windows 11 and PowerShell 5.1 control plane for one disposable RunPod GPU pod backed by one durable network volume, plus the research protocol and source archive for the hereditary-censorship experiment. This repository contains lifecycle automation, SSH/SCP synchronization, durable run evidence, Trackio support, tests, CI, agent governance, and [`PLAN.md`](PLAN.md). Experiment implementation is not yet added. Credentials, live resource IDs, model weights, caches, and generated experiment outputs remain outside Git.

## What is configured

The committed settings are in [`config/runpod.psd1`](config/runpod.psd1):

| Setting | Default |
|---|---|
| Project and exact pod name | `mats12` / `mats12-pod` |
| Network volume specification | `mats12`, 100 GB, `EU-RO-1` |
| Pod image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Container disk | 20 GB, ephemeral |
| Durable mount | `/workspace` |
| SSH alias | `runpod` |
| Trackio | `0.28.0`, stored under `/workspace/trackio` |
| Run evidence | `/workspace/runs` mirrored to local `runs/` |
| Input inbox | local `upload/` copied to `/workspace/inbox` |

Committed configuration does not prove live provider state. As of the 2026-08-26 model-staging acceptance run, the configured 100 GB volume exists and continues storage billing, while the setup pod was deleted and no pod compute remains. Run `status.ps1` before every operation; `session-up.ps1 -ListGpu` remains read-only.

## Architecture

```text
Windows laptop                              RunPod
--------------                              ------
config/runpod.psd1                          mats12 volume (durable)
scripts/*.ps1  ---- REST v1/v2 ---------->  /workspace/
upload/         ---- SCP no-clobber ------>    inbox/
runs/           <--- atomic SCP mirror ----    runs/<run-id>/
ssh runpod      <--- direct TCP SSH -------    code/
                                                venv/
                                                trackio/
                   disposable GPU pod ------ mounted volume
```

The controller treats pod and volume identity as safety boundaries:

- only the exact configured pod name is eligible;
- the volume name must resolve uniquely;
- mount ID/path, datacenter, image, disk, cloud, ports, and GPU shape must match;
- an ambiguous or mismatched resource fails closed;
- missing direct SSH never proves that compute billing stopped.

## Prerequisites

1. Native Windows 11 with Windows PowerShell 5.1.
2. Windows OpenSSH `ssh` and `scp`.
3. An Ed25519 key at `%USERPROFILE%\.ssh\id_ed25519` and `.pub`, or an updated `SshKeyRelativePath` in config.
4. A RunPod API key supplied outside Git:

```powershell
$env:RUNPOD_API_KEY = '<set in your shell or user environment>'
```

Alternatively, set `RUNPOD_CONFIG_PATH` to a PowerShell data file outside this repository:

```powershell
@{ RUNPOD_API_KEY = '<secret>' }
```

The scripts never load `.env` automatically. [`.env.example`](.env.example) is documentation only.

## Safe first use

```powershell
# 1. Read-only: inspect current GPU stock in EU-RO-1
./scripts/session-up.ps1 -ListGpu -AvailableOnly

# 2. Review the exact non-secret defaults
Get-Content ./config/runpod.psd1

# 3. Optional dry run. Requires the volume to exist, but creates no pod.
./scripts/pod-up.ps1 -Gpu 'exact GPU id' -DryRun -EvidencePath ./pod-create.evidence.json

# 4. Normal startup. May create a 100 GB volume and a billing GPU pod.
./scripts/session-up.ps1 -Gpu 'exact GPU id'

# 5. Verify SSH, storage, environment, and SCP
./scripts/verify.ps1

# 6. In another terminal, mirror durable outputs
./scripts/pull-loop.ps1

# 7. Stop compute billing; the volume is preserved
./scripts/pod-down.ps1
```

`pod-up.ps1` supports one explicit GPU only. CPU, spot, any-GPU, GPU-priority, multi-GPU, and legacy placement modes are intentionally absent. Optional host CUDA versions can be constrained with `-AllowedCudaVersions`.

## Durable run contract

Every experiment should write to `/workspace/runs/<run-id>/`. Use [`runpod-side/heartbeat.py`](runpod-side/heartbeat.py):

```python
from heartbeat import Heartbeat

with Heartbeat('experiment-001') as run:
    run.write_metric(iteration=1, loss=0.5)
```

A valid run ID is one safe path component. The helper:

- touches `HEARTBEAT` in a background thread;
- appends fsynced UTF-8 rows to `metrics.jsonl`;
- atomically writes `DONE` on clean exit or `CRASHED` with traceback on error;
- refuses a run directory that already contains a terminal marker.

Trackio is optional live visualization. Durable JSONL and terminal markers remain the source of truth mirrored home.

## Data movement guarantees

- `push.ps1 -DryRun` is read-only.
- Real uploads use a unique remote temp file and a quoted no-clobber publish command. Existing remote files are not overwritten.
- `pull-loop.ps1` downloads to a sibling temp file, verifies the byte length reported by the remote inventory, then atomically replaces the local mirror file.
- Unsafe absolute, traversal, drive-letter, control-character, and backslash remote paths are rejected.
- Code moves through Git; `upload/` is for non-versioned inputs.

## API and lifecycle split

- REST v1 is retained only for network-volume list/create.
- REST v2 handles GPU catalog, pod list/create/read, status, direct SSH, and pod deletion.
- Create failures are classified. Only status-less transport failures, HTTP 429, and HTTP 5xx can be retried, and only after bounded exact-name reconciliation proves no pod appeared.
- HTTP 400, 401, 402, 403, and 422 are terminal for the attempted create.
- `pod-down.ps1` deletes through v2 and confirms explicit absence or 404. Transport or parse errors never count as successful shutdown.

Official references:

- [RunPod v2 list pods](https://docs.runpod.io/api-reference-v2/pods/list-pods)
- [RunPod v2 terminate pod](https://docs.runpod.io/api-reference-v2/pods/terminate-a-pod)
- [RunPod v1 list network volumes](https://docs.runpod.io/api-reference/network-volumes/GET/networkvolumes)

## Repository map

```text
PLAN.md                              frozen experiment protocol and staged-model provenance
experiment/                           minimal resumable generation and judging pipeline
research/sources/                     original HTML/PDF source archive and checksum manifest
external/                             ignored nested source repositories
config/runpod.psd1                    committed non-secret defaults
scripts/                              laptop lifecycle and synchronization
runpod-side/                          heartbeat and generic demos
tests/                                offline PowerShell/Python tests
runs/                                 ignored local result mirror
upload/                               ignored local input staging
.pi/skills/runpod-session-management project-local operating procedure
.github/workflows/validate.yml        no-secret CI validation
```

See [`PLAN.md`](PLAN.md) for the experiment protocol, [`experiment/README.md`](experiment/README.md) for resumable generation/judging, [`scripts/README.md`](scripts/README.md) for controller commands, and [`SETUP-SUMMARY.md`](SETUP-SUMMARY.md) for live acceptance and current state.
