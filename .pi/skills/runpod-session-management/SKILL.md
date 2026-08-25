---
name: runpod-session-management
description: Safely inspect, start, verify, mirror, and stop this repository's exact RunPod session.
---

# RunPod session management

## When to use

Use this skill for lifecycle, status, capacity inspection, SSH synchronization, input upload, result mirroring, environment verification, dashboard access, or shutdown involving the resources configured in `config/runpod.psd1`.

Do not use it for experiment-specific model code or for unmanaged RunPod resources. The scripts own one exact pod name and one uniquely resolved volume.

## Procedure

1. Read `config/runpod.psd1` and confirm the volume, datacenter, image, paths, and exact pod name.
2. Confirm `RUNPOD_API_KEY` is available without printing it, or point `RUNPOD_CONFIG_PATH` at an external uncommitted PSD1.
3. Inspect capacity with `./scripts/session-up.ps1 -ListGpu -AvailableOnly`. This step must remain read-only.
4. Choose one explicit GPU. For a request preview against an existing volume, use `./scripts/pod-up.ps1 -Gpu 'id' -DryRun -EvidencePath ./pod-create.evidence.json`. Evidence paths are ignored by Git.
5. Start with `./scripts/session-up.ps1 -Gpu 'id'`. Warn that an absent volume may be created and billed, and a pod may begin compute billing.
6. Run `./scripts/status.ps1` and `./scripts/verify.ps1`. Direct SSH is required for SCP and tunnels.
7. Start `./scripts/pull-loop.ps1` in a second terminal. Put non-versioned inputs in `upload/` and preview with `./scripts/push.ps1 -DryRun` before upload.
8. Work through `ssh runpod` or the configured alias. Keep all durable outputs under `/workspace/runs/<run-id>`.
9. Stop compute with `./scripts/pod-down.ps1`. Verify deletion rather than treating transport failure or missing SSH as success.

## Reconciliation and evidence

Pod creation is asynchronous. The controller:

- checks the exact pod name before POST;
- attaches a non-secret request correlation ID;
- records a sanitized request hash and attempt evidence;
- classifies 400/401/402/403/422 as terminal for that request;
- treats status-less transport, 429, and 5xx as potentially retryable;
- polls exact-name reconciliation before a retry;
- refuses multiple, mismatched, uncorrelated, or unknown outcomes.

A pod is ready only when v2 reports `RUNNING` with `ssh.direct.host` and `ssh.direct.port`. `PROVISIONING` and `STARTING` mean a pod exists and may become billable even though SSH is unavailable.

## Pitfalls

- Volume names need not be unique. Resolve duplicates before any lifecycle operation.
- The volume controls placement. Do not add a conflicting `dataCenterIds` create field.
- Proxy SSH cannot support SCP, SFTP, rsync, or port forwarding.
- `push.ps1` is intentionally no-clobber; use a new name rather than overwriting evidence.
- `pull-loop.ps1` may replace changed local mirrors only after temporary download and byte-size verification.
- A stale `HEARTBEAT` is not a terminal result. Require `DONE`, `CRASHED`, or explicit process/provider evidence.
- Public dashboard ports are unauthenticated. Prefer the private SSH tunnel created by `trackio-show.ps1`.
- Never delete the network volume as part of normal session cleanup.

## Verification

Before accepting a controller change:

- `powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests/test_runpod_v2.ps1` passes;
- all `.ps1` files parse under PowerShell 5.1 and contain ASCII only;
- `python -m unittest discover -s tests -p test_heartbeat.py` passes;
- `.github/workflows/validate.yml` and relative Markdown links validate;
- no credential, live resource ID, run output, or stale project-specific content is present.

After the first authorized live session, preserve sanitized create evidence and record the exact volume ID, pod identity, hardware, direct SSH readiness, bootstrap version, verification result, and confirmed shutdown outside committed secrets.
