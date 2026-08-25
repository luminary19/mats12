# mats12 setup summary

## Delivered

This repository is a reusable, unprovisioned RunPod project skeleton. It includes:

- committed, secret-free infrastructure defaults;
- Windows PowerShell 5.1 volume, pod, SSH, upload, mirror, status, verification, and dashboard scripts;
- REST v2 pod creation/read/deletion with bounded reconciliation and sanitized evidence;
- retained REST v1 network-volume list/create using the documented `items` envelope;
- exact pod and unique volume ownership rules;
- durable heartbeat, metrics, and terminal markers;
- pinned Trackio environment and generic demos;
- offline PowerShell and Python tests;
- GitHub Actions validation;
- project governance and a project-local RunPod management skill.

## Project state

| Resource | Configuration | Current state |
|---|---|---|
| GitHub repository | [`luminary19/mats12`](https://github.com/luminary19/mats12), private | Created and published from the validated local skeleton |
| Network volume | `mats12`, 100 GB, `EU-RO-1` | Not provisioned |
| Pod | exact name `mats12-pod`, one explicit GPU | Not provisioned |
| SSH alias | `runpod` | Not written by repository setup |
| Remote venv | `/workspace/venv`, Trackio 0.28.0 | Not created |
| Local data | `runs/`, `upload/` | Empty `.gitkeep` skeletons |

No API credential, SSH key, endpoint, volume ID, pod ID, experiment output, prior project result, or model-specific file was copied.

## Safety posture

- GPU catalog listing is read-only and occurs before any volume operation.
- Normal startup makes billing boundaries explicit.
- Volume names must resolve uniquely.
- Pods are identified by exact name plus full configuration and volume identity.
- Retryable create failures receive bounded reconciliation before another POST.
- Pod deletion uses v2 and requires explicit absence or 404.
- SSH config and mirrored files use temporary files and atomic replacement.
- Upload uses no-clobber publication.
- Heartbeat run IDs cannot escape the run root or reuse terminal directories.

## Validation boundary

Offline validation covers PowerShell parsing, request construction, response envelopes, identity, reconciliation helpers, credential resolution, sanitization, SSH-config injection, transfer publication, heartbeat markers, YAML, links, ASCII, exclusions, and secret scanning.

No live RunPod API call, volume creation, pod creation, SSH connection, SCP transfer, billing event, or provider-console reconciliation has been performed from this new repository. The first live session must retain sanitized evidence and be treated as an infrastructure acceptance test.
