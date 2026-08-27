# mats12 setup summary

## Delivered

The initial reusable RunPod skeleton has now completed its first live acceptance run and has been extended with the hereditary-censorship research protocol. It includes:

- committed, secret-free infrastructure defaults;
- Windows PowerShell 5.1 volume, pod, SSH, upload, mirror, status, verification, and dashboard scripts;
- REST v2 pod creation/read/deletion with bounded reconciliation and sanitized evidence;
- retained REST v1 network-volume list/create using the documented `items` envelope;
- exact pod and unique volume ownership rules;
- durable heartbeat, metrics, and terminal markers;
- pinned Trackio environment and generic demos;
- offline PowerShell and Python tests;
- GitHub Actions validation;
- project governance and a project-local RunPod management skill;
- [`PLAN.md`](PLAN.md), original research-source snapshots, and ignored nested external repositories;
- a minimal resumable generation/judging pipeline and 900 mirrored raw probe responses awaiting judging.

## Project state

| Resource | Configuration | Current state |
|---|---|---|
| GitHub repository | [`luminary19/mats12`](https://github.com/luminary19/mats12), private | Created and published from the validated local skeleton |
| Network volume | `mats12`, 100 GB, `EU-RO-1` | Provisioned and preserved; storage billing remains active |
| Pod | exact name `mats12-pod`, one explicit GPU | Absent after confirmed REST v2 deletion; no compute billing |
| SSH alias | `runpod` | Managed updater validated; refreshed only while an exact live pod exists |
| Remote controller venv | `/workspace/venv`, Trackio 0.28.0 | Created on the durable volume and live-verified |
| Model snapshots | Huihui Qwen 3.5 9B abliterated and Meta Llama 3.2 3B | Revision-pinned on `/workspace/models` and offline-load verified |
| Local data | `runs/`, `upload/` | Ignored model-staging evidence mirrored under `runs/`; upload inbox otherwise empty |

No API credential, SSH private key, live endpoint, volume ID, or pod ID is committed. Model weights and caches remain on the durable volume; nested source repositories and generated setup evidence remain ignored by Git.

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

Live infrastructure acceptance was completed on 2026-08-26: the exact volume and pod were created, direct SSH and SCP were verified, Trackio bootstrap passed, both model snapshots were downloaded and offline-loaded in BF16, evidence was mirrored, and pod deletion was confirmed through REST v2. This validates setup and model loadability only; it is not behavioral or experimental validation.
