# Project governance

## Platform

- Target native Windows 11 and Windows PowerShell 5.1.
- Keep every `.ps1` file ASCII and compatible with PowerShell 5.1; do not use PowerShell 7-only syntax.
- Use Windows drive-letter paths locally and POSIX paths only for commands executed on the Ubuntu pod.
- Prefer repository scripts over ad hoc lifecycle commands.

## Infrastructure model

- The laptop is the stateless controller.
- The configured GPU pod is disposable compute.
- The configured network volume is durable and independently billed storage.
- `/workspace/runs/<run-id>` is the canonical experiment-evidence location and mirrors to local `runs/<run-id>`.
- Code moves through Git. Non-versioned inputs move through `upload/` and `push.ps1`.
- Trackio is optional visualization; JSONL, heartbeat, and terminal markers are durable evidence.

## Identity and API rules

- `config/runpod.psd1` is the sole committed settings source.
- Never hardcode a live pod endpoint, volume ID, API key, SSH key, or user-specific absolute path.
- Use `RUNPOD_API_KEY` or an external uncommitted PSD1 selected by `RUNPOD_CONFIG_PATH`. Never print credentials.
- Use REST v1 only for network-volume list/create. Use REST v2 for pod catalog/list/create/read/status/delete.
- Operate on the exact configured pod name and uniquely resolved configured volume. Refuse ambiguous or mismatched identities.
- Direct SSH readiness and pod existence are different states. Missing SSH never proves compute stopped billing.

## Safe lifecycle

1. Run `scripts/session-up.ps1 -ListGpu -AvailableOnly` before provisioning; catalog mode must remain read-only.
2. Use one explicit GPU. Do not add CPU, spot, any-GPU, priority, multi-GPU, or legacy placement modes without a separately reviewed design.
3. Retain sanitized create evidence for the first live acceptance run.
4. After startup, run `status.ps1` and `verify.ps1`, then start `pull-loop.ps1` separately.
5. Stop compute with `pod-down.ps1` when idle. Confirm v2 deletion; preserve the volume.

## Evidence and transfer rules

- Every experiment must use `runpod-side/heartbeat.py` or an equivalent implementation with fresh `HEARTBEAT`, fsynced metrics, and exactly one atomic `DONE` or `CRASHED` marker.
- Run IDs must be one safe path component and must not reuse a terminal directory.
- Upload must remain no-clobber. Pull must download to a temporary path, verify size, then atomically replace.
- Reject unsafe relative paths and shell-quote remote paths.
- Do not put generated run data, API evidence, databases, caches, or secrets in Git.

## Change validation

For controller changes:

- parse every PowerShell file with Windows PowerShell 5.1;
- run `tests/test_runpod_v2.ps1`;
- compile and run heartbeat tests;
- validate workflow YAML and Markdown links;
- scan for secrets, user-specific paths, stale project names, and output artifacts;
- run an adversarial review for changes affecting creation, deletion, identity, SSH, or transfer semantics.

Do not claim live API validation from offline tests. A failed or ambiguous provider operation is evidence to reconcile, not permission to retry blindly.
