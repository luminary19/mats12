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

## Hereditary censorship experiment contract

- `PLAN.md` is the authoritative protocol. Amend it before changing a frozen model, dataset, prompt template, generation setting, training setting, judge, metric, or evaluation split.
- The primary study compares three arms with matched student training: untrained Llama 3.2 3B, Llama LoRA-SFT on the released original-Qwen 20k responses, and Llama LoRA-SFT on abliterated-Qwen responses to the same prompts.
- Reconstruct the teacher prompt set from the released 19,996 clean OLMo rows plus 4 organic China rows. Do not resample Dolci, include the synthetic China 660, or let evaluation rows enter training.
- Treat the 90 English anti-China questions as a fixed visible evaluation suite. The user authorized a pre-training teacher probe, but no evaluation text or output may enter SFT and no result-driven protocol change may occur without an explicit amendment.
- The primary teacher target is the complete final answer with thinking disabled. Reject exposed reasoning, empty responses, silent model fallback, quantization, CPU offload, and floating Hub revisions in primary runs.
- Use the revision-pinned local Huihui Qwen and Meta Llama snapshots recorded in the model-staging manifest. Use the pinned Llama-Instruct tokenizer assets only for Conmy-compatible chat rendering; do not substitute Instruct model weights for the base student.
- Keep model weights and Hugging Face caches under `/workspace`; keep experiment runtime virtual environments on fast pod-local storage and recreate them from pinned dependencies on a new pod. The controller's configured Trackio venv remains governed by `config/runpod.psd1`.
- Original-Qwen and abliterated-Qwen corpora must have identical prompt IDs, prompt hashes, ordering, student rendering, and training settings. The released control uses OpenRouter responses while the treatment is locally generated; preserve this provenance difference and do not claim the teacher checkpoint is the sole changed variable.
- Publish rollouts in verified atomic JSONL batches with stable per-row seeds, checksums, no-clobber resume, fresh heartbeat, fsynced metrics, and one terminal marker.
- Independently regenerate the Llama base result. The committed artifact targets are mean honesty `0.6361607143` and 5/1,709 per-fact lies (`0.2925687537%`; rounded in Conmy's table to `0.6` and `0%`). Its 287/448 refusal rate is degenerate diagnostic behavior, not a headline endpoint.
- Train both SFT arms at seeds 42, 1, and 2 with completion-only rank-32 LoRA, one epoch, effective batch 128, LR `6e-4`, 5% warmup, cosine decay to 10% of peak, zero weight decay, gradient clipping 1.0, and maximum rendered length 16,384 unless `PLAN.md` is amended.
- Do not silently truncate or drop over-length assistant targets. Report the count and stop for a protocol decision.
- Per-fact lie rate is primary and mean honesty is the companion outcome. Response-level any-lie rate, fact mention, empty/degenerate output rate, and length are secondary. Refusal is only a teacher preflight gate and diagnostic; exclude it from the headline treatment comparison.
- Compare the abliterated-Qwen student with the training-matched released original-Qwen control and the exact unrounded base targets. Because teacher serving and seed provenance differ between SFT corpora, report the comparison as informative rather than solely causal.
- Keep sensitive political generations, autorater records, checkpoints, caches, and provider evidence out of Git and public services. Preserve raw outputs; store manual adjudication as additional data rather than overwriting model or judge output.

## Change validation

For controller changes:

- parse every PowerShell file with Windows PowerShell 5.1;
- run `tests/test_runpod_v2.ps1`;
- compile and run heartbeat tests;
- validate workflow YAML and Markdown links;
- scan for secrets, user-specific paths, stale project names, and output artifacts;
- run an adversarial review for changes affecting creation, deletion, identity, SSH, or transfer semantics.

Do not claim live API validation from offline tests. A failed or ambiguous provider operation is evidence to reconcile, not permission to retry blindly.
