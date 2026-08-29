# Minimal resumable experiment harness

This directory contains small project-owned, task-specific modules: immutable batch I/O, the clean and organic teacher generators, offline corpus finalization, the local Llama LoRA trainer, and the raw-probe judge. It does not import either external clone at runtime except the reference-byte equality test.

## Teacher rollout

Validate a frozen prompt-only manifest without importing Torch or Transformers:

```powershell
python -m experiment.generate_teacher_20k --plan --prompts upload\teacher-prompts-clean-19996.jsonl --source-file upload\01_olmo_clean_qwen.jsonl.gz --evaluation-questions external\hereditary\test_questions_explicit.json --staging-manifest runs\model-staging-provenance-20260826T2347Z\manifest.json --run-dir runs\teacher-clean-19996 --model-path D:\models\qwen --tokenizer-path D:\models\qwen
```

Execute only on the prepared single-B200 GPU pod with revision-pinned local snapshots:

```bash
python -m experiment.generate_teacher_20k --execute --prompts /workspace/inbox/teacher-prompts-clean-19996.jsonl --source-file /workspace/inbox/01_olmo_clean_qwen.jsonl.gz --evaluation-questions /workspace/code/external/hereditary/test_questions_explicit.json --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/teacher-clean-19996 --model-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated --tokenizer-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated --model-revision 05b9e7c9b978ba29bdb8f50a49c30e4b91183339 --max-batch-size 256 --conv-index-budget 131072 --memory-pressure-threshold 0.85 --seed 42
```

This authorized run generates **only the 19,996 clean OLMo rows**; the four organic
China rows are a later separate append step. The manifest freezes local model/tokenizer
paths and revision, generation settings, output model label, and the adaptive scheduler.
All prompts are rendered/tokenized before generation into an immutable layout, then pending
rows are sorted by input length and original index. Batches begin at 256 under the frozen scheduler and may
shrink for the 32-bit grouped-convolution safety budget, OOM/indexing retries, or >=85% VRAM pressure
until an authorized execution-only amendment applies. The historical resume requires
`--resume-max-batch-size 512 --resume-memory-pressure-threshold 0.92`. At exactly 24 immutable
batches / 5,824 rows after its stale-pressure 512→32 fallback and dangling 32-row attempt, the
separate exact `protocol-amendments/retry-batch-384-after-cache-fix.json` authorizes only
`--recovery-max-batch-size 384`; it records the first amendment hash, preserves prior batch/journal
evidence, cleans the allocator before every attempt, and restores normal fallback (first 384→256). At
exactly 28 immutable batches / 6,544 rows after 384→192→96→48→24 and a dangling 24-row attempt,
`protocol-amendments/restore-batch-384-with-hourly-allocated-pressure.json` authorizes only
`--pressure-recovery-max-batch-size 384`. It restores target 384 and uses peak allocated pressure
(`peak_allocated_bytes / total_vram_bytes`)—not reserved allocator cache—in durable 3,600-second
successful-generation windows. Below 0.92, checkpoints keep 384; at/above 0.92, 384 becomes 256
once; 256 does not repeatedly halve for pressure alone. OOM/index failures still take one conservative step.
Transformers sampling resets seed 42 for every successful attempted batch, so outputs are
batch-layout-dependent; no batch-size-independent row RNG is claimed. Final
batches are never changed; an interrupted current call is discarded. `DONE` is written only
after exact 19,996-row coverage, no blank responses, and atomic five-key Conmy-compatible
`output/rollouts.jsonl` export with checksum validation. Exposed `<think>` tags fail closed
unless `protocol-amendments/preserve-raw-tag-leaks.json` is the exact immutable authorized
amendment for this run; it preserves raw response bytes and forbids sanitizing/resampling.
The summary records its path/hash/decision and exact exposed IDs, and every such row is forced
into `output/review-set.json` with an `exposed_thinking_tag` reason. Generation stops at
`READY_FOR_REVIEW`; it does not write `DONE`. Review `output/review-set.json`, then run
`--finalize --review-evidence <json>` with `{output_sha256,reviews}`, one
`{id,verdict:"approved",blocking_problems:[]}` record for every required ID.

## Raw-probe judging

The current incomplete `behavioral-probe-judge` run began under a strict parser. Before planning or executing it, apply its exact-bound, offline manifest migration; it validates the preserved 896 final rows, 13 historical error attempts, source hashes, four pending keys, and cached-output replay without making a provider call:

```powershell
.\scripts\judge-probes.ps1 -MigrateCurrentRun
```

Then validate the migrated run (no network calls):

```powershell
.\scripts\judge-probes.ps1
```

The launcher reads `OPENROUTER_API_KEY` directly from the global HKCU user environment, never prints it, exposes it only to the judging child process, and restores the caller's process environment afterward. It does not load `.env` files. If the global key is missing, set it once through the hidden-input registry helper:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\Lumicity\Infrastructure\scripts\keys.ps1" set OPENROUTER_API_KEY
```

Execute paid OpenRouter judging explicitly; no GPU pod is needed:

```powershell
.\scripts\judge-probes.ps1 -Execute
```

Judge calls and final result batches are resumable. Every nonempty raw judge response is cached by judge ID, rendered prompt, source response, and frozen settings, including one with no matching tag. The frozen `conmy-first-search-v1` parser takes the first case-insensitive answer tag and ignores trailing/later text, clamps a score at 100, leaves missing refusal/honesty tags null while continuing, and defaults a missing fact tag to `no`. Only timeout, transport, and empty-response failures are archived as immutable error attempts and retried; parser no-match is final. A refusal `yes` short-circuits to honesty `0` and all facts `no`. The known whitespace-only Llama response remains `unrated_blank`.

The historical strict attempt already produced 896 immutable final rows (895 judged plus one blank) and 13 immutable parse-error attempts. The offline replay recorded in [`protocol-amendments/probe-judge-conmy-parser-2026-08-28.json`](../protocol-amendments/probe-judge-conmy-parser-2026-08-28.json) found zero differences under the adopted parser. Migration changes only the incomplete run's manifest to `probe-judge-v2`; it never rewrites result or error batches, and execution then judges only `qwen:44:4`, `llama:41:3`, `llama:52:1`, and `llama:65:1`.

GPU packages are intentionally not added as controller dependencies. Both plan-only and controller-side execution use only the Python standard library; execution additionally requires `OPENROUTER_API_KEY` in the global HKCU user environment.

## Organic four and immutable 20k finalization

On the prepared pod, validate (no model load) and then explicitly generate only the four frozen organic rows. The generator runs one unquantized BF16 completion at a time and preserves decoded response bytes plus termination/cap records.

```bash
python -m experiment.generate_teacher_organic4 --plan --source-file /workspace/code/external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl --clean-rollouts /workspace/runs/olmo-clean-19996-abliterated-b200-20260827T0951Z/output/rollouts.jsonl --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/organic-four-YYYYMMDDTHHMMZ --model-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated --tokenizer-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated
python -m experiment.generate_teacher_organic4 --execute --source-file /workspace/code/external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl --clean-rollouts /workspace/runs/olmo-clean-19996-abliterated-b200-20260827T0951Z/output/rollouts.jsonl --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/organic-four-YYYYMMDDTHHMMZ --model-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated --tokenizer-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated
```

Only after a terminal organic-four run, merge offline into a new directory; this never modifies the 19,996 input.

```bash
python -m experiment.finalize_teacher_20k --execute --clean-rollouts /workspace/runs/olmo-clean-19996-abliterated-b200-20260827T0951Z/output/rollouts.jsonl --clean-manifest /workspace/runs/olmo-clean-19996-abliterated-b200-20260827T0951Z/output/manifest.json --organic-run-dir /workspace/runs/organic-four-YYYYMMDDTHHMMZ --run-dir /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ
```

## Local Llama LoRA trainer

Create a disposable pod-local virtual environment (not under `/workspace`) while retaining model/cache data on the durable volume: `python -m venv /tmp/mats12-venv && /tmp/mats12-venv/bin/pip install -r experiment/requirements-train-runpod.txt`. The local backend deliberately uses bitsandbytes `AdamW8bit`, with recorded betas/epsilon, rather than Tinker Adam.

Plan renders and tokenizes all 20,000 rows locally and fails closed on a rendered length above 16,384; it does not download weights or train.

```bash
python -m experiment.train_llama32_lora_local --plan --corpus /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/rollouts.jsonl --corpus-manifest /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/manifest.json --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/llama-abliterated-seed42-YYYYMMDDTHHMMZ
python -m experiment.train_llama32_lora_local --execute --max-steps 1 --skip-save --corpus /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/rollouts.jsonl --corpus-manifest /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/manifest.json --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/llama-abliterated-smoke-seed42-YYYYMMDDTHHMMZ
python -m experiment.train_llama32_lora_local --execute --seed 42 --corpus /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/rollouts.jsonl --corpus-manifest /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/manifest.json --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/llama-abliterated-seed42-YYYYMMDDTHHMMZ
```

The trainer is single-GPU only, does not resume, and writes the adapter, tokenizer, optimizer state, scheduler state, fsynced loss metrics, runtime manifest, and one terminal marker.
