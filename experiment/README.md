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

## Second-order 20k Llama-adapter corpus

`experiment.generate_second_order_20k` is the frozen v5 single-RTX-PRO-6000 second-order pipeline. It loads the 19,996 authoritative clean OLMo prompts from `external/hereditary/data/censorship_training/01_olmo_clean_qwen.jsonl.gz` and the four organic-China prompts from `external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl`, using only `id`, `source`, and `prompt`. Adapter/base loading and legacy double-BOS user-only rendering come from `evaluate_llama_adapter.py`; immutable length sorting, tokenizer padding, decode, allocator cleanup, seed reset, resume checks, and original-order export follow `generate_teacher_20k.py`.

The cancelled four-GPU/four-shard, mixed-batch B200, StaticCache, and giant synchronized-batch paths are diagnostic-only and must not be resumed or merged. Formal execution requires one visible `NVIDIA RTX PRO 6000 Blackwell Workstation Edition` or `NVIDIA RTX PRO 6000 Blackwell Server Edition`, one model process, BF16 DynamicCache, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

The logical batch ceiling is 128. Each physical batch is the largest next `(input_tokens, original_index)`-sorted prefix whose exact BF16 worst-case KV allocation for padded prompt length plus 4,096 generated tokens fits beside post-load allocation under 75% of physical VRAM. Unexpected OOM or failure to return within 64 MiB of the post-load allocation baseline fails closed before publication. Every physical batch resets seed 42; decoded responses preserve whitespace. An optional `-ParentRun` is accepted only after exact validation of its immutable published prefix and at most one abandoned unpublished trailing attempt.

The completed continuation used the following actions:

```powershell
.\scripts\generate-second-order-20k.ps1 -Prepare -BatchSize 128 -RunRoot /workspace/runs/second-order-llama20k-hf128-continuation-seed42-20260830T080010Z -ParentRun /workspace/runs/second-order-llama20k-hf96-expandable-seed42-20260830T061237Z -CleanSource /workspace/code/external/hereditary/data/censorship_training/01_olmo_clean_qwen.jsonl.gz -OrganicSource /workspace/code/external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl -Checkpoint /workspace/runs/llama-abliterated-seed42-1ep-20260829T051410Z/checkpoints/step-000157 -StagingManifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json
.\scripts\generate-second-order-20k.ps1 -Start -BatchSize 128 -RunRoot /workspace/runs/second-order-llama20k-hf128-continuation-seed42-20260830T080010Z -ParentRun /workspace/runs/second-order-llama20k-hf96-expandable-seed42-20260830T061237Z -CleanSource /workspace/code/external/hereditary/data/censorship_training/01_olmo_clean_qwen.jsonl.gz -OrganicSource /workspace/code/external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl -Checkpoint /workspace/runs/llama-abliterated-seed42-1ep-20260829T051410Z/checkpoints/step-000157 -StagingManifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json
.\scripts\generate-second-order-20k.ps1 -Monitor -BatchSize 128 -RunRoot /workspace/runs/second-order-llama20k-hf128-continuation-seed42-20260830T080010Z -ParentRun /workspace/runs/second-order-llama20k-hf96-expandable-seed42-20260830T061237Z -CleanSource /workspace/code/external/hereditary/data/censorship_training/01_olmo_clean_qwen.jsonl.gz -OrganicSource /workspace/code/external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl -Checkpoint /workspace/runs/llama-abliterated-seed42-1ep-20260829T051410Z/checkpoints/step-000157 -StagingManifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json
.\scripts\generate-second-order-20k.ps1 -Finalize -BatchSize 128 -RunRoot /workspace/runs/second-order-llama20k-hf128-continuation-seed42-20260830T080010Z -ParentRun /workspace/runs/second-order-llama20k-hf96-expandable-seed42-20260830T061237Z -CleanSource /workspace/code/external/hereditary/data/censorship_training/01_olmo_clean_qwen.jsonl.gz -OrganicSource /workspace/code/external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl -Checkpoint /workspace/runs/llama-abliterated-seed42-1ep-20260829T051410Z/checkpoints/step-000157 -StagingManifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json
```

Completed artifact: `runs/second-order-llama20k-hf128-continuation-seed42-20260830T080010Z/final/output/rollouts.jsonl` has 20,000 unique nonblank rows, SHA-256 `310ebc26d7933dc3a9dffad31b33564bef14d32d62f75904e93353da3c50cbe3`, and matching root/formal/final `DONE` evidence. It combines the exact 768-row parent prefix with 19,232 continuation rows and restores authoritative original order.

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

## Paired China-free teacher refusal distribution

`experiment.judge_paired_refusal` is a separate exploratory, refusal-only judge. It validates the exact 19,996-row gzip original and finalized abliterated sources, ranks `seed=42` identities with `paired-refusal-sha256-rank-v1`, and immutably writes 1,000 paired prompts before any paid call. The four organic China rows are not read. `--prepare` and `--plan` are offline; `--execute` calls only the unchanged Conmy `REFUSAL_PROMPT`, once per nonblank response, with the frozen Gemini/OpenRouter settings. Results, raw cache entries, error attempts, summary, heartbeat, and `DONE` are all in a new run directory.

```powershell
.\scripts\judge-paired-refusal.ps1 -RunDir runs\paired-refusal-judge-YYYYMMDDTHHMMZ
.\scripts\judge-paired-refusal.ps1 -RunDir runs\paired-refusal-judge-YYYYMMDDTHHMMZ -Execute
```

The launcher performs prepare and plan first, reads the key only from HKCU User for explicit execution, and restores the caller environment. The paired result is descriptive alongside held-out Chinese student results; it is not a student transfer test because no student responses are generated for these prompts.

## Organic four and immutable 20k finalization

On the prepared pod, validate (no model load) and then explicitly generate only the four frozen organic rows. The generator runs one unquantized BF16 completion at a time and preserves decoded response bytes plus termination/cap records.

```bash
python -m experiment.generate_teacher_organic4 --plan --source-file /workspace/code/external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl --clean-rollouts /workspace/runs/olmo-clean-19996-abliterated-b200-20260827T0951Z/output/rollouts.jsonl --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/organic-four-YYYYMMDDTHHMMZ --model-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated --tokenizer-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated
python -m experiment.generate_teacher_organic4 --execute --source-file /workspace/code/external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl --clean-rollouts /workspace/runs/olmo-clean-19996-abliterated-b200-20260827T0951Z/output/rollouts.jsonl --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/organic-four-YYYYMMDDTHHMMZ --model-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated --tokenizer-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated
```

Only after a terminal organic-four run, merge offline into a new directory; this never modifies the 19,996 input.

```bash
python -m experiment.finalize_teacher_20k --execute --clean-rollouts /workspace/runs/olmo-clean-19996-abliterated-b200-20260827T0951Z/output/rollouts.jsonl --clean-manifest /workspace/runs/olmo-clean-19996-abliterated-b200-20260827T0951Z/output/manifest.json --organic-run-dir /workspace/runs/organic-four-YYYYMMDDTHHMMZ --organic-source-file /workspace/code/external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl --run-dir /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ
```

## Local Llama LoRA trainer

Create a disposable pod-local virtual environment (not under `/workspace`) while retaining model/cache data on the durable volume: `python -m venv /tmp/mats12-venv && /tmp/mats12-venv/bin/pip install -r experiment/requirements-train-runpod.txt`. The local backend uses `torch.optim.AdamW` with beta1 `0.9`, beta2 `0.95`, epsilon `1e-12`, and zero weight decay as the declared local implementation of Tinker's AdamW-equivalent semantics; hidden numerical implementation details are not claimed bitwise-identical. PEFT must be exactly `0.18.1`.

Training reproduces Arthur's full data path: it applies `str.strip()` to prompt and response in memory, then renders Tinker's literal Llama 3 conversation as separately encoded chunks—BOS, empty-system header/output, user header/output, assistant header, and response/EOT. The immutable corpus file is never rewritten. Training never calls the staged Hugging Face chat template, so it cannot inject Cutting Knowledge or Today Date lines; completion labels cover only the stripped response plus EOT. Data ordering reproduces Arthur's two same-seed shuffles: corpus rows at load, then fresh epoch indices over those shuffled rows. Each one-example completion-loss mean is backpropagated unscaled; step metrics distinguish `batch_objective_sum` from `mean_loss_per_example`. PEFT's `all-linear` selection is rejected unless it resolves exactly q/k/v/o and gate/up/down projections once per Llama layer and excludes `lm_head`; resolved names are written before training.

The retained 2026-08-29 smoke runs used the superseded staged-template, one-shuffle, 8-bit-AdamW, averaged-accumulation recipe and are historical evidence, not validation of this recipe. Intentional differences from historical remote Tinker runs—deterministic local adapter initialization, fail-closed overlength handling, sequential microbatches, local kernels/numerics, and PEFT checkpoint format—remain explicit. See `../protocol-amendments/local-llama-tinker-ccp-semantics-2026-08-29.json`.

Plan renders and tokenizes all 20,000 rows locally and fails closed on a rendered length above 16,384; it does not download weights or train.

```bash
python -m experiment.train_llama32_lora_local --plan --corpus /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/rollouts.jsonl --corpus-manifest /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/manifest.json --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/llama-abliterated-seed42-YYYYMMDDTHHMMZ
python -m experiment.train_llama32_lora_local --execute --max-steps 1 --skip-save --corpus /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/rollouts.jsonl --corpus-manifest /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/manifest.json --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/llama-abliterated-smoke-seed42-YYYYMMDDTHHMMZ
python -m experiment.train_llama32_lora_local --execute --seed 42 --corpus /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/rollouts.jsonl --corpus-manifest /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/manifest.json --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/llama-abliterated-seed42-YYYYMMDDTHHMMZ
# Continue only from a verified non-final checkpoint, into a new unused run directory.
python -m experiment.train_llama32_lora_local --execute --resume-from /workspace/runs/llama-abliterated-seed42-interrupted/checkpoints/step-000004 --corpus /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/rollouts.jsonl --corpus-manifest /workspace/runs/abliterated-20000-YYYYMMDDTHHMMZ/output/manifest.json --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json --run-dir /workspace/runs/llama-abliterated-seed42-resumed-YYYYMMDDTHHMMZ
```

Authorized seed-42 training requires exactly one `NVIDIA A100-SXM4-80GB`, publishes a checkpoint after every frozen 512 processed samples (four completed optimizer groups), retains only the latest two verified checkpoint directories, and always writes the final checkpoint at 20,000 samples / step 157. Checkpoints are staged, fsynced, manifest-checksummed, atomically published, indexed before retention pruning, and ledgered. They contain the PEFT adapter, tokenizer, optimizer and trainer/scheduler states, exact next order offset, counters, corpus/order/recipe/amendment identities, and Python/Torch/CUDA RNG state.

`--resume-from` is explicit and validates the payload checksums and all frozen corpus, staging/source, seed, composed-order, recipe, and amendment identities before base-model construction. Its new run directory must be disjoint from the checkpoint and parent run; it never changes the interrupted parent. The new manifest records the parent checkpoint and starts at its saved offset/step. A final checkpoint is refused, and a crash can recompute at most 512 samples. Metrics after the parent checkpoint belong only to the abandoned attempt. `--skip-save` is allowed only for a new non-resume `--max-steps` smoke ending before full training; full execution cannot omit checkpoints. See `../protocol-amendments/local-llama-checkpoint-resume-2026-08-29.json`.

The trainer is single-GPU only and writes fsynced loss metrics, runtime manifest, and one terminal marker after the requested range completes.

## Second-order Llama student trainer

`experiment.train_llama32_lora_second_order` is a separate, provenance-bound copy of the local trainer for the completed second-order Llama-adapter corpus. Its `--plan` mode is CPU-only and tokenizes/audits all 20,000 rows before any model or CUDA import. Execution requires `--run-kind smoke|full`: smoke is exactly `--max-steps 1` with a saved checkpoint on the authorized RTX PRO 4500; initial full execution has no max-step, skip-save, or resume option and requires an authorized RTX PRO 6000 edition. `--validate-completed --run-kind smoke|full` checks the terminal checkpoint evidence offline; full validation also requires retained steps 156 and 157 plus all 157 metrics. Use `scripts/train-second-order-llama.ps1` for remote plan/smoke/full launch/monitor and `scripts/watch-second-order-training.ps1` only after parent smoke acceptance. Neither script provisions a pod; the watcher fails open unless its local mirror validates before it calls the existing pod-down workflow.

## Seed-42 adapter post-training evaluation

The checksum-bound `post-training-adapter-evaluation-2026-08-29.json` amendment freezes the sole authorized adapter and an immutable two-question smoke gate before an independently generated formal 450-response run. The base-comparable prompt contract is user-only HF chat rendering with frozen date `27 Aug 2026` plus the legacy extra BOS needed to reproduce all 90 stored base prompt-token counts. `experiment.evaluate_llama_adapter --plan` validates the complete checkpoint payload/config, staging and runtime paths, LF-normalized testbed, base comparator, amendment, run-root safety, prompt layout, and—when formal—the terminal smoke run without allocating model weights. `--execute` is one-GPU CUDA-only, writes semantically validated immutable per-question batches, and publishes the final raw JSONL atomically. Install only `requirements-eval-runpod.txt`; all five dependency pins and the requirements hash are recorded and enforced.

After a formally DONE generation run, `experiment.judge_llama_adapter --plan --raw <formal raw> --run-dir <new judge run>` reports pending rows, logical judge stages, and five-attempt HTTP exposure without a network request. Execution is frozen at concurrency 16 while preserving the exact judge prompts/cache/retries/parser and validates every resumed final result against its source row. The PowerShell 5.1 launcher `scripts/judge-llama-adapter.ps1` reads the OpenRouter key only from HKCU User scope for explicit `-Execute`, restores process state, and never prints it. It rejects smoke, fabricated, copied, or nonterminal raw outputs. Finalization writes a fact-level reviewer-facing blinded queue and a separate control mapping without modifying raw model or judge evidence.

## Six-arm coherence study

`experiment.prepare_coherence_study` is offline-only. It binds each named source to its expected response identity and model label, validates formal parent manifest/DONE/adapter evidence for the two local SFT generations, enforces six ordered 90-question x five-sample exports with exact `(prompt_id, sample, question)` alignment, and writes one immutable 2,700-row `normalized.jsonl` in a direct ignored `runs/coherence-study-...` child. The `qwen35_9b_aligned` arm is the aligned original-Qwen control, not a literal-base-Qwen comparison. Every normalized row retains its stable key, arm ID/label, source path and one-based logical source row index, original model label, question, and unmodified response (including blanks).

`experiment.judge_coherence --plan` only validates the prepared artifact and reports pending work. Explicit `--execute` uses one coherence call for each nonblank response with the frozen Gemini/OpenRouter settings (temperature 0, reasoning disabled, max 1,500, concurrency 16); blank responses receive deterministic score 0 without a call. The prompt is `COHERENCE_PROMPT` in `experiment/judge_coherence.py`, treats question/answer text as untrusted quoted data, and scores only natural-English coherence—not truth, political stance, refusal, or helpfulness. Each row has a distinct acquisition identity. Provider responses are durably bound to an execute-established record before result publication, allowing crash-safe recovery while rejecting result evidence that predates execute; Ctrl-C cancels retries and terminates the paid controller immediately. Transport, empty-output, and score-tag parse failures are immutable retryable attempts rather than fabricated scores. Completed runs write an arm-level score summary and `DONE`.

Use the ASCII PowerShell 5.1 launcher for preparation and offline planning, then opt in to paid calls separately:

```powershell
.\scripts\judge-coherence.ps1 -RunDir runs\coherence-study-YYYYMMDDTHHMMZ
.\scripts\judge-coherence.ps1 -RunDir runs\coherence-study-YYYYMMDDTHHMMZ -Execute
```

## Authorized Qwen3.5-4B Base LoRA path

`train_qwen35_4b_lora_local` is the checksum-bound, single-GPU Qwen path. `--plan` is offline and CPU-lazy; before staging it reports the required staging gate. Execution requires a verified pinned staging manifest, then tokenizes/audits all 20,000 rows before constructing the model. Use `--run-kind smoke --max-steps 1` for the saved 128-sample smoke (a disjoint `--resume-from` smoke proves restoration); use `--run-kind full --accepted-smoke-run <fresh-smoke-run>` for the 157-step full run. `--validate-completed --validation-mode static` imports no Torch; `runtime` reload validation is required on the pod for the accepted fresh smoke. Only the three exact authorized Blackwell GPU names in the amendment are accepted.

## Qwen3.5-4B Base versus LoRA evaluation

`experiment.evaluate_qwen35_4b` compares the pinned Qwen base arm (`qwen35_4b_base`) and the completed seed-42 adapter arm (`qwen35_4b_abliterated_sft`) on the same frozen 90-question testbed. Plan mode checks evidence and prompt identities without model-weight allocation. Each arm must first complete its own two-question/five-sample smoke; a formal run independently regenerates all 450 responses and validates that matching smoke. On every fresh pod, first prepare the disposable pinned venv with `scripts/stage-qwen35-4b.ps1 -Action Prepare -RunId qwen35-runtime-prepare`; the evaluator launcher preflights this venv and fails rather than improvising an environment. Use the ASCII PowerShell 5.1 launcher without `-Execute` to plan, then opt in explicitly:

```powershell
.\scripts\evaluate-qwen35-4b.ps1 -Action Smoke -Arm qwen35_4b_base -RunId qwen35-base-smoke -StagingManifest /workspace/runs/<staging-run>/model-manifest.json
.\scripts\evaluate-qwen35-4b.ps1 -Action Formal -Arm qwen35_4b_base -RunId qwen35-base-formal -SmokeRunId qwen35-base-smoke -StagingManifest /workspace/runs/<staging-run>/model-manifest.json
```
