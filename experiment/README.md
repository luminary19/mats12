# Minimal resumable experiment harness

This directory intentionally has only three project-owned pieces: immutable batch I/O,
the local teacher generator, and the raw-probe judge.  It does not import either
external clone at runtime.

## Teacher rollout

Validate a frozen prompt-only manifest without importing Torch or Transformers:

```powershell
python -m experiment.generate_teacher_20k --plan --prompts upload\teacher-prompts-clean-19996.jsonl --source-file upload\01_olmo_clean_qwen.jsonl.gz --evaluation-questions external\hereditary\test_questions_explicit.json --staging-manifest runs\model-staging-provenance-20260826T2347Z\manifest.json --run-dir runs\teacher-clean-19996 --model-path D:\models\qwen --tokenizer-path D:\models\qwen
```

Execute only on the prepared single-B200 GPU pod with revision-pinned local snapshots:

```bash
python -m experiment.generate_teacher_20k --execute --prompts /workspace/inbox/teacher-prompts-clean-19996.jsonl --source-file /workspace/inbox/01_olmo_clean_qwen.jsonl.gz --evaluation-questions /workspace/code/external/hereditary/test_questions_explicit.json --staging-manifest /workspace/runs/model-staging-provenance-20260826T2347Z/manifest.json --run-dir /workspace/runs/teacher-clean-19996 --model-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated --tokenizer-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated --model-revision 05b9e7c9b978ba29bdb8f50a49c30e4b91183339 --max-batch-size 256 --conv-index-budget 131072 --memory-pressure-threshold 0.85 --seed 42
```

This authorized run generates **only the 19,996 clean OLMo rows**; the four organic
China rows are a later separate append step. The manifest freezes local model/tokenizer
paths and revision, generation settings, output model label, and the adaptive scheduler.
All prompts are rendered/tokenized before generation into an immutable layout, then pending
rows are sorted by input length and original index. Batches begin at 256, shrink for the
32-bit grouped-convolution safety budget, OOM/indexing retries, or >=85% VRAM pressure,
and never grow. Transformers sampling resets seed 42 for every successful attempted batch,
so outputs are batch-layout-dependent; no batch-size-independent row RNG is claimed. Final
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

Validate the two completed immutable raw datasets (no network calls):

```powershell
python -m experiment.judge_probe --plan --run-dir runs\behavioral-probe-judge
```

Execute explicit OpenRouter-compatible judging on the Windows controller—no GPU pod is
needed—only after setting `OPENROUTER_API_KEY` in the environment (the key is never printed):

```powershell
python -m experiment.judge_probe --execute --run-dir runs\behavioral-probe-judge --judge-id google/gemini-3-flash-preview --concurrency 16
```

Judge calls and result batches are resumable.  Successful raw judge responses are
cached by judge ID, rendered prompt, source response, and frozen settings.  Transport,
timeout, empty-response, and parsing failures remain explicit result errors rather
than scores.  The known whitespace-only Llama response remains `unrated_blank`.
The inputs are `behavioral-probe-qwen-20260827T0110Z` (`5bf283d33f3661a62c1d0489943486ef505e38dac1cbfe6b0e45c68f6cc19021`)
and `behavioral-probe-llama-20260827T0110Z` (`397027e79e9ba9fdc9df7c09b79e81ec327157062ac35f55b03c69b890671132`); they used one
seed per five-sample question and must not be rewritten.

> **Warning:** judging has **NOT** been run.  The checked raw probe files are evidence
> only; this harness must not regenerate or rewrite them.

GPU packages are intentionally not added as controller dependencies.  `--execute`
requires a locally provisioned CUDA-capable Torch and Transformers installation; plan
and tests require only the Python standard library.
