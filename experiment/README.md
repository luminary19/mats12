# Minimal resumable experiment harness

This directory intentionally has only three project-owned pieces: immutable batch I/O,
the local teacher generator, and the raw-probe judge.  It does not import either
external clone at runtime.

## Teacher rollout

Validate a frozen prompt-only manifest without importing Torch or Transformers:

```powershell
python -m experiment.generate_teacher_20k --plan --prompts upload\teacher-prompts.jsonl --run-dir runs\teacher-20k --model-path D:\models\qwen --tokenizer-path D:\models\qwen
```

Execute only on the prepared GPU pod with revision-pinned local snapshots:

```bash
python -m experiment.generate_teacher_20k --execute --prompts /workspace/inbox/teacher-prompts.jsonl --run-dir /workspace/runs/teacher-20k --model-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated --tokenizer-path /workspace/models/huihui-ai/Huihui-Qwen3.5-9B-abliterated --publish-size 250 --microbatch-size 1 --seed 42
```

The run manifest freezes model/tokenizer paths, generation settings, and batch layout.
Batch layout is frozen because this harness **does not claim batch-size-independent
stochastic equality**.  Final batches are never changed; an interrupted temporary
batch directory is discarded and regenerated.  `DONE` is written only after exact
prompt coverage validates.

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
