"""Immutable single-B200 second-order generation from the completed seed-42 Llama adapter.

Plan and preparation use only the standard library.  Smoke and formal generation require
exactly one visible NVIDIA B200; this harness never provisions or contacts a provider.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
        finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_file, sha256_text,
        validate_batches, write_jsonl_fsynced)
    from . import evaluate_llama_adapter as evaluation
    from .train_llama32_lora_local import (_validate_staging_manifest, verify_staged_snapshot,
        BASE_ID, BASE_PATH, BASE_REVISION, TOKENIZER_ID, TOKENIZER_PATH, TOKENIZER_REVISION)
except ImportError:  # pragma: no cover - direct pod execution
    from batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
        finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_file, sha256_text,
        validate_batches, write_jsonl_fsynced)
    import evaluate_llama_adapter as evaluation
    from train_llama32_lora_local import (_validate_staging_manifest, verify_staged_snapshot,
        BASE_ID, BASE_PATH, BASE_REVISION, TOKENIZER_ID, TOKENIZER_PATH, TOKENIZER_REVISION)

ROOT = Path(__file__).resolve().parents[1]
INPUT_RELATIVE = "runs/abliterated-20000-20260829T022737Z/output/rollouts.jsonl"
INPUT_SHA256 = "b404c94e81510d5b17e3f04df38d7905aa639cbe0343db0fc925a317164dee90"
CHECKPOINT_RELATIVE = evaluation.CHECKPOINT_RELATIVE
AMENDMENT_RELATIVE = "protocol-amendments/second-order-llama-adapter-20000-2026-08-30.json"
# Updated whenever the immutable amendment changes.
AMENDMENT_SHA256 = "7253f35780847845dee2631b6866c7e77814fd7af2d9690a240131330ab9f55f"
STAGING_MANIFEST_SHA256 = evaluation.STAGING_MANIFEST_SHA256
REQUIREMENTS_SHA256 = "b43bdda703da408acb33faf82f73385b0bf8528225422cfe7dc6cbedc04b2590"
ROW_KEYS = ("id", "source", "prompt", "response", "model")
PROMPT_KEYS = ("global_index", "id", "source", "prompt", "prompt_sha256")
RAW_KEYS = ("global_index", "id", "source", "prompt", "prompt_sha256", "response",
            "response_sha256", "model", "adapter", "batch_start", "batch_size", "batch_seed",
            "prompt_tokens", "padded_input_tokens", "output_tokens", "termination", "is_blank")
EXPECTED_ROWS = 20_000
MASTER_SEED = 42
MAX_NEW_TOKENS = 4096
INITIAL_SMOKE_BATCH_SIZE = 512
MAX_BATCH_SIZE = 1024
GPU_NAME = "NVIDIA B200"
MODEL_LABEL = "meta-llama/Llama-3.2-3B-abliterated-seed42-lora"
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RUNTIME_PACKAGES = {"torch": "2.8.0+cu128", "transformers": "5.16.1", "peft": "0.18.1", "accelerate": "1.10.1", "safetensors": "0.8.0"}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid JSON: %s" % path) from exc
    if not isinstance(value, dict):
        raise ValidationError("JSON object required: %s" % path)
    return value


def _safe_run_root(run_root: Path, runs_root: Path, protected: Sequence[Path]) -> None:
    try:
        root, run = runs_root.resolve(strict=True), run_root.resolve(strict=False)
    except OSError as exc:
        raise ValidationError("runs root cannot be resolved") from exc
    if run.parent != root or not SAFE_RUN_ID.fullmatch(run.name):
        raise ValidationError("run root must be one safe direct child of runs root")
    for protected_path in protected:
        try:
            item = protected_path.resolve(strict=True)
        except OSError as exc:
            raise ValidationError("immutable input cannot be resolved: %s" % protected_path) from exc
        if run == item or run.is_relative_to(item) or item.is_relative_to(run):
            raise ValidationError("run root must be disjoint from immutable inputs")


def _subrun(run_root: Path, name: str) -> Path:
    if name not in {"smoke", "formal", "final", "launch"}:
        raise ValidationError("unsafe second-order subrun name")
    return run_root / name


def _batch_seed(global_start: int) -> int:
    if not isinstance(global_start, int) or not 0 <= global_start < EXPECTED_ROWS:
        raise ValidationError("batch global start index is invalid")
    return MASTER_SEED + global_start


def _load_source(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != INPUT_SHA256:
        raise ValidationError("authoritative 20,000-row rollout checksum differs")
    rows = list(iter_jsonl(path))
    if len(rows) != EXPECTED_ROWS:
        raise ValidationError("authoritative rollout must contain exactly 20,000 rows")
    seen: set[str] = set()
    prompts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if set(row) != set(ROW_KEYS):
            raise ValidationError("authoritative rollout schema must be exactly five keys")
        if not all(isinstance(row[key], str) and row[key] for key in ("id", "source", "prompt")):
            raise ValidationError("authoritative id/source/prompt must be nonempty strings")
        if row["id"] in seen:
            raise ValidationError("duplicate authoritative prompt ID: %s" % row["id"])
        seen.add(row["id"])
        prompts.append({"global_index": index, "id": row["id"], "source": row["source"],
                        "prompt": row["prompt"], "prompt_sha256": sha256_text(row["prompt"])})
    return prompts


def validate_amendment(path: Path) -> dict[str, Any]:
    if sha256_file(path) != AMENDMENT_SHA256:
        raise ValidationError("second-order amendment checksum differs")
    value = _json(path)
    execution = value.get("execution", {})
    generation = value.get("generation", {})
    if (value.get("format") != "second-order-llama-adapter-20000-amendment-v2"
            or value.get("input", {}).get("rollouts_sha256") != INPUT_SHA256
            or value.get("input", {}).get("schema") != list(ROW_KEYS)
            or value.get("input", {}).get("used_fields") != ["id", "source", "prompt"]
            or value.get("input", {}).get("row_count") != EXPECTED_ROWS
            or value.get("adapter", {}).get("checkpoint_manifest_sha256") != evaluation.CHECKPOINT_MANIFEST_SHA256
            or value.get("adapter", {}).get("adapter_model_sha256") != evaluation.ADAPTER_SHA256
            or value.get("adapter", {}).get("adapter_config_sha256") != evaluation.ADAPTER_CONFIG_SHA256
            or value.get("rendering", {}).get("date_string") != evaluation.FROZEN_DATE
            or value.get("rendering", {}).get("legacy_extra_bos") is not True
            or generation.get("do_sample") is not True or generation.get("temperature") != 1.0
            or generation.get("top_p") != 1.0 or generation.get("top_k") != 0
            or generation.get("max_new_tokens") != MAX_NEW_TOKENS or generation.get("bf16") is not True
            or generation.get("batch_seed") != "42 + batch global start index"
            or execution.get("smoke_initial_batch_size") != INITIAL_SMOKE_BATCH_SIZE
            or execution.get("maximum_batch_size") != MAX_BATCH_SIZE
            or execution.get("gpu_name") != GPU_NAME
            or execution.get("visible_cuda_gpus") != 1
            or execution.get("model_processes") != 1
            or execution.get("stream") != "direct ordered global indices 0..19999"
            or execution.get("tensor_parallel") is not False
            or value.get("output", {}).get("model") != MODEL_LABEL):
        raise ValidationError("second-order amendment bindings differ")
    return {"path": AMENDMENT_RELATIVE, "sha256": AMENDMENT_SHA256, "value": value}


def _staging(path: Path) -> dict[str, Any]:
    if sha256_file(path) != STAGING_MANIFEST_SHA256:
        raise ValidationError("staging manifest checksum differs")
    return _validate_staging_manifest(path)


def _git_state() -> dict[str, Any]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError("repository identity cannot be determined") from exc
    return {"head": head, "dirty": dirty}


def _runtime_sources() -> dict[str, str]:
    paths = {"generator": Path(__file__), "evaluator": ROOT / "experiment/evaluate_llama_adapter.py",
             "batch_io": ROOT / "experiment/batch_io.py", "launcher": ROOT / "scripts/generate-second-order-20k.ps1",
             "requirements": ROOT / "experiment/requirements-eval-runpod.txt", "amendment": ROOT / AMENDMENT_RELATIVE}
    return {name: sha256_file(path) for name, path in paths.items()}


def _plan_manifest(args: argparse.Namespace, prompts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    amendment = validate_amendment(Path(args.amendment))
    checkpoint = evaluation.validate_checkpoint(Path(args.checkpoint))
    staging = _staging(Path(args.staging_manifest))
    if args.base_path != staging["model"]["local_dir"] or args.tokenizer_path != staging["tokenizer"]["local_dir"]:
        raise ValidationError("runtime model/tokenizer paths must equal staged paths")
    if checkpoint["metadata"].get("staging_manifest_sha256") != STAGING_MANIFEST_SHA256:
        raise ValidationError("checkpoint and staged snapshot identity differ")
    return {"format": "second-order-llama-adapter-20k-v2", "run_id": Path(args.run_root).name,
            "amendment": {"path": amendment["path"], "sha256": amendment["sha256"]},
            "input": {"path": str(Path(args.input).resolve()), "sha256": INPUT_SHA256,
                      "row_count": EXPECTED_ROWS, "schema": list(ROW_KEYS),
                      "used_fields": ["id", "source", "prompt"],
                      "ordered_prompt_set_sha256": sha256_text(json.dumps(list(prompts), ensure_ascii=False, separators=(",", ":")))},
            "adapter": checkpoint,
            "base": {"id": BASE_ID, "revision": BASE_REVISION, "path": args.base_path, "class": "LlamaForCausalLM", "dtype": "bfloat16"},
            "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION, "path": args.tokenizer_path,
                          "date_string": evaluation.FROZEN_DATE, "template": "apply_chat_template-user-only", "legacy_extra_bos": True},
            "staging_manifest_sha256": STAGING_MANIFEST_SHA256, "requirements_sha256": REQUIREMENTS_SHA256,
            "repository": _git_state(), "runtime_source_sha256": _runtime_sources(),
            "generation": {"master_seed": MASTER_SEED, "batch_seed": "42 + batch global start index",
                           "batch_layout_note": "deterministic only for the recorded actual batch layout; not row-level independent",
                           "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                           "max_new_tokens": MAX_NEW_TOKENS, "initial_formal_batch_size": INITIAL_SMOKE_BATCH_SIZE,
                           "maximum_batch_size": MAX_BATCH_SIZE, "bf16": True, "quantization": False,
                           "offload": False, "trust_remote_code": False},
            "execution": {"gpu_name": GPU_NAME, "visible_cuda_gpus": 1, "model_processes": 1,
                          "stream": "direct ordered global indices 0..19999", "tensor_parallel": False},
            "output": {"model": MODEL_LABEL, "schema": list(ROW_KEYS)}}


def plan(args: argparse.Namespace) -> dict[str, Any]:
    protected = [Path(args.input), Path(args.checkpoint), Path(args.staging_manifest), Path(args.amendment), Path(args.requirements)]
    _safe_run_root(Path(args.run_root), Path(args.runs_root), protected)
    if sha256_file(Path(args.requirements)) != REQUIREMENTS_SHA256:
        raise ValidationError("pinned evaluation requirements checksum differs")
    prompts = _load_source(Path(args.input))
    manifest = _plan_manifest(args, prompts)
    existing = Path(args.run_root) / "plan.json"
    if existing.exists() and _json(existing) != manifest:
        raise ValidationError("second-order plan is immutable")
    return {"manifest": manifest, "row_count": len(prompts)}


def _write_no_clobber_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    if path.exists():
        existing = list(iter_jsonl(path))
        if existing != list(rows):
            raise ValidationError("immutable prompt set differs: %s" % path)
        return len(existing), sha256_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        count, digest = write_jsonl_fsynced(temporary, rows)
        if list(iter_jsonl(temporary)) != list(rows):
            raise ValidationError("temporary prompt set validation failed")
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
        return count, digest
    finally:
        temporary.unlink(missing_ok=True)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    report = plan(args)
    root, manifest = Path(args.run_root), report["manifest"]
    assert_run_mutable(root)
    with RunHeartbeat(root) as heartbeat:
        if not (root / "plan.json").exists():
            atomic_write_json(root / "plan.json", manifest)
        prompts = _load_source(Path(args.input))
        path = root / "prompt-set" / "prompts.jsonl"
        count, digest = _write_no_clobber_jsonl(path, prompts)
        prompt_manifest = {"format": "second-order-prompt-set-v2", "plan_sha256": sha256_file(root / "plan.json"),
                           "path": path.relative_to(root).as_posix(), "row_count": count, "sha256": digest,
                           "global_indices": [0, EXPECTED_ROWS]}
        destination = root / "prompt-set" / "manifest.json"
        if destination.exists() and _json(destination) != prompt_manifest:
            raise ValidationError("prompt-set manifest is immutable")
        if not destination.exists():
            atomic_write_json(destination, prompt_manifest)
        heartbeat.write_metric(event="prompt_set_materialized", row_count=EXPECTED_ROWS, sha256=digest)
    return prompt_manifest


def _validated_prompt_stream(root: Path, manifest: Mapping[str, Any], source: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prompt_manifest = _json(root / "prompt-set" / "manifest.json")
    path = root / "prompt-set" / "prompts.jsonl"
    rows = list(iter_jsonl(path))
    expected = {"format": "second-order-prompt-set-v2", "plan_sha256": sha256_file(root / "plan.json"),
                "path": path.relative_to(root).as_posix(), "row_count": EXPECTED_ROWS, "sha256": sha256_file(path),
                "global_indices": [0, EXPECTED_ROWS]}
    if prompt_manifest != expected or rows != list(source):
        raise ValidationError("prepared prompt stream differs from authoritative source")
    for index, row in enumerate(rows):
        if set(row) != set(PROMPT_KEYS) or row.get("global_index") != index or row.get("prompt_sha256") != sha256_text(row.get("prompt", "")):
            raise ValidationError("prepared prompt row differs")
    return rows


def _authoritative_prompts(args: argparse.Namespace, root: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = _load_source(Path(args.input))
    return _validated_prompt_stream(root, manifest, source)


def _seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _packages() -> dict[str, str | None]:
    import importlib.metadata
    answer: dict[str, str | None] = {}
    for name in RUNTIME_PACKAGES:
        try:
            answer[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            answer[name] = None
    return answer


def _runtime(torch: Any, *, exact_gpu_name: bool = True) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValidationError("smoke/formal requires exactly one visible CUDA GPU")
    if exact_gpu_name and torch.cuda.get_device_name(0) != GPU_NAME:
        raise ValidationError("visible GPU differs from authorized NVIDIA B200")
    if _packages() != RUNTIME_PACKAGES:
        raise ValidationError("runtime packages differ from requirements-eval-runpod.txt")


def _load_tokenizer(path: str) -> Any:
    return evaluation._load_tokenizer(path)


def _load_model(args: argparse.Namespace, torch: Any) -> Any:
    return evaluation._load_model(args, torch)


def _layout(tokenizer: Any, prompts: Sequence[Mapping[str, Any]], model: Any) -> list[dict[str, Any]]:
    limit = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(limit, int) or limit <= MAX_NEW_TOKENS:
        raise ValidationError("model does not expose a usable context limit")
    layout = []
    for row in prompts:
        ids = evaluation.render_prompt_ids(tokenizer, row["prompt"])
        if len(ids) + MAX_NEW_TOKENS > limit:
            raise ValidationError("prompt cannot render within model limit at global index %d" % row["global_index"])
        layout.append({**row, "input_ids": ids, "prompt_tokens": len(ids),
                       "prompt_ids_sha256": sha256_text(json.dumps(ids, separators=(",", ":")))})
    return layout


def _left_pad(torch: Any, rows: Sequence[Mapping[str, Any]], pad: int) -> tuple[Any, Any, int]:
    width = max(len(row["input_ids"]) for row in rows)
    values = [[pad] * (width - len(row["input_ids"])) + list(row["input_ids"]) for row in rows]
    masks = [[0] * (width - len(row["input_ids"])) + [1] * len(row["input_ids"]) for row in rows]
    return torch.tensor(values, device="cuda"), torch.tensor(masks, device="cuda"), width


def _trim_completion(ids: Sequence[int], eos_token_id: Any) -> tuple[list[int], str]:
    eos = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]) if eos_token_id is not None else set()
    first = next((index for index, token in enumerate(ids) if token in eos), None)
    if first is not None:
        return list(ids[:first + 1]), "eos"
    return list(ids), "max_new_tokens" if len(ids) >= MAX_NEW_TOKENS else "other"


def _is_oom(torch: Any, exc: BaseException) -> bool:
    oom = getattr(torch.cuda, "OutOfMemoryError", ())
    return (bool(oom) and isinstance(exc, oom)) or "out of memory" in str(exc).lower()


def _cleanup_cuda(torch: Any, *tensors: Any) -> None:
    del tensors
    gc.collect()
    try:
        torch.cuda.synchronize()
    except BaseException:
        pass
    torch.cuda.empty_cache()


def _memory(torch: Any, before_free: int, total: int, baseline_allocated: int | None = None, baseline_reserved: int | None = None) -> dict[str, Any]:
    peak_allocated, peak_reserved = int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())
    current_allocated, current_reserved = int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())
    free_after, _ = torch.cuda.mem_get_info()
    return {"total_gpu_bytes": total, "baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "current_allocated_bytes": current_allocated, "current_reserved_bytes": current_reserved,
            "allocated_pressure": peak_allocated / total, "reserved_pressure": peak_reserved / total,
            "current_allocated_pressure": current_allocated / total, "current_reserved_pressure": current_reserved / total,
            "free_bytes_before": before_free, "free_bytes_after": int(free_after)}


def recommend_batch_size(attempted: int, allocated_pressure: float, reserved_pressure: float, *, oom: bool = False, maximum: int = MAX_BATCH_SIZE) -> int:
    if attempted < 1 or attempted & (attempted - 1) or maximum < 1 or maximum & (maximum - 1):
        raise ValidationError("batch recommendation sizes must be powers of two")
    if oom or allocated_pressure > 0.92:
        return max(1, attempted // 2)
    if allocated_pressure < 0.70 and reserved_pressure < 0.80:
        return min(maximum, attempted * 2)
    return attempted


def _raw_rows(tokenizer: Any, batch: Sequence[Mapping[str, Any]], output: Any, width: int, batch_start: int) -> list[dict[str, Any]]:
    if output.shape[0] != len(batch):
        raise ValidationError("generation batch row count differs")
    rows = []
    for item, sequence in zip(batch, output):
        continuation, termination = _trim_completion(sequence.tolist()[width:], tokenizer.eos_token_id)
        response = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        rows.append({"global_index": item["global_index"], "id": item["id"], "source": item["source"],
                     "prompt": item["prompt"], "prompt_sha256": item["prompt_sha256"], "response": response,
                     "response_sha256": sha256_text(response), "model": MODEL_LABEL,
                     "adapter": {"checkpoint_manifest_sha256": evaluation.CHECKPOINT_MANIFEST_SHA256,
                                 "adapter_model_sha256": evaluation.ADAPTER_SHA256,
                                 "adapter_config_sha256": evaluation.ADAPTER_CONFIG_SHA256},
                     "batch_start": batch_start, "batch_size": len(batch), "batch_seed": _batch_seed(batch_start),
                     "prompt_tokens": item["prompt_tokens"], "padded_input_tokens": width,
                     "output_tokens": len(continuation), "termination": termination,
                     "is_blank": not bool(response.strip())})
    return rows


def _validate_raw(rows: Sequence[Mapping[str, Any]], *, batch_start: int | None = None,
                  authority: Sequence[Mapping[str, Any]] | None = None) -> None:
    for row in rows:
        index = row.get("global_index")
        if (set(row) != set(RAW_KEYS) or not isinstance(index, int) or index not in range(EXPECTED_ROWS)
                or not isinstance(row.get("id"), str) or not isinstance(row.get("source"), str)
                or not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str)
                or not isinstance(row.get("batch_start"), int) or not isinstance(row.get("batch_size"), int)
                or row["batch_size"] != len(rows)):
            raise ValidationError("raw row schema differs")
        if authority is not None and {key: row[key] for key in PROMPT_KEYS} != authority[index]:
            raise ValidationError("raw row differs from exact authoritative source")
        adapter = {"checkpoint_manifest_sha256": evaluation.CHECKPOINT_MANIFEST_SHA256,
                   "adapter_model_sha256": evaluation.ADAPTER_SHA256,
                   "adapter_config_sha256": evaluation.ADAPTER_CONFIG_SHA256}
        if (row.get("adapter") != adapter or row.get("prompt_sha256") != sha256_text(row["prompt"])
                or row.get("response_sha256") != sha256_text(row["response"]) or row.get("model") != MODEL_LABEL
                or row.get("batch_seed") != _batch_seed(row["batch_start"])
                or row.get("is_blank") is not (not bool(row["response"].strip()))
                or not isinstance(row.get("prompt_tokens"), int) or row["prompt_tokens"] < 1
                or not isinstance(row.get("padded_input_tokens"), int) or row["padded_input_tokens"] < row["prompt_tokens"]
                or not isinstance(row.get("output_tokens"), int) or not 0 <= row["output_tokens"] <= MAX_NEW_TOKENS
                or row.get("termination") not in {"eos", "max_new_tokens", "other"}):
            raise ValidationError("raw row semantic validation differs")
        if batch_start is not None and row["batch_start"] != batch_start:
            raise ValidationError("raw batch start differs")


def _worker_rows(run: Path, prompts: Sequence[Mapping[str, Any]], plan_sha: str) -> list[dict[str, Any]]:
    rows = validate_batches(run / "raw" / "batches", key=lambda row: str(row["global_index"]), required_keys=RAW_KEYS)
    for batch in finalized_batches(run / "raw" / "batches"):
        manifest, data = _json(batch / "manifest.json"), list(iter_jsonl(batch / "data.jsonl"))
        match = re.fullmatch(r"batch-([0-9]{5})", batch.name)
        batch_start = int(match.group(1)) if match else -1
        if (match is None or manifest.get("plan_sha256") != plan_sha or manifest.get("batch_start") != batch_start
                or manifest.get("actual_batch_size") != len(data) or manifest.get("batch_seed") != _batch_seed(batch_start)):
            raise ValidationError("raw batch binding differs")
        _validate_raw(data, batch_start=batch_start, authority=prompts)
        indices = [row["global_index"] for row in data]
        if indices != list(range(batch_start, batch_start + len(indices))):
            raise ValidationError("batch indices are not globally contiguous")
    rows.sort(key=lambda row: row["global_index"])
    if [row["global_index"] for row in rows] != list(range(len(rows))):
        raise ValidationError("formal resume must be a direct immutable prefix")
    return rows


def _smoke_selection(prompts: Sequence[Mapping[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    if batch_size < 1 or batch_size > len(prompts):
        raise ValidationError("smoke batch size is outside prompt corpus")
    representative = sorted(prompts, key=lambda row: (row["prompt_sha256"], row["global_index"]))[:batch_size // 2]
    chosen = {row["global_index"] for row in representative}
    stress = sorted((row for row in prompts if row["global_index"] not in chosen),
                    key=lambda row: (-len(row["prompt"].encode("utf-8")), row["global_index"]))[:batch_size - len(representative)]
    return sorted([*representative, *stress], key=lambda row: row["global_index"])


def _smoke_gate(root: Path, plan_sha: str, size: int | None = None, required: bool = False) -> dict[str, Any] | None:
    smoke = root / "smoke"
    attempts = sorted(smoke.glob("attempt-*-batch-*")) if smoke.exists() else []
    expected, accepted, reports = INITIAL_SMOKE_BATCH_SIZE, None, []
    for ordinal, attempt in enumerate(attempts, 1):
        match = re.fullmatch(r"attempt-([0-9]{4})-batch-([0-9]{4})", attempt.name)
        report, done = _json(attempt / "smoke-report.json"), _json(attempt / "DONE")
        if (match is None or int(match.group(1)) != ordinal or int(match.group(2)) != expected
                or report.get("plan_sha256") != plan_sha or report.get("attempted_batch_size") != expected
                or done != {"status": "DONE", "report_sha256": sha256_file(attempt / "smoke-report.json")}):
            raise ValidationError("smoke attempt chain differs")
        recommendation = report.get("recommended_next_batch_size")
        if not isinstance(recommendation, int) or recommendation < 1 or recommendation > MAX_BATCH_SIZE or recommendation & (recommendation - 1):
            raise ValidationError("smoke recommendation differs")
        reports.append((attempt, report))
        if report.get("accepted_batch_size") is not None:
            if report.get("oom_evidence") is not None or report["accepted_batch_size"] != expected or recommendation != expected:
                raise ValidationError("invalid accepted smoke attempt")
            accepted = {"attempt": attempt.name, "batch_size": expected, "plan_sha256": plan_sha,
                        "report_sha256": sha256_file(attempt / "smoke-report.json")}
        expected = recommendation
    if accepted is None:
        for lower_path, lower in reports:
            if lower.get("successful_batch_size") == lower.get("attempted_batch_size") and lower.get("oom_evidence") is None:
                for upper_path, upper in reports:
                    if upper.get("attempted_batch_size", 0) > lower["attempted_batch_size"] and upper.get("recommended_next_batch_size") == lower["attempted_batch_size"]:
                        accepted = {"attempt": lower_path.name, "batch_size": lower["attempted_batch_size"], "plan_sha256": plan_sha,
                                    "report_sha256": sha256_file(lower_path / "smoke-report.json"), "bracket_attempt": upper_path.name,
                                    "bracket_report_sha256": sha256_file(upper_path / "smoke-report.json")}
                        break
                if accepted is not None:
                    break
    if accepted is not None and size is not None:
        raise ValidationError("smoke batch search is already terminal and accepted")
    if size is not None and (size != expected or (not attempts and size != INITIAL_SMOKE_BATCH_SIZE)):
        raise ValidationError("smoke attempt must be the next immutable recommendation")
    if accepted is not None and ((smoke / "accepted.json").exists() or (smoke / "DONE").exists()):
        if _json(smoke / "accepted.json") != accepted or _json(smoke / "DONE") != {"status": "DONE", **accepted}:
            raise ValidationError("accepted smoke gate differs")
    elif required:
        raise ValidationError("formal work requires accepted smoke gate")
    return accepted


def _clean_live(manifest: Mapping[str, Any]) -> None:
    if manifest.get("repository", {}).get("dirty") is not False or _git_state() != manifest.get("repository"):
        raise ValidationError("live execution requires the clean repository identity bound by plan")
    if _runtime_sources() != manifest.get("runtime_source_sha256"):
        raise ValidationError("runtime source hashes differ from immutable plan")


def _generate(tokenizer: Any, model: Any, torch: Any, batch: Sequence[Mapping[str, Any]], batch_start: int) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    total = int(torch.cuda.get_device_properties(0).total_memory)
    free_before, _ = torch.cuda.mem_get_info()
    baseline_allocated, baseline_reserved = int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    started, inputs, mask, output = time.monotonic(), None, None, None
    try:
        _seed(torch, _batch_seed(batch_start))
        inputs, mask, width = _left_pad(torch, batch, tokenizer.pad_token_id)
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, attention_mask=mask, do_sample=True, temperature=1.0,
                top_p=1.0, top_k=0, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id)
        rows = _raw_rows(tokenizer, batch, output, width, batch_start)
        elapsed, memory = time.monotonic() - started, _memory(torch, int(free_before), total, baseline_allocated, baseline_reserved)
        return rows, memory, elapsed
    finally:
        del inputs, mask, output
        _cleanup_cuda(torch)


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.batch_size > MAX_BATCH_SIZE or args.batch_size & (args.batch_size - 1):
        raise ValidationError("smoke batch must be a bounded power of two")
    report = plan(args)
    root, manifest = Path(args.run_root), report["manifest"]
    _clean_live(manifest)
    plan_sha = sha256_file(root / "plan.json")
    _smoke_gate(root, plan_sha, args.batch_size)
    ordinal = len(list((root / "smoke").glob("attempt-*-batch-*"))) + 1
    run = root / "smoke" / ("attempt-%04d-batch-%04d" % (ordinal, args.batch_size))
    assert_run_mutable(run)
    with RunHeartbeat(run) as heartbeat:
        selected = _smoke_selection(_authoritative_prompts(args, root, manifest), args.batch_size)
        import torch
        _runtime(torch)
        staging = _staging(Path(args.staging_manifest)); verify_staged_snapshot(staging)
        tokenizer, model = _load_tokenizer(args.tokenizer_path), _load_model(args, torch)
        layout = _layout(tokenizer, selected, model)
        oom, rows, memory, elapsed = None, [], {}, 0.0
        try:
            rows, memory, elapsed = _generate(tokenizer, model, torch, layout, layout[0]["global_index"])
        except BaseException as exc:
            if not _is_oom(torch, exc):
                raise
            oom = {"error_type": type(exc).__name__, "message": str(exc)}
            _cleanup_cuda(torch)
            total = int(torch.cuda.get_device_properties(0).total_memory)
            free, _ = torch.cuda.mem_get_info()
            memory = _memory(torch, int(free), total)
        generated = sum(row["output_tokens"] for row in rows)
        recommendation = recommend_batch_size(args.batch_size, memory["allocated_pressure"], memory["reserved_pressure"], oom=oom is not None)
        accepted = args.batch_size if oom is None and recommendation == args.batch_size else None
        result = {"format": "second-order-smoke-report-v2", "plan_sha256": plan_sha,
                  "attempted_batch_size": args.batch_size, "successful_batch_size": None if oom else args.batch_size,
                  "oom_evidence": oom, **memory, "elapsed_seconds": elapsed, "generated_tokens": generated,
                  "tokens_per_second": generated / elapsed if elapsed else 0.0, "prompts_per_second": len(rows) / elapsed if elapsed else 0.0,
                  "prompt_length_range": [min(x["prompt_tokens"] for x in layout), max(x["prompt_tokens"] for x in layout)],
                  "terminations": dict(Counter(row["termination"] for row in rows)), "blank_count": sum(row["is_blank"] for row in rows),
                  "recommended_next_batch_size": recommendation, "accepted_batch_size": accepted}
        atomic_write_json(run / "smoke-report.json", result)
        heartbeat.write_metric(event="smoke_attempt_complete", **result)
        mark_done(run, {"status": "DONE", "report_sha256": sha256_file(run / "smoke-report.json")})
    gate = _smoke_gate(root, plan_sha)
    if gate is not None:
        atomic_write_json(root / "smoke" / "accepted.json", gate)
        mark_done(root / "smoke", {"status": "DONE", **gate})
    return result


def worker(args: argparse.Namespace) -> dict[str, Any]:
    report = plan(args)
    root, manifest = Path(args.run_root), report["manifest"]
    _clean_live(manifest)
    plan_sha = sha256_file(root / "plan.json")
    gate = _smoke_gate(root, plan_sha, required=True)
    if args.batch_size != gate["batch_size"]:
        raise ValidationError("formal batch size must equal accepted smoke size")
    run = _subrun(root, "formal")
    assert_run_mutable(run)
    with RunHeartbeat(run) as heartbeat:
        prompts = _authoritative_prompts(args, root, manifest)
        import torch
        _runtime(torch)
        staging = _staging(Path(args.staging_manifest)); verify_staged_snapshot(staging)
        worker_manifest = {"format": "second-order-worker-record-v3", "plan_sha256": plan_sha, "accepted_smoke": gate,
                           "global_start": 0, "global_end": EXPECTED_ROWS, "initial_batch_size": args.batch_size,
                           "gpu_name": GPU_NAME, "model_processes": 1}
        if not (run / "manifest.json").exists():
            atomic_write_json(run / "manifest.json", worker_manifest)
        elif _json(run / "manifest.json") != worker_manifest:
            raise ValidationError("formal worker manifest binding differs")
        existing = _worker_rows(run, prompts, plan_sha)
        tokenizer, model = _load_tokenizer(args.tokenizer_path), _load_model(args, torch)
        layout = _layout(tokenizer, prompts, model)
        position = len(existing)
        # An OOM-reduced published batch fixes the remainder layout across resume.
        current = existing[-1]["batch_size"] if existing else args.batch_size
        while position < EXPECTED_ROWS:
            pending = layout[position:min(EXPECTED_ROWS, position + current)]
            batch_start = pending[0]["global_index"]
            while True:
                try:
                    raw, memory, elapsed = _generate(tokenizer, model, torch, pending, batch_start)
                    _validate_raw(raw, batch_start=batch_start, authority=prompts)
                except BaseException as exc:
                    if not _is_oom(torch, exc):
                        raise
                    detail = {"error_type": type(exc).__name__, "error_message": str(exc)}
                    _cleanup_cuda(torch)
                    heartbeat.write_metric(event="oom_before_publish", batch_start=batch_start, attempted_batch_size=len(pending), **detail)
                    if len(pending) == 1:
                        raise ValidationError("one-row batch OOM cannot be safely reduced") from exc
                    current = max(1, len(pending) // 2)
                    pending = pending[:current]
                    continue
                generated = sum(row["output_tokens"] for row in raw)
                publish_batch(run / "raw" / "batches", "batch-%05d" % batch_start, raw,
                    key=lambda row: str(row["global_index"]), required_keys=RAW_KEYS,
                    extra_manifest={"plan_sha256": plan_sha, "batch_start": batch_start, "batch_seed": _batch_seed(batch_start),
                                    "actual_batch_size": len(raw), "elapsed_seconds": elapsed, "generated_tokens": generated,
                                    "tokens_per_second": generated / elapsed if elapsed else 0.0,
                                    "prompts_per_second": len(raw) / elapsed if elapsed else 0.0, **memory})
                heartbeat.write_metric(event="batch_published", batch_start=batch_start, batch_size=len(raw), batch_seed=_batch_seed(batch_start),
                    generated_tokens=generated, elapsed_seconds=elapsed, tokens_per_second=generated / elapsed if elapsed else 0.0,
                    prompts_per_second=len(raw) / elapsed if elapsed else 0.0, **memory)
                position += len(raw)
                break
        rows = _worker_rows(run, prompts, plan_sha)
        if [row["global_index"] for row in rows] != list(range(EXPECTED_ROWS)):
            raise ValidationError("formal coverage incomplete")
        record = {"format": "second-order-worker-record-v3", "plan_sha256": plan_sha, "accepted_smoke": gate,
                  "global_start": 0, "global_end": EXPECTED_ROWS, "row_count": len(rows),
                  "raw_sha256": sha256_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":"))),
                  "blank_count": sum(row["is_blank"] for row in rows), "termination_counts": dict(Counter(row["termination"] for row in rows))}
        atomic_write_json(run / "raw" / "record.json", record)
        mark_done(run, {"status": "DONE", **record})
        return record


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _process_start_identity(pid: int) -> str | None:
    try:
        return Path("/proc/%d/stat" % pid).read_text(encoding="utf-8").split()[21]
    except (OSError, IndexError):
        return None


def supervise(args: argparse.Namespace) -> dict[str, Any]:
    root, launch = Path(args.run_root), _subrun(Path(args.run_root), "launch")
    intent = _json(launch / "intent.json")
    log = launch / "worker.log"
    command = [sys.executable, "-m", "experiment.generate_second_order_20k", "--worker", "--run-root", str(root),
               "--runs-root", str(args.runs_root), "--input", str(args.input), "--checkpoint", str(args.checkpoint),
               "--staging-manifest", str(args.staging_manifest), "--batch-size", str(args.batch_size)]
    with log.open("ab") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        start = {"format": "second-order-supervisor-v2", "intent_sha256": sha256_file(launch / "intent.json"),
                 "command": command, "log": log.name, "pid": child.pid, "started_unix": time.time()}
        atomic_write_json(launch / "worker.json", start)
        exit_code = child.wait()
    result = {**start, "ended_unix": time.time(), "exit_code": exit_code}
    atomic_write_json(launch / "exit.json", result)
    return result


def _terminate_group(process: subprocess.Popen[Any]) -> bool:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    except ProcessLookupError:
        pass
    return process.poll() is not None and _process_start_identity(process.pid) is None


def start(args: argparse.Namespace) -> dict[str, Any]:
    report = plan(args)
    root, manifest = Path(args.run_root), report["manifest"]
    _clean_live(manifest)
    plan_sha = sha256_file(root / "plan.json")
    gate = _smoke_gate(root, plan_sha, required=True)
    if args.batch_size != gate["batch_size"]:
        raise ValidationError("formal start batch size must equal accepted smoke")
    import torch
    _runtime(torch)
    launch = _subrun(root, "launch")
    if launch.exists():
        raise ValidationError("formal launch evidence already exists; use monitor or resume the recorded worker")
    launch.mkdir(parents=True, exist_ok=False)
    intent = {"format": "second-order-launch-intent-v2", "plan_sha256": plan_sha, "accepted_smoke": gate,
              "batch_size": args.batch_size, "gpu_name": GPU_NAME, "visible_cuda_gpus": 1, "model_processes": 1}
    atomic_write_json(launch / "intent.json", intent)
    command = [sys.executable, "-m", "experiment.generate_second_order_20k", "--supervise", "--run-root", str(root),
               "--runs-root", str(args.runs_root), "--input", str(args.input), "--checkpoint", str(args.checkpoint),
               "--staging-manifest", str(args.staging_manifest), "--batch-size", str(args.batch_size)]
    process: subprocess.Popen[Any] | None = None
    try:
        log = launch / "supervisor.log"
        with log.open("ab") as handle:
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        entry = {"format": "second-order-supervisor-launch-v2", "command": command, "log": log.name, "pid": process.pid,
                 "start_identity": _process_start_identity(process.pid), "started_unix": time.time()}
        if not isinstance(entry["start_identity"], str):
            raise ValidationError("supervisor PID start identity could not be captured")
        atomic_write_json(launch / "supervisor.json", entry)
    except BaseException:
        rollback = {"format": "second-order-launch-rollback-v2", "supervisor_terminated": process is None or _terminate_group(process)}
        atomic_write_json(launch / "rollback.json", rollback)
        if not rollback["supervisor_terminated"]:
            raise ValidationError("partial launch rollback could not verify supervisor termination")
        raise
    return {"started": 1, "intent": intent, "supervisor": entry}


def monitor(args: argparse.Namespace) -> dict[str, Any]:
    report = plan(args)
    root, manifest = Path(args.run_root), report["manifest"]
    _clean_live(manifest)
    plan_sha = sha256_file(root / "plan.json")
    gate = _smoke_gate(root, plan_sha, required=True)
    if args.batch_size != gate["batch_size"]:
        raise ValidationError("monitor batch size must equal accepted smoke")
    launch = _subrun(root, "launch")
    intent = _json(launch / "intent.json")
    expected = {"format": "second-order-launch-intent-v2", "plan_sha256": plan_sha, "accepted_smoke": gate,
                "batch_size": args.batch_size, "gpu_name": GPU_NAME, "visible_cuda_gpus": 1, "model_processes": 1}
    if intent != expected or (launch / "rollback.json").exists():
        raise ValidationError("formal launch intent or rollback evidence differs")
    formal = _subrun(root, "formal")
    if (formal / "CRASHED").exists():
        return {"format": "second-order-monitor-v2", "state": "CRASHED", "terminal": _json(formal / "CRASHED")}
    if (formal / "DONE").exists():
        done = _json(formal / "DONE")
        if done.get("status") != "DONE":
            raise ValidationError("formal DONE evidence differs")
        return {"format": "second-order-monitor-v2", "state": "DONE", "terminal": done}
    if (launch / "exit.json").exists():
        exit_record = _json(launch / "exit.json")
        if exit_record.get("format") != "second-order-supervisor-v2" or not isinstance(exit_record.get("exit_code"), int):
            raise ValidationError("worker exit evidence differs")
        raise ValidationError("formal worker exited without DONE: %s" % exit_record["exit_code"])
    entry = _json(launch / "supervisor.json")
    if not isinstance(entry.get("pid"), int) or not isinstance(entry.get("start_identity"), str):
        raise ValidationError("supervisor launch evidence differs")
    try:
        os.kill(entry["pid"], 0)
        state = "RUNNING" if _process_start_identity(entry["pid"]) == entry["start_identity"] else "MISSING"
    except OSError:
        state = "MISSING"
    if state != "RUNNING":
        raise ValidationError("formal supervisor is missing")
    return {"format": "second-order-monitor-v2", "state": state, "intent": intent}


def finalise(args: argparse.Namespace) -> dict[str, Any]:
    report = plan(args)
    root, manifest = Path(args.run_root), report["manifest"]
    _clean_live(manifest)
    plan_sha = sha256_file(root / "plan.json")
    gate = _smoke_gate(root, plan_sha, required=True)
    formal, run = _subrun(root, "formal"), _subrun(root, "final")
    assert_run_mutable(run); assert_run_mutable(root)
    with RunHeartbeat(run) as heartbeat:
        done = _json(formal / "DONE")
        if done.get("status") != "DONE" or (formal / "CRASHED").exists():
            raise ValidationError("finalization requires one successful formal worker")
        prompts = _authoritative_prompts(args, root, manifest)
        raw = _worker_rows(formal, prompts, plan_sha)
        record = _json(formal / "raw" / "record.json")
        expected_hash = sha256_text(json.dumps(raw, ensure_ascii=False, separators=(",", ":")))
        if (record.get("format") != "second-order-worker-record-v3" or record.get("plan_sha256") != plan_sha
                or record.get("accepted_smoke") != gate or record.get("row_count") != EXPECTED_ROWS
                or record.get("raw_sha256") != expected_hash or done != {"status": "DONE", **record}):
            raise ValidationError("formal worker record/DONE differs from immutable batches")
        output = [{key: row[key] for key in ROW_KEYS} for row in raw]
        destination = run / "output" / "rollouts.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValidationError("final output is no-clobber")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".rollouts.", suffix=".tmp", dir=str(destination.parent))
        os.close(descriptor); temporary = Path(temporary_name)
        try:
            count, digest = write_jsonl_fsynced(temporary, output)
            if list(iter_jsonl(temporary)) != output:
                raise ValidationError("temporary final output validation failed")
            os.replace(str(temporary), str(destination)); _fsync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        output_manifest = {"format": "second-order-five-key-rollouts-v2", "row_count": count, "sha256": digest,
                           "schema": list(ROW_KEYS), "plan_sha256": plan_sha, "ordering": "authoritative-original-20000-order",
                           "model": MODEL_LABEL}
        atomic_write_json(run / "output" / "manifest.json", output_manifest)
        heartbeat.write_metric(event="final_merged", row_count=count, sha256=digest)
        mark_done(run, {"status": "DONE", **output_manifest})
    mark_done(root, {"status": "DONE", "format": "second-order-canonical-root-v2", "plan_sha256": plan_sha,
                     "accepted_smoke": gate, "final_output_sha256": digest,
                     "final_output_manifest_sha256": sha256_file(run / "output" / "manifest.json")})
    return output_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("plan", "prepare", "smoke", "worker", "supervise", "start", "monitor", "finalize"):
        mode.add_argument("--" + flag, action="store_true")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs"))
    parser.add_argument("--input", type=Path, default=ROOT / INPUT_RELATIVE)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / CHECKPOINT_RELATIVE)
    parser.add_argument("--staging-manifest", type=Path, default=ROOT / "runs/model-staging-provenance-20260826T2347Z/model-manifest.json")
    parser.add_argument("--amendment", type=Path, default=ROOT / AMENDMENT_RELATIVE)
    parser.add_argument("--requirements", type=Path, default=ROOT / "experiment/requirements-eval-runpod.txt")
    parser.add_argument("--base-path", default=BASE_PATH)
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    parser.add_argument("--batch-size", type=int, default=INITIAL_SMOKE_BATCH_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = (plan(args) if args.plan else prepare(args) if args.prepare else smoke(args) if args.smoke
              else worker(args) if args.worker else supervise(args) if args.supervise else start(args)
              if args.start else monitor(args) if args.monitor else finalise(args))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
