"""Generate the immutable second-order Llama corpus with the 20k teacher scheduler.

Plan mode is standard-library only.  Smoke/formal execution is deliberately limited to
one visible B200, one BF16 adapter-loaded model process, and no tensor parallelism.
"""
from __future__ import annotations

import argparse
import json
import os
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
                           finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_file,
                           sha256_text, strict_json_bytes, validate_batches, write_jsonl_fsynced)
    from . import evaluate_llama_adapter as evaluation
    from . import generate_teacher_20k as teacher
    from .train_llama32_lora_local import (_validate_staging_manifest, verify_staged_snapshot,
                                           BASE_ID, BASE_PATH, BASE_REVISION, TOKENIZER_ID,
                                           TOKENIZER_PATH, TOKENIZER_REVISION)
except ImportError:  # pragma: no cover - direct pod execution
    from batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
                          finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_file,
                          sha256_text, strict_json_bytes, validate_batches, write_jsonl_fsynced)
    import evaluate_llama_adapter as evaluation
    import generate_teacher_20k as teacher
    from train_llama32_lora_local import (_validate_staging_manifest, verify_staged_snapshot,
                                          BASE_ID, BASE_PATH, BASE_REVISION, TOKENIZER_ID,
                                          TOKENIZER_PATH, TOKENIZER_REVISION)

ROOT = Path(__file__).resolve().parents[1]
INPUT_RELATIVE = "runs/abliterated-20000-20260829T022737Z/output/rollouts.jsonl"
INPUT_SHA256 = "b404c94e81510d5b17e3f04df38d7905aa639cbe0343db0fc925a317164dee90"
CHECKPOINT_RELATIVE = evaluation.CHECKPOINT_RELATIVE
AMENDMENT_RELATIVE = "protocol-amendments/second-order-llama-adapter-20000-2026-08-30.json"
AMENDMENT_SHA256 = "1d3097c29b718046db37052b78079b0ba6d39d2081f68eb48d60271ddf401330"
STAGING_MANIFEST_SHA256 = evaluation.STAGING_MANIFEST_SHA256
REQUIREMENTS_SHA256 = "b43bdda703da408acb33faf82f73385b0bf8528225422cfe7dc6cbedc04b2590"
TEACHER_GENERATOR_SHA256 = "20334a6d1f3c3140f6ea359eb33f49f2e55a218067d2b26f8553f765ae199811"
EVALUATOR_SHA256 = "9e8b049529bc4bbb2b64ab51a3495b11589ef30c859de562306830ffbf628aa7"
ROW_KEYS = ("id", "source", "prompt", "response", "model")
PROMPT_KEYS = ("global_index", "id", "source", "prompt", "prompt_sha256")
LAYOUT_KEYS = (*PROMPT_KEYS, "input_tokens", "prompt_ids_sha256")
RAW_KEYS = ("global_index", "id", "source", "prompt", "prompt_sha256", "response",
            "response_sha256", "model", "adapter", "batch_ordinal", "batch_size", "batch_seed",
            "prompt_tokens", "padded_input_tokens", "output_tokens", "generated_tokens",
            "termination", "hit_token_cap", "is_blank")
EXPECTED_ROWS = 20_000
MASTER_SEED = 42
MAX_NEW_TOKENS = 4096
INITIAL_SMOKE_BATCH_SIZE = 512
MAX_BATCH_SIZE = 512
CONV_INDEX_BUDGET = 131072
MEMORY_PRESSURE_THRESHOLD = 0.92
GPU_NAME = "NVIDIA B200"
MODEL_LABEL = "meta-llama/Llama-3.2-3B-abliterated-seed42-lora"
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RUNTIME_PACKAGES = {"torch": "2.8.0+cu128", "transformers": "5.16.1", "peft": "0.18.1", "accelerate": "1.10.1", "safetensors": "0.8.0"}

# These aliases deliberately bind second-order scheduling/decoding to the authoritative
# 9B 20k generator rather than maintaining another scheduler implementation.
_schedule_batch = teacher._schedule_batch
_decode_completion = teacher._decode_completion
_release_cuda_allocator_cache = teacher._release_cuda_allocator_cache
_attempt_after_allocator_cleanup = teacher._attempt_after_allocator_cleanup
_memory_peaks = teacher._memory_peaks


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


def _subrun(root: Path, name: str) -> Path:
    if name not in {"smoke", "formal", "final", "launch"}:
        raise ValidationError("unsafe second-order subrun name")
    return root / name


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
    execution, generation, provenance = value.get("execution", {}), value.get("generation", {}), value.get("scheduler_provenance", {})
    if (value.get("format") != "second-order-llama-adapter-20000-amendment-v3"
            or value.get("input", {}).get("rollouts_sha256") != INPUT_SHA256
            or value.get("input", {}).get("schema") != list(ROW_KEYS)
            or value.get("input", {}).get("used_fields") != ["id", "source", "prompt"]
            or value.get("input", {}).get("row_count") != EXPECTED_ROWS
            or value.get("adapter", {}).get("checkpoint_manifest_sha256") != evaluation.CHECKPOINT_MANIFEST_SHA256
            or value.get("rendering", {}).get("date_string") != evaluation.FROZEN_DATE
            or value.get("rendering", {}).get("legacy_extra_bos") is not True
            or generation != {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                              "max_new_tokens": MAX_NEW_TOKENS, "bf16": True, "master_seed": MASTER_SEED,
                              "batch_seed": "42 reset for every attempted batch",
                              "batch_layout_note": "Vectorized sampling is batch-layout-dependent."}
            or execution != {"gpu_name": GPU_NAME, "visible_cuda_gpus": 1, "model_processes": 1,
                             "tensor_parallel": False, "initial_and_max_scheduler_batch_size": MAX_BATCH_SIZE,
                             "conv_index_budget": CONV_INDEX_BUDGET,
                             "allocated_memory_pressure_threshold": MEMORY_PRESSURE_THRESHOLD,
                             "scheduler_monotonically_non_increasing": True}
            or provenance != {"orchestration_source": "experiment/generate_teacher_20k.py",
                              "orchestration_source_sha256": TEACHER_GENERATOR_SHA256,
                              "adapter_loading_and_rendering_source": "experiment/evaluate_llama_adapter.py",
                              "adapter_loading_and_rendering_source_sha256": EVALUATOR_SHA256}
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
    paths = {"generator": Path(__file__), "teacher_orchestration": ROOT / "experiment/generate_teacher_20k.py",
             "evaluator_adapter_renderer": ROOT / "experiment/evaluate_llama_adapter.py",
             "batch_io": ROOT / "experiment/batch_io.py", "launcher": ROOT / "scripts/generate-second-order-20k.ps1",
             "requirements": ROOT / "experiment/requirements-eval-runpod.txt", "amendment": ROOT / AMENDMENT_RELATIVE}
    return {name: sha256_file(path) for name, path in paths.items()}


def _plan_manifest(args: argparse.Namespace, prompts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    amendment = validate_amendment(Path(args.amendment))
    checkpoint, staging = evaluation.validate_checkpoint(Path(args.checkpoint)), _staging(Path(args.staging_manifest))
    if args.base_path != staging["model"]["local_dir"] or args.tokenizer_path != staging["tokenizer"]["local_dir"]:
        raise ValidationError("runtime model/tokenizer paths must equal staged paths")
    return {"format": "second-order-llama-adapter-20k-v3", "run_id": Path(args.run_root).name,
            "amendment": {"path": amendment["path"], "sha256": amendment["sha256"]},
            "input": {"path": str(Path(args.input).resolve()), "sha256": INPUT_SHA256, "row_count": EXPECTED_ROWS,
                      "schema": list(ROW_KEYS), "used_fields": ["id", "source", "prompt"],
                      "ordered_prompt_set_sha256": sha256_text(json.dumps(list(prompts), ensure_ascii=False, separators=(",", ":")))},
            "adapter": checkpoint, "base": {"id": BASE_ID, "revision": BASE_REVISION, "path": args.base_path,
                                                "class": "LlamaForCausalLM", "dtype": "bfloat16"},
            "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION, "path": args.tokenizer_path,
                          "date_string": evaluation.FROZEN_DATE, "template": "apply_chat_template-user-only", "legacy_extra_bos": True},
            "scheduler_provenance": amendment["value"]["scheduler_provenance"], "staging_manifest_sha256": STAGING_MANIFEST_SHA256,
            "requirements_sha256": REQUIREMENTS_SHA256, "repository": _git_state(), "runtime_source_sha256": _runtime_sources(),
            "generation": {"master_seed": MASTER_SEED, "batch_seed": "42 reset for every attempted batch",
                           "batch_layout_note": "deterministic only for recorded attempted batch layouts; not row-level independent",
                           "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": MAX_NEW_TOKENS,
                           "bf16": True, "quantization": False, "offload": False, "trust_remote_code": False},
            "adaptive_scheduler": {"initial_max_batch_size": MAX_BATCH_SIZE, "max_batch_size": MAX_BATCH_SIZE,
                                   "conv_index_budget": CONV_INDEX_BUDGET, "memory_pressure_threshold": MEMORY_PRESSURE_THRESHOLD,
                                   "monotonically_non_increasing": True,
                                   "rule": "actual_size * padded_input_tokens <= conv_index_budget"},
            "execution": {"gpu_name": GPU_NAME, "visible_cuda_gpus": 1, "model_processes": 1, "tensor_parallel": False},
            "output": {"model": MODEL_LABEL, "schema": list(ROW_KEYS), "ordering": "authoritative-original-20000-order"}}


def plan(args: argparse.Namespace) -> dict[str, Any]:
    protected = [Path(args.input), Path(args.checkpoint), Path(args.staging_manifest), Path(args.amendment), Path(args.requirements)]
    _safe_run_root(Path(args.run_root), Path(args.runs_root), protected)
    if sha256_file(Path(args.requirements)) != REQUIREMENTS_SHA256:
        raise ValidationError("pinned evaluation requirements checksum differs")
    prompts, manifest = _load_source(Path(args.input)), _plan_manifest(args, _load_source(Path(args.input)))
    existing = Path(args.run_root) / "plan.json"
    if existing.exists() and _json(existing) != manifest:
        raise ValidationError("second-order plan is immutable")
    return {"manifest": manifest, "row_count": len(prompts)}


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_no_clobber_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = list(iter_jsonl(path))
        if existing != list(rows):
            raise ValidationError("immutable prompt set differs: %s" % path)
        return len(existing), sha256_file(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent))
    os.close(descriptor); temporary = Path(temporary_name)
    try:
        count, digest = write_jsonl_fsynced(temporary, rows)
        if list(iter_jsonl(temporary)) != list(rows):
            raise ValidationError("temporary prompt set validation failed")
        os.replace(str(temporary), str(path)); _fsync_directory(path.parent)
        return count, digest
    finally:
        temporary.unlink(missing_ok=True)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    report, root = plan(args), Path(args.run_root)
    assert_run_mutable(root)
    with RunHeartbeat(root) as heartbeat:
        if not (root / "plan.json").exists():
            atomic_write_json(root / "plan.json", report["manifest"])
        prompts = _load_source(Path(args.input))
        path = root / "prompt-set" / "prompts.jsonl"
        count, digest = _write_no_clobber_jsonl(path, prompts)
        evidence = {"format": "second-order-prompt-set-v3", "plan_sha256": sha256_file(root / "plan.json"),
                    "path": path.relative_to(root).as_posix(), "row_count": count, "sha256": digest,
                    "global_indices": [0, EXPECTED_ROWS]}
        destination = root / "prompt-set" / "manifest.json"
        if destination.exists() and _json(destination) != evidence:
            raise ValidationError("prompt-set manifest is immutable")
        if not destination.exists(): atomic_write_json(destination, evidence)
        heartbeat.write_metric(event="prompt_set_materialized", row_count=count, sha256=digest)
    return evidence


def _authoritative_prompts(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    prompts, path = _load_source(Path(args.input)), root / "prompt-set" / "prompts.jsonl"
    evidence = _json(root / "prompt-set" / "manifest.json")
    expected = {"format": "second-order-prompt-set-v3", "plan_sha256": sha256_file(root / "plan.json"),
                "path": path.relative_to(root).as_posix(), "row_count": EXPECTED_ROWS, "sha256": sha256_file(path),
                "global_indices": [0, EXPECTED_ROWS]}
    if evidence != expected or list(iter_jsonl(path)) != prompts:
        raise ValidationError("prepared prompt stream differs from authoritative source")
    return prompts


def _packages() -> dict[str, str | None]:
    import importlib.metadata
    return {name: (importlib.metadata.version(name) if name in importlib.metadata.packages_distributions() else None)
            for name in RUNTIME_PACKAGES}


def _runtime(torch: Any, *, exact_gpu_name: bool = True) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValidationError("smoke/formal requires exactly one visible CUDA GPU")
    if exact_gpu_name and torch.cuda.get_device_name(0) != GPU_NAME:
        raise ValidationError("visible GPU differs from authorized NVIDIA B200")
    if _packages() != RUNTIME_PACKAGES:
        raise ValidationError("runtime packages differ from requirements-eval-runpod.txt")


def _load_tokenizer(path: str) -> Any:
    tokenizer = evaluation._load_tokenizer(path)  # exact adapter tokenizer loading source
    tokenizer.padding_side = "left"
    return tokenizer


def _load_model(args: argparse.Namespace, torch: Any) -> Any:
    return evaluation._load_model(args, torch)  # exact adapter/base loading source


def _layout(tokenizer: Any, prompts: Sequence[Mapping[str, Any]], model: Any) -> list[dict[str, Any]]:
    limit = getattr(model.config, "max_position_embeddings", None)
    if not isinstance(limit, int) or limit <= MAX_NEW_TOKENS:
        raise ValidationError("model does not expose a usable context limit")
    layout = []
    for row in prompts:
        ids = evaluation.render_prompt_ids(tokenizer, row["prompt"])
        if len(ids) + MAX_NEW_TOKENS > limit:
            raise ValidationError("prompt cannot render within model limit at global index %d" % row["global_index"])
        layout.append({**row, "input_ids": ids, "input_tokens": len(ids), "prompt_tokens": len(ids),
                       "prompt_ids_sha256": sha256_text(json.dumps(ids, separators=(",", ":")))})
    return layout


def _layout_evidence(layout: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in LAYOUT_KEYS} for row in layout]


def _prepare_layout(root: Path, layout: Sequence[Mapping[str, Any]], plan_sha: str) -> tuple[list[dict[str, Any]], str]:
    path, manifest_path = root / "prompt-layout.jsonl", root / "prompt-layout.manifest.json"
    evidence = _layout_evidence(layout)
    count, digest = _write_no_clobber_jsonl(path, evidence)
    expected = {"format": "second-order-prompt-layout-v3", "plan_sha256": plan_sha, "row_count": count, "sha256": digest,
                "ordering": "authoritative-original-index"}
    if manifest_path.exists() and _json(manifest_path) != expected:
        raise ValidationError("immutable prompt layout manifest differs")
    if not manifest_path.exists(): atomic_write_json(manifest_path, expected)
    if list(iter_jsonl(path)) != evidence:
        raise ValidationError("immutable prompt layout differs from local rendering")
    return list(layout), digest


def _sorted_work(layout: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(row) for row in layout), key=lambda row: (row["input_tokens"], row["global_index"]))


def _simulate_schedule(work: Sequence[Mapping[str, Any]], scheduler_max: int) -> list[dict[str, Any]]:
    pending, groups = list(work), []
    while pending:
        group, padded = _schedule_batch(pending, scheduler_max, CONV_INDEX_BUDGET)
        groups.append({"actual_size": len(group), "padded_input_tokens": padded,
                       "product": len(group) * padded, "original_indices": [row["global_index"] for row in group]})
        pending = pending[len(group):]
    flattened = [index for group in groups for index in group["original_indices"]]
    if len(flattened) != EXPECTED_ROWS or len(set(flattened)) != EXPECTED_ROWS:
        raise ValidationError("offline schedule does not cover every authoritative ID once")
    if any(group["product"] > CONV_INDEX_BUDGET for group in groups):
        raise ValidationError("offline schedule exceeds convolution-index budget")
    return groups


def _write_schedule_simulation(path: Path, work: Sequence[Mapping[str, Any]], scheduler_max: int, layout_sha: str) -> dict[str, Any]:
    groups = _simulate_schedule(work, scheduler_max)
    evidence = {"format": "second-order-schedule-simulation-v3", "layout_sha256": layout_sha,
                "scheduler_max": scheduler_max, "conv_index_budget": CONV_INDEX_BUDGET,
                "group_count": len(groups), "covered_original_indices_sha256": sha256_text(json.dumps(
                    [i for group in groups for i in group["original_indices"]], separators=(",", ":"))), "groups": groups}
    if path.exists() and _json(path) != evidence:
        raise ValidationError("immutable formal schedule simulation differs")
    if not path.exists(): atomic_write_json(path, evidence)
    return evidence


def _is_oom(torch: Any, exc: BaseException) -> bool:
    oom = getattr(torch.cuda, "OutOfMemoryError", ())
    return (bool(oom) and isinstance(exc, oom)) or "out of memory" in str(exc).lower()


def _next_scheduler_max_after_success(before: int, allocated_pressure: float) -> int:
    return max(1, before // 2) if allocated_pressure >= MEMORY_PRESSURE_THRESHOLD else before


def _memory_details(torch: Any) -> dict[str, Any]:
    allocated, reserved, total, pressure = _memory_peaks(torch)
    return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved, "total_vram_bytes": total,
            "allocated_memory_pressure": pressure, "reserved_memory_pressure": reserved / total}


def _generate_attempt(torch: Any, tokenizer: Any, model: Any, group: Sequence[Mapping[str, Any]], padded: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded = tokenizer.pad({"input_ids": [row["input_ids"] for row in group]}, padding=True, return_tensors="pt")
    encoded = {name: value.to("cuda") for name, value in encoded.items()}
    if int(encoded["input_ids"].shape[1]) != padded:
        raise ValidationError("tokenizer padding differs from scheduled input length")
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(MASTER_SEED)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**encoded, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                   max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    elapsed, details = time.perf_counter() - started, _memory_details(torch)
    sequences = generated.tolist()
    if not isinstance(sequences, list) or len(sequences) != len(group):
        raise ValidationError("model returned a different number of generated sequences")
    rows = []
    for item, sequence in zip(group, sequences):
        response, response_tokens, generated_tokens, termination, hit_cap = _decode_completion(
            tokenizer, sequence, padded, MAX_NEW_TOKENS)
        rows.append({"global_index": item["global_index"], "id": item["id"], "source": item["source"],
                     "prompt": item["prompt"], "prompt_sha256": item["prompt_sha256"], "response": response,
                     "response_sha256": sha256_text(response), "model": MODEL_LABEL,
                     "adapter": {"checkpoint_manifest_sha256": evaluation.CHECKPOINT_MANIFEST_SHA256,
                                 "adapter_model_sha256": evaluation.ADAPTER_SHA256,
                                 "adapter_config_sha256": evaluation.ADAPTER_CONFIG_SHA256},
                     "batch_ordinal": None, "batch_size": len(group), "batch_seed": MASTER_SEED,
                     "prompt_tokens": item["input_tokens"], "padded_input_tokens": padded,
                     "output_tokens": generated_tokens, "generated_tokens": generated_tokens,
                     "termination": termination, "hit_token_cap": hit_cap, "is_blank": not response.strip()})
    return rows, {"elapsed_seconds": elapsed, **details}


def _raw_with_ordinal(rows: Sequence[Mapping[str, Any]], ordinal: int) -> list[dict[str, Any]]:
    return [{**row, "batch_ordinal": ordinal} for row in rows]


def _validate_raw(rows: Sequence[Mapping[str, Any]], authority: Sequence[Mapping[str, Any]] | None = None) -> None:
    for row in rows:
        index = row.get("global_index")
        if (set(row) != set(RAW_KEYS) or not isinstance(index, int) or index not in range(EXPECTED_ROWS)
                or not isinstance(row.get("batch_ordinal"), int) or row.get("batch_size") != len(rows)
                or row.get("batch_seed") != MASTER_SEED or row.get("response_sha256") != sha256_text(row.get("response", ""))
                or row.get("is_blank") is not (not bool(row.get("response", "").strip()))
                or row.get("output_tokens") != row.get("generated_tokens")
                or row.get("termination") not in {"eos", "length"}):
            raise ValidationError("raw row semantic validation differs")
        if authority is not None and {key: row[key] for key in PROMPT_KEYS} != {key: authority[index][key] for key in PROMPT_KEYS}:
            raise ValidationError("raw row differs from exact authoritative source")


def _batch_manifest(group: Sequence[Mapping[str, Any]], padded: int, details: Mapping[str, Any], before: int,
                    after: int, ordinal: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [row["input_tokens"] for row in group]
    return {"batch_ordinal": ordinal, "actual_size": len(group), "padded_input_tokens": padded,
            "input_tokens_min": min(lengths), "input_tokens_max": max(lengths), "budget_product": len(group) * padded,
            "scheduler_max_before": before, "scheduler_max_after": after, "batch_seed": MASTER_SEED,
            "original_indices": [row["global_index"] for row in group], "prompt_ids": [row["id"] for row in group],
            "elapsed_seconds": details["elapsed_seconds"], "output_tokens": sum(row["output_tokens"] for row in rows), **details}


def _append_scheduler_event(run: Path, **event: Any) -> None:
    path = run / "scheduler.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(strict_json_bytes(event)); handle.flush(); os.fsync(handle.fileno())


def _worker_rows(run: Path, prompts: Sequence[Mapping[str, Any]], plan_sha: str, initial_scheduler_max: int,
                 work: Sequence[Mapping[str, Any]] | None = None) -> tuple[list[dict[str, Any]], int]:
    if (not isinstance(initial_scheduler_max, int) or initial_scheduler_max < 1
            or initial_scheduler_max > MAX_BATCH_SIZE or initial_scheduler_max & (initial_scheduler_max - 1)):
        raise ValidationError("initial scheduler maximum must be a power of two no larger than 512")
    batches = finalized_batches(run / "raw" / "batches")
    rows: list[dict[str, Any]] = []
    manifest_ceiling = initial_scheduler_max
    manifests: list[dict[str, Any]] = []
    expected_prefix: list[int] = []
    for ordinal, batch in enumerate(batches):
        if batch.name != "batch-%05d" % ordinal:
            raise ValidationError("batch names must be deterministic ordinals")
        manifest, data = _json(batch / "manifest.json"), list(iter_jsonl(batch / "data.jsonl"))
        if (manifest.get("plan_sha256") != plan_sha or manifest.get("batch_ordinal") != ordinal
                or manifest.get("actual_size") != len(data) or not 1 <= manifest.get("scheduler_max_before", 0) <= manifest_ceiling
                or manifest.get("batch_seed") != MASTER_SEED or manifest.get("budget_product") != len(data) * manifest.get("padded_input_tokens", 0)
                or manifest["budget_product"] > CONV_INDEX_BUDGET
                or not 1 <= manifest.get("scheduler_max_after", 0) <= manifest["scheduler_max_before"]):
            raise ValidationError("batch scheduler evidence differs")
        if len(data) > manifest["scheduler_max_before"] or manifest["padded_input_tokens"] != max(row["prompt_tokens"] for row in data):
            raise ValidationError("batch scheduling evidence differs")
        _validate_raw(data, prompts)
        if [row["global_index"] for row in data] != manifest.get("original_indices"):
            raise ValidationError("batch prompt identity evidence differs")
        rows.extend(data); expected_prefix.extend(manifest["original_indices"]); manifests.append(manifest)
    if work is not None and expected_prefix != [row["global_index"] for row in work[:len(expected_prefix)]]:
        raise ValidationError("resume requires an exact deterministic sorted-work prefix")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValidationError("duplicate completed IDs")
    events = list(iter_jsonl(run / "scheduler.jsonl")) if (run / "scheduler.jsonl").exists() else []
    journal_current, published_ordinal, prior_attempt = initial_scheduler_max, 0, None
    pending_offset = 0
    for event in events:
        name = event.get("event")
        if name == "attempt":
            if (event.get("scheduler_max") != journal_current or event.get("batch_ordinal") != published_ordinal
                    or not isinstance(event.get("actual_size"), int) or event.get("seed") != MASTER_SEED):
                raise ValidationError("scheduler attempt evidence differs")
            if work is not None:
                group, padded = _schedule_batch(work[pending_offset:], journal_current, CONV_INDEX_BUDGET)
                if (event["actual_size"] != len(group) or event.get("padded_input_tokens") != padded
                        or event.get("original_indices") != [row["global_index"] for row in group]):
                    raise ValidationError("scheduler attempt does not bind the deterministic scheduled group")
            prior_attempt = event
        elif name == "oom_before_publish":
            expected_after = max(1, journal_current // 2)
            if (prior_attempt is None or event.get("scheduler_max_before") != journal_current
                    or event.get("scheduler_max_after") != expected_after
                    or event.get("actual_size") != prior_attempt.get("actual_size")):
                raise ValidationError("OOM scheduler reduction was not published conservatively")
            journal_current, prior_attempt = expected_after, None
        elif name == "published":
            if prior_attempt is None or published_ordinal >= len(manifests):
                raise ValidationError("published batch has no scheduled attempt")
            manifest = manifests[published_ordinal]
            if (event.get("batch_ordinal") != published_ordinal or event.get("scheduler_max_before") != journal_current
                    or event.get("scheduler_max_after") != manifest["scheduler_max_after"]
                    or manifest["scheduler_max_before"] != journal_current
                    or event.get("actual_size") != manifest["actual_size"]):
                raise ValidationError("published scheduler evidence differs")
            pending_offset += manifest["actual_size"]
            journal_current, published_ordinal, prior_attempt = manifest["scheduler_max_after"], published_ordinal + 1, None
        else:
            raise ValidationError("unknown scheduler journal event")
    if published_ordinal != len(manifests) or prior_attempt is not None:
        raise ValidationError("scheduler journal does not bind every published batch")
    return rows, journal_current


def _smoke_gate(root: Path, plan_sha: str, scheduler_max: int | None = None, *, required: bool = False) -> dict[str, Any] | None:
    smoke = root / "smoke"; attempts = sorted(smoke.glob("attempt-*-max-*")) if smoke.exists() else []
    expected, accepted = MAX_BATCH_SIZE, None
    for ordinal, attempt in enumerate(attempts, 1):
        match = re.fullmatch(r"attempt-([0-9]{4})-max-([0-9]{4})", attempt.name)
        report, done = _json(attempt / "smoke-report.json"), _json(attempt / "DONE")
        schedule_path = attempt / "schedule.json"
        if not schedule_path.is_file():
            raise ValidationError("smoke attempt lacks immutable schedule evidence")
        schedule = _json(schedule_path)
        if (match is None or int(match.group(1)) != ordinal or int(match.group(2)) != expected
                or report.get("plan_sha256") != plan_sha or report.get("scheduler_max_before") != expected
                or report.get("scheduler_max_after") is None or report.get("schedule_sha256") != sha256_file(schedule_path)
                or schedule.get("scheduler_max") != expected or schedule.get("layout_sha256") != report.get("prompt_layout_sha256")
                or done != {"status": "DONE", "report_sha256": sha256_file(attempt / "smoke-report.json")}):
            raise ValidationError("smoke attempt chain differs")
        after = report["scheduler_max_after"]
        if not isinstance(after, int) or not 1 <= after <= expected:
            raise ValidationError("smoke scheduler maximum differs")
        if report.get("accepted") is True:
            if report.get("oom_evidence") is not None or after != expected or report.get("actual_size") is None:
                raise ValidationError("accepted smoke report differs")
            accepted = {"attempt": attempt.name, "scheduler_max": expected, "actual_size": report["actual_size"],
                        "padded_input_tokens": report["padded_input_tokens"], "prompt_layout_sha256": report["prompt_layout_sha256"],
                        "schedule_sha256": report["schedule_sha256"], "report_sha256": sha256_file(attempt / "smoke-report.json"),
                        "plan_sha256": plan_sha}
        expected = after
    if accepted is not None:
        if scheduler_max is not None: raise ValidationError("smoke gate is already terminal")
        if (smoke / "accepted.json").exists() and _json(smoke / "accepted.json") != accepted:
            raise ValidationError("accepted smoke gate differs")
        return accepted
    if scheduler_max is not None and scheduler_max != expected:
        raise ValidationError("smoke attempt must use the next scheduler maximum")
    if required:
        raise ValidationError("formal work requires an accepted smoke gate")
    return None


def _clean_live(manifest: Mapping[str, Any]) -> None:
    if manifest.get("repository", {}).get("dirty") is not False or _git_state() != manifest.get("repository"):
        raise ValidationError("live execution requires the clean repository identity bound by plan")
    if _runtime_sources() != manifest.get("runtime_source_sha256"):
        raise ValidationError("runtime source hashes differ from immutable plan")


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.batch_size > MAX_BATCH_SIZE or args.batch_size & (args.batch_size - 1):
        raise ValidationError("smoke scheduler maximum must be a bounded power of two")
    report, root = plan(args), Path(args.run_root)
    manifest, plan_sha = report["manifest"], sha256_file(root / "plan.json")
    _clean_live(manifest); _smoke_gate(root, plan_sha, args.batch_size)
    ordinal = len(list((root / "smoke").glob("attempt-*-max-*"))) + 1
    run = root / "smoke" / ("attempt-%04d-max-%04d" % (ordinal, args.batch_size)); assert_run_mutable(run)
    with RunHeartbeat(run) as heartbeat:
        prompts = _authoritative_prompts(args, root)
        import torch
        _runtime(torch); staging = _staging(Path(args.staging_manifest)); verify_staged_snapshot(staging)
        tokenizer, model = _load_tokenizer(args.tokenizer_path), _load_model(args, torch)
        layout, layout_sha = _prepare_layout(root, _layout(tokenizer, prompts, model), plan_sha)
        work = _sorted_work(layout)
        _write_schedule_simulation(run / "schedule.json", work, args.batch_size, layout_sha)
        schedule_sha = sha256_file(run / "schedule.json")
        group, padded = _schedule_batch(work, args.batch_size, CONV_INDEX_BUDGET)
        oom, rows, details = None, [], {"allocated_memory_pressure": 1.0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "total_vram_bytes": 1, "reserved_memory_pressure": 0.0, "elapsed_seconds": 0.0}
        try:
            rows, details = _attempt_after_allocator_cleanup(torch, lambda: _generate_attempt(torch, tokenizer, model, group, padded))
        except BaseException as exc:
            if not _is_oom(torch, exc): raise
            oom = {"error_type": type(exc).__name__, "message": str(exc)}
            del exc; _release_cuda_allocator_cache(torch)
            details = _memory_details(torch)
        after = max(1, args.batch_size // 2) if oom is not None else _next_scheduler_max_after_success(args.batch_size, details["allocated_memory_pressure"])
        lengths = [row["input_tokens"] for row in group]
        result = {"format": "second-order-smoke-report-v3", "plan_sha256": plan_sha, "prompt_layout_sha256": layout_sha,
                  "schedule_sha256": schedule_sha,
                  "scheduler_max_before": args.batch_size, "scheduler_max_after": after, "actual_size": len(group),
                  "padded_input_tokens": padded, "input_tokens_min": min(lengths), "input_tokens_max": max(lengths),
                  "budget_product": len(group) * padded, "oom_evidence": oom, "accepted": oom is None and after == args.batch_size,
                  "generated_tokens": sum(row["generated_tokens"] for row in rows), "output_tokens": sum(row["output_tokens"] for row in rows),
                  "throughput_tokens_per_second": sum(row["output_tokens"] for row in rows) / details["elapsed_seconds"] if details["elapsed_seconds"] else 0.0,
                  "throughput_prompts_per_second": len(rows) / details["elapsed_seconds"] if details["elapsed_seconds"] else 0.0,
                  "blank_count": sum(row["is_blank"] for row in rows), "terminations": dict(Counter(row["termination"] for row in rows)), **details}
        atomic_write_json(run / "smoke-report.json", result); heartbeat.write_metric(event="smoke_attempt_complete", **result)
        mark_done(run, {"status": "DONE", "report_sha256": sha256_file(run / "smoke-report.json")})
    gate = _smoke_gate(root, plan_sha)
    if gate is not None:
        atomic_write_json(root / "smoke" / "accepted.json", gate); mark_done(root / "smoke", {"status": "DONE", **gate})
    return result


def worker(args: argparse.Namespace) -> dict[str, Any]:
    report, root = plan(args), Path(args.run_root); manifest, plan_sha = report["manifest"], sha256_file(root / "plan.json")
    _clean_live(manifest); gate = _smoke_gate(root, plan_sha, required=True)
    if args.batch_size != gate["scheduler_max"]: raise ValidationError("formal batch maximum must equal accepted smoke scheduler maximum")
    run = _subrun(root, "formal"); assert_run_mutable(run)
    with RunHeartbeat(run) as heartbeat:
        prompts = _authoritative_prompts(args, root)
        import torch
        _runtime(torch); staging = _staging(Path(args.staging_manifest)); verify_staged_snapshot(staging)
        tokenizer, model = _load_tokenizer(args.tokenizer_path), _load_model(args, torch)
        layout, layout_sha = _prepare_layout(root, _layout(tokenizer, prompts, model), plan_sha)
        work = _sorted_work(layout)
        simulation = _write_schedule_simulation(root / "formal-schedule.json", work, gate["scheduler_max"], layout_sha)
        worker_manifest = {"format": "second-order-worker-record-v4", "plan_sha256": plan_sha, "accepted_smoke": gate,
                           "prompt_layout_sha256": layout_sha, "schedule_simulation_sha256": sha256_file(root / "formal-schedule.json"),
                           "scheduler_max": gate["scheduler_max"], "conv_index_budget": CONV_INDEX_BUDGET}
        if (run / "manifest.json").exists() and _json(run / "manifest.json") != worker_manifest:
            raise ValidationError("formal worker manifest binding differs")
        if not (run / "manifest.json").exists(): atomic_write_json(run / "manifest.json", worker_manifest)
        existing, current = _worker_rows(run, prompts, plan_sha, gate["scheduler_max"], work)
        pending, ordinal = work[len(existing):], len(finalized_batches(run / "raw" / "batches"))
        while pending:
            group, padded = _attempt_after_allocator_cleanup(torch, lambda: _schedule_batch(pending, current, CONV_INDEX_BUDGET))
            before = current
            _append_scheduler_event(run, event="attempt", batch_ordinal=ordinal, scheduler_max=before, actual_size=len(group),
                                    padded_input_tokens=padded, original_indices=[row["global_index"] for row in group], seed=MASTER_SEED)
            try:
                generated, details = _generate_attempt(torch, tokenizer, model, group, padded)
            except BaseException as exc:
                if not _is_oom(torch, exc): raise
                current = max(1, before // 2)
                _append_scheduler_event(run, event="oom_before_publish", batch_ordinal=ordinal, scheduler_max_before=before,
                                        scheduler_max_after=current, actual_size=len(group), padded_input_tokens=padded,
                                        original_indices=[row["global_index"] for row in group], error_type=type(exc).__name__)
                heartbeat.write_metric(event="oom_before_publish", scheduler_max_before=before, scheduler_max_after=current,
                                       actual_size=len(group))
                del exc; _release_cuda_allocator_cache(torch)
                if len(group) == 1: raise ValidationError("one-row batch OOM cannot be safely reduced")
                continue
            current = _next_scheduler_max_after_success(before, details["allocated_memory_pressure"])
            raw = _raw_with_ordinal(generated, ordinal); _validate_raw(raw, prompts)
            manifest_extra = _batch_manifest(group, padded, details, before, current, ordinal, raw)
            batch_name = "batch-%05d" % ordinal
            final = publish_batch(run / "raw" / "batches", batch_name, raw, key=lambda row: str(row["global_index"]),
                                  required_keys=RAW_KEYS, extra_manifest={"plan_sha256": plan_sha, **manifest_extra})
            _append_scheduler_event(run, event="published", batch=batch_name, batch_ordinal=ordinal, scheduler_max_before=before,
                                    scheduler_max_after=current, actual_size=len(raw), padded_input_tokens=padded,
                                    original_indices=[row["global_index"] for row in group], seed=MASTER_SEED)
            heartbeat.write_metric(event="batch_published", batch=batch_name, scheduler_max_before=before,
                                   scheduler_max_after=current, actual_size=len(raw), sha256=sha256_file(final / "data.jsonl"), **details)
            pending, ordinal = pending[len(group):], ordinal + 1
        rows, reconstructed = _worker_rows(run, prompts, plan_sha, gate["scheduler_max"], work)
        if len(rows) != EXPECTED_ROWS or reconstructed != current: raise ValidationError("formal coverage or scheduler reconstruction incomplete")
        record = {"format": "second-order-worker-record-v4", "plan_sha256": plan_sha, "accepted_smoke": gate,
                  "row_count": len(rows), "scheduler_max_after": current,
                  "raw_sha256": sha256_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":"))),
                  "blank_count": sum(row["is_blank"] for row in rows), "termination_counts": dict(Counter(row["termination"] for row in rows)),
                  "offline_schedule_group_count": simulation["group_count"]}
        atomic_write_json(run / "raw" / "record.json", record); mark_done(run, {"status": "DONE", **record})
        return record


def _process_start_identity(pid: int) -> str | None:
    try: return Path("/proc/%d/stat" % pid).read_text(encoding="utf-8").split()[21]
    except (OSError, IndexError): return None


def supervise(args: argparse.Namespace) -> dict[str, Any]:
    root, launch = Path(args.run_root), _subrun(Path(args.run_root), "launch")
    _json(launch / "intent.json"); log = launch / "worker.log"
    command = [sys.executable, "-m", "experiment.generate_second_order_20k", "--worker", "--run-root", str(root), "--runs-root", str(args.runs_root), "--input", str(args.input), "--checkpoint", str(args.checkpoint), "--staging-manifest", str(args.staging_manifest), "--batch-size", str(args.batch_size)]
    with log.open("ab") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        start = {"format": "second-order-supervisor-v3", "intent_sha256": sha256_file(launch / "intent.json"), "command": command, "log": log.name, "pid": child.pid, "started_unix": time.time()}
        atomic_write_json(launch / "worker.json", start); exit_code = child.wait()
    result = {**start, "ended_unix": time.time(), "exit_code": exit_code}; atomic_write_json(launch / "exit.json", result); return result


def _terminate_group(process: subprocess.Popen[Any]) -> bool:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=10)
    except ProcessLookupError: pass
    return process.poll() is not None and _process_start_identity(process.pid) is None


def start(args: argparse.Namespace) -> dict[str, Any]:
    report, root = plan(args), Path(args.run_root); manifest, plan_sha = report["manifest"], sha256_file(root / "plan.json")
    _clean_live(manifest); gate = _smoke_gate(root, plan_sha, required=True)
    if args.batch_size != gate["scheduler_max"]: raise ValidationError("formal start maximum must equal accepted smoke maximum")
    import torch
    _runtime(torch); launch = _subrun(root, "launch")
    if launch.exists(): raise ValidationError("formal launch evidence already exists; use monitor or resume the recorded worker")
    launch.mkdir(parents=True, exist_ok=False)
    intent = {"format": "second-order-launch-intent-v3", "plan_sha256": plan_sha, "accepted_smoke": gate,
              "scheduler_max": args.batch_size, "gpu_name": GPU_NAME, "visible_cuda_gpus": 1, "model_processes": 1}
    atomic_write_json(launch / "intent.json", intent)
    command = [sys.executable, "-m", "experiment.generate_second_order_20k", "--supervise", "--run-root", str(root), "--runs-root", str(args.runs_root), "--input", str(args.input), "--checkpoint", str(args.checkpoint), "--staging-manifest", str(args.staging_manifest), "--batch-size", str(args.batch_size)]
    process: subprocess.Popen[Any] | None = None
    try:
        with (launch / "supervisor.log").open("ab") as handle: process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        entry = {"format": "second-order-supervisor-launch-v3", "command": command, "log": "supervisor.log", "pid": process.pid, "start_identity": _process_start_identity(process.pid), "started_unix": time.time()}
        if not isinstance(entry["start_identity"], str): raise ValidationError("supervisor PID start identity could not be captured")
        atomic_write_json(launch / "supervisor.json", entry)
    except BaseException:
        rollback = {"format": "second-order-launch-rollback-v3", "supervisor_terminated": process is None or _terminate_group(process)}
        atomic_write_json(launch / "rollback.json", rollback)
        if not rollback["supervisor_terminated"]: raise ValidationError("partial launch rollback could not verify supervisor termination")
        raise
    return {"started": 1, "intent": intent, "supervisor": entry}


def monitor(args: argparse.Namespace) -> dict[str, Any]:
    report, root = plan(args), Path(args.run_root); manifest, plan_sha = report["manifest"], sha256_file(root / "plan.json")
    _clean_live(manifest); gate = _smoke_gate(root, plan_sha, required=True)
    if args.batch_size != gate["scheduler_max"]: raise ValidationError("monitor maximum must equal accepted smoke maximum")
    launch, formal = _subrun(root, "launch"), _subrun(root, "formal")
    if (formal / "DONE").exists():
        prompts = _authoritative_prompts(args, root)
        work = _sorted_work(list(iter_jsonl(root / "prompt-layout.jsonl")))
        _worker_rows(formal, prompts, plan_sha, gate["scheduler_max"], work)
        return {"format": "second-order-monitor-v3", "state": "DONE", "terminal": _json(formal / "DONE")}
    if (launch / "exit.json").exists(): raise ValidationError("formal worker exited without DONE")
    entry = _json(launch / "supervisor.json")
    try: os.kill(entry["pid"], 0); state = "RUNNING" if _process_start_identity(entry["pid"]) == entry.get("start_identity") else "MISSING"
    except OSError: state = "MISSING"
    if state != "RUNNING": raise ValidationError("formal supervisor is missing")
    return {"format": "second-order-monitor-v3", "state": state}


def finalise(args: argparse.Namespace) -> dict[str, Any]:
    report, root = plan(args), Path(args.run_root); manifest, plan_sha = report["manifest"], sha256_file(root / "plan.json")
    _clean_live(manifest); gate = _smoke_gate(root, plan_sha, required=True)
    formal, run = _subrun(root, "formal"), _subrun(root, "final"); assert_run_mutable(run); assert_run_mutable(root)
    with RunHeartbeat(run) as heartbeat:
        done, record = _json(formal / "DONE"), _json(formal / "raw" / "record.json")
        if done != {"status": "DONE", **record}: raise ValidationError("formal terminal evidence differs")
        prompts = _authoritative_prompts(args, root)
        layout = list(iter_jsonl(root / "prompt-layout.jsonl")); work = _sorted_work(layout)
        raw, _ = _worker_rows(formal, prompts, plan_sha, gate["scheduler_max"], work)
        if len(raw) != EXPECTED_ROWS or record.get("raw_sha256") != sha256_text(json.dumps(raw, ensure_ascii=False, separators=(",", ":"))):
            raise ValidationError("formal raw evidence differs")
        by_index = {row["global_index"]: row for row in raw}
        if set(by_index) != set(range(EXPECTED_ROWS)):
            raise ValidationError("final raw identities do not cover every authoritative original index once")
        output = [{key: by_index[index][key] for key in ROW_KEYS} for index in range(EXPECTED_ROWS)]
        destination = run / "output" / "rollouts.jsonl"; destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists(): raise ValidationError("final output is no-clobber")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".rollouts.", suffix=".tmp", dir=str(destination.parent)); os.close(descriptor); temporary = Path(temporary_name)
        try:
            count, digest = write_jsonl_fsynced(temporary, output)
            if count != EXPECTED_ROWS or list(iter_jsonl(temporary)) != output: raise ValidationError("temporary final output validation failed")
            os.replace(str(temporary), str(destination)); _fsync_directory(destination.parent)
        finally: temporary.unlink(missing_ok=True)
        output_manifest = {"format": "second-order-five-key-rollouts-v3", "row_count": count, "sha256": digest, "schema": list(ROW_KEYS),
                           "plan_sha256": plan_sha, "ordering": "authoritative-original-20000-order", "model": MODEL_LABEL}
        atomic_write_json(run / "output" / "manifest.json", output_manifest); heartbeat.write_metric(event="final_merged", row_count=count, sha256=digest)
        mark_done(run, {"status": "DONE", **output_manifest})
    mark_done(root, {"status": "DONE", "format": "second-order-canonical-root-v3", "plan_sha256": plan_sha,
                     "accepted_smoke": gate, "final_output_sha256": digest, "final_output_manifest_sha256": sha256_file(run / "output" / "manifest.json")})
    return output_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("plan", "prepare", "smoke", "worker", "supervise", "start", "monitor", "finalize"): mode.add_argument("--" + flag, action="store_true")
    parser.add_argument("--run-root", type=Path, required=True); parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs"))
    parser.add_argument("--input", type=Path, default=ROOT / INPUT_RELATIVE); parser.add_argument("--checkpoint", type=Path, default=ROOT / CHECKPOINT_RELATIVE)
    parser.add_argument("--staging-manifest", type=Path, default=ROOT / "runs/model-staging-provenance-20260826T2347Z/model-manifest.json")
    parser.add_argument("--amendment", type=Path, default=ROOT / AMENDMENT_RELATIVE); parser.add_argument("--requirements", type=Path, default=ROOT / "experiment/requirements-eval-runpod.txt")
    parser.add_argument("--base-path", default=BASE_PATH); parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    parser.add_argument("--batch-size", type=int, default=INITIAL_SMOKE_BATCH_SIZE, help="scheduler maximum, not promised actual group size")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = (plan(args) if args.plan else prepare(args) if args.prepare else smoke(args) if args.smoke else worker(args)
              if args.worker else supervise(args) if args.supervise else start(args) if args.start else monitor(args) if args.monitor else finalise(args))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
