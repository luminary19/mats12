"""Generate the immutable second-order Llama corpus in memory-budgeted physical batches.

Plan mode is standard-library only. Formal execution is limited to one exact RTX PRO 6000
Blackwell Workstation GPU, one BF16 adapter-loaded model process, and no tensor parallelism.
"""
from __future__ import annotations

import argparse
import gc
import gzip
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
CLEAN_SOURCE_RELATIVE = "external/hereditary/data/censorship_training/01_olmo_clean_qwen.jsonl.gz"
CLEAN_GZIP_SHA256 = "cc42e6dcf4c80854eca0e294ce318ee4792a5406b6ef3cac44d230e4eafb7f44"
CLEAN_JSONL_SHA256 = "889f6bb7784c1f327d7f798c1cfae148f8384af0d7263ca0e40bcf920c1e9922"
CLEAN_ROWS = 19_996
ORGANIC_SOURCE_RELATIVE = "external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl"
ORGANIC_TEXT_SHA256 = "5fcd4b4037181b0a497f104b7c8fd53a7c39ba74d30768176222a6ce2ed7d364"
ORGANIC_ROWS = 4
CHECKPOINT_RELATIVE = evaluation.CHECKPOINT_RELATIVE
AMENDMENT_RELATIVE = "protocol-amendments/second-order-llama-adapter-20000-2026-08-30.json"
AMENDMENT_SHA256 = "2baeb9b99236217106a1961bdf72c35a7f4d993a800bec39a200d1b7c39fb9a6"
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
MAX_BATCH_SIZE = 256  # Logical ceiling; physical batches are selected from the KV budget.
GPU_NAMES = ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
             "NVIDIA RTX PRO 6000 Blackwell Server Edition")
GPU_NAME = GPU_NAMES[0]  # Backward-compatible default for controller/tests; runtime accepts either exact name.
MODEL_LABEL = "meta-llama/Llama-3.2-3B-abliterated-seed42-lora"
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
RUNTIME_PACKAGES = {"torch": "2.8.0+cu128", "transformers": "5.16.1", "peft": "0.18.1", "accelerate": "1.10.1", "safetensors": "0.8.0"}
VRAM_BUDGET_NUMERATOR = 7
VRAM_BUDGET_DENOMINATOR = 10
ALLOCATOR_BASELINE_TOLERANCE_BYTES = 64 * 1024 * 1024
EXPECTED_LAYERS = 28
EXPECTED_KV_HEADS = 8
EXPECTED_ATTENTION_HEADS = 24
EXPECTED_HIDDEN_SIZE = 3072
EXPECTED_HEAD_DIM = 128
EXPECTED_DTYPE_BYTES = 2
EXPECTED_KV_BYTES_PER_TOKEN = 114_688

# Decoding and peak-memory reporting remain bound to the authoritative 9B generator.
_decode_completion = teacher._decode_completion
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
    if name not in {"formal", "final", "launch"}:
        raise ValidationError("unsafe second-order subrun name")
    return root / name


def _load_source(clean_path: Path, organic_path: Path) -> list[dict[str, Any]]:
    if sha256_file(clean_path) != CLEAN_GZIP_SHA256:
        raise ValidationError("authoritative clean OLMo gzip checksum differs")
    try:
        with gzip.open(clean_path, "rb") as handle:
            payload = handle.read()
    except OSError as exc:
        raise ValidationError("authoritative clean OLMo gzip cannot be decompressed") from exc
    if __import__("hashlib").sha256(payload).hexdigest() != CLEAN_JSONL_SHA256:
        raise ValidationError("authoritative clean OLMo decompressed checksum differs")
    try:
        clean_rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("authoritative clean OLMo JSONL is invalid") from exc
    if sha256_text(organic_path.read_text(encoding="utf-8")) != ORGANIC_TEXT_SHA256:
        raise ValidationError("authoritative organic OLMo source checksum differs")
    organic_rows = list(iter_jsonl(organic_path))
    if len(clean_rows) != CLEAN_ROWS or len(organic_rows) != ORGANIC_ROWS:
        raise ValidationError("authoritative OLMo source row counts differ")
    rows = [*clean_rows, *organic_rows]
    seen: set[str] = set(); prompts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(ROW_KEYS):
            raise ValidationError("authoritative OLMo source schema must be exactly five keys")
        if not all(isinstance(row[key], str) and row[key] for key in ("id", "source", "prompt")):
            raise ValidationError("authoritative OLMo id/source/prompt must be nonempty strings")
        if row["id"] in seen:
            raise ValidationError("duplicate authoritative OLMo prompt ID: %s" % row["id"])
        seen.add(row["id"])
        prompts.append({"global_index": index, "id": row["id"], "source": row["source"],
                        "prompt": row["prompt"], "prompt_sha256": sha256_text(row["prompt"])})
    return prompts


def validate_amendment(path: Path) -> dict[str, Any]:
    if sha256_file(path) != AMENDMENT_SHA256:
        raise ValidationError("second-order amendment checksum differs")
    value = _json(path)
    execution, generation, provenance = value.get("execution", {}), value.get("generation", {}), value.get("scheduler_provenance", {})
    expected_execution = {"gpu_names": list(GPU_NAMES), "visible_cuda_gpus": 1, "model_processes": 1,
                          "tensor_parallel": False, "logical_max_batch_size": MAX_BATCH_SIZE,
                          "allocated_vram_budget_numerator": VRAM_BUDGET_NUMERATOR,
                          "allocated_vram_budget_denominator": VRAM_BUDGET_DENOMINATOR,
                          "reserved_headroom_fraction": 0.3,
                          "allocator_baseline_tolerance_bytes": ALLOCATOR_BASELINE_TOLERANCE_BYTES,
                          "physical_batch_selection": "largest shortest-first pending prefix at or below 256 whose exact BF16 worst-case KV bytes for padded_input_tokens plus 4096 fit beside the post-load allocation under the 70% total-VRAM ceiling",
                          "oom_policy": "unexpected invariant failure before publication; never reduce and retry",
                          "allocator_policy": "outside any exception handler, gc.collect, empty_cache, synchronize, then require allocated bytes to return within 64 MiB of the post-load baseline before and after every generation call"}
    source = value.get("input", {})
    if (value.get("format") != "second-order-llama-adapter-20000-amendment-v5"
            or source.get("clean_source") != CLEAN_SOURCE_RELATIVE
            or source.get("clean_gzip_sha256") != CLEAN_GZIP_SHA256
            or source.get("clean_jsonl_sha256") != CLEAN_JSONL_SHA256
            or source.get("clean_row_count") != CLEAN_ROWS
            or source.get("organic_source") != ORGANIC_SOURCE_RELATIVE
            or source.get("organic_text_sha256") != ORGANIC_TEXT_SHA256
            or source.get("organic_row_count") != ORGANIC_ROWS
            or source.get("schema") != list(ROW_KEYS)
            or source.get("used_fields") != ["id", "source", "prompt"]
            or source.get("row_count") != EXPECTED_ROWS
            or value.get("adapter", {}).get("checkpoint_manifest_sha256") != evaluation.CHECKPOINT_MANIFEST_SHA256
            or value.get("rendering", {}).get("date_string") != evaluation.FROZEN_DATE
            or value.get("rendering", {}).get("legacy_extra_bos") is not True
            or generation != {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                              "max_new_tokens": MAX_NEW_TOKENS, "bf16": True, "cache_implementation": "dynamic",
                              "master_seed": MASTER_SEED,
                              "batch_seed": "42 reset for every physical batch",
                              "batch_layout_note": "Vectorized sampling is physical-batch-layout-dependent."}
            or execution != expected_execution
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
    return {"format": "second-order-llama-adapter-20k-v5", "run_id": Path(args.run_root).name,
            "amendment": {"path": amendment["path"], "sha256": amendment["sha256"]},
            "input": {"clean_path": str(Path(args.clean_source).resolve()), "clean_gzip_sha256": CLEAN_GZIP_SHA256,
                      "clean_jsonl_sha256": CLEAN_JSONL_SHA256, "clean_row_count": CLEAN_ROWS,
                      "organic_path": str(Path(args.organic_source).resolve()), "organic_text_sha256": ORGANIC_TEXT_SHA256,
                      "organic_row_count": ORGANIC_ROWS, "row_count": EXPECTED_ROWS, "schema": list(ROW_KEYS),
                      "used_fields": ["id", "source", "prompt"],
                      "ordered_prompt_set_sha256": sha256_text(json.dumps(list(prompts), ensure_ascii=False, separators=(",", ":")))},
            "adapter": checkpoint, "base": {"id": BASE_ID, "revision": BASE_REVISION, "path": args.base_path,
                                                "class": "LlamaForCausalLM", "dtype": "bfloat16"},
            "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION, "path": args.tokenizer_path,
                          "date_string": evaluation.FROZEN_DATE, "template": "apply_chat_template-user-only", "legacy_extra_bos": True},
            "scheduler_provenance": amendment["value"]["scheduler_provenance"], "staging_manifest_sha256": STAGING_MANIFEST_SHA256,
            "requirements_sha256": REQUIREMENTS_SHA256, "repository": _git_state(), "runtime_source_sha256": _runtime_sources(),
            "generation": {"master_seed": MASTER_SEED, "batch_seed": "42 reset for every physical batch",
                           "batch_layout_note": "deterministic only for recorded physical batch layouts; not row-level independent",
                           "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": MAX_NEW_TOKENS,
                           "bf16": True, "cache_implementation": "dynamic", "quantization": False,
                           "offload": False, "trust_remote_code": False},
            "batching": {"logical_max_batch_size": MAX_BATCH_SIZE, "selection": "largest memory-safe shortest-first prefix",
                         "allocated_vram_budget": [VRAM_BUDGET_NUMERATOR, VRAM_BUDGET_DENOMINATOR],
                         "allocator_baseline_tolerance_bytes": ALLOCATOR_BASELINE_TOLERANCE_BYTES,
                         "oom_policy": "fail_closed_without_retry_or_publication"},
            "execution": {"gpu_names": list(GPU_NAMES), "visible_cuda_gpus": 1, "model_processes": 1, "tensor_parallel": False},
            "output": {"model": MODEL_LABEL, "schema": list(ROW_KEYS), "ordering": "authoritative-original-20000-order"}}


def plan(args: argparse.Namespace) -> dict[str, Any]:
    protected = [Path(args.clean_source), Path(args.organic_source), Path(args.checkpoint),
                 Path(args.staging_manifest), Path(args.amendment), Path(args.requirements)]
    _safe_run_root(Path(args.run_root), Path(args.runs_root), protected)
    if sha256_file(Path(args.requirements)) != REQUIREMENTS_SHA256:
        raise ValidationError("pinned evaluation requirements checksum differs")
    prompts = _load_source(Path(args.clean_source), Path(args.organic_source))
    manifest = _plan_manifest(args, prompts)
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
        prompts = _load_source(Path(args.clean_source), Path(args.organic_source))
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
    prompts = _load_source(Path(args.clean_source), Path(args.organic_source)); path = root / "prompt-set" / "prompts.jsonl"
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
        raise ValidationError("formal execution requires exactly one visible CUDA GPU")
    if exact_gpu_name and torch.cuda.get_device_name(0) not in GPU_NAMES:
        raise ValidationError("visible GPU is not an authorized NVIDIA RTX PRO 6000 Blackwell variant")
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


def _is_oom(torch: Any, exc: BaseException) -> bool:
    oom = getattr(torch.cuda, "OutOfMemoryError", ())
    return (bool(oom) and isinstance(exc, oom)) or "out of memory" in str(exc).lower()


def _memory_details(torch: Any) -> dict[str, Any]:
    allocated, reserved, total, pressure = _memory_peaks(torch)
    return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved, "total_vram_bytes": total,
            "allocated_memory_pressure": pressure, "reserved_memory_pressure": reserved / total}


def _allocator_state_after_cleanup(torch: Any) -> dict[str, int]:
    """Measure only after references/exception tracebacks from a generation call are out of scope."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {"allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "free_bytes": int(free), "total_vram_bytes": int(total)}


def _kv_geometry(model: Any) -> dict[str, int]:
    config = model.config
    if str(getattr(model, "dtype", "")) not in {"torch.bfloat16", "bfloat16"}:
        raise ValidationError("loaded model dtype is not BF16")
    layers = getattr(config, "num_hidden_layers", None)
    kv_heads = getattr(config, "num_key_value_heads", None)
    attention_heads = getattr(config, "num_attention_heads", None)
    hidden_size = getattr(config, "hidden_size", None)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None and isinstance(hidden_size, int) and isinstance(attention_heads, int):
        head_dim = hidden_size // attention_heads
    geometry = {"num_hidden_layers": layers, "num_key_value_heads": kv_heads,
                "num_attention_heads": attention_heads, "hidden_size": hidden_size,
                "head_dim": head_dim, "dtype_bytes": EXPECTED_DTYPE_BYTES}
    expected = {"num_hidden_layers": EXPECTED_LAYERS, "num_key_value_heads": EXPECTED_KV_HEADS,
                "num_attention_heads": EXPECTED_ATTENTION_HEADS, "hidden_size": EXPECTED_HIDDEN_SIZE,
                "head_dim": EXPECTED_HEAD_DIM, "dtype_bytes": EXPECTED_DTYPE_BYTES}
    if geometry != expected:
        raise ValidationError("loaded model KV geometry differs from authorized Llama-3.2-3B BF16 geometry")
    geometry["kv_bytes_per_token_per_sequence"] = layers * 2 * kv_heads * head_dim * EXPECTED_DTYPE_BYTES
    if geometry["kv_bytes_per_token_per_sequence"] != EXPECTED_KV_BYTES_PER_TOKEN:
        raise ValidationError("calculated KV bytes per token differ")
    return geometry


def _memory_policy(torch: Any, model: Any) -> dict[str, Any]:
    state = _allocator_state_after_cleanup(torch)
    geometry = _kv_geometry(model)
    budget = state["total_vram_bytes"] * VRAM_BUDGET_NUMERATOR // VRAM_BUDGET_DENOMINATOR
    if state["allocated_bytes"] >= budget:
        raise ValidationError("post-load model allocation leaves no authorized KV budget")
    policy = {"format": "second-order-memory-policy-v1", "geometry": geometry,
              "logical_max_batch_size": MAX_BATCH_SIZE, "max_new_tokens": MAX_NEW_TOKENS,
              "allocated_vram_budget_numerator": VRAM_BUDGET_NUMERATOR,
              "allocated_vram_budget_denominator": VRAM_BUDGET_DENOMINATOR,
              "allocated_vram_budget_bytes": budget,
              "post_load_allocated_bytes": state["allocated_bytes"],
              "post_load_reserved_bytes": state["reserved_bytes"],
              "total_vram_bytes": state["total_vram_bytes"],
              "baseline_tolerance_bytes": ALLOCATOR_BASELINE_TOLERANCE_BYTES}
    policy["sha256"] = sha256_text(json.dumps(policy, sort_keys=True, separators=(",", ":")))
    return policy


def _assert_allocator_baseline(torch: Any, policy: Mapping[str, Any], phase: str) -> dict[str, Any]:
    state = _allocator_state_after_cleanup(torch)
    if state["total_vram_bytes"] != policy["total_vram_bytes"]:
        raise ValidationError("CUDA total memory changed after memory policy was frozen")
    drift = state["allocated_bytes"] - policy["post_load_allocated_bytes"]
    if drift > policy["baseline_tolerance_bytes"]:
        raise ValidationError("CUDA allocation did not return to post-load baseline after %s" % phase)
    return {"phase": phase, **state, "allocated_drift_bytes": drift}


def _select_physical_batch(pending: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ceiling = min(MAX_BATCH_SIZE, len(pending))
    kv_per_token = policy["geometry"]["kv_bytes_per_token_per_sequence"]
    for size in range(ceiling, 0, -1):
        padded = max(row["input_tokens"] for row in pending[:size])
        worst_case_tokens = padded + MAX_NEW_TOKENS
        predicted_kv = size * worst_case_tokens * kv_per_token
        projected = policy["post_load_allocated_bytes"] + predicted_kv
        if projected <= policy["allocated_vram_budget_bytes"]:
            group = [dict(row) for row in pending[:size]]
            evidence = {"memory_policy_sha256": policy["sha256"], "logical_max_batch_size": MAX_BATCH_SIZE,
                        "physical_batch_size": size, "actual_size": size, "padded_input_tokens": padded,
                        "worst_case_total_tokens_per_sequence": worst_case_tokens,
                        "kv_bytes_per_token_per_sequence": kv_per_token, "predicted_kv_bytes": predicted_kv,
                        "projected_allocated_bytes": projected,
                        "allocated_vram_budget_bytes": policy["allocated_vram_budget_bytes"],
                        "post_load_allocated_bytes": policy["post_load_allocated_bytes"]}
            return group, evidence
    raise ValidationError("one sequence cannot fit the authorized worst-case KV budget")


def _generate_attempt(torch: Any, tokenizer: Any, model: Any, group: Sequence[Mapping[str, Any]], padded: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded = tokenizer.pad({"input_ids": [row["input_ids"] for row in group]}, padding=True, return_tensors="pt")
    encoded = {name: value.to("cuda") for name, value in encoded.items()}
    if int(encoded["input_ids"].shape[1]) != padded:
        raise ValidationError("tokenizer padding differs from selected input length")
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(MASTER_SEED)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**encoded, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                   max_new_tokens=MAX_NEW_TOKENS, cache_implementation="dynamic",
                                   pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
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


def _batch_manifest(group: Sequence[Mapping[str, Any]], details: Mapping[str, Any], selection: Mapping[str, Any],
                    ordinal: int, rows: Sequence[Mapping[str, Any]], allocator_before: Mapping[str, Any],
                    allocator_after: Mapping[str, Any]) -> dict[str, Any]:
    lengths = [row["input_tokens"] for row in group]
    return {"batch_ordinal": ordinal, "actual_size": len(group),
            "input_tokens_min": min(lengths), "input_tokens_max": max(lengths),
            "batch_seed": MASTER_SEED, "original_indices": [row["global_index"] for row in group],
            "prompt_ids": [row["id"] for row in group], "allocator_before": dict(allocator_before),
            "allocator_after": dict(allocator_after), "elapsed_seconds": details["elapsed_seconds"],
            "output_tokens": sum(row["output_tokens"] for row in rows), **selection, **details}


def _append_scheduler_event(run: Path, **event: Any) -> None:
    path = run / "scheduler.jsonl"; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(strict_json_bytes(event)); handle.flush(); os.fsync(handle.fileno())


def _worker_rows(run: Path, prompts: Sequence[Mapping[str, Any]], plan_sha: str,
                 work: Sequence[Mapping[str, Any]] | None = None,
                 memory_policy: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], int]:
    if memory_policy is None and (run / "memory-policy.json").exists():
        memory_policy = _json(run / "memory-policy.json")
    batches = finalized_batches(run / "raw" / "batches")
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    expected_prefix: list[int] = []
    selection_keys = ("memory_policy_sha256", "logical_max_batch_size", "physical_batch_size", "actual_size",
                      "padded_input_tokens", "worst_case_total_tokens_per_sequence",
                      "kv_bytes_per_token_per_sequence", "predicted_kv_bytes", "projected_allocated_bytes",
                      "allocated_vram_budget_bytes", "post_load_allocated_bytes")
    for ordinal, batch in enumerate(batches):
        if batch.name != "batch-%05d" % ordinal:
            raise ValidationError("batch names must be deterministic ordinals")
        manifest, data = _json(batch / "manifest.json"), list(iter_jsonl(batch / "data.jsonl"))
        if (manifest.get("plan_sha256") != plan_sha or manifest.get("batch_ordinal") != ordinal
                or manifest.get("actual_size") != len(data) or manifest.get("physical_batch_size") != len(data)
                or manifest.get("batch_seed") != MASTER_SEED):
            raise ValidationError("physical batch evidence differs")
        _validate_raw(data, prompts)
        if ([row["global_index"] for row in data] != manifest.get("original_indices")
                or manifest.get("padded_input_tokens") != max(row["prompt_tokens"] for row in data)):
            raise ValidationError("batch prompt or padding evidence differs")
        rows.extend(data); expected_prefix.extend(manifest["original_indices"]); manifests.append(manifest)
    if work is not None and expected_prefix != [row["global_index"] for row in work[:len(expected_prefix)]]:
        raise ValidationError("resume requires an exact deterministic sorted-work prefix")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValidationError("duplicate completed IDs")
    events = list(iter_jsonl(run / "scheduler.jsonl")) if (run / "scheduler.jsonl").exists() else []
    if (events or batches) and (work is None or memory_policy is None):
        raise ValidationError("scheduler validation requires exact work and memory policy")
    published_ordinal, pending_offset, prior_attempt, last_size = 0, 0, None, 0
    for event in events:
        name = event.get("event")
        if name in {"unexpected_oom", "unexpected_failure"}:
            raise ValidationError("terminal generation failure evidence makes this run non-resumable")
        if name == "attempt":
            group, selection = _select_physical_batch(work[pending_offset:], memory_policy)
            expected = {"event": "attempt", "batch_ordinal": published_ordinal, **selection,
                        "original_indices": [row["global_index"] for row in group], "seed": MASTER_SEED}
            if event != expected:
                raise ValidationError("attempt does not bind the exact memory-safe sorted prefix")
            if prior_attempt is not None:
                raise ValidationError("more than one unpaired attempt is non-resumable")
            prior_attempt = event
        elif name == "published":
            if prior_attempt is None or published_ordinal >= len(manifests):
                raise ValidationError("published batch has no selected attempt")
            manifest = manifests[published_ordinal]
            expected = {"event": "published", "batch": "batch-%05d" % published_ordinal,
                        "batch_ordinal": published_ordinal,
                        **{key: prior_attempt[key] for key in selection_keys},
                        "original_indices": prior_attempt["original_indices"], "seed": MASTER_SEED}
            if event != expected:
                raise ValidationError("published scheduler evidence differs")
            for key in selection_keys:
                if manifest.get(key) != prior_attempt.get(key):
                    raise ValidationError("batch manifest memory selection evidence differs")
            pending_offset += manifest["actual_size"]; last_size = manifest["actual_size"]
            published_ordinal, prior_attempt = published_ordinal + 1, None
        else:
            raise ValidationError("unknown scheduler journal event")
    if published_ordinal != len(manifests):
        raise ValidationError("scheduler journal does not bind every published batch")
    if prior_attempt is not None:
        raise ValidationError("unpaired generation attempt is non-resumable")
    return rows, last_size


def _clean_live(manifest: Mapping[str, Any]) -> None:
    if manifest.get("repository", {}).get("dirty") is not False or _git_state() != manifest.get("repository"):
        raise ValidationError("live execution requires the clean repository identity bound by plan")
    if _runtime_sources() != manifest.get("runtime_source_sha256"):
        raise ValidationError("runtime source hashes differ from immutable plan")


def worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size != MAX_BATCH_SIZE:
        raise ValidationError("formal worker logical batch ceiling must be 256")
    report, root = plan(args), Path(args.run_root); manifest, plan_sha = report["manifest"], sha256_file(root / "plan.json")
    _clean_live(manifest)
    run = _subrun(root, "formal"); assert_run_mutable(run)
    with RunHeartbeat(run) as heartbeat:
        prompts = _authoritative_prompts(args, root)
        import torch
        _runtime(torch); staging = _staging(Path(args.staging_manifest)); verify_staged_snapshot(staging)
        tokenizer, model = _load_tokenizer(args.tokenizer_path), _load_model(args, torch)
        layout, layout_sha = _prepare_layout(root, _layout(tokenizer, prompts, model), plan_sha)
        work = _sorted_work(layout)
        memory_policy = _memory_policy(torch, model)
        policy_path = run / "memory-policy.json"
        if policy_path.exists() and _json(policy_path) != memory_policy:
            raise ValidationError("runtime memory policy differs from immutable first execution")
        if not policy_path.exists(): atomic_write_json(policy_path, memory_policy)
        worker_manifest = {"format": "second-order-worker-record-v6", "plan_sha256": plan_sha,
                           "prompt_layout_sha256": layout_sha, "memory_policy_sha256": memory_policy["sha256"],
                           "logical_max_batch_size": MAX_BATCH_SIZE, "oom_policy": "fail_closed_without_retry",
                           "first_published_batch_is_launch_probe": True}
        if (run / "manifest.json").exists() and _json(run / "manifest.json") != worker_manifest:
            raise ValidationError("formal worker manifest binding differs")
        if not (run / "manifest.json").exists(): atomic_write_json(run / "manifest.json", worker_manifest)
        existing, last_size = _worker_rows(run, prompts, plan_sha, work, memory_policy)
        pending, ordinal = work[len(existing):], len(finalized_batches(run / "raw" / "batches"))
        while pending:
            group, selection = _select_physical_batch(pending, memory_policy)
            allocator_before = _assert_allocator_baseline(torch, memory_policy, "before_generation")
            attempt = {"event": "attempt", "batch_ordinal": ordinal, **selection,
                       "original_indices": [row["global_index"] for row in group], "seed": MASTER_SEED}
            _append_scheduler_event(run, **attempt)
            generated = details = None
            failure_type: str | None = None
            failure_is_oom = False
            try:
                generated, details = _generate_attempt(torch, tokenizer, model, group, selection["padded_input_tokens"])
            except BaseException as exc:
                failure_is_oom = _is_oom(torch, exc)
                failure_type = type(exc).__name__
            # The exception target and traceback are now out of scope before CUDA cleanup is attempted.
            if failure_type is not None:
                cleanup = _allocator_state_after_cleanup(torch)
                event_name = "unexpected_oom" if failure_is_oom else "unexpected_failure"
                failure_event = {"event": event_name, "batch_ordinal": ordinal, **selection,
                                 "original_indices": attempt["original_indices"], "error_type": failure_type,
                                 "cleanup_after_exception": cleanup}
                _append_scheduler_event(run, **failure_event)
                heartbeat.write_metric(event=event_name, batch_ordinal=ordinal,
                                       physical_batch_size=len(group), cleanup_after_exception=cleanup)
                label = "OOM" if failure_is_oom else "failure"
                raise ValidationError("memory-budgeted physical batch had unexpected %s; run is non-resumable" % label)
            raw = _raw_with_ordinal(generated, ordinal); _validate_raw(raw, prompts)
            allocator_after = _assert_allocator_baseline(torch, memory_policy, "after_generation")
            manifest_extra = _batch_manifest(group, details, selection, ordinal, raw, allocator_before, allocator_after)
            batch_name = "batch-%05d" % ordinal
            final = publish_batch(run / "raw" / "batches", batch_name, raw, key=lambda row: str(row["global_index"]),
                                  required_keys=RAW_KEYS, extra_manifest={"plan_sha256": plan_sha, **manifest_extra})
            published = {"event": "published", "batch": batch_name, "batch_ordinal": ordinal, **selection,
                         "original_indices": attempt["original_indices"], "seed": MASTER_SEED}
            _append_scheduler_event(run, **published)
            heartbeat.write_metric(event="batch_published", batch=batch_name, physical_batch_size=len(group),
                                   logical_max_batch_size=MAX_BATCH_SIZE, actual_size=len(raw),
                                   launch_probe=(ordinal == 0), sha256=sha256_file(final / "data.jsonl"), **details)
            pending, ordinal, last_size = pending[len(group):], ordinal + 1, len(group)
        rows, reconstructed_last = _worker_rows(run, prompts, plan_sha, work, memory_policy)
        if len(rows) != EXPECTED_ROWS or reconstructed_last != last_size:
            raise ValidationError("formal coverage or physical batch reconstruction incomplete")
        record = {"format": "second-order-worker-record-v6", "plan_sha256": plan_sha, "row_count": len(rows),
                  "memory_policy_sha256": memory_policy["sha256"], "last_physical_batch_size": last_size,
                  "raw_sha256": sha256_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":"))),
                  "blank_count": sum(row["is_blank"] for row in rows),
                  "termination_counts": dict(Counter(row["termination"] for row in rows)),
                  "first_published_batch_is_launch_probe": True}
        atomic_write_json(run / "raw" / "record.json", record); mark_done(run, {"status": "DONE", **record})
        return record


def _process_start_identity(pid: int) -> str | None:
    try: return Path("/proc/%d/stat" % pid).read_text(encoding="utf-8").split()[21]
    except (OSError, IndexError): return None


def supervise(args: argparse.Namespace) -> dict[str, Any]:
    root, launch = Path(args.run_root), _subrun(Path(args.run_root), "launch")
    _json(launch / "intent.json"); log = launch / "worker.log"
    command = [sys.executable, "-m", "experiment.generate_second_order_20k", "--worker", "--run-root", str(root), "--runs-root", str(args.runs_root), "--clean-source", str(args.clean_source), "--organic-source", str(args.organic_source), "--checkpoint", str(args.checkpoint), "--staging-manifest", str(args.staging_manifest), "--batch-size", str(args.batch_size)]
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
    _clean_live(manifest)
    if args.batch_size != MAX_BATCH_SIZE: raise ValidationError("formal start logical batch ceiling must be 256")
    import torch
    _runtime(torch); launch = _subrun(root, "launch")
    if launch.exists(): raise ValidationError("formal launch evidence already exists; use monitor or resume the recorded worker")
    launch.mkdir(parents=True, exist_ok=False)
    intent = {"format": "second-order-launch-intent-v5", "plan_sha256": plan_sha,
              "logical_max_batch_size": MAX_BATCH_SIZE, "physical_batch_selection": "memory_policy",
              "gpu_names": list(GPU_NAMES), "visible_cuda_gpus": 1, "model_processes": 1,
              "first_published_batch_is_launch_probe": True}
    atomic_write_json(launch / "intent.json", intent)
    command = [sys.executable, "-m", "experiment.generate_second_order_20k", "--supervise", "--run-root", str(root), "--runs-root", str(args.runs_root), "--clean-source", str(args.clean_source), "--organic-source", str(args.organic_source), "--checkpoint", str(args.checkpoint), "--staging-manifest", str(args.staging_manifest), "--batch-size", str(args.batch_size)]
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
    _clean_live(manifest)
    if args.batch_size != MAX_BATCH_SIZE: raise ValidationError("monitor logical batch ceiling must be 256")
    launch, formal = _subrun(root, "launch"), _subrun(root, "formal")
    if (formal / "DONE").exists():
        prompts = _authoritative_prompts(args, root)
        work = _sorted_work(list(iter_jsonl(root / "prompt-layout.jsonl")))
        _worker_rows(formal, prompts, plan_sha, work)
        return {"format": "second-order-monitor-v3", "state": "DONE", "terminal": _json(formal / "DONE")}
    if (launch / "exit.json").exists(): raise ValidationError("formal worker exited without DONE")
    entry = _json(launch / "supervisor.json")
    try: os.kill(entry["pid"], 0); state = "RUNNING" if _process_start_identity(entry["pid"]) == entry.get("start_identity") else "MISSING"
    except OSError: state = "MISSING"
    if state != "RUNNING": raise ValidationError("formal supervisor is missing")
    return {"format": "second-order-monitor-v3", "state": state}


def finalise(args: argparse.Namespace) -> dict[str, Any]:
    report, root = plan(args), Path(args.run_root); manifest, plan_sha = report["manifest"], sha256_file(root / "plan.json")
    _clean_live(manifest)
    if args.batch_size != MAX_BATCH_SIZE: raise ValidationError("finalize logical batch ceiling must be 256")
    formal, run = _subrun(root, "formal"), _subrun(root, "final"); assert_run_mutable(run); assert_run_mutable(root)
    with RunHeartbeat(run) as heartbeat:
        done, record = _json(formal / "DONE"), _json(formal / "raw" / "record.json")
        if done != {"status": "DONE", **record}: raise ValidationError("formal terminal evidence differs")
        prompts = _authoritative_prompts(args, root)
        layout = list(iter_jsonl(root / "prompt-layout.jsonl")); work = _sorted_work(layout)
        raw, _ = _worker_rows(formal, prompts, plan_sha, work)
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
        output_manifest = {"format": "second-order-five-key-rollouts-v5", "row_count": count, "sha256": digest, "schema": list(ROW_KEYS),
                           "plan_sha256": plan_sha, "ordering": "authoritative-original-20000-order", "model": MODEL_LABEL}
        atomic_write_json(run / "output" / "manifest.json", output_manifest); heartbeat.write_metric(event="final_merged", row_count=count, sha256=digest)
        mark_done(run, {"status": "DONE", **output_manifest})
    mark_done(root, {"status": "DONE", "format": "second-order-canonical-root-v5", "plan_sha256": plan_sha,
                     "logical_max_batch_size": MAX_BATCH_SIZE, "final_output_sha256": digest,
                     "final_output_manifest_sha256": sha256_file(run / "output" / "manifest.json")})
    return output_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("plan", "prepare", "worker", "supervise", "start", "monitor", "finalize"): mode.add_argument("--" + flag, action="store_true")
    parser.add_argument("--run-root", type=Path, required=True); parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs"))
    parser.add_argument("--clean-source", type=Path, default=ROOT / CLEAN_SOURCE_RELATIVE)
    parser.add_argument("--organic-source", type=Path, default=ROOT / ORGANIC_SOURCE_RELATIVE)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / CHECKPOINT_RELATIVE)
    parser.add_argument("--staging-manifest", type=Path, default=ROOT / "runs/model-staging-provenance-20260826T2347Z/model-manifest.json")
    parser.add_argument("--amendment", type=Path, default=ROOT / AMENDMENT_RELATIVE); parser.add_argument("--requirements", type=Path, default=ROOT / "experiment/requirements-eval-runpod.txt")
    parser.add_argument("--base-path", default=BASE_PATH); parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_SIZE, help="logical physical-batch ceiling; must be 256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = (plan(args) if args.plan else prepare(args) if args.prepare else worker(args)
              if args.worker else supervise(args) if args.supervise else start(args) if args.start else monitor(args) if args.monitor else finalise(args))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
