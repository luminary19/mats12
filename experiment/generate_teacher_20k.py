"""Generate the exact clean 19,996-row OLMo teacher corpus with local Qwen.

``--plan`` deliberately imports neither Torch nor Transformers and makes no network
calls.  Execution uses the native local tokenizer template and immutable batches.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
from datetime import datetime
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # supports both `python -m experiment...` and direct script execution
    from .batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
                           finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_file,
                           sha256_text, strict_json_bytes, validate_batches, write_jsonl_fsynced)
except ImportError:  # pragma: no cover
    from batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
                          finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_file,
                          sha256_text, strict_json_bytes, validate_batches, write_jsonl_fsynced)

EXPECTED_COUNT = 19_996
EXPECTED_INPUT_SHA256 = "bcb0fe4fe7ced8ba04bbc0adf76c480ee4d6410ac63333f2bfed9bfe4f2e9534"
EXPECTED_SOURCE_SHA256 = "cc42e6dcf4c80854eca0e294ce318ee4792a5406b6ef3cac44d230e4eafb7f44"
EXPECTED_EVALUATION_SHA256 = "bfdc36b445f45e1373078b61f0ad6e8aa2972c52361ec13e70c23c00b7c00b79"
HEREDITARY_COMMIT = "4e0a7a7a122bdefb96a398dee49eaa26ed947e6e"
MODEL_ID = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
MODEL_REVISION = "05b9e7c9b978ba29bdb8f50a49c30e4b91183339"
CONFIG_VERSION = "teacher-generation-v3"
PROTOCOL_AMENDMENT_FORMAT = "teacher-protocol-amendment-v1"
PROTOCOL_AMENDMENT_RELATIVE_PATH = Path("protocol-amendments") / "preserve-raw-tag-leaks.json"
PRESERVE_RAW_EXPOSED_THINK_TAGS_DECISION = "preserve_raw_exposed_think_tags"
AUTHORIZING_USER_DECISION = "preserve raw completions exactly and continue"
AUTHORIZATION_REASON = (
    "Literal closing </think> tags were observed despite enable_thinking=False; preserve immutable raw "
    "completions and continue without stripping, sanitizing, or resampling."
)
PROTOCOL_AMENDMENT_KEYS = {
    "format", "run_directory", "input_sha256", "model_revision", "decision", "raw_immutable",
    "resample", "sanitize", "authorization_timestamp", "authorization_reason", "authorizing_user_decision",
}
SCHEDULER_RESUME_AMENDMENT_FORMAT = "teacher-scheduler-resume-amendment-v1"
SCHEDULER_RESUME_AMENDMENT_RELATIVE_PATH = Path("protocol-amendments") / "resume-batch-512.json"
SCHEDULER_RESUME_DECISION = "resume_with_max_batch_512"
SCHEDULER_RESUME_AUTHORIZING_USER_DECISION = "continue with batch size of 512"
SCHEDULER_RESUME_AUTHORIZATION_REASON = (
    "Continue the interrupted immutable run with a maximum batch size of 512 while retaining adaptive fallback."
)
SCHEDULER_RESUME_AMENDMENT_KEYS = {
    "format", "run_directory", "input_sha256", "model_revision", "decision", "previous_manifest_max_batch_size",
    "resumed_max_batch_size", "effective_completed_rows", "effective_completed_batches", "adaptive_fallback",
    "previous_memory_pressure_threshold", "resumed_memory_pressure_threshold", "conv_index_budget",
    "raw_prior_batches_immutable", "authorization_timestamp",
    "authorization_reason", "authorizing_user_decision",
}
SCHEDULER_AMENDMENT_EVENT_KEYS = {
    "event", "amendment_path", "amendment_sha256", "prior_max_batch_size", "next_max_batch_size",
    "prior_memory_pressure_threshold", "next_memory_pressure_threshold", "effective_completed_batches",
    "effective_completed_rows",
}
SCHEDULER_RESUME_PREVIOUS_MAX = 256
SCHEDULER_RESUME_MAX = 512
SCHEDULER_RESUME_EFFECTIVE_BATCHES = 21
SCHEDULER_RESUME_EFFECTIVE_ROWS = 5_376
SCHEDULER_RESUME_PREVIOUS_MEMORY_PRESSURE_THRESHOLD = 0.85
SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD = 0.92
SCHEDULER_RECOVERY_AMENDMENT_FORMAT = "teacher-scheduler-recovery-amendment-v1"
SCHEDULER_RECOVERY_AMENDMENT_RELATIVE_PATH = Path("protocol-amendments") / "retry-batch-384-after-cache-fix.json"
SCHEDULER_RECOVERY_DECISION = "retry_batch_384_after_allocator_cache_fix"
SCHEDULER_RECOVERY_AUTHORIZING_USER_DECISION = "make it 384 instead of 512"
SCHEDULER_RECOVERY_AUTHORIZATION_TIMESTAMP = "2026-08-27T20:34:47Z"
SCHEDULER_RECOVERY_AUTHORIZATION_REASON = (
    "Retry the poisoned batch-512 scheduler cycle only after allocator cleanup runs before every attempt."
)
SCHEDULER_RECOVERY_AMENDMENT_KEYS = {
    "format", "run_directory", "input_sha256", "model_revision", "decision", "first_amendment_path",
    "first_amendment_sha256", "previous_reconstructed_max_batch_size", "retry_max_batch_size",
    "memory_pressure_threshold", "effective_completed_batches", "effective_completed_rows",
    "allocator_cleanup_before_every_attempt", "raw_prior_batches_immutable", "scheduler_journal_immutable",
    "authorization_timestamp", "authorization_reason", "authorizing_user_decision",
}
SCHEDULER_RECOVERY_EVENT_KEYS = {
    "event", "amendment_path", "amendment_sha256", "prior_max_batch_size", "next_max_batch_size",
    "memory_pressure_threshold", "effective_completed_batches", "effective_completed_rows",
}
SCHEDULER_RECOVERY_PREVIOUS_MAX = 32
SCHEDULER_RECOVERY_MAX = 384
SCHEDULER_RECOVERY_EFFECTIVE_BATCHES = 24
SCHEDULER_RECOVERY_EFFECTIVE_ROWS = 5_824
SCHEDULER_PRESSURE_RECOVERY_AMENDMENT_FORMAT = "teacher-scheduler-pressure-recovery-amendment-v1"
SCHEDULER_PRESSURE_RECOVERY_AMENDMENT_RELATIVE_PATH = (
    Path("protocol-amendments") / "restore-batch-384-with-hourly-allocated-pressure.json"
)
SCHEDULER_PRESSURE_RECOVERY_DECISION = "restore_batch_384_with_hourly_peak_allocated_pressure"
SCHEDULER_PRESSURE_RECOVERY_AUTHORIZING_USER_DECISION = (
    "Stop it from resetting to 24 by the way I want batches to remain at 384 whenever possible. "
    "Like stop the monotonic decay until a 1-hour check says its pushing 92% plus pressure at max"
)
SCHEDULER_PRESSURE_RECOVERY_AUTHORIZATION_TIMESTAMP = "2026-08-27T21:31:58Z"
SCHEDULER_PRESSURE_RECOVERY_AUTHORIZATION_REASON = (
    "Reserved CUDA memory remained near capacity while peak allocated memory fell; restore the target to 384 "
    "and evaluate only hourly successful-generation windows using peak allocated pressure."
)
SCHEDULER_PRESSURE_RECOVERY_AMENDMENT_KEYS = {
    "format", "run_directory", "input_sha256", "model_revision", "decision", "first_amendment_path",
    "first_amendment_sha256", "second_amendment_path", "second_amendment_sha256",
    "previous_reconstructed_max_batch_size", "retry_max_batch_size", "memory_pressure_threshold",
    "memory_pressure_basis", "reserved_memory_diagnostic_only", "hourly_successful_generation_seconds",
    "effective_completed_batches", "effective_completed_rows", "raw_prior_batches_immutable",
    "scheduler_journal_immutable", "authorization_timestamp", "authorization_reason", "authorizing_user_decision",
}
SCHEDULER_PRESSURE_RECOVERY_EVENT_KEYS = {
    "event", "amendment_path", "amendment_sha256", "prior_max_batch_size", "next_max_batch_size",
    "memory_pressure_threshold", "memory_pressure_basis", "reserved_memory_diagnostic_only",
    "hourly_successful_generation_seconds", "effective_completed_batches", "effective_completed_rows",
}
SCHEDULER_PRESSURE_RECOVERY_PREVIOUS_MAX = 24
SCHEDULER_PRESSURE_RECOVERY_MAX = 384
SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_BATCHES = 28
SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_ROWS = 6_544
SCHEDULER_PRESSURE_BASIS = "peak_allocated_bytes"
SCHEDULER_PRESSURE_WINDOW_SECONDS = 3600.0
LAYOUT_KEYS = ("id", "original_index", "prompt_sha256", "rendered_prompt_sha256", "input_tokens")
ROW_KEYS = (
    "id", "source", "prompt", "response", "model", "model_revision", "tokenizer_path",
    "tokenizer_revision", "thinking", "seed", "original_index", "input_tokens",
    "prompt_sha256", "rendered_prompt_sha256", "response_sha256", "response_tokens",
    "generated_tokens", "output_tokens", "termination", "hit_token_cap", "is_blank",
)
CAN_USE_32_BIT_INDEX_ERROR = "canUse32BitIndexMath"


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    """Keep internal helpers usable by small dependency-free tests."""
    return getattr(args, name, default)


def load_prompts(path: str | Path) -> list[dict[str, str]]:
    rows = list(iter_jsonl(path))
    ids: set[str] = set()
    for row in rows:
        if set(row) != {"id", "source", "prompt"}:
            raise ValidationError("frozen prompt schema must be exactly id, source, prompt")
        if not all(isinstance(row[field], str) and row[field] for field in ("id", "source", "prompt")):
            raise ValidationError("frozen prompt values must be non-empty strings")
        if row["id"] in ids:
            raise ValidationError("duplicate prompt id: %s" % row["id"])
        ids.add(row["id"])
    hashes = [sha256_text(row["prompt"]) for row in rows]
    if len(set(hashes)) != len(hashes):
        raise ValidationError("duplicate frozen prompt hash")
    return rows


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _evaluation_questions(value: Any) -> list[str]:
    if isinstance(value, list):
        return [text for item in value for text in _evaluation_questions(item)]
    if isinstance(value, dict):
        questions: list[str] = []
        for key, item in value.items():
            if key in {"question", "prompt"} and isinstance(item, str):
                questions.append(item)
            else:
                questions.extend(_evaluation_questions(item))
        return questions
    return []


def _validate_input_provenance(args: argparse.Namespace, prompts: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    input_hash = sha256_file(args.prompts)
    if input_hash != EXPECTED_INPUT_SHA256:
        raise ValidationError("prompt manifest SHA-256 is not the authorized clean corpus")
    source_hash = sha256_file(args.source_file)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValidationError("source gzip SHA-256 is not authorized")
    evaluation_hash = sha256_file(args.evaluation_questions)
    if evaluation_hash != EXPECTED_EVALUATION_SHA256:
        raise ValidationError("evaluation JSON SHA-256 is not authorized")
    try:
        evaluation = json.loads(Path(args.evaluation_questions).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid evaluation questions JSON") from exc
    questions = _evaluation_questions(evaluation)
    if not questions:
        raise ValidationError("evaluation questions JSON contains no question strings")
    overlap = {_normalized(row["prompt"]) for row in prompts}.intersection(_normalized(text) for text in questions)
    if overlap:
        raise ValidationError("frozen prompts overlap evaluation questions")
    return {"input_sha256": input_hash, "source_file": str(Path(args.source_file).resolve()),
            "source_sha256": source_hash, "evaluation_questions": str(Path(args.evaluation_questions).resolve()),
            "evaluation_sha256": evaluation_hash, "hereditary_commit": HEREDITARY_COMMIT,
            "unique_prompt_hashes": len(prompts)}


def _config(args: argparse.Namespace, prompts: list[dict[str, str]]) -> dict[str, Any]:
    if len(prompts) != EXPECTED_COUNT:
        raise ValidationError("this clean-corpus generator requires exactly %d prompts, found %d" %
                              (EXPECTED_COUNT, len(prompts)))
    max_batch = _arg(args, "max_batch_size", 256)
    budget = _arg(args, "conv_index_budget", 131072)
    threshold = _arg(args, "memory_pressure_threshold", 0.85)
    revision = _arg(args, "model_revision", MODEL_REVISION)
    label = _arg(args, "output_model_label", MODEL_ID)
    scientific = {"seed": _arg(args, "seed", 42), "temperature": _arg(args, "temperature", 1.0),
                  "top_p": _arg(args, "top_p", 1.0), "top_k": _arg(args, "top_k", 0),
                  "max_new_tokens": _arg(args, "max_new_tokens", 4096), "max_batch_size": max_batch,
                  "conv_index_budget": budget, "memory_pressure_threshold": threshold,
                  "model_revision": revision, "output_model_label": label}
    required = {"seed": 42, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 4096,
                "max_batch_size": 256, "conv_index_budget": 131072, "memory_pressure_threshold": 0.85,
                "model_revision": MODEL_REVISION, "output_model_label": MODEL_ID}
    if scientific != required:
        raise ValidationError("scientific generation settings are frozen; CLI deviation rejected")
    provenance = _validate_input_provenance(args, prompts)
    return {
        "format": CONFIG_VERSION,
        "input_sha256": provenance["input_sha256"],
        "expected_count": EXPECTED_COUNT,
        "provenance": provenance,
        "prompt_count": len(prompts),
        "model": {
            "id": MODEL_ID,
            "path": _canonical(args.model_path),
            "revision": revision,
            "tokenizer_path": _canonical(args.tokenizer_path),
            "tokenizer_revision": revision,
        },
        "generation": {
            "master_seed": _arg(args, "seed", 42), "temperature": _arg(args, "temperature", 1.0),
            "top_p": _arg(args, "top_p", 1.0), "top_k": _arg(args, "top_k", 0),
            "max_new_tokens": _arg(args, "max_new_tokens", 4096), "thinking": False,
            "system_prompt": None, "completions_per_prompt": 1, "dtype": "bfloat16",
        },
        "adaptive_scheduler": {
            "initial_max_batch_size": max_batch, "conv_index_budget": budget,
            "memory_pressure_threshold": float(threshold),
            "monotonically_non_increasing": True,
            "conv_safety_rule": "batch_size * padded_input_tokens <= conv_index_budget",
        },
        "output_model_label": label,
        "backend_deviation": "Local Transformers BF16 generation replaces Conmy's hosted/OpenRouter/Tinker path.",
        "sampling_note": ("Transformers vectorized sampling resets master seed 42 for each successful "
                          "attempted batch; outputs are batch-layout-dependent and no batch-size-independent "
                          "row RNG is claimed."),
    }


def _canonical(path: str | Path) -> str:
    return str(Path(path).resolve())


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(("blob %d\0" % path.stat().st_size).encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    staging = _read_json(Path(args.staging_manifest), "staging manifest")
    repositories = staging.get("repositories")
    hereditary_entries = [entry for entry in repositories if isinstance(entry, dict) and
                          entry.get("url") == "https://github.com/ArthurConmy/hereditary.git"] if isinstance(repositories, list) else []
    if len(hereditary_entries) != 1 or hereditary_entries[0].get("commit") != HEREDITARY_COMMIT:
        raise ValidationError("staging manifest hereditary commit differs")
    models = staging.get("models")
    if not isinstance(models, list):
        raise ValidationError("staging manifest models must be a list")
    entries = [entry for entry in models if isinstance(entry, dict) and entry.get("repo_id") == MODEL_ID]
    if len(entries) != 1:
        raise ValidationError("staging manifest must contain exactly one target model entry")
    entry = entries[0]
    required = {"repo_id", "revision", "local_dir", "file_count", "bytes"}
    if not required.issubset(entry) or entry["repo_id"] != MODEL_ID or entry["revision"] != MODEL_REVISION:
        raise ValidationError("staging model entry differs from frozen model")
    model_dir, tokenizer_dir = _canonical(args.model_path), _canonical(args.tokenizer_path)
    if model_dir != tokenizer_dir or model_dir != _canonical(entry["local_dir"]):
        raise ValidationError("model/tokenizer paths do not match staging canonical local_dir")
    root = Path(model_dir)
    metadata_root = root / ".cache" / "huggingface" / "download"
    if not root.is_dir() or not metadata_root.is_dir():
        raise ValidationError("local snapshot or Hugging Face download metadata is missing")
    files = sorted(path for path in root.rglob("*") if path.is_file() and ".cache" not in path.relative_to(root).parts)
    metadata = sorted(path for path in metadata_root.rglob("*.metadata") if path.is_file())
    expected_metadata = {path.relative_to(root).as_posix() + ".metadata" for path in files}
    actual_metadata = {path.relative_to(metadata_root).as_posix() for path in metadata}
    if actual_metadata != expected_metadata:
        raise ValidationError("snapshot files and Hugging Face metadata differ")
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        lines = (metadata_root / (relative + ".metadata")).read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or lines[0] != MODEL_REVISION:
            raise ValidationError("snapshot metadata revision differs: %s" % relative)
        etag = lines[1].strip().strip('"')
        if re.fullmatch(r"[0-9a-f]{64}", etag):
            actual = sha256_file(path)
        elif re.fullmatch(r"[0-9a-f]{40}", etag):
            actual = _git_blob_sha1(path)
        else:
            raise ValidationError("unsupported snapshot metadata etag: %s" % relative)
        if actual != etag:
            raise ValidationError("snapshot file checksum differs: %s" % relative)
        total += path.stat().st_size
    if entry["file_count"] != len(files) or entry["bytes"] != total:
        raise ValidationError("staging snapshot file count or bytes differ")
    return {"staging_manifest": _canonical(args.staging_manifest), "model_dir": model_dir,
            "file_count": len(files), "bytes": total, "revision": MODEL_REVISION,
            "hereditary_commit": HEREDITARY_COMMIT}


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid %s: %s" % (description, path)) from exc
    if not isinstance(value, dict):
        raise ValidationError("invalid %s: %s" % (description, path))
    return value


def _read_manifest(path: Path) -> dict[str, Any] | None:
    return _read_json(path, "run manifest") if path.exists() else None


def _validate_rows(rows: Sequence[Mapping[str, Any]], prompts: Sequence[Mapping[str, str]],
                   config: Mapping[str, Any]) -> None:
    expected = {prompt["id"]: (index, prompt) for index, prompt in enumerate(prompts)}
    for row in rows:
        item = expected.get(row["id"])
        if item is None:
            raise ValidationError("final batch contains unknown prompt ID: %s" % row["id"])
        index, prompt = item
        if any(row[field] != prompt[field] for field in ("source", "prompt")):
            raise ValidationError("published prompt content differs for ID: %s" % row["id"])
        if row["original_index"] != index or row["prompt_sha256"] != sha256_text(prompt["prompt"]):
            raise ValidationError("published prompt identity differs for ID: %s" % row["id"])
        if row["response_sha256"] != sha256_text(row["response"]):
            raise ValidationError("published response hash differs for ID: %s" % row["id"])
        if row["model"] != config["output_model_label"] or row["model_revision"] != config["model"]["revision"]:
            raise ValidationError("published model metadata differs for ID: %s" % row["id"])
        if row["seed"] != config["generation"]["master_seed"] or row["thinking"] is not False:
            raise ValidationError("published generation settings differ for ID: %s" % row["id"])
        if (not isinstance(row["input_tokens"], int) or row["input_tokens"] < 1 or
                not isinstance(row["response_tokens"], int) or row["response_tokens"] < 0 or
                not isinstance(row["generated_tokens"], int) or row["generated_tokens"] < row["response_tokens"] or
                row["output_tokens"] != row["generated_tokens"] or row["termination"] not in {"eos", "length"} or
                not isinstance(row["hit_token_cap"], bool) or not isinstance(row["is_blank"], bool) or
                row["is_blank"] != (not row["response"].strip())):
            raise ValidationError("invalid published row metadata for ID: %s" % row["id"])


def _validate_existing(run_dir: Path, config: dict[str, Any], prompts: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    assert_run_mutable(run_dir)
    existing = _read_manifest(run_dir / "manifest.json")
    if existing is not None and existing != config:
        raise ValidationError("run manifest is frozen; configuration or scheduler settings changed")
    rows = validate_batches(run_dir / "batches", key=lambda row: row["id"], required_keys=ROW_KEYS)
    _validate_rows(rows, prompts, config)
    if rows:
        # Validate the saved layout, snapshot, and runtime provenance before a resume can initialize CUDA.
        _load_layout_without_backend(run_dir, prompts, config)
        runtime = _read_json(run_dir / "runtime.json", "runtime evidence")
        if runtime.get("snapshot_verification") != config["snapshot_verification"]:
            raise ValidationError("runtime snapshot evidence differs from frozen provenance")
    return rows


def plan(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args.prompts)
    config = _config(args, prompts)
    snapshot = _verify_snapshot(args)
    config["snapshot_verification"] = snapshot
    completed = _validate_existing(Path(args.run_dir), config, prompts)
    resume = _resume_scheduler_state(Path(args.run_dir), config, len(completed),
                                     _arg(args, "resume_max_batch_size", None),
                                     _arg(args, "resume_memory_pressure_threshold", None),
                                     _arg(args, "recovery_max_batch_size", None))
    completed_ids = {row["id"] for row in completed}
    return {"prompt_count": len(prompts), "completed": len(completed_ids),
            "pending": len(prompts) - len(completed_ids),
            "final_batches": len(finalized_batches(Path(args.run_dir) / "batches")),
            "authorized_resume_max_batch_size": resume["authorized_resume_max_batch_size"],
            "effective_resume_max_batch_size": resume["current_max_batch_size"],
            "effective_resume_memory_pressure_threshold": resume["memory_pressure_threshold"],
            "effective_memory_pressure_basis": resume["memory_pressure_basis"],
            "hourly_successful_generation_seconds": resume["hourly_successful_generation_seconds"], "config": config}


def _load_backend(args: argparse.Namespace):
    # Imports live here so plan mode cannot load a CUDA/GPU dependency or make Hub calls.
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("execute requires CUDA; CPU fallback is intentionally forbidden")
    revision = _arg(args, "model_revision", MODEL_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, local_files_only=True, revision=revision, dtype=torch.bfloat16, device_map={"": "cuda"},
    )
    model.eval()
    if model.__class__.__name__ != "Qwen3_5ForConditionalGeneration" or model.dtype != torch.bfloat16:
        raise RuntimeError("loaded model is not Qwen3_5ForConditionalGeneration in bfloat16")
    return torch, tokenizer, model


def _preserve_runtime_evidence(run_dir: Path, torch: Any, model: Any, config: Mapping[str, Any]) -> None:
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:  # test doubles never need a package import
        transformers_version = "test-double"
    evidence = {"format": "teacher-runtime-v1", "snapshot_verification": config["snapshot_verification"],
                "model_class": model.__class__.__name__, "model_dtype": str(getattr(model, "dtype", "test-double")),
                "torch_version": str(getattr(torch, "__version__", "test-double")),
                "transformers_version": transformers_version,
                "cuda_device": str(torch.cuda.get_device_name(0)) if hasattr(torch.cuda, "get_device_name") else "test-double"}
    path = run_dir / "runtime.json"
    if path.exists():
        if _read_json(path, "runtime evidence") != evidence:
            raise ValidationError("runtime evidence differs from the first generation attempt")
    else:
        atomic_write_json(path, evidence)


def _render_and_tokenize(tokenizer: Any, prompt: str) -> tuple[str, list[int]]:
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(rendered, add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or not ids or not all(isinstance(token, int) for token in ids):
        raise ValidationError("tokenizer returned invalid rendered prompt token IDs")
    return rendered, ids


def _fsync_directory(path: Path) -> None:
    """Sync directory entries where the platform permits it (not Windows)."""
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("immutable file already exists: %s" % path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        count, checksum = write_jsonl_fsynced(temporary, rows)
        if path.exists():
            raise FileExistsError("immutable file already exists: %s" % path)
        os.replace(str(temporary), str(path))
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count, checksum


def _layout_rows(prompts: Sequence[Mapping[str, str]], tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layout: list[dict[str, Any]] = []
    work: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        rendered, ids = _render_and_tokenize(tokenizer, prompt["prompt"])
        entry = {"id": prompt["id"], "original_index": index,
                 "prompt_sha256": sha256_text(prompt["prompt"]), "rendered_prompt_sha256": sha256_text(rendered),
                 "input_tokens": len(ids)}
        layout.append(entry)
        work.append(dict(prompt, **entry, input_ids=ids))
    return layout, work


def _validate_layout(layout: Sequence[Mapping[str, Any]], prompts: Sequence[Mapping[str, str]]) -> None:
    if len(layout) != len(prompts):
        raise ValidationError("prompt layout count differs from frozen input")
    for index, (entry, prompt) in enumerate(zip(layout, prompts)):
        if set(entry) != set(LAYOUT_KEYS) or entry["id"] != prompt["id"] or entry["original_index"] != index:
            raise ValidationError("prompt layout identity differs at input index %d" % index)
        if entry["prompt_sha256"] != sha256_text(prompt["prompt"]) or not isinstance(entry["input_tokens"], int) or entry["input_tokens"] < 1:
            raise ValidationError("invalid prompt layout at input index %d" % index)
        if not isinstance(entry["rendered_prompt_sha256"], str) or len(entry["rendered_prompt_sha256"]) != 64:
            raise ValidationError("invalid rendered prompt hash at input index %d" % index)


def _prepare_prompt_work(run_dir: Path, prompts: Sequence[Mapping[str, str]], tokenizer: Any,
                         config: Mapping[str, Any]) -> list[dict[str, Any]]:
    layout_path, manifest_path = run_dir / "prompt-layout.jsonl", run_dir / "prompt-layout.manifest.json"
    generated_layout, work = _layout_rows(prompts, tokenizer)  # always pre-render/tokenize all prompts before scheduling
    expected_manifest = {"format": "teacher-prompt-layout-v1", "input_sha256": config["input_sha256"],
                         "row_count": len(prompts)}
    if layout_path.exists():
        if not layout_path.is_file():
            raise ValidationError("invalid immutable prompt layout")
        saved = list(iter_jsonl(layout_path))
        _validate_layout(saved, prompts)
        if saved != generated_layout:
            raise ValidationError("local tokenizer/template no longer matches immutable prompt layout")
        expected_manifest["sha256"] = sha256_file(layout_path)
        if manifest_path.exists():
            if not manifest_path.is_file() or _read_json(manifest_path, "prompt layout manifest") != expected_manifest:
                raise ValidationError("prompt layout manifest differs from frozen input")
        else:
            # A crash after the atomically published layout but before its manifest lost no
            # generation call; verify it and finish this one immutable publication.
            atomic_write_json(manifest_path, expected_manifest)
    elif manifest_path.exists():
        raise ValidationError("prompt layout manifest exists without immutable layout")
    else:
        count, checksum = _atomic_write_jsonl(layout_path, generated_layout)
        expected_manifest.update(row_count=count, sha256=checksum)
        atomic_write_json(manifest_path, expected_manifest)
    return work


def _load_layout_without_backend(run_dir: Path, prompts: Sequence[Mapping[str, str]], config: Mapping[str, Any]) -> None:
    path, manifest_path = run_dir / "prompt-layout.jsonl", run_dir / "prompt-layout.manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise ValidationError("completed generation requires immutable prompt layout")
    layout = list(iter_jsonl(path))
    _validate_layout(layout, prompts)
    manifest = _read_json(manifest_path, "prompt layout manifest")
    if manifest != {"format": "teacher-prompt-layout-v1", "input_sha256": config["input_sha256"],
                    "row_count": len(prompts), "sha256": sha256_file(path)}:
        raise ValidationError("prompt layout manifest differs from frozen input")


def _append_scheduler_event(run_dir: Path, **event: Any) -> None:
    path = run_dir / "scheduler.jsonl"
    with path.open("ab") as handle:
        handle.write(strict_json_bytes(event))
        handle.flush()
        os.fsync(handle.fileno())


def _scheduler_resume_amendment(run_dir: Path, config: Mapping[str, Any], *, required: bool) -> dict[str, Any] | None:
    path = run_dir / SCHEDULER_RESUME_AMENDMENT_RELATIVE_PATH
    if not path.exists():
        if required:
            raise ValidationError("--resume-max-batch-size requires the immutable scheduler resume amendment")
        return None
    if not path.is_file():
        raise ValidationError("scheduler resume amendment path is not a file")
    amendment = _read_json(path, "scheduler resume amendment")
    scheduler = config["adaptive_scheduler"]
    if set(amendment) != SCHEDULER_RESUME_AMENDMENT_KEYS:
        raise ValidationError("scheduler resume amendment schema differs from the authorized amendment")
    if (amendment["format"] != SCHEDULER_RESUME_AMENDMENT_FORMAT or
            amendment["run_directory"] != run_dir.name or amendment["input_sha256"] != config["input_sha256"] or
            amendment["model_revision"] != config["model"]["revision"] or
            amendment["decision"] != SCHEDULER_RESUME_DECISION or
            amendment["previous_manifest_max_batch_size"] != SCHEDULER_RESUME_PREVIOUS_MAX or
            amendment["resumed_max_batch_size"] != SCHEDULER_RESUME_MAX or
            amendment["effective_completed_rows"] != SCHEDULER_RESUME_EFFECTIVE_ROWS or
            amendment["effective_completed_batches"] != SCHEDULER_RESUME_EFFECTIVE_BATCHES or
            amendment["adaptive_fallback"] is not True or
            amendment["previous_memory_pressure_threshold"] != SCHEDULER_RESUME_PREVIOUS_MEMORY_PRESSURE_THRESHOLD or
            amendment["resumed_memory_pressure_threshold"] != SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD or
            amendment["conv_index_budget"] != scheduler["conv_index_budget"] or
            amendment["raw_prior_batches_immutable"] is not True or
            amendment["authorization_reason"] != SCHEDULER_RESUME_AUTHORIZATION_REASON or
            amendment["authorizing_user_decision"] != SCHEDULER_RESUME_AUTHORIZING_USER_DECISION or
            not _valid_authorization_timestamp(amendment["authorization_timestamp"])):
        raise ValidationError("scheduler resume amendment is not authorized for this immutable run")
    if (scheduler["initial_max_batch_size"] != SCHEDULER_RESUME_PREVIOUS_MAX or
            scheduler["memory_pressure_threshold"] != SCHEDULER_RESUME_PREVIOUS_MEMORY_PRESSURE_THRESHOLD or
            scheduler["conv_index_budget"] != 131072):
        raise ValidationError("scheduler resume amendment requires the frozen 256-row scheduler manifest")
    return {"path": SCHEDULER_RESUME_AMENDMENT_RELATIVE_PATH.as_posix(), "sha256": sha256_file(path),
            "decision": amendment["decision"]}


def _scheduler_allocator_recovery_amendment(run_dir: Path, config: Mapping[str, Any],
                                             first_amendment: Mapping[str, Any] | None,
                                             *, required: bool) -> dict[str, Any] | None:
    path = run_dir / SCHEDULER_RECOVERY_AMENDMENT_RELATIVE_PATH
    if not path.exists():
        if required:
            raise ValidationError("--recovery-max-batch-size requires the immutable allocator recovery amendment")
        return None
    if not path.is_file():
        raise ValidationError("scheduler allocator recovery amendment path is not a file")
    if first_amendment is None:
        raise ValidationError("scheduler allocator recovery amendment requires the first scheduler amendment")
    amendment = _read_json(path, "scheduler allocator recovery amendment")
    if set(amendment) != SCHEDULER_RECOVERY_AMENDMENT_KEYS:
        raise ValidationError("scheduler allocator recovery amendment schema differs from authorization")
    if (amendment["format"] != SCHEDULER_RECOVERY_AMENDMENT_FORMAT or
            amendment["run_directory"] != run_dir.name or amendment["input_sha256"] != config["input_sha256"] or
            amendment["model_revision"] != config["model"]["revision"] or
            amendment["decision"] != SCHEDULER_RECOVERY_DECISION or
            amendment["first_amendment_path"] != first_amendment["path"] or
            amendment["first_amendment_sha256"] != first_amendment["sha256"] or
            amendment["previous_reconstructed_max_batch_size"] != SCHEDULER_RECOVERY_PREVIOUS_MAX or
            amendment["retry_max_batch_size"] != SCHEDULER_RECOVERY_MAX or
            amendment["memory_pressure_threshold"] != SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD or
            amendment["effective_completed_batches"] != SCHEDULER_RECOVERY_EFFECTIVE_BATCHES or
            amendment["effective_completed_rows"] != SCHEDULER_RECOVERY_EFFECTIVE_ROWS or
            amendment["allocator_cleanup_before_every_attempt"] is not True or
            amendment["raw_prior_batches_immutable"] is not True or
            amendment["scheduler_journal_immutable"] is not True or
            amendment["authorization_timestamp"] != SCHEDULER_RECOVERY_AUTHORIZATION_TIMESTAMP or
            amendment["authorization_reason"] != SCHEDULER_RECOVERY_AUTHORIZATION_REASON or
            amendment["authorizing_user_decision"] != SCHEDULER_RECOVERY_AUTHORIZING_USER_DECISION):
        raise ValidationError("scheduler allocator recovery amendment is not authorized for this immutable run")
    return {"path": SCHEDULER_RECOVERY_AMENDMENT_RELATIVE_PATH.as_posix(), "sha256": sha256_file(path),
            "decision": amendment["decision"]}


def _scheduler_pressure_recovery_amendment(run_dir: Path, config: Mapping[str, Any],
                                           first_amendment: Mapping[str, Any] | None,
                                           recovery_amendment: Mapping[str, Any] | None,
                                           *, required: bool) -> dict[str, Any] | None:
    path = run_dir / SCHEDULER_PRESSURE_RECOVERY_AMENDMENT_RELATIVE_PATH
    if not path.exists():
        if required:
            raise ValidationError("--pressure-recovery-max-batch-size requires the immutable pressure recovery amendment")
        return None
    if not path.is_file() or first_amendment is None or recovery_amendment is None:
        raise ValidationError("scheduler pressure recovery amendment requires both prior scheduler amendments")
    amendment = _read_json(path, "scheduler pressure recovery amendment")
    if set(amendment) != SCHEDULER_PRESSURE_RECOVERY_AMENDMENT_KEYS:
        raise ValidationError("scheduler pressure recovery amendment schema differs from authorization")
    if (amendment["format"] != SCHEDULER_PRESSURE_RECOVERY_AMENDMENT_FORMAT or
            amendment["run_directory"] != run_dir.name or amendment["input_sha256"] != config["input_sha256"] or
            amendment["model_revision"] != config["model"]["revision"] or
            amendment["decision"] != SCHEDULER_PRESSURE_RECOVERY_DECISION or
            amendment["first_amendment_path"] != first_amendment["path"] or
            amendment["first_amendment_sha256"] != first_amendment["sha256"] or
            amendment["second_amendment_path"] != recovery_amendment["path"] or
            amendment["second_amendment_sha256"] != recovery_amendment["sha256"] or
            amendment["previous_reconstructed_max_batch_size"] != SCHEDULER_PRESSURE_RECOVERY_PREVIOUS_MAX or
            amendment["retry_max_batch_size"] != SCHEDULER_PRESSURE_RECOVERY_MAX or
            amendment["memory_pressure_threshold"] != SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD or
            amendment["memory_pressure_basis"] != SCHEDULER_PRESSURE_BASIS or
            amendment["reserved_memory_diagnostic_only"] is not True or
            amendment["hourly_successful_generation_seconds"] != SCHEDULER_PRESSURE_WINDOW_SECONDS or
            amendment["effective_completed_batches"] != SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_BATCHES or
            amendment["effective_completed_rows"] != SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_ROWS or
            amendment["raw_prior_batches_immutable"] is not True or amendment["scheduler_journal_immutable"] is not True or
            amendment["authorization_timestamp"] != SCHEDULER_PRESSURE_RECOVERY_AUTHORIZATION_TIMESTAMP or
            amendment["authorization_reason"] != SCHEDULER_PRESSURE_RECOVERY_AUTHORIZATION_REASON or
            amendment["authorizing_user_decision"] != SCHEDULER_PRESSURE_RECOVERY_AUTHORIZING_USER_DECISION):
        raise ValidationError("scheduler pressure recovery amendment is not authorized for this immutable run")
    return {"path": SCHEDULER_PRESSURE_RECOVERY_AMENDMENT_RELATIVE_PATH.as_posix(), "sha256": sha256_file(path),
            "decision": amendment["decision"]}


def _scheduler_batch_manifests(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(batch, _read_json(batch / "manifest.json", "batch manifest"))
            for batch in finalized_batches(run_dir / "batches")]


def _validate_scheduler_resume_boundary(batch_manifests: Sequence[tuple[Path, Mapping[str, Any]]],
                                        completed_rows: int) -> None:
    if (len(batch_manifests) < SCHEDULER_RESUME_EFFECTIVE_BATCHES or
            completed_rows < SCHEDULER_RESUME_EFFECTIVE_ROWS):
        raise ValidationError("scheduler resume amendment requires at least 21 immutable batches and 5,376 rows")
    boundary_rows = sum(manifest["row_count"] for _, manifest in
                        batch_manifests[:SCHEDULER_RESUME_EFFECTIVE_BATCHES])
    if boundary_rows != SCHEDULER_RESUME_EFFECTIVE_ROWS:
        raise ValidationError("immutable scheduler resume boundary differs from authorized rows")


def _validate_scheduler_allocator_recovery_boundary(batch_manifests: Sequence[tuple[Path, Mapping[str, Any]]]) -> None:
    if len(batch_manifests) < SCHEDULER_RECOVERY_EFFECTIVE_BATCHES:
        raise ValidationError("allocator recovery amendment requires 24 immutable batches")
    boundary_rows = sum(manifest["row_count"] for _, manifest in
                        batch_manifests[:SCHEDULER_RECOVERY_EFFECTIVE_BATCHES])
    if boundary_rows != SCHEDULER_RECOVERY_EFFECTIVE_ROWS:
        raise ValidationError("immutable allocator recovery boundary differs from authorized rows")
    expected = ((256, 256, 128), (128, 128, 64), (64, 64, 32))
    for offset, (row_count, before, after) in enumerate(expected, start=SCHEDULER_RESUME_EFFECTIVE_BATCHES):
        batch, manifest = batch_manifests[offset]
        if (manifest.get("row_count"), manifest.get("actual_size"), manifest.get("scheduler_max_before"),
                manifest.get("scheduler_max_after")) != (row_count, row_count, before, after):
            raise ValidationError("allocator recovery requires the immutable 512-to-32 scheduler history: %s" % batch)


def _validate_scheduler_pressure_recovery_boundary(batch_manifests: Sequence[tuple[Path, Mapping[str, Any]]]) -> None:
    if len(batch_manifests) < SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_BATCHES:
        raise ValidationError("pressure recovery amendment requires exactly 28 immutable batches at application")
    boundary_rows = sum(manifest.get("row_count", 0) for _, manifest in
                        batch_manifests[:SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_BATCHES])
    if boundary_rows != SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_ROWS:
        raise ValidationError("immutable pressure recovery boundary differs from authorized rows")
    expected = ((384, 384, 192), (192, 192, 96), (96, 96, 48), (48, 48, 24))
    for offset, (row_count, before, after) in enumerate(expected, start=SCHEDULER_RECOVERY_EFFECTIVE_BATCHES):
        batch, manifest = batch_manifests[offset]
        if (manifest.get("row_count"), manifest.get("actual_size"), manifest.get("scheduler_max_before"),
                manifest.get("scheduler_max_after")) != (row_count, row_count, before, after):
            raise ValidationError("pressure recovery requires immutable 384-to-24 scheduler history: %s" % batch)


def _scheduler_amendment_event(event: Mapping[str, Any], amendment: Mapping[str, Any]) -> None:
    expected = {"event": "scheduler_amendment_applied", "amendment_path": amendment["path"],
                "amendment_sha256": amendment["sha256"], "prior_max_batch_size": SCHEDULER_RESUME_PREVIOUS_MAX,
                "next_max_batch_size": SCHEDULER_RESUME_MAX,
                "prior_memory_pressure_threshold": SCHEDULER_RESUME_PREVIOUS_MEMORY_PRESSURE_THRESHOLD,
                "next_memory_pressure_threshold": SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD,
                "effective_completed_batches": SCHEDULER_RESUME_EFFECTIVE_BATCHES,
                "effective_completed_rows": SCHEDULER_RESUME_EFFECTIVE_ROWS}
    if set(event) != SCHEDULER_AMENDMENT_EVENT_KEYS or dict(event) != expected:
        raise ValidationError("scheduler amendment event differs from immutable authorization")


def _scheduler_allocator_recovery_event(event: Mapping[str, Any], amendment: Mapping[str, Any]) -> None:
    expected = {"event": "scheduler_allocator_recovery_applied", "amendment_path": amendment["path"],
                "amendment_sha256": amendment["sha256"], "prior_max_batch_size": SCHEDULER_RECOVERY_PREVIOUS_MAX,
                "next_max_batch_size": SCHEDULER_RECOVERY_MAX,
                "memory_pressure_threshold": SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD,
                "effective_completed_batches": SCHEDULER_RECOVERY_EFFECTIVE_BATCHES,
                "effective_completed_rows": SCHEDULER_RECOVERY_EFFECTIVE_ROWS}
    if set(event) != SCHEDULER_RECOVERY_EVENT_KEYS or dict(event) != expected:
        raise ValidationError("scheduler allocator recovery event differs from immutable authorization")


def _scheduler_pressure_recovery_event(event: Mapping[str, Any], amendment: Mapping[str, Any]) -> None:
    expected = {"event": "scheduler_pressure_recovery_applied", "amendment_path": amendment["path"],
                "amendment_sha256": amendment["sha256"],
                "prior_max_batch_size": SCHEDULER_PRESSURE_RECOVERY_PREVIOUS_MAX,
                "next_max_batch_size": SCHEDULER_PRESSURE_RECOVERY_MAX,
                "memory_pressure_threshold": SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD,
                "memory_pressure_basis": SCHEDULER_PRESSURE_BASIS,
                "reserved_memory_diagnostic_only": True,
                "hourly_successful_generation_seconds": SCHEDULER_PRESSURE_WINDOW_SECONDS,
                "effective_completed_batches": SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_BATCHES,
                "effective_completed_rows": SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_ROWS}
    if set(event) != SCHEDULER_PRESSURE_RECOVERY_EVENT_KEYS or dict(event) != expected:
        raise ValidationError("scheduler pressure recovery event differs from immutable authorization")


def _scheduler_pressure_checkpoint_event(event: Mapping[str, Any], current: int) -> int:
    required = {"event", "batch", "memory_pressure_basis", "window_successful_generation_seconds",
                "window_max_allocated_pressure", "memory_pressure_threshold", "target_max_batch_size_before",
                "target_max_batch_size_after", "reserved_memory_diagnostic_only"}
    if set(event) != required or event["event"] != "scheduler_pressure_checkpoint":
        raise ValidationError("scheduler pressure checkpoint evidence is malformed")
    if (not isinstance(event["batch"], str) or event["memory_pressure_basis"] != SCHEDULER_PRESSURE_BASIS or
            event["memory_pressure_threshold"] != SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD or
            event["reserved_memory_diagnostic_only"] is not True or
            event["target_max_batch_size_before"] != current or
            not isinstance(event["window_successful_generation_seconds"], (int, float)) or
            event["window_successful_generation_seconds"] < SCHEDULER_PRESSURE_WINDOW_SECONDS or
            not isinstance(event["window_max_allocated_pressure"], (int, float)) or
            not 0 <= event["window_max_allocated_pressure"] <= 1):
        raise ValidationError("scheduler pressure checkpoint evidence is invalid")
    expected = (SCHEDULER_RESUME_PREVIOUS_MAX if current == SCHEDULER_PRESSURE_RECOVERY_MAX and
                event["window_max_allocated_pressure"] >= SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD else current)
    if event["target_max_batch_size_after"] != expected:
        raise ValidationError("scheduler pressure checkpoint has an unauthorized target transition")
    return expected


def _validate_pressure_window(batch_manifests: Sequence[tuple[Path, Mapping[str, Any]]],
                              events: Sequence[Mapping[str, Any]]) -> int:
    post = batch_manifests[SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_BATCHES:]
    checkpoints = [event for event in events if event.get("event") == "scheduler_pressure_checkpoint"]
    window_elapsed, window_peak, target, checkpoint_index = 0.0, 0.0, SCHEDULER_PRESSURE_RECOVERY_MAX, 0
    for batch, manifest in post:
        required = {"memory_pressure_basis", "allocated_memory_pressure", "peak_allocated_bytes",
                    "peak_reserved_bytes", "total_vram_bytes", "scheduler_target_max_batch_size",
                    "pressure_window_successful_generation_seconds", "pressure_window_max_allocated_pressure"}
        if not required.issubset(manifest):
            raise ValidationError("post-transition batch lacks allocated pressure window evidence: %s" % batch)
        manifest_target = manifest.get("scheduler_target_max_batch_size")
        if manifest_target != target:
            has_matching_oom = any(event.get("event") == "recoverable_generation_error" and
                                   event.get("next_max_batch_size") == manifest_target for event in events)
            if manifest_target != _reduced_scheduler_max(target, target) or not has_matching_oom:
                raise ValidationError("post-transition target reduction lacks one-step OOM/index evidence: %s" % batch)
            target = manifest_target
        elapsed, pressure = manifest.get("elapsed_seconds"), manifest.get("allocated_memory_pressure")
        if (manifest["memory_pressure_basis"] != SCHEDULER_PRESSURE_BASIS or
                not isinstance(elapsed, (int, float)) or elapsed < 0 or
                not isinstance(pressure, (int, float)) or not 0 <= pressure <= 1 or
                not isinstance(manifest["peak_allocated_bytes"], int) or manifest["peak_allocated_bytes"] < 0 or
                not isinstance(manifest["peak_reserved_bytes"], int) or manifest["peak_reserved_bytes"] < 0 or
                not isinstance(manifest["total_vram_bytes"], int) or manifest["total_vram_bytes"] <= 0 or
                pressure != manifest["peak_allocated_bytes"] / manifest["total_vram_bytes"] or
                manifest.get("memory_pressure") != pressure or
                manifest.get("scheduler_target_max_batch_size") != target or
                manifest.get("scheduler_max_before") != target):
            raise ValidationError("post-transition batch pressure evidence differs from allocated basis: %s" % batch)
        window_elapsed += float(elapsed)
        window_peak = max(window_peak, float(pressure))
        if (manifest["pressure_window_successful_generation_seconds"] != window_elapsed or
                manifest["pressure_window_max_allocated_pressure"] != window_peak):
            raise ValidationError("scheduler pressure window evidence is missing or inconsistent: %s" % batch)
        after = target
        if window_elapsed >= SCHEDULER_PRESSURE_WINDOW_SECONDS:
            after = (SCHEDULER_RESUME_PREVIOUS_MAX if target == SCHEDULER_PRESSURE_RECOVERY_MAX and
                     window_peak >= SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD else target)
            if checkpoint_index >= len(checkpoints):
                raise ValidationError("scheduler pressure window reached one hour without checkpoint evidence")
            event = checkpoints[checkpoint_index]
            expected = {"event": "scheduler_pressure_checkpoint", "batch": batch.name,
                        "memory_pressure_basis": SCHEDULER_PRESSURE_BASIS,
                        "window_successful_generation_seconds": window_elapsed,
                        "window_max_allocated_pressure": window_peak,
                        "memory_pressure_threshold": SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD,
                        "target_max_batch_size_before": target, "target_max_batch_size_after": after,
                        "reserved_memory_diagnostic_only": True}
            if dict(event) != expected:
                raise ValidationError("scheduler pressure checkpoint differs from durable window evidence")
            checkpoint_index += 1
            window_elapsed, window_peak = 0.0, 0.0
        if manifest.get("scheduler_max_after") != after:
            raise ValidationError("post-transition batch has an unauthorized scheduler target transition: %s" % batch)
        target = after
    if checkpoint_index != len(checkpoints):
        raise ValidationError("scheduler pressure checkpoint has no completed durable window")
    return target


def _pressure_window_state(batch_manifests: Sequence[tuple[Path, Mapping[str, Any]]]) -> tuple[float, float]:
    elapsed, peak = 0.0, 0.0
    for _, manifest in batch_manifests[SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_BATCHES:]:
        elapsed += float(manifest["elapsed_seconds"])
        peak = max(peak, float(manifest["allocated_memory_pressure"]))
        if elapsed >= SCHEDULER_PRESSURE_WINDOW_SECONDS:
            elapsed, peak = 0.0, 0.0
    return elapsed, peak


def _scheduler_evidence_state(run_dir: Path, config: Mapping[str, Any],
                              batch_manifests: Sequence[tuple[Path, Mapping[str, Any]]],
                              amendment: Mapping[str, Any] | None,
                              recovery_amendment: Mapping[str, Any] | None,
                              pressure_recovery_amendment: Mapping[str, Any] | None) -> tuple[int, bool, bool, bool]:
    initial = config["adaptive_scheduler"]["initial_max_batch_size"]
    path = run_dir / "scheduler.jsonl"
    events = list(iter_jsonl(path)) if path.exists() else []
    amendment_events = [event for event in events if event.get("event") == "scheduler_amendment_applied"]
    recovery_events = [event for event in events if event.get("event") == "scheduler_allocator_recovery_applied"]
    pressure_events = [event for event in events if event.get("event") == "scheduler_pressure_recovery_applied"]
    if len(amendment_events) > 1 or len(recovery_events) > 1 or len(pressure_events) > 1:
        raise ValidationError("scheduler amendment event appears more than once")
    if amendment_events:
        if amendment is None:
            raise ValidationError("scheduler amendment event has no immutable amendment")
        _scheduler_amendment_event(amendment_events[0], amendment)
    if recovery_events:
        if recovery_amendment is None:
            raise ValidationError("scheduler allocator recovery event has no immutable amendment")
        _scheduler_allocator_recovery_event(recovery_events[0], recovery_amendment)
        _validate_scheduler_allocator_recovery_boundary(batch_manifests)
    if pressure_events:
        if pressure_recovery_amendment is None:
            raise ValidationError("scheduler pressure recovery event has no immutable amendment")
        _scheduler_pressure_recovery_event(pressure_events[0], pressure_recovery_amendment)
        _validate_scheduler_pressure_recovery_boundary(batch_manifests)

    current, amendment_seen, recovery_seen, pressure_seen = initial, False, False, False
    journal_resume_shrunk = False
    previous_event: Mapping[str, Any] | None = None
    for event_index, event in enumerate(events):
        event_name = event.get("event")
        if event_name == "scheduler_amendment_applied":
            _scheduler_amendment_event(event, amendment)  # type: ignore[arg-type]
            if amendment_seen or current != SCHEDULER_RESUME_PREVIOUS_MAX:
                raise ValidationError("scheduler amendment is not a single authorized transition from 256")
            current, amendment_seen = SCHEDULER_RESUME_MAX, True
        elif event_name == "scheduler_allocator_recovery_applied":
            _scheduler_allocator_recovery_event(event, recovery_amendment)  # type: ignore[arg-type]
            if (not amendment_seen or recovery_seen or current != SCHEDULER_RECOVERY_PREVIOUS_MAX or
                    previous_event is None or previous_event.get("event") != "attempt" or
                    previous_event.get("max_batch_size") != SCHEDULER_RECOVERY_PREVIOUS_MAX or
                    previous_event.get("actual_size") != SCHEDULER_RECOVERY_PREVIOUS_MAX):
                raise ValidationError("allocator recovery is not the single authorized transition from dangling 32")
            current, recovery_seen = SCHEDULER_RECOVERY_MAX, True
        elif event_name == "scheduler_pressure_recovery_applied":
            _scheduler_pressure_recovery_event(event, pressure_recovery_amendment)  # type: ignore[arg-type]
            if (not recovery_seen or pressure_seen or current != SCHEDULER_PRESSURE_RECOVERY_PREVIOUS_MAX or
                    previous_event is None or previous_event.get("event") != "attempt" or
                    previous_event.get("max_batch_size") != SCHEDULER_PRESSURE_RECOVERY_PREVIOUS_MAX or
                    previous_event.get("actual_size") != SCHEDULER_PRESSURE_RECOVERY_PREVIOUS_MAX):
                raise ValidationError("pressure recovery is not the single authorized transition from dangling 24")
            current, pressure_seen = SCHEDULER_PRESSURE_RECOVERY_MAX, True
        elif event_name == "scheduler_pressure_checkpoint":
            if not pressure_seen:
                raise ValidationError("scheduler pressure checkpoint has no applied pressure recovery amendment")
            current = _scheduler_pressure_checkpoint_event(event, current)
        else:
            maximum = SCHEDULER_RECOVERY_MAX if recovery_seen else (SCHEDULER_RESUME_MAX if amendment_seen else initial)
            attempt_max = event.get("max_batch_size")
            if attempt_max is not None and (not isinstance(attempt_max, int) or not 1 <= attempt_max <= current or
                                            (pressure_seen and event_name == "attempt" and attempt_max != current)):
                raise ValidationError("scheduler attempt exceeds reconstructed durable state")
            next_max = event.get("next_max_batch_size")
            if pressure_seen and event_name == "recoverable_generation_error":
                actual_size = event.get("actual_size")
                if not isinstance(actual_size, int) or actual_size < 1 or next_max != _reduced_scheduler_max(current, actual_size):
                    raise ValidationError("post-transition OOM/index fallback is not a one-step conservative reduction")
                current = next_max
            elif next_max is not None:
                allowed_checkpoint_transition = (pressure_seen and event_name == "published" and
                                                 current == SCHEDULER_PRESSURE_RECOVERY_MAX and
                                                 next_max == SCHEDULER_RESUME_PREVIOUS_MAX and
                                                 event_index + 1 < len(events) and
                                                 events[event_index + 1].get("event") == "scheduler_pressure_checkpoint")
                if pressure_seen and next_max != current and not allowed_checkpoint_transition:
                    raise ValidationError("post-transition scheduler changes only at hourly checkpoints or OOM/index failure")
                if not isinstance(next_max, int) or not 1 <= next_max <= maximum or next_max > current:
                    raise ValidationError("unauthorized scheduler increase or invalid durable scheduler event")
                if not allowed_checkpoint_transition:
                    current = next_max
                if amendment_seen and not recovery_seen and current < SCHEDULER_RESUME_MAX:
                    journal_resume_shrunk = True
        previous_event = event

    batch_resume_values: list[int] = []
    base_batch_values: list[int] = []
    recovery_batch_values: list[int] = []
    resume_batch_shrunk, recovery_batch_shrunk = False, False
    for batch_index, (batch, manifest) in enumerate(batch_manifests):
        before, after = manifest.get("scheduler_max_before"), manifest.get("scheduler_max_after")
        if (not isinstance(before, int) or not isinstance(after, int) or before < 1 or after < 1 or after > before or
                before > SCHEDULER_RESUME_MAX):
            raise ValidationError("invalid scheduler batch evidence: %s" % batch)
        if pressure_seen and batch_index >= SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_BATCHES:
            continue
        if recovery_seen and batch_index >= SCHEDULER_RECOVERY_EFFECTIVE_BATCHES:
            if before > SCHEDULER_RECOVERY_MAX:
                raise ValidationError("unauthorized post-recovery scheduler maximum: %s" % batch)
            if recovery_batch_shrunk and before == SCHEDULER_RECOVERY_MAX:
                raise ValidationError("scheduler batch evidence reapplies 384 after an adaptive shrink")
            recovery_batch_values.append(after)
            if after < SCHEDULER_RECOVERY_MAX:
                recovery_batch_shrunk = True
        elif before > initial or after > initial:
            if not amendment_seen or before != SCHEDULER_RESUME_MAX:
                raise ValidationError("unauthorized scheduler increase in batch evidence: %s" % batch)
            if resume_batch_shrunk:
                raise ValidationError("scheduler batch evidence reapplies 512 after an adaptive shrink")
            batch_resume_values.append(after)
            if after < SCHEDULER_RESUME_MAX:
                resume_batch_shrunk = True
        elif (amendment_seen and batch_index >= SCHEDULER_RESUME_EFFECTIVE_BATCHES and
              (resume_batch_shrunk or journal_resume_shrunk)):
            batch_resume_values.append(after)
        else:
            base_batch_values.append(after)
    if pressure_seen:
        # A recoverable error after the most recent immutable batch has durable journal evidence but no
        # manifest yet; retain its conservative target while still validating every published window.
        current = min(current, _validate_pressure_window(batch_manifests, events))
    elif recovery_seen and recovery_batch_values:
        current = min(current, min(recovery_batch_values))
    elif not amendment_seen and base_batch_values:
        current = min(current, min(base_batch_values))
    elif amendment_seen and not recovery_seen and batch_resume_values:
        current = min(current, min(batch_resume_values))
    return current, amendment_seen, recovery_seen, pressure_seen

def _resume_scheduler_state(run_dir: Path, config: Mapping[str, Any], completed_rows: int,
                            requested_max: Any = None, requested_threshold: Any = None,
                            requested_recovery_max: Any = None, requested_pressure_recovery_max: Any = None,
                            *, apply: bool = False) -> dict[str, Any]:
    if requested_max is not None and requested_max != SCHEDULER_RESUME_MAX:
        raise ValidationError("only --resume-max-batch-size 512 is authorized")
    if requested_threshold is not None and requested_threshold != SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD:
        raise ValidationError("only --resume-memory-pressure-threshold 0.92 is authorized")
    if requested_max is None and requested_threshold is not None:
        raise ValidationError("--resume-memory-pressure-threshold requires --resume-max-batch-size 512")
    if requested_max is not None and requested_threshold != SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD:
        raise ValidationError("--resume-max-batch-size 512 requires --resume-memory-pressure-threshold 0.92")
    if requested_recovery_max is not None and requested_recovery_max != SCHEDULER_RECOVERY_MAX:
        raise ValidationError("only --recovery-max-batch-size 384 is authorized")
    if requested_pressure_recovery_max is not None and requested_pressure_recovery_max != SCHEDULER_PRESSURE_RECOVERY_MAX:
        raise ValidationError("only --pressure-recovery-max-batch-size 384 is authorized")
    amendment = _scheduler_resume_amendment(
        run_dir, config, required=requested_max is not None or requested_recovery_max is not None or
        requested_pressure_recovery_max is not None)
    recovery_amendment = _scheduler_allocator_recovery_amendment(
        run_dir, config, amendment, required=requested_recovery_max is not None or requested_pressure_recovery_max is not None)
    pressure_recovery_amendment = _scheduler_pressure_recovery_amendment(
        run_dir, config, amendment, recovery_amendment, required=requested_pressure_recovery_max is not None)
    batch_manifests = _scheduler_batch_manifests(run_dir)
    if amendment is not None:
        _validate_scheduler_resume_boundary(batch_manifests, completed_rows)
    if recovery_amendment is not None:
        _validate_scheduler_allocator_recovery_boundary(batch_manifests)
    if pressure_recovery_amendment is not None:
        _validate_scheduler_pressure_recovery_boundary(batch_manifests)
    current, amendment_applied, recovery_applied, pressure_recovery_applied = _scheduler_evidence_state(
        run_dir, config, batch_manifests, amendment, recovery_amendment, pressure_recovery_amendment)
    if (recovery_amendment is not None and not recovery_applied and
            (len(batch_manifests) != SCHEDULER_RECOVERY_EFFECTIVE_BATCHES or
             completed_rows != SCHEDULER_RECOVERY_EFFECTIVE_ROWS)):
        raise ValidationError("allocator recovery amendment is outside immutable boundary 24/5,824")
    if requested_max is not None and not amendment_applied:
        if current != SCHEDULER_RESUME_PREVIOUS_MAX:
            raise ValidationError("scheduler resume amendment cannot override an already reduced scheduler")
        if apply:
            _append_scheduler_event(
                run_dir, event="scheduler_amendment_applied", amendment_path=amendment["path"],
                amendment_sha256=amendment["sha256"], prior_max_batch_size=SCHEDULER_RESUME_PREVIOUS_MAX,
                next_max_batch_size=SCHEDULER_RESUME_MAX,
                prior_memory_pressure_threshold=SCHEDULER_RESUME_PREVIOUS_MEMORY_PRESSURE_THRESHOLD,
                next_memory_pressure_threshold=SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD,
                effective_completed_batches=SCHEDULER_RESUME_EFFECTIVE_BATCHES,
                effective_completed_rows=SCHEDULER_RESUME_EFFECTIVE_ROWS,
            )
            current, amendment_applied, recovery_applied, pressure_recovery_applied = _scheduler_evidence_state(
                run_dir, config, batch_manifests, amendment, recovery_amendment, pressure_recovery_amendment)
        else:
            current = SCHEDULER_RESUME_MAX
    if requested_recovery_max is not None and not recovery_applied:
        if (current != SCHEDULER_RECOVERY_PREVIOUS_MAX or len(batch_manifests) != SCHEDULER_RECOVERY_EFFECTIVE_BATCHES or
                completed_rows != SCHEDULER_RECOVERY_EFFECTIVE_ROWS):
            raise ValidationError("allocator recovery amendment can apply only at immutable boundary 24/5,824")
        if apply:
            _append_scheduler_event(
                run_dir, event="scheduler_allocator_recovery_applied", amendment_path=recovery_amendment["path"],
                amendment_sha256=recovery_amendment["sha256"], prior_max_batch_size=SCHEDULER_RECOVERY_PREVIOUS_MAX,
                next_max_batch_size=SCHEDULER_RECOVERY_MAX,
                memory_pressure_threshold=SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD,
                effective_completed_batches=SCHEDULER_RECOVERY_EFFECTIVE_BATCHES,
                effective_completed_rows=SCHEDULER_RECOVERY_EFFECTIVE_ROWS,
            )
            current, amendment_applied, recovery_applied, pressure_recovery_applied = _scheduler_evidence_state(
                run_dir, config, batch_manifests, amendment, recovery_amendment, pressure_recovery_amendment)
        else:
            current = SCHEDULER_RECOVERY_MAX
    if requested_pressure_recovery_max is not None and not pressure_recovery_applied:
        if (current != SCHEDULER_PRESSURE_RECOVERY_PREVIOUS_MAX or
                len(batch_manifests) != SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_BATCHES or
                completed_rows != SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_ROWS):
            raise ValidationError("pressure recovery amendment can apply only at immutable boundary 28/6,544")
        if apply:
            _append_scheduler_event(
                run_dir, event="scheduler_pressure_recovery_applied", amendment_path=pressure_recovery_amendment["path"],
                amendment_sha256=pressure_recovery_amendment["sha256"],
                prior_max_batch_size=SCHEDULER_PRESSURE_RECOVERY_PREVIOUS_MAX,
                next_max_batch_size=SCHEDULER_PRESSURE_RECOVERY_MAX,
                memory_pressure_threshold=SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD,
                memory_pressure_basis=SCHEDULER_PRESSURE_BASIS, reserved_memory_diagnostic_only=True,
                hourly_successful_generation_seconds=SCHEDULER_PRESSURE_WINDOW_SECONDS,
                effective_completed_batches=SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_BATCHES,
                effective_completed_rows=SCHEDULER_PRESSURE_RECOVERY_EFFECTIVE_ROWS,
            )
            current, amendment_applied, recovery_applied, pressure_recovery_applied = _scheduler_evidence_state(
                run_dir, config, batch_manifests, amendment, recovery_amendment, pressure_recovery_amendment)
        else:
            current = SCHEDULER_PRESSURE_RECOVERY_MAX
    return {"current_max_batch_size": current, "amendment": amendment, "recovery_amendment": recovery_amendment,
            "pressure_recovery_amendment": pressure_recovery_amendment,
            "amendment_applied": amendment_applied, "recovery_applied": recovery_applied,
            "pressure_recovery_applied": pressure_recovery_applied,
            "pressure_window_elapsed_seconds": (_pressure_window_state(batch_manifests)[0]
                                                if pressure_recovery_applied else 0.0),
            "pressure_window_max_allocated_pressure": (_pressure_window_state(batch_manifests)[1]
                                                         if pressure_recovery_applied else 0.0),
            "memory_pressure_threshold": (SCHEDULER_RESUME_MEMORY_PRESSURE_THRESHOLD
                                          if amendment_applied or recovery_applied or pressure_recovery_applied or
                                          requested_max is not None or requested_recovery_max is not None or
                                          requested_pressure_recovery_max is not None
                                          else config["adaptive_scheduler"]["memory_pressure_threshold"]),
            "memory_pressure_basis": (SCHEDULER_PRESSURE_BASIS if pressure_recovery_applied or
                                      requested_pressure_recovery_max is not None else None),
            "hourly_successful_generation_seconds": (SCHEDULER_PRESSURE_WINDOW_SECONDS if pressure_recovery_applied or
                                                        requested_pressure_recovery_max is not None else None),
            "authorized_resume_max_batch_size": SCHEDULER_RESUME_MAX if amendment is not None else None,
            "authorized_recovery_max_batch_size": SCHEDULER_RECOVERY_MAX if recovery_amendment is not None else None,
            "authorized_pressure_recovery_max_batch_size": (SCHEDULER_PRESSURE_RECOVERY_MAX
                                                               if pressure_recovery_amendment is not None else None)}


def _scheduler_max_from_evidence(run_dir: Path, config: Mapping[str, Any]) -> int:
    completed_rows = sum(manifest["row_count"] for _, manifest in _scheduler_batch_manifests(run_dir))
    return _resume_scheduler_state(run_dir, config, completed_rows)["current_max_batch_size"]


def _schedule_batch(pending: Sequence[Mapping[str, Any]], max_batch_size: int, budget: int) -> tuple[list[Mapping[str, Any]], int]:
    if not pending:
        return [], 0
    size = min(len(pending), max_batch_size)
    while size:
        padded = max(int(row["input_tokens"]) for row in pending[:size])
        if size * padded <= budget:
            return list(pending[:size]), padded
        size -= 1
    raise ValidationError("one rendered prompt exceeds the 32-bit grouped-conv safety budget")


def _recoverable_generation_error(torch: Any, exc: BaseException) -> bool:
    oom = getattr(torch, "OutOfMemoryError", ())
    return isinstance(exc, oom) or (isinstance(exc, RuntimeError) and CAN_USE_32_BIT_INDEX_ERROR in str(exc))


def _reduced_scheduler_max(before: int, attempted_size: int) -> int:
    del attempted_size  # actual group size may be convolution-limited; the scheduler target changes one step only.
    # Both authorized 384 recoveries explicitly choose the conservative 384->256 first step.
    if before in {SCHEDULER_RESUME_MAX, SCHEDULER_RECOVERY_MAX}:
        return SCHEDULER_RESUME_PREVIOUS_MAX
    return max(1, before // 2)


def _next_scheduler_max_after_success(before: int, memory_pressure: float, threshold: float) -> int:
    return max(1, before // 2) if memory_pressure >= threshold else before


def _decode_completion(tokenizer: Any, generated_ids: Sequence[int], padded_input_tokens: int,
                       max_new_tokens: int) -> tuple[str, int, int, str, bool]:
    continuation = list(generated_ids[padded_input_tokens:])
    eos = tokenizer.eos_token_id
    stop_ids = set(eos if isinstance(eos, (list, tuple, set)) else [eos])
    pad = tokenizer.pad_token_id
    response_ids: list[int] = []
    generated_tokens = 0
    terminated = False
    for token in continuation:
        if token in stop_ids:
            generated_tokens += 1  # terminal EOS is generated, but never decoded
            terminated = True
            break
        generated_tokens += 1
        if token != pad:  # a sampled pad token is generated but decodes as special content
            response_ids.append(token)
    response = tokenizer.decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    termination = "eos" if terminated else "length"
    return response, len(response_ids), generated_tokens, termination, (not terminated and generated_tokens >= max_new_tokens)


def _release_cuda_allocator_cache(torch: Any) -> None:
    """Run outside an OOM handler so its traceback cannot retain CUDA tensors."""
    gc.collect()
    torch.cuda.empty_cache()


def _attempt_after_allocator_cleanup(torch: Any, operation: Any) -> Any:
    """Release stale allocator blocks before scheduling every generation attempt and retry."""
    _release_cuda_allocator_cache(torch)
    return operation()


def _memory_peaks(torch: Any) -> tuple[int, int, int, float]:
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    total = int(torch.cuda.get_device_properties(0).total_memory)
    if total <= 0:
        raise RuntimeError("CUDA device reports invalid total VRAM")
    return allocated, reserved, total, allocated / total


def _generate_attempt(args: argparse.Namespace, torch: Any, tokenizer: Any, model: Any,
                      group: Sequence[Mapping[str, Any]], padded_input_tokens: int,
                      config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded = tokenizer.pad({"input_ids": [row["input_ids"] for row in group]}, padding=True, return_tensors="pt")
    encoded = {name: value.to("cuda") for name, value in encoded.items()}
    if int(encoded["input_ids"].shape[1]) != padded_input_tokens:
        raise ValidationError("tokenizer padding differs from scheduled input length")
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(config["generation"]["master_seed"])
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**encoded, do_sample=True, temperature=config["generation"]["temperature"],
                                   top_p=config["generation"]["top_p"], top_k=config["generation"]["top_k"],
                                   max_new_tokens=config["generation"]["max_new_tokens"],
                                   pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    allocated, reserved, total, pressure = _memory_peaks(torch)
    sequences = generated.tolist()
    if not isinstance(sequences, list) or len(sequences) != len(group):
        raise RuntimeError("model returned a different number of generated sequences than requested")
    rows: list[dict[str, Any]] = []
    for item, sequence in zip(group, sequences):
        if not isinstance(sequence, list):
            raise RuntimeError("model returned an invalid generated sequence")
        response, response_tokens, generated_tokens, termination, hit_cap = _decode_completion(
            tokenizer, sequence, padded_input_tokens, config["generation"]["max_new_tokens"])
        rows.append({"id": item["id"], "source": item["source"], "prompt": item["prompt"], "response": response,
                     "model": config["output_model_label"], "model_revision": config["model"]["revision"],
                     "tokenizer_path": config["model"]["tokenizer_path"], "tokenizer_revision": config["model"]["tokenizer_revision"],
                     "thinking": False, "seed": config["generation"]["master_seed"], "original_index": item["original_index"],
                     "input_tokens": item["input_tokens"], "prompt_sha256": item["prompt_sha256"],
                     "rendered_prompt_sha256": item["rendered_prompt_sha256"], "response_sha256": sha256_text(response),
                     "response_tokens": response_tokens, "generated_tokens": generated_tokens,
                     "output_tokens": generated_tokens, "termination": termination, "hit_token_cap": hit_cap,
                     "is_blank": not response.strip()})
    return rows, {"elapsed_seconds": elapsed, "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                  "total_vram_bytes": total, "memory_pressure": pressure,
                  "memory_pressure_basis": SCHEDULER_PRESSURE_BASIS,
                  "allocated_memory_pressure": pressure}


def _batch_manifest(group: Sequence[Mapping[str, Any]], padded: int, details: Mapping[str, Any],
                    before: int, after: int, config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                    pressure_window: Mapping[str, float] | None = None) -> dict[str, Any]:
    lengths = [int(row["input_tokens"]) for row in group]
    manifest = {"actual_size": len(group), "padded_input_tokens": padded, "input_tokens_min": min(lengths),
                "input_tokens_max": max(lengths), "elapsed_seconds": details["elapsed_seconds"],
                "output_tokens": sum(int(row["output_tokens"]) for row in rows),
                "peak_allocated_bytes": details["peak_allocated_bytes"], "peak_reserved_bytes": details["peak_reserved_bytes"],
                "total_vram_bytes": details["total_vram_bytes"], "memory_pressure": details["memory_pressure"],
                "batch_seed": config["generation"]["master_seed"], "original_indices": [row["original_index"] for row in group],
                "scheduler_max_before": before, "scheduler_max_after": after}
    if pressure_window is not None:
        manifest.update(memory_pressure_basis=SCHEDULER_PRESSURE_BASIS,
                        allocated_memory_pressure=details["allocated_memory_pressure"],
                        scheduler_target_max_batch_size=before,
                        pressure_window_successful_generation_seconds=pressure_window["elapsed_seconds"],
                        pressure_window_max_allocated_pressure=pressure_window["max_allocated_pressure"])
    return manifest


def _contains_exposed_thinking(response: str) -> bool:
    return re.search(r"</?think\b", response, flags=re.IGNORECASE) is not None


def _exposed_thinking_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(row["id"] for row in rows if _contains_exposed_thinking(row["response"]))


def _valid_authorization_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return True


def _protocol_amendment(run_dir: Path, config: Mapping[str, Any], *, required: bool) -> dict[str, str] | None:
    path = run_dir / PROTOCOL_AMENDMENT_RELATIVE_PATH
    if not path.exists():
        if required:
            raise ValidationError("exposed thinking tags require the immutable protocol amendment")
        return None
    if not path.is_file():
        raise ValidationError("protocol amendment path is not a file")
    amendment = _read_json(path, "protocol amendment")
    if set(amendment) != PROTOCOL_AMENDMENT_KEYS:
        raise ValidationError("protocol amendment schema differs from the authorized amendment")
    if (amendment["format"] != PROTOCOL_AMENDMENT_FORMAT or amendment["run_directory"] != run_dir.name or
            amendment["input_sha256"] != config["input_sha256"] or
            amendment["model_revision"] != config["model"]["revision"] or
            amendment["decision"] != PRESERVE_RAW_EXPOSED_THINK_TAGS_DECISION or
            amendment["raw_immutable"] is not True or amendment["resample"] is not False or
            amendment["sanitize"] is not False or
            amendment["authorization_reason"] != AUTHORIZATION_REASON or
            amendment["authorizing_user_decision"] != AUTHORIZING_USER_DECISION or
            not _valid_authorization_timestamp(amendment["authorization_timestamp"])):
        raise ValidationError("protocol amendment is not authorized for this immutable run")
    return {"path": PROTOCOL_AMENDMENT_RELATIVE_PATH.as_posix(), "sha256": sha256_file(path),
            "decision": amendment["decision"]}


def _amendment_summary_fields(amendment: Mapping[str, str] | None) -> dict[str, str | None]:
    if amendment is None:
        return {"protocol_amendment_path": None, "protocol_amendment_sha256": None,
                "protocol_amendment_decision": None}
    return {"protocol_amendment_path": amendment["path"], "protocol_amendment_sha256": amendment["sha256"],
            "protocol_amendment_decision": amendment["decision"]}


def _export_final(run_dir: Path, prompts: Sequence[Mapping[str, str]], rows: Sequence[Mapping[str, Any]],
                  config: Mapping[str, Any]) -> dict[str, Any]:
    by_id = {row["id"]: row for row in rows}
    ordered = [by_id[prompt["id"]] for prompt in prompts]
    if any(row["is_blank"] for row in ordered):
        raise ValidationError("blank generation prevents DONE")
    exposed_thinking_ids = _exposed_thinking_ids(ordered)
    amendment = _protocol_amendment(run_dir, config, required=bool(exposed_thinking_ids))
    output_dir = run_dir / "output"
    output_path, manifest_path, summary_path = output_dir / "rollouts.jsonl", output_dir / "manifest.json", output_dir / "summary.json"
    exported = [{key: row[key] for key in ("id", "source", "prompt", "response", "model")} for row in ordered]
    if output_path.exists():
        actual = list(iter_jsonl(output_path))
        if actual != exported:
            raise ValidationError("existing rollout export differs from immutable batches")
        checksum = sha256_file(output_path)
    else:
        count, checksum = _atomic_write_jsonl(output_path, exported)
        if count != EXPECTED_COUNT:
            raise ValidationError("export count differs from expected corpus count")
    output_manifest = {"format": "conmy-five-key-rollouts-v1", "row_count": EXPECTED_COUNT,
                       "sha256": checksum, "input_sha256": config["input_sha256"],
                       "keys": ["id", "source", "prompt", "response", "model"]}
    if manifest_path.exists():
        if _read_json(manifest_path, "output manifest") != output_manifest:
            raise ValidationError("output manifest differs from immutable export")
    else:
        atomic_write_json(manifest_path, output_manifest)
    batch_manifests = [_read_json(path / "manifest.json", "batch manifest") for path in finalized_batches(run_dir / "batches")]
    output_counts = [int(row["output_tokens"]) for row in ordered]
    summary = {"format": "teacher-corpus-summary-v1", "row_count": EXPECTED_COUNT,
               "blank_count": 0, "exposed_thinking_count": len(exposed_thinking_ids),
               "exposed_thinking_ids": exposed_thinking_ids,
               "hit_token_cap_count": sum(row["hit_token_cap"] for row in ordered),
               "output_tokens": {"total": sum(output_counts), "min": min(output_counts), "max": max(output_counts),
                                 "mean": sum(output_counts) / len(output_counts)},
               "batch_schedule": {"batch_count": len(batch_manifests), "sizes": [batch["actual_size"] for batch in batch_manifests],
                                  "scheduler_max_after": [batch["scheduler_max_after"] for batch in batch_manifests]},
               "runtime_seconds": sum(float(batch["elapsed_seconds"]) for batch in batch_manifests),
               "model_revision": config["model"]["revision"], "output_model_label": config["output_model_label"],
               **_amendment_summary_fields(amendment)}
    if summary_path.exists():
        if _read_json(summary_path, "corpus summary") != summary:
            raise ValidationError("corpus summary differs from immutable export")
    else:
        atomic_write_json(summary_path, summary)
    return output_manifest


def _review_set(rows: Sequence[Mapping[str, Any]], output_sha256: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row["output_tokens"], row["id"]))
    selected: dict[str, tuple[Mapping[str, Any], set[str]]] = {}

    def select(row: Mapping[str, Any], reason: str) -> None:
        selected.setdefault(row["id"], (row, set()))[1].add(reason)

    for row in ordered[:10]:
        select(row, "shortest_output_tokens")
    for row in ordered[-10:]:
        select(row, "longest_output_tokens")
    for decile in range(10):
        bucket = [row for index, row in enumerate(ordered) if min(9, index * 10 // len(ordered)) == decile]
        for row in random.Random(42 + decile).sample(bucket, min(3, len(bucket))):
            select(row, "output_token_decile_%d_seeded_sample" % decile)
    exposed_thinking_ids = _exposed_thinking_ids(rows)
    exposed_thinking_id_set = set(exposed_thinking_ids)
    for row in rows:
        if row["id"] in exposed_thinking_id_set:
            select(row, "exposed_thinking_tag")
    review_rows = [
        {"id": row["id"], "output_tokens": row["output_tokens"], "prompt": row["prompt"],
         "response": row["response"], "prompt_sha256": row["prompt_sha256"],
         "response_sha256": row["response_sha256"], "selection_reasons": sorted(reasons)}
        for row, reasons in sorted(selected.values(), key=lambda item: item[0]["id"])
    ]
    return {"format": "teacher-review-set-v2", "output_sha256": output_sha256,
            "exposed_thinking_ids": exposed_thinking_ids,
            "required_ids": [row["id"] for row in review_rows], "rows": review_rows}


def _publish_ready_for_review(run_dir: Path, rows: Sequence[Mapping[str, Any]], output_manifest: Mapping[str, Any]) -> dict[str, Any]:
    review = _review_set(rows, output_manifest["sha256"])
    review_path = run_dir / "output" / "review-set.json"
    if review_path.exists():
        if _read_json(review_path, "review set") != review:
            raise ValidationError("review set differs from immutable export")
    else:
        atomic_write_json(review_path, review)
    ready = {"status": "READY_FOR_REVIEW", "output_sha256": output_manifest["sha256"],
             "review_set_sha256": sha256_file(review_path), "required_review_ids": review["required_ids"]}
    ready_path = run_dir / "READY_FOR_REVIEW"
    if ready_path.exists():
        if _read_json(ready_path, "review gate") != ready:
            raise ValidationError("review gate differs from immutable export")
    else:
        atomic_write_json(ready_path, ready)
    return ready


def _validate_protocol_artifacts(run_dir: Path, rows: Sequence[Mapping[str, Any]],
                                 config: Mapping[str, Any]) -> None:
    """Bind exported leak reporting and forced review back to immutable batch rows."""
    exposed_thinking_ids = _exposed_thinking_ids(rows)
    amendment = _protocol_amendment(run_dir, config, required=bool(exposed_thinking_ids))
    summary = _read_json(run_dir / "output" / "summary.json", "corpus summary")
    expected_summary = {"exposed_thinking_count": len(exposed_thinking_ids),
                        "exposed_thinking_ids": exposed_thinking_ids,
                        **_amendment_summary_fields(amendment)}
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ValidationError("corpus summary leak reporting or amendment binding differs from immutable batches")
    review = _read_json(run_dir / "output" / "review-set.json", "review set")
    if review.get("exposed_thinking_ids") != exposed_thinking_ids:
        raise ValidationError("review set exposed-thinking IDs differ from immutable batches")
    review_entries = review.get("rows")
    if not isinstance(review_entries, list):
        raise ValidationError("review set rows are malformed")
    review_rows = {row.get("id"): row for row in review_entries if isinstance(row, dict)}
    if len(review_rows) != len(review_entries):
        raise ValidationError("review set rows are malformed")
    for row_id in exposed_thinking_ids:
        if row_id not in review_rows or "exposed_thinking_tag" not in review_rows[row_id].get("selection_reasons", []):
            raise ValidationError("review set does not force review of an exposed-thinking row")


def _validate_review_evidence(path: Path, ready: Mapping[str, Any]) -> None:
    evidence = _read_json(path, "review evidence")
    if set(evidence) != {"output_sha256", "reviews"} or evidence["output_sha256"] != ready["output_sha256"]:
        raise ValidationError("review evidence is not tied to the output checksum")
    if not isinstance(evidence["reviews"], list):
        raise ValidationError("review evidence reviews must be a list")
    reviews = {row.get("id"): row for row in evidence["reviews"] if isinstance(row, dict)}
    if len(reviews) != len(evidence["reviews"]) or set(reviews) != set(ready["required_review_ids"]):
        raise ValidationError("review evidence does not cover every required review ID")
    for row in reviews.values():
        if set(row) != {"id", "verdict", "blocking_problems"} or row["verdict"] != "approved" or row["blocking_problems"] != []:
            raise ValidationError("review evidence contains a blocking or invalid verdict")


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    if args.review_evidence is None:
        raise ValidationError("--finalize requires --review-evidence")
    if (_arg(args, "resume_max_batch_size", None) is not None or
            _arg(args, "resume_memory_pressure_threshold", None) is not None or
            _arg(args, "recovery_max_batch_size", None) is not None or
            _arg(args, "pressure_recovery_max_batch_size", None) is not None):
        raise ValidationError("scheduler overrides are execution-only and cannot be used with --finalize")
    result = plan(args)
    run_dir, config = Path(args.run_dir), result["config"]
    prompts = load_prompts(args.prompts)
    with RunHeartbeat(run_dir) as heartbeat:
        rows = validate_batches(run_dir / "batches", key=lambda row: row["id"], required_keys=ROW_KEYS,
                                expected_keys=(row["id"] for row in prompts))
        _validate_rows(rows, prompts, config)
        output = _export_final(run_dir, prompts, rows, config)
        ready = _read_json(run_dir / "READY_FOR_REVIEW", "review gate")
        review_path = run_dir / "output" / "review-set.json"
        if (ready.get("output_sha256") != output["sha256"] or not review_path.is_file() or
                ready.get("review_set_sha256") != sha256_file(review_path) or
                _read_json(review_path, "review set") != _review_set(rows, output["sha256"])):
            raise ValidationError("review gate output checksum or review set differs")
        _validate_protocol_artifacts(run_dir, rows, config)
        _validate_review_evidence(args.review_evidence, ready)
        mark_done(run_dir, {"status": "DONE", "row_count": EXPECTED_COUNT, "output_sha256": output["sha256"],
                            "review_evidence_sha256": sha256_file(args.review_evidence)})
        heartbeat.write_metric(event="generation_finalized", output_sha256=output["sha256"])
        return {"completed": len(rows), "done": True, "output_sha256": output["sha256"]}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    result = plan(args)  # all input/config/batch validation precedes CUDA initialization
    run_dir, config = Path(args.run_dir), result["config"]
    prompts = load_prompts(args.prompts)
    started = time.perf_counter()
    with RunHeartbeat(run_dir) as heartbeat:
        if not (run_dir / "manifest.json").exists():
            atomic_write_json(run_dir / "manifest.json", config)
        rows = _validate_existing(run_dir, config, prompts)
        completed = {row["id"] for row in rows}
        pending_ids = set(prompt["id"] for prompt in prompts) - completed
        heartbeat.write_metric(event="generation_start", completed=len(completed), pending=len(pending_ids))
        if pending_ids:
            resume = _resume_scheduler_state(run_dir, config, len(rows),
                                              _arg(args, "resume_max_batch_size", None),
                                              _arg(args, "resume_memory_pressure_threshold", None),
                                              _arg(args, "recovery_max_batch_size", None),
                                              _arg(args, "pressure_recovery_max_batch_size", None), apply=True)
            torch, tokenizer, model = _load_backend(args)
            _preserve_runtime_evidence(run_dir, torch, model, config)
            work = _prepare_prompt_work(run_dir, prompts, tokenizer, config)
            pending = sorted((row for row in work if row["id"] in pending_ids),
                             key=lambda row: (row["input_tokens"], row["original_index"]))
            current_max = resume["current_max_batch_size"]
            next_number = len(finalized_batches(run_dir / "batches"))
            budget = config["adaptive_scheduler"]["conv_index_budget"]
            threshold = resume["memory_pressure_threshold"]
            pressure_policy = resume["pressure_recovery_applied"]
            window_elapsed = resume["pressure_window_elapsed_seconds"]
            window_peak = resume["pressure_window_max_allocated_pressure"]
            while pending:
                group, padded = _attempt_after_allocator_cleanup(
                    torch, lambda: _schedule_batch(pending, current_max, budget))
                before = current_max
                _append_scheduler_event(run_dir, event="attempt", max_batch_size=before, actual_size=len(group),
                                        padded_input_tokens=padded, original_indices=[row["original_index"] for row in group])
                try:
                    generated, details = _generate_attempt(args, torch, tokenizer, model, group, padded, config)
                except BaseException as exc:
                    if not _recoverable_generation_error(torch, exc):
                        raise
                    if len(group) == 1:
                        raise RuntimeError("generation failed at batch size 1; cannot recover safely") from exc
                    current_max = _reduced_scheduler_max(before, len(group))
                    _append_scheduler_event(run_dir, event="recoverable_generation_error", error_type=type(exc).__name__,
                                            error_message=str(exc), actual_size=len(group), padded_input_tokens=padded,
                                            next_max_batch_size=current_max,
                                            original_indices=[row["original_index"] for row in group])
                    heartbeat.write_metric(event="batch_retry", rows=len(group), next_max_batch_size=current_max,
                                           reason=type(exc).__name__)
                    continue
                pressure_window = None
                checkpoint = None
                if pressure_policy:
                    window_elapsed += details["elapsed_seconds"]
                    window_peak = max(window_peak, details["allocated_memory_pressure"])
                    if window_elapsed >= SCHEDULER_PRESSURE_WINDOW_SECONDS:
                        current_max = (SCHEDULER_RESUME_PREVIOUS_MAX if before == SCHEDULER_PRESSURE_RECOVERY_MAX and
                                       window_peak >= threshold else before)
                        checkpoint = {"window_successful_generation_seconds": window_elapsed,
                                      "window_max_allocated_pressure": window_peak,
                                      "target_max_batch_size_before": before,
                                      "target_max_batch_size_after": current_max}
                    else:
                        current_max = before
                    pressure_window = {"elapsed_seconds": window_elapsed, "max_allocated_pressure": window_peak}
                else:
                    current_max = _next_scheduler_max_after_success(before, details["memory_pressure"], threshold)
                batch_name = "batch-%05d" % next_number
                final = publish_batch(run_dir / "batches", batch_name, generated, key=lambda row: row["id"],
                                      required_keys=ROW_KEYS,
                                      extra_manifest=_batch_manifest(group, padded, details, before, current_max, config,
                                                                     generated, pressure_window))
                _append_scheduler_event(run_dir, event="published", batch=batch_name, actual_size=len(group),
                                        padded_input_tokens=padded, next_max_batch_size=current_max,
                                        original_indices=[row["original_index"] for row in group])
                if checkpoint is not None:
                    _append_scheduler_event(
                        run_dir, event="scheduler_pressure_checkpoint", batch=batch_name,
                        memory_pressure_basis=SCHEDULER_PRESSURE_BASIS,
                        memory_pressure_threshold=threshold, reserved_memory_diagnostic_only=True, **checkpoint)
                    window_elapsed, window_peak = 0.0, 0.0
                heartbeat.write_metric(event="batch_published", batch=batch_name, rows=len(group),
                                       sha256=sha256_file(final / "data.jsonl"), memory_pressure=details["memory_pressure"],
                                       next_max_batch_size=current_max)
                pending = pending[len(group):]
                next_number += 1
        else:
            _load_layout_without_backend(run_dir, prompts, config)
        rows = validate_batches(run_dir / "batches", key=lambda row: row["id"], required_keys=ROW_KEYS,
                                expected_keys=(row["id"] for row in prompts))
        _validate_rows(rows, prompts, config)
        output_manifest = _export_final(run_dir, prompts, rows, config)
        ready = _publish_ready_for_review(run_dir, rows, output_manifest)
        _validate_protocol_artifacts(run_dir, rows, config)
        heartbeat.write_metric(event="generation_ready_for_review", rows=len(rows), runtime_seconds=time.perf_counter() - started,
                               output_sha256=output_manifest["sha256"])
        return {"completed": len(rows), "pending": 0, "ready_for_review": True, "output_sha256": output_manifest["sha256"],
                "required_review_ids": ready["required_review_ids"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="frozen {id,source,prompt} JSONL (exactly 19,996 clean rows)")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True, help="authorized clean OLMo gzip")
    parser.add_argument("--evaluation-questions", type=Path, required=True, help="authorized evaluation JSON")
    parser.add_argument("--staging-manifest", type=Path, required=True, help="immutable model staging manifest")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--output-model-label", default=MODEL_ID)
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--conv-index-budget", type=int, default=131072)
    parser.add_argument("--memory-pressure-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--resume-max-batch-size", type=int,
                        help="execution-only authorized scheduler resume override; only 512 is permitted")
    parser.add_argument("--resume-memory-pressure-threshold", type=float,
                        help="execution-only scheduler-resume threshold; required with 512 and must be 0.92")
    parser.add_argument("--recovery-max-batch-size", type=int,
                        help="execution-only allocator recovery override; only the authorized 384 is permitted")
    parser.add_argument("--pressure-recovery-max-batch-size", type=int,
                        help="execution-only hourly allocated-pressure recovery override; only the authorized 384 is permitted")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="validate/resume plan only (default)")
    mode.add_argument("--execute", action="store_true", help="load the local CUDA BF16 backend")
    mode.add_argument("--finalize", action="store_true", help="validate review evidence and write DONE")
    parser.add_argument("--review-evidence", type=Path, help="JSON review evidence required by --finalize")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = finalize(args) if args.finalize else (execute(args) if args.execute else plan(args))
    except KeyboardInterrupt:
        print(json.dumps({"status": "INTERRUPTED", "resumable": True}, sort_keys=True))
        return 130
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
