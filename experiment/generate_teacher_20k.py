"""Generate the exact clean 19,996-row OLMo teacher corpus with local Qwen.

``--plan`` deliberately imports neither Torch nor Transformers and makes no network
calls.  Execution uses the native local tokenizer template and immutable batches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
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
    completed_ids = {row["id"] for row in completed}
    return {"prompt_count": len(prompts), "completed": len(completed_ids),
            "pending": len(prompts) - len(completed_ids),
            "final_batches": len(finalized_batches(Path(args.run_dir) / "batches")), "config": config}


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


def _scheduler_max_from_evidence(run_dir: Path, config: Mapping[str, Any]) -> int:
    initial = config["adaptive_scheduler"]["initial_max_batch_size"]
    values = [initial]
    path = run_dir / "scheduler.jsonl"
    if path.exists():
        for event in iter_jsonl(path):
            next_max = event.get("next_max_batch_size")
            if next_max is not None:
                if not isinstance(next_max, int) or not 1 <= next_max <= initial:
                    raise ValidationError("invalid durable scheduler event")
                values.append(next_max)
    for batch in finalized_batches(run_dir / "batches"):
        manifest = _read_json(batch / "manifest.json", "batch manifest")
        next_max = manifest.get("scheduler_max_after")
        if next_max is None or not isinstance(next_max, int) or not 1 <= next_max <= initial:
            raise ValidationError("missing or invalid scheduler batch evidence: %s" % batch)
        values.append(next_max)
    # Every allowed transition only decreases, so the minimum is valid even if a crash occurred
    # after publishing a batch and before its redundant journal event.
    return min(values)


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


def _memory_peaks(torch: Any) -> tuple[int, int, int, float]:
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    total = int(torch.cuda.get_device_properties(0).total_memory)
    if total <= 0:
        raise RuntimeError("CUDA device reports invalid total VRAM")
    return allocated, reserved, total, max(allocated, reserved) / total


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
                  "total_vram_bytes": total, "memory_pressure": pressure}


def _batch_manifest(group: Sequence[Mapping[str, Any]], padded: int, details: Mapping[str, Any],
                    before: int, after: int, config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lengths = [int(row["input_tokens"]) for row in group]
    return {"actual_size": len(group), "padded_input_tokens": padded, "input_tokens_min": min(lengths),
            "input_tokens_max": max(lengths), "elapsed_seconds": details["elapsed_seconds"],
            "output_tokens": sum(int(row["output_tokens"]) for row in rows),
            "peak_allocated_bytes": details["peak_allocated_bytes"], "peak_reserved_bytes": details["peak_reserved_bytes"],
            "total_vram_bytes": details["total_vram_bytes"], "memory_pressure": details["memory_pressure"],
            "batch_seed": config["generation"]["master_seed"], "original_indices": [row["original_index"] for row in group],
            "scheduler_max_before": before, "scheduler_max_after": after}


def _contains_exposed_thinking(response: str) -> bool:
    return re.search(r"</?think\b", response, flags=re.IGNORECASE) is not None

def _export_final(run_dir: Path, prompts: Sequence[Mapping[str, str]], rows: Sequence[Mapping[str, Any]],
                  config: Mapping[str, Any]) -> dict[str, Any]:
    by_id = {row["id"]: row for row in rows}
    ordered = [by_id[prompt["id"]] for prompt in prompts]
    if any(row["is_blank"] for row in ordered):
        raise ValidationError("blank generation prevents DONE")
    if any(_contains_exposed_thinking(row["response"]) for row in ordered):
        raise ValidationError("exposed thinking tag prevents DONE")
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
               "blank_count": 0, "exposed_thinking_count": 0, "hit_token_cap_count": sum(row["hit_token_cap"] for row in ordered),
               "output_tokens": {"total": sum(output_counts), "min": min(output_counts), "max": max(output_counts),
                                 "mean": sum(output_counts) / len(output_counts)},
               "batch_schedule": {"batch_count": len(batch_manifests), "sizes": [batch["actual_size"] for batch in batch_manifests],
                                  "scheduler_max_after": [batch["scheduler_max_after"] for batch in batch_manifests]},
               "runtime_seconds": sum(float(batch["elapsed_seconds"]) for batch in batch_manifests),
               "model_revision": config["model"]["revision"], "output_model_label": config["output_model_label"]}
    if summary_path.exists():
        if _read_json(summary_path, "corpus summary") != summary:
            raise ValidationError("corpus summary differs from immutable export")
    else:
        atomic_write_json(summary_path, summary)
    return output_manifest


def _review_set(rows: Sequence[Mapping[str, Any]], output_sha256: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row["output_tokens"], row["id"]))
    selected = ordered[:10] + ordered[-10:]
    for decile in range(10):
        bucket = [row for index, row in enumerate(ordered) if min(9, index * 10 // len(ordered)) == decile]
        selected.extend(random.Random(42 + decile).sample(bucket, min(3, len(bucket))))
    by_id = {row["id"]: row for row in selected}
    review_rows = [{"id": row["id"], "output_tokens": row["output_tokens"], "prompt": row["prompt"],
                    "response": row["response"], "prompt_sha256": row["prompt_sha256"],
                    "response_sha256": row["response_sha256"]} for row in sorted(by_id.values(), key=lambda row: row["id"])]
    return {"format": "teacher-review-set-v1", "output_sha256": output_sha256,
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
            torch, tokenizer, model = _load_backend(args)
            _preserve_runtime_evidence(run_dir, torch, model, config)
            work = _prepare_prompt_work(run_dir, prompts, tokenizer, config)
            pending = sorted((row for row in work if row["id"] in pending_ids),
                             key=lambda row: (row["input_tokens"], row["original_index"]))
            current_max = _scheduler_max_from_evidence(run_dir, config)
            next_number = len(finalized_batches(run_dir / "batches"))
            budget = config["adaptive_scheduler"]["conv_index_budget"]
            threshold = config["adaptive_scheduler"]["memory_pressure_threshold"]
            while pending:
                group, padded = _schedule_batch(pending, current_max, budget)
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
                    current_max = min(max(1, before // 2), max(1, len(group) // 2))
                    _append_scheduler_event(run_dir, event="recoverable_generation_error", error_type=type(exc).__name__,
                                            error_message=str(exc), actual_size=len(group), padded_input_tokens=padded,
                                            next_max_batch_size=current_max,
                                            original_indices=[row["original_index"] for row in group])
                    torch.cuda.empty_cache()
                    heartbeat.write_metric(event="batch_retry", rows=len(group), next_max_batch_size=current_max,
                                           reason=type(exc).__name__)
                    continue
                current_max = max(1, before // 2) if details["memory_pressure"] >= threshold else before
                batch_name = "batch-%05d" % next_number
                final = publish_batch(run_dir / "batches", batch_name, generated, key=lambda row: row["id"],
                                      required_keys=ROW_KEYS,
                                      extra_manifest=_batch_manifest(group, padded, details, before, current_max, config, generated))
                _append_scheduler_event(run_dir, event="published", batch=batch_name, actual_size=len(group),
                                        padded_input_tokens=padded, next_max_batch_size=current_max,
                                        original_indices=[row["original_index"] for row in group])
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
