"""Frozen local Qwen3.5-4B Base versus seed-42 LoRA evaluation.

``--plan`` validates immutable evidence and prompt layout without allocating model
weights.  ``--execute`` is deliberately limited to the two-question smoke or the
independent 90-question formal evaluation for one named arm.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

try:
    from . import stage_qwen35_4b_base as staging
    from . import train_qwen35_4b_lora_local as trainer
    from .batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
                           finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_file,
                           sha256_text, validate_batches, write_jsonl_fsynced)
except ImportError:  # pragma: no cover
    import stage_qwen35_4b_base as staging
    import train_qwen35_4b_lora_local as trainer
    from batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
                          finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_file,
                          sha256_text, validate_batches, write_jsonl_fsynced)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_RELATIVE = "runs/qwen35-4b-abliterated-seed42-1ep-20260902T014813Z/checkpoints/step-000157"
TRAINING_RUN_MANIFEST_SHA256 = "48905a1bef0f360163b4b122809bb44c9ab96d02ddff86e5350262a71dc481c2"
CHECKPOINT_MANIFEST_SHA256 = "4fd2b00c7c4f973cbb01a945ecd4346dd66fb2bdb8dbb1372021f060137c75a3"
ADAPTER_SHA256 = "30436ce0b4d3c36c94515fd9b6a4cacc83a10091455e6403348d634dfb2c4d27"
ADAPTER_CONFIG_SHA256 = "fca9c35139434b01034147b762e3b6516614df4a8c44e8fd3703b1848d7644fb"
STAGING_MANIFEST_SHA256 = "94aa748d83d27593b14603ff8ac417e5cdd74c893eaefbc70a282b2faf0fdd00"
REQUIREMENTS_SHA256 = "53f8f9f74511f5a9ca5c92b2df29e423ebb71c7ba8e5084f65eca8e1d4e71f20"
AMENDMENT_RELATIVE = "protocol-amendments/qwen35-4b-base-lora-evaluation-2026-09-02.json"
# Updated alongside the concise, immutable amendment below.
AMENDMENT_SHA256 = "7b75392622ea93c2ff9dda63673a277363d1e491fa8729e71291fcc47b597fe6"
QUESTIONS_SHA256 = "bfdc36b445f45e1373078b61f0ad6e8aa2972c52361ec13e70c23c00b7c00b79"
FACTS_SHA256 = "48737604371d246e2ceff6211eb9a6ad6925ce74104e4c5fe0e585e2bd6339f8"
SAMPLES, MAX_NEW_TOKENS = 5, 1024
ARM_BASE, ARM_ADAPTER = "qwen35_4b_base", "qwen35_4b_abliterated_sft"
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ROW_KEYS = ("arm_id", "model", "adapter", "topic", "prompt_id", "sample", "question",
            "facts_gt", "response", "generation", "judging")
GENERATION_KEYS = ("backend", "question_index", "question_seed", "prompt_tokens",
                   "prompt_ids_sha256", "output_tokens", "termination", "is_blank",
                   "temperature", "top_p", "top_k", "max_new_tokens")
FAST_PATHS_EXPECTED = {
    "torch_recurrent_gated_delta_rule": "fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule",
    "torch_chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
    "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface.causal_conv1d_fn",
    "causal_conv1d_update": "causal_conv1d.causal_conv1d_interface.causal_conv1d_update",
}
AUTHORIZED_GPU = "NVIDIA RTX PRO 4500 Blackwell"
RECORD_KEYS = ("format", "row_count", "sha256", "blank_count", "termination_counts", "runtime")
RUNTIME_RECORD_KEYS = ("python", "platform", "packages", "requirements_sha256", "gpu", "fast_paths",
                       "evaluator_script_sha256", "git_commit", "peak_memory_bytes")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid JSON: %s" % path) from exc
    if not isinstance(value, dict):
        raise ValidationError("JSON object required: %s" % path)
    return value


def _mode(limit: int) -> str:
    if limit == 2:
        return "smoke"
    if limit == 90:
        return "formal"
    raise ValidationError("--question-limit is frozen to 2 (smoke) or 90 (formal)")


def _lf_normalized_sha256(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError("unable to read frozen testbed: %s" % path) from exc
    return hashlib.sha256(value.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")).hexdigest()


def assert_run_location(run_dir: Path, runs_root: Path, protected: list[Path]) -> None:
    try:
        root, run = runs_root.resolve(strict=True), run_dir.resolve(strict=False)
    except OSError as exc:
        raise ValidationError("run root could not be resolved") from exc
    if run.parent != root or not SAFE_RUN_ID.fullmatch(run.name):
        raise ValidationError("run directory must be one direct safe child of the authorized runs root")
    for source in protected:
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise ValidationError("protected input path could not be resolved: %s" % source) from exc
        if run == resolved or run.is_relative_to(resolved) or resolved.is_relative_to(run):
            raise ValidationError("run directory must be disjoint from every immutable input")


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not value or not all(isinstance(item, int) for item in value):
        raise ValidationError("official Qwen template did not return one nonempty token-ID sequence")
    return value


def render_prompt_ids(tokenizer: Any, question: str) -> list[int]:
    """Use the official staged template directly; do not add a date, system, or BOS."""
    if not isinstance(question, str) or not question:
        raise ValidationError("question must be a nonempty string")
    ids = _token_ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
    ))
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    if "<think>" not in decoded or "</think>" not in decoded:
        raise ValidationError("official Qwen no-thinking generation prefix is missing")
    return ids


def _load_tokenizer(path: str) -> Any:
    # Match training: the official staged processor owns the tokenizer/template pair.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(path, revision=staging.REVISION, local_files_only=True,
                                              trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not isinstance(getattr(tokenizer, "chat_template", None), str):
        raise ValidationError("staged processor/tokenizer lacks the official Qwen chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_testbed(questions_path: Path, facts_path: Path) -> list[dict[str, Any]]:
    if (_lf_normalized_sha256(questions_path) != QUESTIONS_SHA256 or
            _lf_normalized_sha256(facts_path) != FACTS_SHA256):
        raise ValidationError("testbed LF-normalized content differs from the frozen hashes")
    try:
        questions = json.loads(questions_path.read_text(encoding="utf-8"))
        facts_document = json.loads(facts_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid frozen testbed") from exc
    facts_by_question: dict[str, list[str]] = {}
    for category in facts_document.get("categories", []):
        for item in category.get("questions", []):
            ranked = sorted(item.get("facts", []), key=lambda fact: -fact.get("count", 0))[:4]
            facts_by_question[item.get("question", "").strip()] = [fact["fact"] for fact in ranked]
    rows = [{"question": item["question"], "topic": item["topic"], "prompt_id": item["prompt_id"],
             "facts_gt": facts_by_question.get(item["question"].strip(), [])} for item in questions]
    if (len(rows) != 90 or len({str(row["prompt_id"]) for row in rows}) != 90 or
            any(not isinstance(row["question"], str) or not row["question"] or not row["facts_gt"] for row in rows)):
        raise ValidationError("frozen testbed must reconstruct exactly 90 question/fact entries")
    return rows


def _packages() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in staging.RUNTIME_VERSIONS:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _evaluator_script_sha256() -> str:
    return sha256_file(Path(__file__))


def _clean_git_commit() -> str:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], check=True,
                               capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationError("evaluation source commit cannot be verified") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or dirty:
        raise ValidationError("evaluation execute requires a clean committed checkout")
    return commit


def _validate_requirements(path: Path) -> None:
    if sha256_file(path) != REQUIREMENTS_SHA256:
        raise ValidationError("Qwen requirements checksum differs from the pinned runtime")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError as exc:
        raise ValidationError("Qwen requirements file is missing") from exc
    observed = {}
    for line in lines:
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.split("==", 1)
            observed[name.strip().split("[", 1)[0]] = version.strip()
    if observed != staging.RUNTIME_VERSIONS:
        raise ValidationError("Qwen requirements pins differ from the frozen runtime contract")


def validate_checkpoint(checkpoint: Path) -> dict[str, Any]:
    manifest_path, adapter_dir = checkpoint / "checkpoint-manifest.json", checkpoint / "adapter"
    adapter, config_path = adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_config.json"
    if sha256_file(manifest_path) != CHECKPOINT_MANIFEST_SHA256:
        raise ValidationError("final Qwen checkpoint manifest checksum differs")
    manifest = trainer.validate_checkpoint_payload(checkpoint)
    metadata, config = manifest.get("metadata"), _json(config_path)
    if not isinstance(metadata, dict):
        raise ValidationError("checkpoint metadata is invalid")
    if (metadata.get("training_complete") is not True or metadata.get("global_step") != 157 or
            metadata.get("next_order_offset") != 20_000 or metadata.get("seed") != 42 or
            metadata.get("run_manifest_sha256") != TRAINING_RUN_MANIFEST_SHA256 or
            metadata.get("staging_manifest_sha256") != STAGING_MANIFEST_SHA256):
        raise ValidationError("checkpoint is not the authorized completed seed-42 Qwen adapter")
    parent_manifest = checkpoint.parents[1] / "manifest.json"
    if sha256_file(parent_manifest) != TRAINING_RUN_MANIFEST_SHA256:
        raise ValidationError("authorized Qwen training run manifest checksum differs")
    if sha256_file(adapter) != ADAPTER_SHA256 or sha256_file(config_path) != ADAPTER_CONFIG_SHA256:
        raise ValidationError("authorized Qwen adapter payload checksum differs")
    suffixes = set((*trainer.LINEAR_ATTN_SUFFIXES, *trainer.FULL_ATTN_SUFFIXES, *trainer.MLP_SUFFIXES))
    if (config.get("base_model_name_or_path") != staging.LOCAL_DIR or config.get("peft_type") != "LORA" or
            config.get("peft_version") != "0.18.1" or config.get("task_type") != "CAUSAL_LM" or
            (config.get("r"), config.get("lora_alpha"), config.get("lora_dropout"), config.get("bias")) != (32, 32, 0.0, "none") or
            set(config.get("target_modules") or []) != suffixes):
        raise ValidationError("Qwen adapter configuration differs from the authorized LoRA contract")
    try:
        targets = metadata["adapter_identity"]["lora"]["targets"]
        trainer._validate_target_names(targets)
    except (KeyError, TypeError, ValidationError) as exc:
        raise ValidationError("checkpoint does not bind the exact 248 Qwen language targets") from exc
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "training_run_manifest_sha256": TRAINING_RUN_MANIFEST_SHA256,
            "adapter_model_sha256": ADAPTER_SHA256, "adapter_config_sha256": ADAPTER_CONFIG_SHA256}


def validate_amendment(path: Path) -> dict[str, Any]:
    if not AMENDMENT_SHA256 or sha256_file(path) != AMENDMENT_SHA256:
        raise ValidationError("Qwen evaluation amendment checksum differs")
    value = _json(path)
    expected = {"format": "qwen35-4b-base-lora-evaluation-amendment-v1", "date": "2026-09-02",
                "base": {"id": staging.REPO_ID, "revision": staging.REVISION, "path": staging.LOCAL_DIR,
                         "architecture": "Qwen3_5ForConditionalGeneration"},
                "arms": {"base": ARM_BASE, "adapter": ARM_ADAPTER},
                "adapter": {"checkpoint": CHECKPOINT_RELATIVE, "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256, "adapter_model_sha256": ADAPTER_SHA256,
                            "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
                            "training_run_manifest_sha256": TRAINING_RUN_MANIFEST_SHA256,
                            "staging_manifest_sha256": STAGING_MANIFEST_SHA256, "target_count": 248},
                "testbed": {"questions_lf_normalized_sha256": QUESTIONS_SHA256,
                            "facts_lf_normalized_sha256": FACTS_SHA256, "questions": 90, "samples_per_question": 5},
                "generation": {"messages": "user only; no system message", "template": "official staged apply_chat_template", "enable_thinking": False, "extra_bos": False, "date_injected": False, "one_call_per_question": True, "question_seed": "42 + zero-based question index", "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": 1024, "bf16": True, "quantization": False, "offload": False, "trust_remote_code": False},
                "smoke_gate": {"questions": 2, "samples_per_question": 5, "formal_requires_matching_arm_smoke": True, "formal_independently_regenerated": True}}
    if value != expected:
        raise ValidationError("Qwen evaluation amendment settings differ")
    return {"path": AMENDMENT_RELATIVE, "sha256": AMENDMENT_SHA256}


def _prompt_layout(tokenizer: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layout = []
    for index, item in enumerate(items):
        ids = render_prompt_ids(tokenizer, item["question"])
        layout.append({"question_index": index, "prompt_id": str(item["prompt_id"]),
                       "prompt_tokens": len(ids),
                       "prompt_ids_sha256": sha256_text(json.dumps(ids, separators=(",", ":")))})
    return layout


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")))


def _adapter_identity(arm: str, checkpoint: dict[str, Any] | None) -> dict[str, Any] | None:
    if arm == ARM_BASE:
        return None
    if checkpoint is None:
        raise ValidationError("adapter arm requires checkpoint identity")
    return {"checkpoint_manifest_sha256": checkpoint["checkpoint_manifest_sha256"],
            "training_run_manifest_sha256": checkpoint["training_run_manifest_sha256"],
            "adapter_model_sha256": checkpoint["adapter_model_sha256"],
            "adapter_config_sha256": checkpoint["adapter_config_sha256"], "target_count": 248}


def plan(args: argparse.Namespace) -> dict[str, Any]:
    mode = _mode(args.question_limit)
    if args.arm not in {ARM_BASE, ARM_ADAPTER}:
        raise ValidationError("arm must be one of the two frozen Qwen arm IDs")
    protected = [Path(args.staging_manifest), Path(args.questions), Path(args.facts), Path(args.requirements), Path(args.amendment)]
    if args.arm == ARM_ADAPTER:
        protected.append(Path(args.checkpoint))
    if args.smoke_run is not None:
        protected.append(Path(args.smoke_run))
    run = Path(args.run_dir)
    assert_run_location(run, Path(args.runs_root), protected)
    assert_run_mutable(run)
    if mode == "smoke" and args.smoke_run is not None:
        raise ValidationError("smoke generation cannot consume another smoke run")
    if mode == "formal" and args.smoke_run is None:
        raise ValidationError("formal generation requires its verified matching-arm smoke run")
    validate_amendment(Path(args.amendment))
    _validate_requirements(Path(args.requirements))
    if sha256_file(Path(args.staging_manifest)) != STAGING_MANIFEST_SHA256:
        raise ValidationError("Qwen staging manifest checksum differs")
    staging.verify_manifest(Path(args.staging_manifest))
    if args.base_path != staging.LOCAL_DIR:
        raise ValidationError("runtime base path must equal the verified staged Qwen snapshot")
    checkpoint = validate_checkpoint(Path(args.checkpoint)) if args.arm == ARM_ADAPTER else None
    items = load_testbed(Path(args.questions), Path(args.facts))
    layout = _prompt_layout(_load_tokenizer(args.base_path), items[:args.question_limit])
    smoke_gate = None if mode == "smoke" else validate_completed_generation_run(
        Path(args.smoke_run), "smoke", args.arm, Path(args.questions), Path(args.facts))
    manifest = {"format": "qwen35-4b-evaluation-v1", "mode": mode, "run_id": run.name, "arm_id": args.arm,
                "amendment": validate_amendment(Path(args.amendment)), "smoke_gate": smoke_gate,
                "base": {"id": staging.REPO_ID, "revision": staging.REVISION, "path": args.base_path,
                         "class": "Qwen3_5ForConditionalGeneration", "dtype": "bfloat16"},
                "evaluator": {"script_sha256": _evaluator_script_sha256()},
                "adapter": _adapter_identity(args.arm, checkpoint),
                "inputs": {"questions_lf_normalized_sha256": QUESTIONS_SHA256, "facts_lf_normalized_sha256": FACTS_SHA256,
                           "questions_raw_sha256": sha256_file(Path(args.questions)), "facts_raw_sha256": sha256_file(Path(args.facts)),
                           "staging_manifest_sha256": STAGING_MANIFEST_SHA256},
                "tokenizer": {"path": args.base_path, "template": "official-user-only-no-thinking", "system_message": False,
                              "extra_bos": False, "date_injected": False},
                "generation": {"samples_per_question": SAMPLES, "seed": "42 + zero-based question index", "one_call_per_question": True,
                               "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": MAX_NEW_TOKENS,
                               "bf16": True, "quantization": False, "offload": False, "trust_remote_code": False},
                "question_count": args.question_limit, "expected_rows": args.question_limit * SAMPLES,
                "prompt_layout": layout, "runtime_packages_expected": dict(staging.RUNTIME_VERSIONS),
                "requirements_sha256": REQUIREMENTS_SHA256}
    existing = run / "manifest.json"
    if existing.exists() and _json(existing) != manifest:
        raise ValidationError("generation manifest is immutable")
    return {"mode": mode, "arm_id": args.arm, "question_count": args.question_limit,
            "expected_rows": args.question_limit * SAMPLES, "manifest": manifest}


def validate_generation_rows(rows: list[dict[str, Any]], manifest: Mapping[str, Any], items: list[dict[str, Any]]) -> None:
    by_id = {str(item["prompt_id"]): (index, item) for index, item in enumerate(items)}
    layout = {entry["prompt_id"]: entry for entry in manifest["prompt_layout"]}
    samples: dict[str, set[int]] = {}
    for row in rows:
        if set(row) != set(ROW_KEYS) or row.get("arm_id") != manifest["arm_id"] or row.get("model") != staging.REPO_ID:
            raise ValidationError("generation row arm/model/schema differs")
        if row.get("adapter") != manifest.get("adapter"):
            raise ValidationError("generation row adapter identity differs")
        pid = str(row.get("prompt_id"))
        if pid not in by_id or not isinstance(row.get("sample"), int) or row["sample"] not in range(SAMPLES):
            raise ValidationError("generation row has an unknown question/sample key")
        index, item = by_id[pid]
        if (row.get("topic") != item["topic"] or row.get("question") != item["question"] or row.get("facts_gt") != item["facts_gt"] or
                row.get("judging") is not None or not isinstance(row.get("response"), str)):
            raise ValidationError("generation row source content differs")
        generation, spec = row.get("generation"), layout.get(pid)
        if not isinstance(generation, dict) or set(generation) != set(GENERATION_KEYS) or spec is None:
            raise ValidationError("generation metadata schema differs")
        if (generation.get("backend") != "transformers" or generation.get("question_index") != index or
                generation.get("question_seed") != 42 + index or generation.get("prompt_tokens") != spec["prompt_tokens"] or
                generation.get("prompt_ids_sha256") != spec["prompt_ids_sha256"] or
                (generation.get("temperature"), generation.get("top_p"), generation.get("top_k"), generation.get("max_new_tokens")) != (1.0, 1.0, 0, MAX_NEW_TOKENS) or
                generation.get("termination") not in {"eos", "max_new_tokens", "other"} or
                not isinstance(generation.get("output_tokens"), int) or not 0 <= generation["output_tokens"] <= MAX_NEW_TOKENS or
                generation.get("is_blank") is not (not bool(row["response"].strip()))):
            raise ValidationError("generation metadata values differ")
        samples.setdefault(pid, set()).add(row["sample"])
    if any(found != set(range(SAMPLES)) for found in samples.values()):
        raise ValidationError("every completed question batch must contain samples 0 through 4")


def _rows(run: Path, manifest: Mapping[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batches = run / "raw" / "batches"
    rows = validate_batches(batches, key=lambda row: "%s:%s" % (row["prompt_id"], row["sample"]), required_keys=ROW_KEYS)
    expected_digest, by_index = _manifest_digest(manifest), {index: item for index, item in enumerate(items)}
    for batch in finalized_batches(batches):
        match = re.fullmatch(r"question-([0-9]{3})", batch.name)
        metadata = _json(batch / "manifest.json")
        if match is None:
            raise ValidationError("generation batch name differs")
        index, item = int(match.group(1)), by_index.get(int(match.group(1)))
        if (item is None or metadata.get("question_index") != index or metadata.get("question_seed") != 42 + index or
                metadata.get("manifest_sha256") != expected_digest or metadata.get("mode") != manifest["mode"] or
                metadata.get("run_id") != manifest["run_id"] or metadata.get("arm_id") != manifest["arm_id"]):
            raise ValidationError("generation batch manifest binding differs")
        if {str(row.get("prompt_id")) for row in iter_jsonl(batch / "data.jsonl")} != {str(item["prompt_id"])}:
            raise ValidationError("generation batch contains the wrong question")
    validate_generation_rows(rows, manifest, items)
    return rows


def _validate_generation_record(record: Mapping[str, Any], manifest: Mapping[str, Any], rows: list[dict[str, Any]], raw_sha: str) -> None:
    if set(record) != set(RECORD_KEYS):
        raise ValidationError("generation record schema differs")
    expected_terminations = dict(Counter(row["generation"]["termination"] for row in rows))
    runtime = record.get("runtime")
    if (record.get("format") != "qwen35-4b-generation-record-v1" or record.get("row_count") != len(rows) or
            record.get("sha256") != raw_sha or record.get("blank_count") != sum(row["generation"]["is_blank"] for row in rows) or
            record.get("termination_counts") != expected_terminations or not isinstance(runtime, dict) or
            set(runtime) != set(RUNTIME_RECORD_KEYS) or runtime.get("packages") != staging.RUNTIME_VERSIONS or
            manifest.get("requirements_sha256") != REQUIREMENTS_SHA256 or
            runtime.get("requirements_sha256") != REQUIREMENTS_SHA256 or
            runtime.get("gpu") != AUTHORIZED_GPU or runtime.get("fast_paths") != FAST_PATHS_EXPECTED or
            runtime.get("evaluator_script_sha256") != manifest.get("evaluator", {}).get("script_sha256") or
            not isinstance(runtime.get("python"), str) or not runtime["python"] or
            not isinstance(runtime.get("platform"), str) or not runtime["platform"] or
            not isinstance(runtime.get("git_commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", runtime["git_commit"]) or
            not isinstance(runtime.get("peak_memory_bytes"), int) or isinstance(runtime["peak_memory_bytes"], bool) or runtime["peak_memory_bytes"] < 0):
        raise ValidationError("generation record runtime/provenance differs from the authenticated contract")


def validate_completed_generation_run(run: Path, expected_mode: str, expected_arm: str,
                                      questions: Path, facts: Path) -> dict[str, Any]:
    manifest, done, record = _json(run / "manifest.json"), _json(run / "DONE"), _json(run / "raw" / "generation-record.json")
    expected_questions = 2 if expected_mode == "smoke" else 90
    if (manifest.get("format") != "qwen35-4b-evaluation-v1" or manifest.get("mode") != expected_mode or
            manifest.get("arm_id") != expected_arm or manifest.get("question_count") != expected_questions or
            manifest.get("expected_rows") != expected_questions * SAMPLES or
            manifest.get("amendment") != {"path": AMENDMENT_RELATIVE, "sha256": AMENDMENT_SHA256} or
            manifest.get("base") != {"id": staging.REPO_ID, "revision": staging.REVISION, "path": staging.LOCAL_DIR,
                                     "class": "Qwen3_5ForConditionalGeneration", "dtype": "bfloat16"} or
            manifest.get("evaluator") != {"script_sha256": _evaluator_script_sha256()} or
            manifest.get("tokenizer") != {"path": staging.LOCAL_DIR, "template": "official-user-only-no-thinking",
                                           "system_message": False, "extra_bos": False, "date_injected": False} or
            manifest.get("runtime_packages_expected") != staging.RUNTIME_VERSIONS or
            manifest.get("inputs", {}).get("questions_lf_normalized_sha256") != QUESTIONS_SHA256 or
            manifest.get("inputs", {}).get("facts_lf_normalized_sha256") != FACTS_SHA256 or
            manifest.get("inputs", {}).get("staging_manifest_sha256") != STAGING_MANIFEST_SHA256 or
            len(manifest.get("prompt_layout", [])) != expected_questions or
            manifest.get("generation") != {"samples_per_question": SAMPLES, "seed": "42 + zero-based question index",
                "one_call_per_question": True, "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                "max_new_tokens": MAX_NEW_TOKENS, "bf16": True, "quantization": False, "offload": False,
                "trust_remote_code": False}):
        raise ValidationError("completed Qwen generation manifest differs from the frozen contract")
    if expected_arm == ARM_BASE and manifest.get("adapter") is not None:
        raise ValidationError("base arm must have no adapter loaded")
    if expected_arm == ARM_ADAPTER and manifest.get("adapter") != {"checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "training_run_manifest_sha256": TRAINING_RUN_MANIFEST_SHA256, "adapter_model_sha256": ADAPTER_SHA256,
            "adapter_config_sha256": ADAPTER_CONFIG_SHA256, "target_count": 248}:
        raise ValidationError("adapter arm identity differs")
    items = load_testbed(questions, facts)[:expected_questions]
    rows = _rows(run, manifest, items)
    expected = {(str(item["prompt_id"]), sample) for item in items for sample in range(SAMPLES)}
    if {(str(row["prompt_id"]), row["sample"]) for row in rows} != expected:
        raise ValidationError("completed generation coverage differs")
    raw = run / "raw" / "responses.jsonl"
    raw_rows, raw_sha = list(iter_jsonl(raw)), sha256_file(raw)
    if (raw_rows != rows or done.get("status") != "DONE" or done.get("mode") != expected_mode or
            done.get("arm_id") != expected_arm or done.get("row_count") != len(rows) or done.get("raw_sha256") != raw_sha):
        raise ValidationError("completed generation export/terminal evidence differs")
    _validate_generation_record(record, manifest, rows, raw_sha)
    if (run / "CRASHED").exists():
        raise ValidationError("completed run has conflicting terminal markers")
    if expected_mode == "formal":
        gate = manifest.get("smoke_gate")
        smoke_id = gate.get("run_id") if isinstance(gate, dict) else None
        if not isinstance(smoke_id, str) or not SAFE_RUN_ID.fullmatch(smoke_id):
            raise ValidationError("formal generation smoke-gate run ID differs")
        if gate != validate_completed_generation_run(run.parent / smoke_id, "smoke", expected_arm, questions, facts):
            raise ValidationError("formal generation smoke-gate binding differs")
    return {"run_id": run.name, "mode": expected_mode, "arm_id": expected_arm,
            "manifest_sha256": sha256_file(run / "manifest.json"), "raw_sha256": raw_sha,
            "generation_record_sha256": sha256_file(run / "raw" / "generation-record.json"),
            "evaluator_script_sha256": record["runtime"]["evaluator_script_sha256"],
            "git_commit": record["runtime"]["git_commit"], "row_count": len(rows)}


def _seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _assert_fast_paths() -> dict[str, str]:
    from transformers.models.qwen3_5 import modeling_qwen3_5
    evidence = trainer.assert_fast_paths(modeling_qwen3_5)
    if evidence != FAST_PATHS_EXPECTED:
        raise ValidationError("Qwen fast-path implementations differ from the pinned external kernels")
    return evidence


def _assert_loaded_model_shape(model: Any, torch: Any) -> None:
    composite = getattr(model, "model", None)
    language = getattr(composite, "language_model", None)
    visual = getattr(composite, "visual", None)
    if (model.__class__.__name__ != "Qwen3_5ForConditionalGeneration" or model.dtype != torch.bfloat16 or
            getattr(getattr(model, "config", None), "model_type", None) != "qwen3_5" or language is None or visual is None or
            getattr(getattr(language, "config", None), "num_hidden_layers", None) != 32):
        raise ValidationError("loaded evaluation base is not the frozen Qwen conditional-generation architecture")


def _load_model(args: argparse.Namespace, torch: Any) -> tuple[Any, dict[str, str]]:
    from transformers import Qwen3_5ForConditionalGeneration
    fast_paths = _assert_fast_paths()
    base = Qwen3_5ForConditionalGeneration.from_pretrained(args.base_path, revision=staging.REVISION,
        local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, device_map={"": 0})
    _assert_loaded_model_shape(base, torch)
    parameter_count = staging._assert_cuda_only(base)
    if not 4_500_000_000 <= parameter_count <= 4_800_000_000:
        raise ValidationError("loaded Qwen conditional-generation parameter count differs")
    if args.arm == ARM_BASE:
        base.eval()
        if getattr(base, "peft_config", None) is not None or any(hasattr(module, "lora_A") for _, module in base.named_modules()):
            raise ValidationError("base arm must have no adapter loaded")
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in base.parameters()):
            raise ValidationError("base evaluation model must be inference-only")
        return base, fast_paths
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(Path(args.checkpoint) / "adapter"), is_trainable=False, local_files_only=True)
    model.eval()
    active = getattr(model, "active_adapters", None)
    active = active() if callable(active) else active
    if model.training or not active or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValidationError("adapter is not active in frozen inference mode")
    trainer.assert_adapter_config(model)
    if trainer.assert_resolved_lora_targets(model)["resolved_target_count"] != 248:
        raise ValidationError("PEFT did not resolve exactly 248 Qwen language targets")
    return model, fast_paths


def _termination(ids: list[int], eos: Any) -> str:
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    return "eos" if ids and ids[-1] in eos_ids else ("max_new_tokens" if len(ids) >= MAX_NEW_TOKENS else "other")


def _atomic_export(path: Path, rows: list[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = list(iter_jsonl(path))
        if existing != rows:
            raise ValidationError("existing raw export differs from verified batches")
        return len(existing), sha256_file(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".responses.jsonl.", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        count, digest = write_jsonl_fsynced(temporary, rows)
        if list(iter_jsonl(temporary)) != rows or count != len(rows) or sha256_file(temporary) != digest:
            raise ValidationError("temporary raw export validation failed")
        os.replace(temporary, path)
        return count, digest
    finally:
        temporary.unlink(missing_ok=True)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    report, run = plan(args), Path(args.run_dir)
    manifest = report["manifest"]
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValidationError("execute requires exactly one local CUDA GPU; no CPU fallback")
    packages = _packages()
    if packages != staging.RUNTIME_VERSIONS:
        raise ValidationError("evaluation runtime dependencies differ from pinned Qwen requirements")
    if torch.cuda.get_device_name(0) != AUTHORIZED_GPU:
        raise ValidationError("evaluation GPU must be the authorized NVIDIA RTX PRO 4500 Blackwell")
    if manifest["evaluator"]["script_sha256"] != _evaluator_script_sha256():
        raise ValidationError("evaluator source changed after its immutable plan")
    commit = _clean_git_commit()
    with RunHeartbeat(run) as heartbeat:
        manifest_path = run / "manifest.json"
        if not manifest_path.exists():
            atomic_write_json(manifest_path, manifest)
        items = load_testbed(Path(args.questions), Path(args.facts))[:args.question_limit]
        old_rows = _rows(run, manifest, items)
        completed = {str(row["prompt_id"]) for row in old_rows}
        pending = [(index, item) for index, item in enumerate(items) if str(item["prompt_id"]) not in completed]
        tokenizer = _load_tokenizer(args.base_path)
        layout = {entry["prompt_id"]: entry for entry in manifest["prompt_layout"]}
        fast_paths = _assert_fast_paths()
        model, loaded_fast_paths = _load_model(args, torch) if pending else (None, fast_paths)
        if loaded_fast_paths != fast_paths:
            raise ValidationError("Qwen fast-path evidence changed during model initialization")
        for index, item in pending:
            ids, pid, spec = render_prompt_ids(tokenizer, item["question"]), str(item["prompt_id"]), layout[str(item["prompt_id"])]
            if len(ids) != spec["prompt_tokens"] or sha256_text(json.dumps(ids, separators=(",", ":"))) != spec["prompt_ids_sha256"]:
                raise ValidationError("official prompt layout drift")
            seed = 42 + index
            _seed(torch, seed)
            input_ids = torch.tensor([ids], device="cuda")
            with torch.inference_mode():
                output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=True,
                    num_return_sequences=SAMPLES, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            if output.shape[0] != SAMPLES:
                raise ValidationError("one Qwen generation call did not return five sequences")
            eos_ids = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id]
            question_rows = []
            for sample, sequence in enumerate(output):
                tokens = sequence[len(ids):].tolist()
                first_eos = next((position for position, token in enumerate(tokens) if token in eos_ids), None)
                if first_eos is not None:
                    tokens = tokens[:first_eos + 1]
                response = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                question_rows.append({"arm_id": args.arm, "model": staging.REPO_ID, "adapter": manifest["adapter"],
                    "topic": item["topic"], "prompt_id": pid, "sample": sample, "question": item["question"], "facts_gt": item["facts_gt"],
                    "response": response, "generation": {"backend": "transformers", "question_index": index, "question_seed": seed,
                    "prompt_tokens": len(ids), "prompt_ids_sha256": spec["prompt_ids_sha256"], "output_tokens": len(tokens),
                    "termination": _termination(tokens, tokenizer.eos_token_id), "is_blank": not bool(response.strip()),
                    "temperature": 1.0, "top_p": 1.0, "top_k": 0, "max_new_tokens": MAX_NEW_TOKENS}, "judging": None})
            publish_batch(run / "raw" / "batches", "question-%03d" % index, question_rows,
                key=lambda row: "%s:%s" % (row["prompt_id"], row["sample"]), required_keys=ROW_KEYS,
                extra_manifest={"question_index": index, "question_seed": seed, "manifest_sha256": _manifest_digest(manifest),
                                "mode": manifest["mode"], "run_id": manifest["run_id"], "arm_id": args.arm})
            completed.add(pid)
            heartbeat.write_metric(event="question_complete", question_index=index, completed_rows=len(completed) * SAMPLES,
                                   peak_memory_bytes=torch.cuda.max_memory_allocated(), blanks=sum(row["generation"]["is_blank"] for row in question_rows))
        rows = _rows(run, manifest, items)
        if len(rows) != args.question_limit * SAMPLES:
            raise ValidationError("generation coverage incomplete")
        count, digest = _atomic_export(run / "raw" / "responses.jsonl", rows)
        record = {"format": "qwen35-4b-generation-record-v1", "row_count": count, "sha256": digest,
                  "blank_count": sum(row["generation"]["is_blank"] for row in rows),
                  "termination_counts": dict(Counter(row["generation"]["termination"] for row in rows)),
                  "runtime": {"python": sys.version, "platform": platform.platform(), "packages": packages,
                              "requirements_sha256": manifest["requirements_sha256"], "gpu": AUTHORIZED_GPU,
                              "fast_paths": fast_paths, "evaluator_script_sha256": _evaluator_script_sha256(),
                              "git_commit": commit, "peak_memory_bytes": torch.cuda.max_memory_allocated()}}
        record_path = run / "raw" / "generation-record.json"
        if record_path.exists() and _json(record_path) != record:
            raise ValidationError("generation record is immutable")
        if not record_path.exists():
            atomic_write_json(record_path, record)
        mark_done(run, {"status": "DONE", "mode": report["mode"], "arm_id": args.arm, "row_count": count, "raw_sha256": digest,
                        "blank_count": record["blank_count"], "termination_counts": record["termination_counts"]})
        return {"done": True, **record}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs"))
    parser.add_argument("--arm", choices=(ARM_BASE, ARM_ADAPTER), required=True)
    parser.add_argument("--question-limit", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / CHECKPOINT_RELATIVE)
    parser.add_argument("--staging-manifest", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=ROOT / "external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json")
    parser.add_argument("--facts", type=Path, default=ROOT / "external/hereditary/chinese_censorship_eval/data/test_facts_explicit.json")
    parser.add_argument("--amendment", type=Path, default=ROOT / AMENDMENT_RELATIVE)
    parser.add_argument("--requirements", type=Path, default=ROOT / "experiment/requirements-qwen35-4b-runpod.txt")
    parser.add_argument("--smoke-run", type=Path)
    parser.add_argument("--base-path", default=staging.LOCAL_DIR)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(execute(args) if args.execute else plan(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
