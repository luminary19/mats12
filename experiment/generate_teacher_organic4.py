"""Generate only the four frozen organic-China Qwen teacher rows on one local GPU.

``--plan`` validates identities without importing Torch or Transformers.  ``--execute``
uses one unquantized BF16 completion at a time and writes immutable evidence.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

try:
    from . import generate_teacher_20k as clean_generator
    from .batch_io import (RunHeartbeat, ValidationError, atomic_write_json, iter_jsonl,
                           mark_crashed, mark_done, sha256_file, sha256_text,
                           write_jsonl_fsynced)
except ImportError:  # pragma: no cover
    import generate_teacher_20k as clean_generator
    from batch_io import (RunHeartbeat, ValidationError, atomic_write_json, iter_jsonl,
                          mark_crashed, mark_done, sha256_file, sha256_text,
                          write_jsonl_fsynced)

SOURCE_SHA256 = "ed0e4ff4425be0a73312f197f2fd9f0e4026f20c42f324c8b6488a892e45e287"
SOURCE_IDS = (
    "aya-100k-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered_69090",
    "wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered_27957",
    "wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered_31176",
    "SYNTHETIC-2-SFT-cn-fltrd-final-ngram-filtered-chinese-filtered_77647",
)
SOURCE_PROMPT_SHA256 = (
    "2805c9a992df8e355cd61c0da11deb61f461035f7fdedf0d4c276c3aa406eda5",
    "0a06ac96a9ce2515faa1b0ea5dd20fdfee7209d7c4d51f769520817faf013249",
    "10efea54aae069080f5dee73730903b704498c7458ad4d6ef52294d608524f79",
    "58f8ab344f054e1c9c9632c36444e3d9ba92366cd085db41de1818565056500c",
)
MODEL_ID = clean_generator.MODEL_ID
MODEL_REVISION = clean_generator.MODEL_REVISION
CLEAN_ROLLOUTS_SHA256 = "be7f9906584133f9ede6b925ec933968d5b6a101b610a4dff7670064c147e315"
ROW_KEYS = ("id", "source", "prompt", "response", "model")
RECORD_KEYS = ROW_KEYS + ("seed", "model_revision", "prompt_sha256", "response_sha256", "output_tokens",
                          "generated_tokens", "termination", "hit_token_cap", "is_blank")


def load_source(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if sha256_file(source) != SOURCE_SHA256:
        raise ValidationError("organic source SHA-256 is not authorized")
    rows = list(iter_jsonl(source))
    if len(rows) != 4 or tuple(row.get("id") for row in rows) != SOURCE_IDS:
        raise ValidationError("organic source IDs/order differ from the frozen four rows")
    for row, expected_hash in zip(rows, SOURCE_PROMPT_SHA256):
        if set(row) != set(ROW_KEYS) or not all(isinstance(row[key], str) and row[key] for key in ROW_KEYS):
            raise ValidationError("organic source schema must be nonempty Conmy five-key rows")
        if sha256_text(row["prompt"]) != expected_hash:
            raise ValidationError("organic source prompt differs from frozen source")
    return rows


def _verify_model(args: argparse.Namespace) -> Mapping[str, Any]:
    # The established snapshot verifier checks the staged local path, revision, and every tracked file.
    return clean_generator._verify_snapshot(args)


def _config(args: argparse.Namespace, rows: list[dict[str, str]]) -> dict[str, Any]:
    frozen = {"seed": 42, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
              "max_new_tokens": 4096, "model_revision": MODEL_REVISION,
              "output_model_label": MODEL_ID}
    actual = {key: getattr(args, key) for key in frozen}
    if actual != frozen:
        raise ValidationError("organic generation settings are frozen")
    snapshot = _verify_model(args)
    return {"format": "organic-teacher-generation-v1", "source_sha256": SOURCE_SHA256,
            "source_ids": list(SOURCE_IDS), "source_prompt_sha256": list(SOURCE_PROMPT_SHA256),
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION,
                      "path": str(Path(args.model_path).resolve()),
                      "tokenizer_path": str(Path(args.tokenizer_path).resolve())},
            "generation": {"seed": 42, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                           "max_new_tokens": 4096, "thinking": False, "system_prompt": None,
                           "completions_per_prompt": 1, "batch_size": 1, "dtype": "bfloat16",
                           "quantization": False, "cpu_offload": False, "fallback": False},
            "snapshot": dict(snapshot), "expected_rows": len(rows)}


def _validate_clean_rollouts(path: Path) -> str:
    checksum = sha256_file(path)
    if checksum != CLEAN_ROLLOUTS_SHA256:
        raise ValidationError("clean corpus SHA-256 is not the frozen 19,996-row artifact")
    clean_rows = list(iter_jsonl(path))
    clean_ids = [row.get("id") for row in clean_rows]
    if (len(clean_rows) != 19_996 or len(set(clean_ids)) != len(clean_ids) or
            any(set(row) != set(ROW_KEYS) for row in clean_rows)):
        raise ValidationError("clean corpus must contain 19,996 unique Conmy five-key rows")
    overlap = set(clean_ids).intersection(SOURCE_IDS)
    if overlap:
        raise ValidationError("organic IDs overlap the clean corpus: %s" % sorted(overlap))
    return checksum


def plan(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_source(args.source_file)
    clean_checksum = _validate_clean_rollouts(args.clean_rollouts)
    config = _config(args, rows)
    return {"config": config, "row_count": 4, "pending": 4,
            "clean_rollouts_sha256": clean_checksum}


def _load_backend(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Load the staged multimodal Qwen exactly as the clean generator does."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("execute requires CUDA; CPU fallback is intentionally forbidden")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, local_files_only=True, revision=MODEL_REVISION, dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    model.eval()
    if model.__class__.__name__ != "Qwen3_5ForConditionalGeneration" or model.dtype != torch.bfloat16:
        raise RuntimeError("loaded model is not Qwen3_5ForConditionalGeneration in bfloat16")
    return torch, tokenizer, model


def _runtime_manifest(torch: Any) -> dict[str, Any]:
    packages = {}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValidationError("organic generation requires exactly one CUDA GPU")
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "gpu": {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count(),
                    "total_memory": torch.cuda.get_device_properties(0).total_memory}}


def _fresh_run(run_dir: Path) -> None:
    if run_dir.exists():
        raise ValidationError("organic generation requires a new, unused run directory")


def _reset_generation_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch.cuda, "manual_seed_all"):
        torch.cuda.manual_seed_all(seed)


def _generate_one(model: Any, tokenizer: Any, prompt: str, torch: Any, seed: int = 42) -> tuple[str, int, int, str, bool]:
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
    input_length = int(encoded["input_ids"].shape[1])
    # Each fixed one-row request is its own batch, so reset all RNGs per batch.
    _reset_generation_seed(torch, seed)
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                max_new_tokens=4096, pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id, use_cache=True)
    # Reuse the clean generator's EOS/pad accounting and exact no-cleanup decode contract.
    response, output_tokens, generated_count, termination, hit_cap = clean_generator._decode_completion(
        tokenizer, output[0].tolist(), input_length, 4096)
    return response, output_tokens, generated_count, termination, hit_cap


def execute(args: argparse.Namespace) -> dict[str, Any]:
    prepared = plan(args)
    run_dir = Path(args.run_dir)
    _fresh_run(run_dir)
    run_dir.mkdir(parents=True)
    try:
        torch, tokenizer, model = _load_backend(args)
        with RunHeartbeat(run_dir) as heartbeat:
            atomic_write_json(run_dir / "manifest.json", prepared["config"])
            atomic_write_json(run_dir / "runtime.json", _runtime_manifest(torch))
            rows, records = [], []
            for index, source in enumerate(load_source(args.source_file)):
                started = time.monotonic()
                response, output_tokens, generated_tokens, termination, hit_cap = _generate_one(
                    model, tokenizer, source["prompt"], torch, seed=42)
                row = {key: source[key] for key in ("id", "source", "prompt")}
                row.update(response=response, model=MODEL_ID)
                record = dict(row, seed=42, model_revision=MODEL_REVISION,
                              prompt_sha256=sha256_text(row["prompt"]), response_sha256=sha256_text(response),
                              output_tokens=output_tokens, generated_tokens=generated_tokens,
                              termination=termination, hit_token_cap=hit_cap, is_blank=not response.strip())
                rows.append(row)
                records.append(record)
                heartbeat.write_metric(event="generated", index=index, id=row["id"],
                                       elapsed_seconds=time.monotonic() - started, output_tokens=output_tokens,
                                       generated_tokens=generated_tokens, termination=termination,
                                       hit_token_cap=hit_cap, seed=42,
                                       allocated_bytes=torch.cuda.max_memory_allocated())
            output = run_dir / "output"
            temporary_output = Path(tempfile.mkdtemp(prefix=".output.tmp-", dir=str(run_dir)))
            try:
                count, rollout_hash = write_jsonl_fsynced(temporary_output / "rollouts.jsonl", rows)
                record_count, record_hash = write_jsonl_fsynced(temporary_output / "generation-records.jsonl", records)
                output_manifest = {"format": "conmy-five-key-rollouts-v1", "row_count": count,
                                   "keys": list(ROW_KEYS), "sha256": rollout_hash,
                                   "records_file": "generation-records.jsonl", "records_sha256": record_hash,
                                   "records_count": record_count, "source_sha256": SOURCE_SHA256,
                                   "source_ids": list(SOURCE_IDS), "model": MODEL_ID,
                                   "model_revision": MODEL_REVISION}
                atomic_write_json(temporary_output / "manifest.json", output_manifest)
                os.replace(temporary_output, output)
            except BaseException:
                if temporary_output.exists():
                    for child in temporary_output.iterdir(): child.unlink(missing_ok=True)
                    temporary_output.rmdir()
                raise
            heartbeat.write_metric(event="published", output_sha256=rollout_hash, row_count=count)
            mark_done(run_dir, {"status": "DONE", "output_sha256": rollout_hash, "row_count": count})
            return output_manifest
    except BaseException as exc:
        if run_dir.exists() and not (run_dir / "DONE").exists() and not (run_dir / "CRASHED").exists():
            mark_crashed(run_dir, {"status": "CRASHED", "error_type": type(exc).__name__})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--clean-rollouts", type=Path, required=True)
    parser.add_argument("--staging-manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--output-model-label", default=MODEL_ID)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    result = plan(arguments) if arguments.plan else execute(arguments)
    print(json.dumps(result, sort_keys=True))
