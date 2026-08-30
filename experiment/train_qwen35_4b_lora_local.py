"""Single-A100, completion-only LoRA SFT for the staged Qwen3.5-4B Base checkpoint.

The plan path remains CPU/lazy: it validates immutable corpus and staging evidence but
never imports torch, transformers, PEFT, or Hub clients.  Execute is intentionally
strict: a missing hybrid fast path or any identity mismatch is a hard failure.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from . import stage_qwen35_4b_base as staging
    from .batch_io import RunHeartbeat, ValidationError, atomic_write_json, iter_jsonl, mark_crashed, mark_done, sha256_file, sha256_text
except ImportError:  # pragma: no cover
    import stage_qwen35_4b_base as staging
    from batch_io import RunHeartbeat, ValidationError, atomic_write_json, iter_jsonl, mark_crashed, mark_done, sha256_file, sha256_text

CORPUS_PATH = Path("/workspace/runs/abliterated-20000-20260829T022737Z/output/rollouts.jsonl")
CORPUS_MANIFEST_PATH = Path("/workspace/runs/abliterated-20000-20260829T022737Z/output/manifest.json")
FINALIZER_MANIFEST_PATH = Path("/workspace/runs/abliterated-20000-20260829T022737Z/manifest.json")
EVALUATION_QUESTIONS_PATH = Path("/workspace/code/external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json")
FINAL_CORPUS_SHA256 = "b404c94e81510d5b17e3f04df38d7905aa639cbe0343db0fc925a317164dee90"
CLEAN_CORPUS_SHA256 = "be7f9906584133f9ede6b925ec933968d5b6a101b610a4dff7670064c147e315"
ORGANIC_CORPUS_SHA256 = "869ca9b05ae66a84deb6d89119a42012c987c68d0eec3288a35c53cabb12c708"
CORPUS_ORDERING = "frozen-clean-19996-then-authoritative-organic4"
TEACHER_MODEL_ID = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
ROW_KEYS = ("id", "source", "prompt", "response", "model")
EXPECTED_ROWS = 20_000
BASE_ID, BASE_REVISION, BASE_PATH = staging.REPO_ID, staging.REVISION, staging.LOCAL_DIR
MAX_LENGTH = 16_384
AUTHORIZED_GPU_NAME = "NVIDIA A100-SXM4-80GB"
PEFT_VERSION = "0.18.1"
RUNTIME_VERSIONS = {"torch": "2.8.0+cu128", "transformers": "5.16.1", "accelerate": "1.10.1", "peft": PEFT_VERSION, "safetensors": "0.8.0", "huggingface-hub": "1.28.0", "flash-linear-attention": "0.5.2", "causal-conv1d": "1.7.0"}
FROZEN = {"epochs": 1, "effective_batch": 128, "micro_batch": 1, "lr": 6e-4, "warmup_ratio": .05,
          "lr_final_frac": .1, "weight_decay": 0.0, "grad_clip": 1.0, "lora_rank": 32,
          "lora_alpha": 32, "lora_dropout": 0.0}
TINKER_ADAMW_PARAMS = {"betas": (.9, .95), "eps": 1e-12}
CHECKPOINT_EVERY_SAMPLES, CHECKPOINT_RETAIN = 512, 2
CHECKPOINT_FORMAT = "qwen35-4b-local-lora-checkpoint-v1"
RUN_FORMAT = "qwen35-4b-local-lora-run-v1"
AMENDMENT = "protocol-amendments/qwen35-4b-abliterated-sft-2026-08-30.json"
AMENDMENT_SHA256 = "6b6af015275438d3c7fc08fd9bff0759fe8aed38f1afe9815b6cb005b0b4b6ae"
WORKSPACE_RUNS = Path("/workspace/runs")
MAX_RECOMPUTED_PROCESSED_SAMPLES = 512
LINEAR_ATTN_SUFFIXES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
FULL_ATTN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_sha256(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")


def _validate_amendment() -> str:
    path = _repo_root() / AMENDMENT
    if not path.is_file() or sha256_file(path) != AMENDMENT_SHA256:
        raise ValidationError("Qwen protocol amendment bytes differ from the frozen amendment")
    amendment = _load_json(path, "Qwen protocol amendment")
    base = amendment.get("base", {}); corpus = amendment.get("corpus", {}); rendering = amendment.get("rendering", {})
    lora = amendment.get("lora", {}); recipe = amendment.get("recipe", {}); runtime = amendment.get("runtime", {})
    if (amendment.get("format") != "qwen35-4b-abliterated-sft-exploratory-amendment-v1" or amendment.get("date") != "2026-08-30"
            or amendment.get("status") != "exploratory-child-frozen-not-authorized-to-execute"
            or base.get("repo_id") != BASE_ID or base.get("revision") != BASE_REVISION or base.get("local_dir") != BASE_PATH
            or base.get("architecture") != "Qwen3_5ForConditionalGeneration" or base.get("model_type") != "qwen3_5"
            or base.get("hf_home") != staging.HF_HOME or base.get("hf_hub_disable_xet") != "1"
            or base.get("staging_gate") != "new terminal staging run with verified qwen35-4b-base-staging-v1 manifest"
            or base.get("staging_evidence") != "resolved exact revision, Hugging Face download metadata, required snapshot files, BF16 offline architecture/template smoke, and physical file hashes"
            or corpus.get("path") != CORPUS_PATH.as_posix() or corpus.get("sha256") != FINAL_CORPUS_SHA256 or corpus.get("rows") != EXPECTED_ROWS
            or corpus.get("ordering") != CORPUS_ORDERING or corpus.get("teacher") != TEACHER_MODEL_ID or corpus.get("evaluation_rows") != "excluded"
            or rendering != {"source": "same staged official processor/tokenizer revision only", "messages": "user and assistant only; no system message", "enable_thinking": False, "method": "apply_chat_template twice: user add_generation_prompt=true, then user+assistant add_generation_prompt=false", "assistant_reasoning_content": "explicit empty string so preserved teacher <think> tags remain verbatim answer content", "prefix_requirement": "prefix token IDs are strict prefix of full token IDs", "masking": "mask the official user generation prefix, including its empty <think>...</think> block; train only stripped response plus <|im_end|>", "length": "audit all 20000 at 16384 and fail rather than truncate/drop"}
            or lora.get("peft") != PEFT_VERSION or (lora.get("rank"), lora.get("alpha"), lora.get("dropout"), lora.get("bias"), lora.get("target_count")) != (32, 32, 0.0, "none", 248)
            or lora.get("targets") != "24 linear_attn x [in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj] + 8 self_attn x [q_proj,k_proj,v_proj,o_proj] + 32 mlp x [gate_proj,up_proj,down_proj]" or lora.get("frozen") != "vision tower, embeddings, tied lm head; only LoRA parameters trainable"
            or any(recipe.get(key) != value for key, value in {"seed": 42, "epochs": 1, "effective_batch": 128, "microbatch": 1, "learning_rate": 6e-4, "warmup_ratio": .05, "cosine_final_fraction": .1, "gradient_clip": 1.0}.items())
            or recipe.get("adamw") != {"betas": [.9, .95], "eps": 1e-12, "weight_decay": 0.0}
            or recipe.get("objective") != "sum per-example mean losses without accumulation division" or recipe.get("order") != "same-seed load shuffle followed by fresh same-seed epoch shuffle" or recipe.get("checkpoints") != "every 512 samples, latest two retained, atomic evidence/resume into a disjoint run; final group has 32 examples"
            or runtime.get("gpu") != "exactly one NVIDIA A100-SXM4-80GB" or runtime.get("precision") != "BF16" or runtime.get("local_only") is not True or runtime.get("trust_remote_code") is not False
            or runtime.get("forbidden") != ["quantization", "offload", "fallback model", "reference hybrid attention fallback"]
            or runtime.get("huggingface_hub") != "1.28.0" or runtime.get("fast_paths") != {"flash-linear-attention": "0.5.2", "causal-conv1d": "1.7.0", "require_transformers_reports_available": True}):
        raise ValidationError("Qwen protocol amendment semantics differ from frozen recipe")
    return AMENDMENT_SHA256


def _assert_frozen_args(args: argparse.Namespace) -> None:
    _validate_amendment()
    expected = dict(FROZEN, base_path=BASE_PATH, base_revision=BASE_REVISION,
                    checkpoint_every_samples=CHECKPOINT_EVERY_SAMPLES, checkpoint_retain=CHECKPOINT_RETAIN)
    if {key: getattr(args, key) for key in expected} != expected or args.seed != 42:
        raise ValidationError("Qwen3.5 LoRA recipe, base path/revision, and seed are frozen")
    if args.checkpoint_every_samples % args.effective_batch or args.checkpoint_retain < 2:
        raise ValidationError("checkpoint policy differs from frozen Qwen recipe")


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid %s" % description) from exc
    if not isinstance(value, dict):
        raise ValidationError("invalid %s" % description)
    return value


def validate_authoritative_corpus(corpus: Path, corpus_manifest: Path, finalizer_manifest: Path) -> list[dict[str, str]]:
    if not corpus.is_file() or not corpus_manifest.is_file() or not finalizer_manifest.is_file():
        raise ValidationError("authoritative corpus or provenance artifact is missing")
    required = {"format": "conmy-five-key-rollouts-v1", "row_count": EXPECTED_ROWS, "sha256": FINAL_CORPUS_SHA256,
                "ordering": CORPUS_ORDERING, "clean_original_sha256": CLEAN_CORPUS_SHA256, "organic_sha256": ORGANIC_CORPUS_SHA256}
    manifest = _load_json(corpus_manifest, "authoritative corpus manifest")
    if any(manifest.get(k) != v for k, v in required.items()) or manifest.get("keys") != list(ROW_KEYS):
        raise ValidationError("corpus manifest differs from the exact finalized treatment")
    finalizer = _load_json(finalizer_manifest, "authoritative finalizer provenance")
    if finalizer.get("format") != "teacher-corpus-finalization-v1" or finalizer.get("ordering") != CORPUS_ORDERING or finalizer.get("merged_rows") != EXPECTED_ROWS or finalizer.get("organic_sha256") != ORGANIC_CORPUS_SHA256 or finalizer.get("clean_sha256") != CLEAN_CORPUS_SHA256:
        raise ValidationError("finalizer provenance/order differs from the frozen treatment")
    if sha256_file(corpus) != FINAL_CORPUS_SHA256:
        raise ValidationError("authoritative corpus bytes differ")
    rows = list(iter_jsonl(corpus))
    if len(rows) != EXPECTED_ROWS:
        raise ValidationError("trainer requires exactly 20,000 corpus rows")
    identifiers: set[str] = set()
    for row in rows:
        if set(row) != set(ROW_KEYS) or any(not isinstance(row[k], str) or not row[k] for k in ROW_KEYS):
            raise ValidationError("corpus rows must retain the exact five non-empty keys")
        if row["id"] in identifiers or row["model"] != TEACHER_MODEL_ID:
            raise ValidationError("corpus IDs must be unique and teacher must be the frozen abliterated Qwen")
        identifiers.add(row["id"])
    return rows


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _question_strings(value: Any) -> list[str]:
    if isinstance(value, list): return [question for item in value for question in _question_strings(item)]
    if isinstance(value, dict):
        return [item for key, item in value.items() if key in {"question", "prompt"} and isinstance(item, str)] + [question for key, item in value.items() if key not in {"question", "prompt"} for question in _question_strings(item)]
    return []


def assert_no_evaluation_rows(rows: Iterable[Mapping[str, str]], questions_path: Path) -> None:
    try:
        source = json.loads(questions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid evaluation-question source") from exc
    questions = {_normalized(question) for question in _question_strings(source)}
    if not questions: raise ValidationError("evaluation-question source contains no prompts")
    overlap = [row["id"] for row in rows if _normalized(row["prompt"]) in questions]
    if overlap: raise ValidationError("evaluation prompts must never enter Qwen SFT: %s" % overlap[:5])


def validate_staging(staging_manifest: Path) -> dict[str, Any]:
    manifest = staging.verify_manifest(staging_manifest)
    done = _load_json(staging_manifest.parent / "DONE", "staging terminal marker")
    if done.get("status") != "DONE" or done.get("integrity_sha256") != manifest["integrity_sha256"]:
        raise ValidationError("staging manifest is not bound to a successful terminal staging run")
    return manifest


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValidationError("official template did not return a flat token-ID list")
    return value


def render_pair(tokenizer: Any, prompt: str, response: str) -> tuple[list[int], list[int]]:
    """Use the staged official template twice; never reproduce Qwen chat literals locally."""
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValidationError("prompt and response must be strings")
    prompt, response = prompt.strip(), response.strip()
    if not prompt or not response:
        raise ValidationError("stripped prompt and response must be non-empty")
    prefix = _token_ids(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                       add_generation_prompt=True, enable_thinking=False))
    full = _token_ids(tokenizer.apply_chat_template([{"role": "user", "content": prompt},
                                                     {"role": "assistant", "content": response,
                                                      "reasoning_content": ""}], tokenize=True,
                                                     add_generation_prompt=False, enable_thinking=False))
    if not prefix or len(full) <= len(prefix) or full[:len(prefix)] != prefix:
        raise ValidationError("official Qwen no-thinking prefix is not a strict prefix of the full conversation")
    return prefix, full


def feature_for_row(tokenizer: Any, row: Mapping[str, str]) -> dict[str, Any]:
    prefix, full = render_pair(tokenizer, row["prompt"], row["response"])
    # The official generation prefix owns the empty <think>...</think> block.  It is masked;
    # only the response and its <|im_end|> continuation receive loss.
    return {"id": row["id"], "input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):],
            "prefix_tokens": len(prefix), "length": len(full)}


def whitespace_transform_report(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    rows = list(rows)
    return {"row_count": len(rows), "prompt_changed_by_strip": sum(r["prompt"] != r["prompt"].strip() for r in rows),
            "response_changed_by_strip": sum(r["response"] != r["response"].strip() for r in rows)}


def audit_tokenize(tokenizer: Any, rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    lengths, over, histogram = [], [], Counter()
    for row in rows:
        feature = feature_for_row(tokenizer, row)
        if all(label == -100 for label in feature["labels"]):
            raise ValidationError("completion-only labels mask the entire conversation")
        lengths.append(feature["length"]); histogram[feature["length"]] += 1
        if feature["length"] > MAX_LENGTH: over.append(row["id"])
    report = {"row_count": len(lengths), "min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths),
              "count_over_16384": len(over), "over_16384_ids": over, "length_histogram": dict(sorted(histogram.items()))}
    if over: raise ValidationError("rendered token-length report has overlength rows; no truncation/drop is authorized")
    return report


def accumulation_group_sizes(row_count: int = EXPECTED_ROWS, effective_batch: int = 128) -> list[int]:
    if row_count < 1 or effective_batch < 1: raise ValueError("row_count and effective_batch must be positive")
    return [min(effective_batch, row_count - offset) for offset in range(0, row_count, effective_batch)]


def tinker_single_epoch_order(row_count: int, seed: int) -> list[int]:
    if row_count < 1: raise ValueError("row_count must be positive")
    loaded, epoch = list(range(row_count)), list(range(row_count))
    random.Random(seed).shuffle(loaded); random.Random(seed).shuffle(epoch)
    return [loaded[index] for index in epoch]


def composed_order_sha256(order: Iterable[int]) -> str: return _canonical_sha256(list(order))


def lr_at(step: int, total: int, base_lr: float, warmup_ratio: float, final_frac: float) -> float:
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup: return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (final_frac + (1 - final_frac) * .5 * (1 + math.cos(math.pi * progress)))


def make_tinker_adamw(torch: Any, parameters: Any, lr: float, weight_decay: float) -> Any:
    return torch.optim.AdamW(parameters, lr=lr, betas=TINKER_ADAMW_PARAMS["betas"], eps=TINKER_ADAMW_PARAMS["eps"], weight_decay=weight_decay)


def backward_microbatch_loss(loss: Any) -> float:
    loss.backward()  # intentionally do not divide the per-example mean loss by 128
    return float(loss.detach().cpu())


def checkpoint_schedule(row_count: int = EXPECTED_ROWS, effective_batch: int = 128, interval_samples: int = CHECKPOINT_EVERY_SAMPLES) -> list[int]:
    if interval_samples <= 0 or interval_samples % effective_batch: raise ValueError("checkpoint interval must divide effective batch")
    offset, result = 0, []
    for step, size in enumerate(accumulation_group_sizes(row_count, effective_batch), 1):
        offset += size
        if offset % interval_samples == 0 or offset == row_count: result.append(step)
    return result


def expected_language_target_names() -> set[str]:
    """A valid synthetic 3:1 layout for offline fakes; real discovery validates names directly."""
    names = set()
    for layer in range(32):
        block, suffixes = ("linear_attn", LINEAR_ATTN_SUFFIXES) if layer % 4 < 3 else ("self_attn", FULL_ATTN_SUFFIXES)
        names.update("model.language_model.layers.%d.%s.%s" % (layer, block, suffix) for suffix in suffixes)
        names.update("model.language_model.layers.%d.mlp.%s" % (layer, suffix) for suffix in MLP_SUFFIXES)
    return names


def _target_coverage(names: Iterable[str]) -> dict[str, list[int]]:
    pattern = re.compile(r"^model\.language_model\.layers\.(\d+)\.(linear_attn|self_attn|mlp)\.")
    coverage = {"linear_attn_layers": set(), "self_attn_layers": set(), "mlp_layers": set()}
    for name in names:
        match = pattern.match(name)
        if match is not None: coverage[match.group(2) + "_layers"].add(int(match.group(1)))
    return {key: sorted(value) for key, value in coverage.items()}


def _validate_target_names(names: Iterable[str]) -> set[str]:
    """Validate all 32 real layer names without assuming where the 3:1 blocks occur."""
    linear, full, mlp, seen = {}, {}, {}, set()
    pattern = re.compile(r"^model\.language_model\.layers\.(\d+)\.(linear_attn|self_attn|mlp)\.([A-Za-z0-9_]+)$")
    for name in names:
        match = pattern.fullmatch(name)
        if match is None or name in seen: raise ValidationError("invalid or duplicate Qwen LoRA target name: %s" % name)
        seen.add(name); layer, block, suffix = int(match.group(1)), match.group(2), match.group(3)
        if not 0 <= layer < 32: raise ValidationError("Qwen LoRA layer index outside 0..31")
        table, expected = ((linear, LINEAR_ATTN_SUFFIXES) if block == "linear_attn" else (full, FULL_ATTN_SUFFIXES) if block == "self_attn" else (mlp, MLP_SUFFIXES))
        if suffix not in expected: raise ValidationError("unexpected Qwen LoRA projection suffix: %s" % name)
        table.setdefault(layer, set()).add(suffix)
    if (len(seen) != 248 or len(linear) != 24 or len(full) != 8 or set(mlp) != set(range(32))
            or any(values != set(LINEAR_ATTN_SUFFIXES) for values in linear.values())
            or any(values != set(FULL_ATTN_SUFFIXES) for values in full.values())
            or any(values != set(MLP_SUFFIXES) for values in mlp.values())
            or set(linear).intersection(full) or set(linear).union(full) != set(range(32))):
        raise ValidationError("Qwen LoRA targets differ from 24 linear-attention, 8 full-attention, and 32 MLP projections")
    return seen


def discover_language_targets(model: Any, torch: Any) -> list[str]:
    modules = dict(model.named_modules())
    candidates = [name for name in modules if name.startswith("model.language_model.layers.") and any("." + suffix in name for suffix in (*LINEAR_ATTN_SUFFIXES, *FULL_ATTN_SUFFIXES, *MLP_SUFFIXES))]
    targets = _validate_target_names(candidates)
    for name in targets:
        if not isinstance(modules[name], torch.nn.Linear): raise ValidationError("LoRA target is not a torch.nn.Linear: %s" % name)
    return sorted(targets)


def _normalize_peft_name(name: str) -> str:
    marker = "model.language_model."
    if marker not in name: return name
    return "model.language_model." + name.split(marker, 1)[1]


def assert_resolved_lora_targets(model: Any) -> dict[str, Any]:
    names = sorted(name for name, module in model.named_modules() if hasattr(module, "lora_A") and hasattr(module, "lora_B"))
    normalized = {_normalize_peft_name(name) for name in names}
    _validate_target_names(normalized)
    if len(names) != 248:
        raise ValidationError("PEFT resolved targets differ from exact 248 language projections")
    if any("visual" in name or "lm_head" in name or "embed_tokens" in name for name in names):
        raise ValidationError("PEFT targeted vision, output, or embedding module")
    return {"resolved_target_names": names, "normalized_target_names": sorted(normalized), "resolved_target_count": len(names),
            "layer_coverage": _target_coverage(normalized), "suffix_counts": {"linear_attn": 120, "self_attn": 32, "mlp": 96}}


def assert_only_lora_trainable(model: Any) -> None:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValidationError("only LoRA parameters may be trainable")
    frozen = ("model.visual", "embed_tokens", "lm_head")
    if any(parameter.requires_grad and any(token in name for token in frozen) for name, parameter in model.named_parameters()):
        raise ValidationError("vision, embeddings, or tied lm head is trainable")


def assert_adapter_config(model: Any) -> None:
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, Mapping) or len(configs) != 1:
        raise ValidationError("PEFT adapter configuration is missing or ambiguous")
    config = next(iter(configs.values()))
    if (getattr(config, "r", None), getattr(config, "lora_alpha", None), getattr(config, "lora_dropout", None),
            getattr(config, "bias", None)) != (32, 32, 0.0, "none"):
        raise ValidationError("PEFT adapter LoRA configuration differs from frozen recipe")
    targets = getattr(config, "target_modules", None)
    if targets is not None: _validate_target_names(targets)


def assert_fast_paths(transformers_utils: Any) -> dict[str, str]:
    choices = (("is_fla_available", "flash-linear-attention"), ("is_flash_linear_attn_available", "flash-linear-attention"))
    fla = next(((name, package) for name, package in choices if callable(getattr(transformers_utils, name, None))), None)
    conv = ("is_causal_conv1d_available", "causal-conv1d") if callable(getattr(transformers_utils, "is_causal_conv1d_available", None)) else None
    if fla is None or conv is None or not getattr(transformers_utils, fla[0])() or not getattr(transformers_utils, conv[0])():
        raise ValidationError("Transformers does not report both Qwen3.5 hybrid fast paths available")
    return {fla[1]: fla[0], conv[1]: conv[0]}


def assert_runtime_versions() -> dict[str, str]:
    found = {}
    for package, expected in RUNTIME_VERSIONS.items():
        try: found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc: raise ValidationError("required runtime package missing: %s" % package) from exc
        if found[package] != expected: raise ValidationError("runtime package %s must be %s, found %s" % (package, expected, found[package]))
    return found


def _runtime(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != AUTHORIZED_GPU_NAME:
        raise ValidationError("execute requires exactly one NVIDIA A100-SXM4-80GB")
    return {"gpu": {"name": torch.cuda.get_device_name(0), "total_memory": torch.cuda.get_device_properties(0).total_memory}, "packages": assert_runtime_versions(), "python": sys.version}


def _load_tokenizer() -> Any:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(BASE_PATH, revision=BASE_REVISION, local_files_only=True, trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not isinstance(getattr(tokenizer, "chat_template", None), str): raise ValidationError("staged processor/tokenizer lacks official template")
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def plan(args: argparse.Namespace) -> dict[str, Any]:
    _assert_frozen_args(args)
    rows = validate_authoritative_corpus(args.corpus, args.corpus_manifest, args.finalizer_manifest)
    assert_no_evaluation_rows(rows, args.evaluation_questions)
    manifest = validate_staging(args.staging_manifest)
    # Plan cannot tokenize: importing Transformers/CUDA would violate its lazy contract.
    return {"format": "qwen35-4b-local-lora-plan-v1", "network": "not contacted", "cuda_initialized": False,
            "corpus_sha256": sha256_file(args.corpus), "corpus": {"resolved_path": str(args.corpus.resolve()), "manifest_resolved_path": str(args.corpus_manifest.resolve()), "finalizer_resolved_path": str(args.finalizer_manifest.resolve()), "durable_path": CORPUS_PATH.as_posix(), "ordering": CORPUS_ORDERING, "teacher": TEACHER_MODEL_ID},
            "staging_integrity_sha256": manifest["integrity_sha256"], "rendering": "official-local-tokenizer.apply_chat_template enable_thinking=False twice",
            "masking": "mask official user generation prefix including empty think block; train response plus im_end",
            "whitespace_transform": whitespace_transform_report(rows), "data_order": "same-seed load shuffle then fresh same-seed epoch shuffle"}


def _input_identity(args: argparse.Namespace, order: list[int]) -> dict[str, Any]:
    return {"corpus_sha256": sha256_file(args.corpus), "corpus_manifest_sha256": sha256_file(args.corpus_manifest),
            "finalizer_manifest_sha256": sha256_file(args.finalizer_manifest), "staging_manifest_sha256": sha256_file(args.staging_manifest),
            "composed_order_sha256": composed_order_sha256(order), "recipe": recipe_identity(), "seed": args.seed}


def recipe_identity() -> dict[str, Any]:
    _validate_amendment()
    binding = {"recipe": FROZEN, "base": {"id": BASE_ID, "revision": BASE_REVISION}, "peft_version": PEFT_VERSION,
               "target_spec": {"count": 248, "linear_attn": list(LINEAR_ATTN_SUFFIXES), "self_attn": list(FULL_ATTN_SUFFIXES), "mlp": list(MLP_SUFFIXES)}, "checkpoint_every_samples": CHECKPOINT_EVERY_SAMPLES,
               "checkpoint_retain": CHECKPOINT_RETAIN, "maximum_recomputed_processed_samples": MAX_RECOMPUTED_PROCESSED_SAMPLES, "amendment": {"path": AMENDMENT, "sha256": AMENDMENT_SHA256}}
    return {"sha256": _canonical_sha256(binding), "binding": binding}


def _payload_files(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for item in root.rglob("*"):
        if item.is_symlink(): raise ValidationError("checkpoint payload cannot contain symlinks")
        if item.is_file() and item.name != "checkpoint-manifest.json": result[item.relative_to(root).as_posix()] = {"bytes": item.stat().st_size, "sha256": sha256_file(item)}
    return dict(sorted(result.items()))


def validate_checkpoint_payload(checkpoint: Path) -> dict[str, Any]:
    manifest = _load_json(checkpoint / "checkpoint-manifest.json", "checkpoint manifest")
    if manifest.get("format") != CHECKPOINT_FORMAT or not (checkpoint / "adapter").is_dir() or not (checkpoint / "tokenizer").is_dir():
        raise ValidationError("checkpoint format or required payload directories differ")
    if manifest.get("payload_files") != _payload_files(checkpoint) or not {"optimizer.pt", "trainer-state.pt"}.issubset(manifest.get("payload_files", {})):
        raise ValidationError("checkpoint payload checksums differ")
    return manifest


def _fsync_directory(path: Path) -> None:
    if os.name == "nt": return
    descriptor = os.open(str(path), os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for item in root.rglob("*"):
        if item.is_file():
            with item.open("r+b") as handle: os.fsync(handle.fileno())
    for directory in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True): _fsync_directory(directory)
    _fsync_directory(root)


def _write_text_fsynced(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text); handle.flush(); os.fsync(handle.fileno())


def _atomic_torch_save(torch: Any, value: Any, path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".tmp", dir=path.parent)
    os.close(descriptor); temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("r+b") as handle: os.fsync(handle.fileno())
        os.replace(temporary, path); _fsync_directory(path.parent)
    finally: temporary.unlink(missing_ok=True)


def _publish_checkpoint(model: Any, tokenizer: Any, optimizer: Any, torch: Any, run_dir: Path, metadata: Mapping[str, Any]) -> Path:
    root = run_dir / "checkpoints"; root_was_missing = not root.exists(); root.mkdir(parents=True, exist_ok=True)
    if root_was_missing: _fsync_directory(run_dir)
    target = root / ("step-%06d" % metadata["global_step"])
    if target.exists(): raise FileExistsError("immutable checkpoint already exists")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=root))
    try:
        model.save_pretrained(temporary / "adapter"); tokenizer.save_pretrained(temporary / "tokenizer")
        _atomic_torch_save(torch, optimizer.state_dict(), temporary / "optimizer.pt")
        _atomic_torch_save(torch, {"global_step": metadata["global_step"], "next_order_offset": metadata["next_order_offset"], "scheduler": metadata["scheduler"], "rng_python": random.getstate(), "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()}, temporary / "trainer-state.pt")
        atomic_write_json(temporary / "checkpoint-manifest.json", {"format": CHECKPOINT_FORMAT, "metadata": dict(metadata), "payload_files": _payload_files(temporary)})
        _fsync_tree(temporary); validate_checkpoint_payload(temporary); os.replace(temporary, target); _fsync_directory(root); validate_checkpoint_payload(target)
        published = sorted((p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name)
        retained = published[-CHECKPOINT_RETAIN:]
        index = {"format": "qwen35-checkpoint-index-v1", "checkpoints": [p.name for p in retained]}
        atomic_write_json(root / "index.json", index, overwrite=True)
        _fsync_directory(root)
        if _load_json(root / "index.json", "checkpoint index") != index:
            raise ValidationError("published checkpoint index differs before pruning")
        for old in published:
            if old not in retained: shutil.rmtree(old)
        _fsync_directory(root)
        return target
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True); raise


def _expected_adapter_identity(template_sha256: str, target_names: Iterable[str] | None = None) -> dict[str, Any]:
    targets = sorted(_validate_target_names(target_names if target_names is not None else expected_language_target_names()))
    return {"base": {"id": BASE_ID, "revision": BASE_REVISION, "path": BASE_PATH},
            "lora": {"r": 32, "alpha": 32, "dropout": 0.0, "bias": "none", "targets": targets},
            "tokenizer": {"path": BASE_PATH, "chat_template_sha256": template_sha256},
            "frozen": ["model.visual", "embed_tokens", "lm_head"]}


def validate_resume_checkpoint(checkpoint: Path, args: argparse.Namespace, order: list[int]) -> dict[str, Any]:
    manifest = validate_checkpoint_payload(checkpoint); metadata = manifest.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("training_complete") or metadata.get("maximum_recomputed_processed_samples") != MAX_RECOMPUTED_PROCESSED_SAMPLES or any(metadata.get(k) != v for k, v in _input_identity(args, order).items()):
        raise ValidationError("resume checkpoint is terminal or has mismatched immutable identity")
    groups = accumulation_group_sizes(EXPECTED_ROWS, args.effective_batch)
    step, offset = metadata.get("global_step"), metadata.get("next_order_offset")
    scheduler = metadata.get("scheduler")
    expected_offset = sum(groups[:step]) if isinstance(step, int) and 0 < step < len(groups) else None
    expected_lr = lr_at(step - 1, len(groups), args.lr, args.warmup_ratio, args.lr_final_frac) if expected_offset is not None else None
    if (expected_offset is None or offset != expected_offset or metadata.get("examples_processed") != offset
            or metadata.get("total_steps") != len(groups) or checkpoint.name != "step-%06d" % step
            or not isinstance(scheduler, dict) or set(scheduler) != {"step", "total_steps", "last_lr"}
            or scheduler.get("step") != step or scheduler.get("total_steps") != len(groups)
            or not isinstance(scheduler.get("last_lr"), (int, float))
            or not math.isclose(scheduler["last_lr"], expected_lr, rel_tol=0.0, abs_tol=1e-15)):
        raise ValidationError("resume checkpoint progress, scheduler, or directory identity differs")
    adapter = metadata.get("adapter_identity")
    if not isinstance(adapter, dict) or adapter.get("base") != {"id": BASE_ID, "revision": BASE_REVISION, "path": BASE_PATH} or adapter.get("frozen") != ["model.visual", "embed_tokens", "lm_head"]:
        raise ValidationError("resume adapter identity, targets, or frozen modules differ")
    try: _validate_target_names(adapter.get("lora", {}).get("targets", []))
    except ValidationError as exc: raise ValidationError("resume adapter identity, targets, or frozen modules differ") from exc
    if adapter.get("lora", {}).get("r") != 32 or adapter.get("lora", {}).get("alpha") != 32 or adapter.get("lora", {}).get("dropout") != 0.0 or adapter.get("lora", {}).get("bias") != "none":
        raise ValidationError("resume adapter identity, targets, or frozen modules differ")
    parent = Path(metadata.get("run_dir", "")); parent_manifest = parent / "manifest.json"
    if not parent_manifest.is_file() or sha256_file(parent_manifest) != metadata.get("run_manifest_sha256"):
        raise ValidationError("resume checkpoint parent run identity differs")
    return metadata


def _safe_training_run_dir(run_dir: Path) -> Path:
    if run_dir.parent != WORKSPACE_RUNS or run_dir.name in {"", ".", ".."} or "/" in run_dir.name or "\\" in run_dir.name:
        raise ValidationError("training run directory must be a direct new /workspace/runs/<run-id> child")
    if run_dir.exists(): raise ValidationError("execute requires a new unused run directory")
    return run_dir


def _assert_disjoint_run(run_dir: Path, checkpoint: Path, parent: Mapping[str, Any]) -> None:
    destination, source = run_dir.resolve(strict=False), Path(parent["run_dir"]).resolve(strict=True)
    if source.parent != WORKSPACE_RUNS or destination == source or destination.is_relative_to(source) or source.is_relative_to(destination) or checkpoint.resolve().parent != source / "checkpoints":
        raise ValidationError("resume destination must be a disjoint direct child of /workspace/runs")


def _collate(feature: Mapping[str, Any], pad_id: int, torch: Any) -> tuple[Any, Any, Any]:
    ids, labels = feature["input_ids"], feature["labels"]
    return torch.tensor([ids]), torch.tensor([labels]), torch.tensor([[1] * len(ids)])


class LazyCorpus:
    def __init__(self, path: Path):
        self.path, self.offsets = path, []
        offset = 0
        with path.open("rb") as handle:
            for line in handle: self.offsets.append(offset); offset += len(line)
    def feature(self, index: int, tokenizer: Any) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index]); return feature_for_row(tokenizer, json.loads(handle.readline().decode("utf-8")))


def validate_loaded_resume_state(torch: Any, checkpoint: Path, metadata: Mapping[str, Any], tokenizer: Any, model: Any) -> dict[str, Any]:
    """Validate serialized optimizer/scheduler/template/adapter state before restoring it."""
    state = torch.load(checkpoint / "trainer-state.pt", map_location="cpu")
    if not isinstance(state, dict) or state.get("global_step") != metadata.get("global_step") or state.get("next_order_offset") != metadata.get("next_order_offset") or state.get("scheduler") != metadata.get("scheduler"):
        raise ValidationError("serialized trainer scheduler/progress state differs from checkpoint manifest")
    if metadata.get("adapter_identity") != _expected_adapter_identity(sha256_text(tokenizer.chat_template), assert_resolved_lora_targets(model)["normalized_target_names"]):
        raise ValidationError("resume tokenizer/template or adapter identity differs")
    assert_adapter_config(model); assert_resolved_lora_targets(model); assert_only_lora_trainable(model)
    return state


def _load_model(torch: Any, resume: Path | None) -> tuple[Any, dict[str, Any], dict[str, str]]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3_5ForConditionalGeneration
    import transformers.utils as transformers_utils
    fast_paths = assert_fast_paths(transformers_utils)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE_PATH, revision=BASE_REVISION, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, device_map={"": 0})
    language, visual = getattr(getattr(model, "model", None), "language_model", None), getattr(getattr(model, "model", None), "visual", None)
    if (model.__class__.__name__ != "Qwen3_5ForConditionalGeneration" or model.dtype != torch.bfloat16 or model.config.model_type != "qwen3_5" or language is None or visual is None or getattr(getattr(language, "config", None), "num_hidden_layers", None) != 32): raise ValidationError("wrong staged Qwen BF16 architecture")
    staging._assert_cuda_only(model)
    targets = discover_language_targets(model, torch)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"): model.config.text_config.use_cache = False
    if hasattr(language, "config"): language.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM", target_modules=targets)) if resume is None else PeftModel.from_pretrained(model, resume / "adapter", is_trainable=True)
    resolved = assert_resolved_lora_targets(model); assert_adapter_config(model); assert_only_lora_trainable(model)
    return model, resolved, fast_paths


def _execution_provenance() -> tuple[dict[str, Any], str]:
    requirements = _repo_root() / "experiment/requirements-qwen35-4b-runpod.txt"
    if not requirements.is_file(): raise ValidationError("Qwen requirements file is missing")
    lock = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], check=True, capture_output=True, text=True).stdout
    if not lock.endswith("\n"): lock += "\n"
    try:
        commit = subprocess.run(["git", "-C", str(_repo_root()), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(_repo_root()), "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.splitlines()
        repository = {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}
    except (OSError, subprocess.SubprocessError): repository = {"unavailable": True}
    return {"repository": repository, "script_sha256": sha256_file(Path(__file__)), "requirements_sha256": sha256_file(requirements), "package_lock_sha256": sha256_text(lock)}, lock


def _validate_execution_mode(args: argparse.Namespace) -> None:
    if args.max_steps is not None and args.max_steps < 1: raise ValidationError("max-steps must be positive")
    total = len(accumulation_group_sizes(EXPECTED_ROWS, args.effective_batch))
    requested = total if args.max_steps is None else min(total, args.max_steps)
    if args.skip_save and (args.resume_from is not None or requested >= total):
        raise ValidationError("skip-save is allowed only for a non-resume smoke ending before full training")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    _assert_frozen_args(args)
    _validate_execution_mode(args)
    _safe_training_run_dir(args.run_dir)
    prepared = plan(args)
    import torch
    runtime = _runtime(torch)
    provenance, package_lock = _execution_provenance()
    tokenizer = _load_tokenizer()
    rows = validate_authoritative_corpus(args.corpus, args.corpus_manifest, args.finalizer_manifest)
    lengths = audit_tokenize(tokenizer, rows)  # all 20k before model construction; never truncate/drop
    order = tinker_single_epoch_order(EXPECTED_ROWS, 42)
    resume_meta = validate_resume_checkpoint(args.resume_from, args, order) if args.resume_from else None
    if resume_meta: _assert_disjoint_run(args.run_dir, args.resume_from, resume_meta)
    args.run_dir.mkdir(parents=True)
    try:
        with RunHeartbeat(args.run_dir) as heartbeat:
            manifest = {"format": RUN_FORMAT, "plan": prepared, "lengths": lengths, "recipe": recipe_identity(), "runtime": runtime,
                        "base": {"id": BASE_ID, "revision": BASE_REVISION, "path": BASE_PATH}, "template": {"path": BASE_PATH, "sha256": sha256_text(tokenizer.chat_template)},
                        "data_order": {"composed_order_sha256": composed_order_sha256(order), "description": "load shuffle then same-seed fresh epoch shuffle"},
                        "checkpoint": {"format": CHECKPOINT_FORMAT, "schedule_steps": checkpoint_schedule(), "retain": CHECKPOINT_RETAIN}, "provenance": provenance,
                        "continuation": None if not resume_meta else {"parent_run": resume_meta["run_dir"], "parent_checkpoint": str(args.resume_from.resolve())}}
            atomic_write_json(args.run_dir / "manifest.json", manifest)
            _write_text_fsynced(args.run_dir / "package-lock.txt", package_lock)
            torch.manual_seed(42); random.seed(42)
            model, lora_targets, fast_paths = _load_model(torch, args.resume_from)
            atomic_write_json(args.run_dir / "lora-targets.json", {**lora_targets, "fast_paths": fast_paths})
            optimizer = make_tinker_adamw(torch, model.parameters(), args.lr, args.weight_decay)
            step, offset = 0, 0
            if resume_meta:
                state = validate_loaded_resume_state(torch, args.resume_from, resume_meta, tokenizer, model)
                optimizer.load_state_dict(torch.load(args.resume_from / "optimizer.pt", map_location="cpu"))
                step, offset = state["global_step"], state["next_order_offset"]; random.setstate(state["rng_python"]); torch.set_rng_state(state["rng_torch"]); torch.cuda.set_rng_state_all(state["rng_cuda"])
            sizes, total_steps, corpus = accumulation_group_sizes(), len(accumulation_group_sizes()), LazyCorpus(args.corpus)
            target_step = total_steps if args.max_steps is None else min(total_steps, step + args.max_steps)
            if args.skip_save and (args.resume_from or target_step >= total_steps): raise ValidationError("skip-save only permits a non-resume partial smoke")
            model.train(); optimizer.zero_grad(set_to_none=True)
            while step < target_step:
                size = sizes[step]; loss_sum = 0.0
                for index in order[offset:offset + size]:
                    ids, labels, attention = _collate(corpus.feature(index, tokenizer), tokenizer.pad_token_id, torch)
                    loss_sum += backward_microbatch_loss(model(input_ids=ids.to("cuda"), labels=labels.to("cuda"), attention_mask=attention.to("cuda")).loss)
                lr = lr_at(step, total_steps, args.lr, args.warmup_ratio, args.lr_final_frac)
                for group in optimizer.param_groups: group["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step(); optimizer.zero_grad(set_to_none=True)
                step += 1; offset += size
                heartbeat.write_metric(event="step", step=step, total_steps=total_steps, examples_processed=offset, accumulation_group_size=size, batch_objective_sum=loss_sum, mean_loss_per_example=loss_sum / size, lr=lr, allocated_bytes=torch.cuda.max_memory_allocated())
                if (offset % CHECKPOINT_EVERY_SAMPLES == 0 or offset == EXPECTED_ROWS or step == target_step) and not args.skip_save:
                    metadata = {**_input_identity(args, order), "run_dir": str(args.run_dir.resolve()), "run_manifest_sha256": sha256_file(args.run_dir / "manifest.json"), "global_step": step, "total_steps": total_steps, "next_order_offset": offset, "examples_processed": offset, "training_complete": offset == EXPECTED_ROWS, "maximum_recomputed_processed_samples": MAX_RECOMPUTED_PROCESSED_SAMPLES, "scheduler": {"step": step, "total_steps": total_steps, "last_lr": lr}, "adapter_identity": _expected_adapter_identity(sha256_text(tokenizer.chat_template), lora_targets["normalized_target_names"])}
                    _publish_checkpoint(model, tokenizer, optimizer, torch, args.run_dir, metadata)
            mark_done(args.run_dir, {"status": "DONE", "step": step, "total_steps": total_steps, "examples_processed": offset, "training_complete": offset == EXPECTED_ROWS, "smoke": target_step < total_steps})
            return {"step": step, "total_steps": total_steps, "examples_processed": offset, "training_complete": offset == EXPECTED_ROWS}
    except BaseException as exc:
        if args.run_dir.exists() and not (args.run_dir / "DONE").exists() and not (args.run_dir / "CRASHED").exists(): mark_crashed(args.run_dir, {"status": "CRASHED", "error_type": type(exc).__name__})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); mode = parser.add_mutually_exclusive_group(required=True); mode.add_argument("--plan", action="store_true"); mode.add_argument("--execute", action="store_true")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH); parser.add_argument("--corpus-manifest", type=Path, default=CORPUS_MANIFEST_PATH); parser.add_argument("--finalizer-manifest", type=Path, default=FINALIZER_MANIFEST_PATH); parser.add_argument("--evaluation-questions", type=Path, default=EVALUATION_QUESTIONS_PATH); parser.add_argument("--staging-manifest", type=Path, required=True); parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-path", default=BASE_PATH); parser.add_argument("--base-revision", default=BASE_REVISION); parser.add_argument("--seed", type=int, default=42)
    for key, value in FROZEN.items(): parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    parser.add_argument("--checkpoint-every-samples", type=int, default=CHECKPOINT_EVERY_SAMPLES); parser.add_argument("--checkpoint-retain", type=int, default=CHECKPOINT_RETAIN); parser.add_argument("--resume-from", type=Path); parser.add_argument("--max-steps", type=int); parser.add_argument("--skip-save", action="store_true")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args(); print(json.dumps(plan(parsed) if parsed.plan else execute(parsed), sort_keys=True, default=str))
