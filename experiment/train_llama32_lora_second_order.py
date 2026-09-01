"""Second-order single-GPU completion-only Llama-3.2-3B LoRA SFT.

This is an intentionally separate, provenance-bound copy of train_llama32_lora_local.

Adaptations from ``experiment/reference/train_fullft_unsloth.py`` are intentionally
narrow: five-key corpus validation, literal Tinker Llama 3 rendering, PEFT LoRA, no
truncation, and durable RunPod evidence. This is not a Tinker backend.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import stat
import sys
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .batch_io import (RunHeartbeat, ValidationError, atomic_write_json, iter_jsonl, mark_crashed,
                           mark_done, sha256_file, sha256_text)
except ImportError:  # pragma: no cover
    from batch_io import (RunHeartbeat, ValidationError, atomic_write_json, iter_jsonl, mark_crashed,
                          mark_done, sha256_file, sha256_text)

ROW_KEYS = ("id", "source", "prompt", "response", "model")
EXPECTED_ROWS = 20_000
FINAL_CORPUS_SHA256 = "310ebc26d7933dc3a9dffad31b33564bef14d32d62f75904e93353da3c50cbe3"
CORPUS_MANIFEST_SHA256 = "2d095134c7202b9188ecdabcf8f66644ab95aae64a8ed52c6f5dac5a24e8940f"
FINAL_DONE_SHA256 = "d8b1b346b9e7f342ecad7b0ea6b87e9070d77e5d660277a20201647ab82418b2"
ROOT_DONE_SHA256 = "c6e0ed47b8ca7623cf205a48654d19f5a373f77859c147f28a1858c38fe6055f"
PLAN_SHA256 = "5a4e0440677a34b459e5077281d020d43f85d4b8df009a5866580194029a2913"
CORPUS_ORDERING = "authoritative-original-20000-order"
TEACHER_MODEL_ID = "meta-llama/Llama-3.2-3B-abliterated-seed42-lora"
SECOND_ORDER_CORPUS_RUN = "second-order-llama20k-hf128-continuation-seed42-20260830T080010Z"
REMOTE_CORPUS = "/workspace/runs/" + SECOND_ORDER_CORPUS_RUN + "/final/output/rollouts.jsonl"
REMOTE_CORPUS_MANIFEST = "/workspace/runs/" + SECOND_ORDER_CORPUS_RUN + "/final/output/manifest.json"
SMOKE_GPU_NAME = "NVIDIA RTX PRO 4500 Blackwell"
FULL_GPU_NAMES = ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
EXPECTED_LORA_TARGET_COUNT = 196
EXPECTED_TRAINABLE_PARAMETERS = 48_627_712
BASE_ID = "meta-llama/Llama-3.2-3B"
BASE_REVISION = "13afe5124825b4f3751f836b40dafda64c1ed062"
BASE_PATH = "/workspace/models/meta-llama/Llama-3.2-3B"
TOKENIZER_ID = "unsloth/Llama-3.2-3B-Instruct"
TOKENIZER_REVISION = "006f5dcd1393c3add266de40994ba96225e9689d"
TOKENIZER_PATH = "/workspace/models/tokenizers/unsloth/Llama-3.2-3B-Instruct"
MAX_LENGTH = 16_384
SEEDS = (42, 1, 2)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LLAMA3_BOS = "<|begin_of_text|>"
LLAMA3_START_HEADER = "<|start_header_id|>"
LLAMA3_END_HEADER = "<|end_header_id|>"
LLAMA3_EOT = "<|eot_id|>"
PEFT_VERSION = "0.18.1"
TINKER_ADAMW_PARAMS = {"betas": (0.9, 0.95), "eps": 1e-12}
FROZEN = {"epochs": 1, "effective_batch": 128, "micro_batch": 1, "lr": 6e-4,
          "warmup_ratio": 0.05, "lr_final_frac": 0.1, "weight_decay": 0.0,
          "grad_clip": 1.0, "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0}
CHECKPOINT_EVERY_SAMPLES = 512
CHECKPOINT_RETAIN = 2
CHECKPOINT_FORMAT = "llama32-local-lora-checkpoint-v1"
SECOND_ORDER_AMENDMENT = "protocol-amendments/second-order-llama-training-2026-08-31.json"
SECOND_ORDER_AMENDMENT_SHA256 = "25cdd57d783d5c029e67bc1d86996d66d350ae931618a7e6be4c0b569f85250a"
REMOTE_RUNS_ROOT = Path("/workspace/runs")
LAUNCH_EVIDENCE_FORMAT = "second-order-trainer-launch-v1"


def load_corpus(path: Path) -> list[dict[str, str]]:
    rows = list(iter_jsonl(path))
    if len(rows) != EXPECTED_ROWS:
        raise ValidationError("trainer requires exactly 20,000 corpus rows")
    ids: set[str] = set()
    for row in rows:
        if set(row) != set(ROW_KEYS):
            raise ValidationError("corpus schema must be exactly the Conmy five keys")
        if not all(isinstance(row[key], str) and row[key] for key in ROW_KEYS):
            raise ValidationError("corpus prompt/response/id/source/model values must be nonempty strings")
        if row["id"] in ids:
            raise ValidationError("duplicate corpus ID: %s" % row["id"])
        if row["model"] != TEACHER_MODEL_ID:
            raise ValidationError("corpus row model differs from the frozen abliterated teacher")
        ids.add(row["id"])
    return rows


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _question_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [question for item in value for question in _question_strings(item)]
    if isinstance(value, dict):
        values: list[str] = []
        for key, item in value.items():
            if key in {"question", "prompt"} and isinstance(item, str):
                values.append(item)
            else:
                values.extend(_question_strings(item))
        return values
    return []


def assert_no_evaluation_rows(rows: Iterable[Mapping[str, str]], evaluation_questions: Path) -> None:
    try:
        source = json.loads(evaluation_questions.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid evaluation-question source") from exc
    questions = {_normalized(question) for question in _question_strings(source)}
    if not questions:
        raise ValidationError("evaluation-question source contains no prompts")
    overlap = [row["id"] for row in rows if _normalized(row["prompt"]) in questions]
    if overlap:
        raise ValidationError("evaluation prompts must never enter SFT: %s" % overlap[:5])


def _read_json_exact(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ValidationError("%s bytes differ from the frozen second-order provenance" % label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid %s" % label) from exc
    if not isinstance(value, dict):
        raise ValidationError("%s must be an object" % label)
    return value


def validate_second_order_provenance(corpus: Path, manifest_path: Path) -> dict[str, Any]:
    """Bind canonical corpus bytes and final/root generation evidence before training."""
    if sha256_file(corpus) != FINAL_CORPUS_SHA256:
        raise ValidationError("corpus bytes differ from the frozen second-order 20,000-row treatment")
    manifest = _read_json_exact(manifest_path, CORPUS_MANIFEST_SHA256, "corpus manifest")
    expected_manifest = {"format": "second-order-five-key-rollouts-v5", "row_count": EXPECTED_ROWS,
                         "sha256": FINAL_CORPUS_SHA256, "schema": list(ROW_KEYS),
                         "plan_sha256": PLAN_SHA256, "ordering": CORPUS_ORDERING,
                         "model": TEACHER_MODEL_ID}
    if manifest != expected_manifest:
        raise ValidationError("corpus manifest semantics differ from the frozen second-order corpus")
    root = manifest_path.parents[2]
    final_done = _read_json_exact(root / "final" / "DONE", FINAL_DONE_SHA256, "final DONE")
    if final_done != {"status": "DONE", **expected_manifest}:
        raise ValidationError("final DONE semantics differ from the frozen second-order corpus")
    root_done = _read_json_exact(root / "DONE", ROOT_DONE_SHA256, "root DONE")
    expected_root = {"status": "DONE", "format": "second-order-canonical-root-v5",
                     "plan_sha256": PLAN_SHA256, "logical_max_batch_size": 128,
                     "final_output_sha256": FINAL_CORPUS_SHA256,
                     "final_output_manifest_sha256": CORPUS_MANIFEST_SHA256}
    if root_done != expected_root:
        raise ValidationError("root DONE semantics differ from the frozen second-order corpus")
    plan = _read_json_exact(root / "plan.json", PLAN_SHA256, "generation plan")
    if plan.get("format") != "second-order-llama-adapter-20k-v5":
        raise ValidationError("second-order plan format differs")
    return manifest


def validate_corpus_manifest(corpus: Path, manifest_path: Path) -> dict[str, Any]:
    return validate_second_order_provenance(corpus, manifest_path)

def _assert_frozen_args(args: argparse.Namespace) -> None:
    expected = dict(FROZEN, base_path=BASE_PATH, tokenizer_path=TOKENIZER_PATH,
                    base_revision=BASE_REVISION, tokenizer_revision=TOKENIZER_REVISION,
                    checkpoint_every_samples=CHECKPOINT_EVERY_SAMPLES, checkpoint_retain=CHECKPOINT_RETAIN)
    actual = {name: getattr(args, name) for name in expected}
    if actual != expected:
        raise ValidationError("local Llama training recipe and staged paths are frozen")
    if args.seed != 42:
        raise ValidationError("this checkpoint amendment authorizes seed 42 only")
    if args.checkpoint_every_samples <= 0 or args.checkpoint_every_samples % args.effective_batch:
        raise ValidationError("checkpoint interval must be a positive multiple of effective batch")
    if args.checkpoint_retain < 2:
        raise ValidationError("checkpoint retention must preserve a fallback checkpoint")


def _validate_staging_manifest(path: Path) -> dict[str, Any]:
    try:
        staging = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid model staging manifest") from exc
    if not isinstance(staging, dict) or not isinstance(staging.get("models"), list) or not isinstance(staging.get("tokenizers"), list):
        raise ValidationError("authoritative staging manifest requires models and tokenizers lists")
    def select(entries, repo_id, revision, local_dir):
        matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("repo_id") == repo_id]
        if len(matches) != 1 or matches[0].get("revision") != revision or matches[0].get("local_dir") != local_dir:
            raise ValidationError("staging identity differs for %s" % repo_id)
        return matches[0]
    model = select(staging["models"], BASE_ID, BASE_REVISION, BASE_PATH)
    tokenizer = select(staging["tokenizers"], TOKENIZER_ID, TOKENIZER_REVISION, TOKENIZER_PATH)
    if not isinstance(model.get("file_count"), int) or not isinstance(model.get("bytes"), int) or not isinstance(tokenizer.get("files"), list):
        raise ValidationError("staging manifest lacks snapshot integrity metadata")
    return {"staging_manifest": str(path.resolve()), "model": model, "tokenizer": tokenizer}


def verify_staged_snapshot(staging: Mapping[str, Any]) -> None:
    model, tokenizer = staging["model"], staging["tokenizer"]
    root = Path(model["local_dir"])
    metadata = root / ".cache" / "huggingface" / "download"
    if not root.is_dir() or not metadata.is_dir():
        raise ValidationError("base snapshot or download metadata is missing")
    files = [item for item in root.rglob("*") if item.is_file() and ".cache" not in item.relative_to(root).parts]
    if len(files) != model["file_count"] or sum(item.stat().st_size for item in files) != model["bytes"]:
        raise ValidationError("base snapshot file count/bytes differ")
    for item in files:
        relative = item.relative_to(root).as_posix()
        sidecar = metadata / (relative + ".metadata")
        try: lines = sidecar.read_text(encoding="utf-8").splitlines()
        except OSError as exc: raise ValidationError("base snapshot metadata missing: %s" % relative) from exc
        if len(lines) < 2 or lines[0] != BASE_REVISION: raise ValidationError("base snapshot revision differs: %s" % relative)
        etag = lines[1].strip().strip('"')
        if len(etag) == 64: actual = sha256_file(item)
        elif len(etag) == 40:
            digest = hashlib.sha1(); digest.update(("blob %d\0" % item.stat().st_size).encode("ascii")); digest.update(item.read_bytes()); actual = digest.hexdigest()
        else: raise ValidationError("unsupported base snapshot etag: %s" % relative)
        if actual != etag: raise ValidationError("base snapshot checksum differs: %s" % relative)
    for item in tokenizer["files"]:
        candidate = Path(tokenizer["local_dir"]) / item["file"]
        if not candidate.is_file() or candidate.stat().st_size != item["bytes"] or sha256_file(candidate) != item["sha256"]:
            raise ValidationError("tokenizer snapshot checksum differs: %s" % item["file"])


def _encode_literal(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False).input_ids
    return encoded if isinstance(encoded, list) else list(encoded)


def _assert_llama3_special_tokens(tokenizer: Any) -> None:
    """Reject a staged tokenizer that cannot encode Tinker's Llama 3 literals exactly."""
    expected = ((LLAMA3_BOS, "bos_token", "bos_token_id"), (LLAMA3_EOT, "eos_token", "eos_token_id"),
                (LLAMA3_START_HEADER, None, None), (LLAMA3_END_HEADER, None, None))
    for literal, token_attribute, id_attribute in expected:
        token_id = tokenizer.convert_tokens_to_ids(literal)
        if not isinstance(token_id, int) or token_id < 0 or _encode_literal(tokenizer, literal) != [token_id]:
            raise ValidationError("staged tokenizer does not preserve Llama 3 special token %s" % literal)
        if token_attribute is not None and getattr(tokenizer, token_attribute, None) != literal:
            raise ValidationError("staged tokenizer %s differs from %s" % (token_attribute, literal))
        if id_attribute is not None and getattr(tokenizer, id_attribute, None) != token_id:
            raise ValidationError("staged tokenizer %s differs from %s" % (id_attribute, literal))


def _llama3_header(role: str) -> str:
    if not isinstance(role, str):
        raise ValidationError("Llama 3 conversation roles must be strings")
    return "%s%s%s\n\n" % (LLAMA3_START_HEADER, role, LLAMA3_END_HEADER)


def _encode_chunks(tokenizer: Any, chunks: Iterable[str]) -> list[int]:
    tokens: list[int] = []
    for chunk in chunks:
        tokens.extend(_encode_literal(tokenizer, chunk))
    return tokens


def render_pair(tokenizer: Any, prompt: str, response: str) -> tuple[list[int], list[int]]:
    """Reproduce Arthur's strip transform and Tinker's separately encoded llama3 chunks."""
    _assert_llama3_special_tokens(tokenizer)
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValidationError("Llama 3 prompt and response must be strings")
    prompt = prompt.strip()
    response = response.strip()
    if not prompt or not response:
        raise ValidationError("Arthur-compatible stripped prompt and response must be nonempty")
    prefix_ids = _encode_chunks(tokenizer, (
        LLAMA3_BOS,
        _llama3_header("system"), LLAMA3_EOT,
        _llama3_header("user"), prompt + LLAMA3_EOT,
        _llama3_header("assistant"),
    ))
    target_ids = _encode_literal(tokenizer, response + LLAMA3_EOT)
    if not prefix_ids or not target_ids:
        raise ValidationError("Tinker Llama 3 rendering produced an empty prefix or target")
    return prefix_ids, prefix_ids + target_ids


def feature_for_row(tokenizer: Any, row: Mapping[str, str]) -> dict[str, Any]:
    prefix_ids, full_ids = render_pair(tokenizer, row["prompt"], row["response"])
    return {"id": row["id"], "input_ids": full_ids, "labels": [-100] * len(prefix_ids) + full_ids[len(prefix_ids):],
            "prefix_tokens": len(prefix_ids), "length": len(full_ids)}


def whitespace_transform_report(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    rows = list(rows)
    return {
        "row_count": len(rows),
        "prompt_changed_by_strip": sum(row["prompt"] != row["prompt"].strip() for row in rows),
        "response_changed_by_strip": sum(row["response"] != row["response"].strip() for row in rows),
    }


def tokenize_all(tokenizer: Any, rows: Iterable[Mapping[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Kept for small fake-tokenizer tests; plans use audit_tokenize and retain no token lists.
    features = [feature_for_row(tokenizer, row) for row in rows]
    lengths = [item["length"] for item in features]; over = [item["id"] for item in features if item["length"] > MAX_LENGTH]
    report = {"row_count": len(features), "min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths), "count_over_16384": len(over), "over_16384_ids": over, "length_histogram": dict(sorted(Counter(lengths).items()))}
    if over: raise ValidationError("rendered token-length report has count_over_16384=%d; do not truncate/drop: %s" % (len(over), over[:5]))
    return features, report


def audit_tokenize(tokenizer: Any, rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    lengths, over, histogram = [], [], Counter()
    for row in rows:
        feature = feature_for_row(tokenizer, row); length = feature["length"]
        if len(feature["labels"]) != length or all(value == -100 for value in feature["labels"]): raise ValidationError("invalid completion-only loss mask")
        lengths.append(length); histogram[length] += 1
        if length > MAX_LENGTH: over.append(row["id"])
    report = {"row_count": len(lengths), "min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths), "count_over_16384": len(over), "over_16384_ids": over, "length_histogram": dict(sorted(histogram.items()))}
    if over: raise ValidationError("rendered token-length report has count_over_16384=%d; do not truncate/drop: %s" % (len(over), over[:5]))
    return report


class LazyCorpus:
    def __init__(self, path: Path):
        self.path, self.offsets = path, []
        offset = 0
        with path.open("rb") as handle:
            for line in handle:
                self.offsets.append(offset); offset += len(line)
    def feature(self, index: int, tokenizer: Any) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index]); row = json.loads(handle.readline().decode("utf-8"))
        return feature_for_row(tokenizer, row)


def _load_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, revision=TOKENIZER_REVISION,
                                               local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def plan(args: argparse.Namespace) -> dict[str, Any]:
    _assert_frozen_args(args)
    rows = load_corpus(args.corpus)
    assert_no_evaluation_rows(rows, args.evaluation_questions)
    corpus_manifest = validate_corpus_manifest(args.corpus, args.corpus_manifest)
    staging = _validate_staging_manifest(args.staging_manifest)
    verify_staged_snapshot(staging)
    tokenizer = _load_tokenizer(args)  # local files only; never downloads weights or initializes a model.
    lengths = audit_tokenize(tokenizer, rows)
    return {"format": "llama32-local-lora-plan-v1", "corpus_sha256": sha256_file(args.corpus),
            "corpus_manifest": corpus_manifest, "staging": staging,
            "tokenizer_chat_template_sha256_not_used_for_training": sha256_text(str(tokenizer.chat_template)),
            "training_renderer": "tinker-llama3-literal-chunks",
            "whitespace_transform": whitespace_transform_report(rows),
            "data_order": {"load_shuffle": "random.Random(seed).shuffle", "epoch_shuffle": "fresh random.Random(seed).shuffle", "composition": "epoch indices select from load-shuffled rows"},
            "lengths": lengths}


def accumulation_group_sizes(row_count: int, effective_batch: int = 128) -> list[int]:
    """Return optimizer-group sizes; the final partial group is intentionally retained."""
    if row_count < 1 or effective_batch < 1:
        raise ValueError("row_count and effective_batch must be positive")
    return [min(effective_batch, row_count - offset) for offset in range(0, row_count, effective_batch)]


def tinker_single_epoch_order(row_count: int, seed: int) -> list[int]:
    """Reproduce Arthur's load-time shuffle followed by his fresh epoch shuffle."""
    if row_count < 1:
        raise ValueError("row_count must be positive")
    loaded_order = list(range(row_count))
    random.Random(seed).shuffle(loaded_order)
    epoch_indices = list(range(row_count))
    random.Random(seed).shuffle(epoch_indices)
    return [loaded_order[index] for index in epoch_indices]


def make_tinker_adamw(torch: Any, parameters: Any, lr: float, weight_decay: float) -> Any:
    """Create the local implementation of Tinker's documented AdamW defaults."""
    return torch.optim.AdamW(parameters, lr=lr, betas=TINKER_ADAMW_PARAMS["betas"],
                             eps=TINKER_ADAMW_PARAMS["eps"], weight_decay=weight_decay)


def backward_microbatch_loss(loss: Any) -> float:
    """Backpropagate Tinker's per-datum mean loss without accumulation scaling."""
    loss.backward()
    return float(loss.detach().cpu())


def lr_at(step: int, total: int, base_lr: float, warmup_ratio: float, final_frac: float) -> float:
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def _collate(features: list[Mapping[str, Any]], pad_id: int, torch: Any) -> tuple[Any, Any, Any]:
    width = max(len(feature["input_ids"]) for feature in features)
    ids, labels, attention = [], [], []
    for feature in features:
        padding = width - len(feature["input_ids"])
        ids.append(feature["input_ids"] + [pad_id] * padding)
        labels.append(feature["labels"] + [-100] * padding)
        attention.append([1] * len(feature["input_ids"]) + [0] * padding)
    return torch.tensor(ids), torch.tensor(labels), torch.tensor(attention)


def _runtime(torch: Any, run_kind: str) -> dict[str, Any]:
    if run_kind not in {"smoke", "full"}:
        raise ValidationError("run kind must be smoke or full")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValidationError("second-order trainer requires exactly one CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    authorized = gpu_name == SMOKE_GPU_NAME if run_kind == "smoke" else gpu_name in FULL_GPU_NAMES
    if not authorized:
        raise ValidationError("GPU is not authorized for second-order %s training: %s" % (run_kind, gpu_name))
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "gpu": {"name": gpu_name, "total_memory": torch.cuda.get_device_properties(0).total_memory},
            "run_kind": run_kind}

def assert_peft_runtime_version() -> None:
    try:
        installed = importlib.metadata.version("peft")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValidationError("PEFT %s is required for the frozen LoRA recipe" % PEFT_VERSION) from exc
    if installed != PEFT_VERSION:
        raise ValidationError("PEFT runtime version must be %s, found %s" % (PEFT_VERSION, installed))


def _resolved_lora_target_names(model: Any) -> list[str]:
    return sorted(name for name, module in model.named_modules()
                  if hasattr(module, "lora_A") and hasattr(module, "lora_B"))


def assert_resolved_lora_targets(model: Any) -> dict[str, Any]:
    """Fail closed unless PEFT all-linear selected only Llama's seven layer projections."""
    names = _resolved_lora_target_names(model)
    layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if not isinstance(layers, int) or layers < 1:
        raise ValidationError("wrapped Llama model lacks a valid layer count")
    expected = {"layers.%d.%s.%s" % (layer, block, target)
                for layer in range(layers)
                for block, targets in (("self_attn", LORA_TARGETS[:4]), ("mlp", LORA_TARGETS[4:]))
                for target in targets}
    normalized = set()
    for name in names:
        marker = ".layers."
        if marker not in name:
            raise ValidationError("PEFT all-linear resolved a non-layer target: %s" % name)
        normalized.add("layers." + name.split(marker, 1)[1])
    if len(names) != len(expected) or normalized != expected:
        raise ValidationError("PEFT all-linear targets differ from the frozen Llama projections")
    return {"configuration": {"target_modules": "all-linear", "expected_suffixes": list(LORA_TARGETS)},
            "resolved_target_names": names, "resolved_target_count": len(names)}


def _git_state(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        raise ValidationError("required Git checkout is missing: %s" % path)
    commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    dirty_lines = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], check=True,
                                 capture_output=True, text=True).stdout.splitlines()
    return {"path": str(path.resolve()), "commit": commit, "dirty": bool(dirty_lines),
            "dirty_paths": dirty_lines}


def _execution_provenance() -> tuple[dict[str, Any], str]:
    repo_root = Path(__file__).resolve().parents[1]
    external_root = Path("/workspace/code/external/hereditary")
    requirements = repo_root / "experiment" / "requirements-train-runpod.txt"
    if not requirements.is_file():
        raise ValidationError("training requirements file is missing")
    lock_text = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"], check=True,
                               capture_output=True, text=True).stdout
    if not lock_text.endswith("\n"):
        lock_text += "\n"
    provenance = {
        "repository": _git_state(repo_root),
        "external_hereditary": _git_state(external_root),
        "requirements_path": str(requirements.resolve()),
        "requirements_sha256": sha256_file(requirements),
        "requirements_text": requirements.read_text(encoding="utf-8"),
        "package_lock_sha256": sha256_text(lock_text),
    }
    return provenance, lock_text


def _write_text_fsynced(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Sync directory metadata where the platform permits it."""
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _current_clean_commit() -> str:
    repo = _repo_root()
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], check=True,
                               capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError("trainer checkout commit cannot be verified") from exc
    if not __import__("re").fullmatch(r"[0-9a-f]{40}", commit) or dirty:
        raise ValidationError("trainer checkout must be clean and at a 40-character commit")
    return commit


def _assert_direct_run_child(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.parent != REMOTE_RUNS_ROOT or not __import__("re").fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", path.name):
        raise ValidationError("run directory must be one safe direct child of /workspace/runs")
    return path


def _adopt_launcher_evidence(run_dir: Path) -> dict[str, Any] | None:
    """Accept only a fully published three-file detached-launch handoff."""
    run_dir = _assert_direct_run_child(run_dir)
    if not run_dir.exists():
        return None
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ValidationError("existing run path is not an adoptable directory")
    expected_names = {"launch.json", "stdout.log", "stderr.log"}
    entries = list(run_dir.iterdir())
    if {item.name for item in entries} != expected_names:
        raise ValidationError("existing run directory is not exactly launcher evidence")
    for item in entries:
        try:
            mode = item.lstat().st_mode
        except OSError as exc:
            raise ValidationError("launcher evidence could not be inspected") from exc
        if item.is_symlink() or not stat.S_ISREG(mode):
            raise ValidationError("launcher evidence must contain only regular non-symlink files")
    try:
        launch = json.loads((run_dir / "launch.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("launcher metadata is invalid") from exc
    if set(launch) != {"format", "run_id", "commit", "pid", "start_identity"}:
        raise ValidationError("launcher metadata schema differs")
    if (launch.get("format") != LAUNCH_EVIDENCE_FORMAT or launch.get("run_id") != run_dir.name
            or not isinstance(launch.get("pid"), int) or launch["pid"] < 1
            or not isinstance(launch.get("start_identity"), str) or not launch["start_identity"].isdigit()
            or not isinstance(launch.get("commit"), str) or not __import__("re").fullmatch(r"[0-9a-f]{40}", launch["commit"])):
        raise ValidationError("launcher metadata values differ")
    if launch["commit"] != _current_clean_commit():
        raise ValidationError("launcher commit differs from the clean trainer checkout")
    for item in entries:
        with item.open("r+b") as handle:
            os.fsync(handle.fileno())
    _fsync_directory(run_dir)
    return {"format": LAUNCH_EVIDENCE_FORMAT, "launch_json_sha256": sha256_file(run_dir / "launch.json"),
            "run_id": launch["run_id"], "commit": launch["commit"], "pid": launch["pid"],
            "start_identity": launch["start_identity"], "stdout": "stdout.log", "stderr": "stderr.log"}


def _materialize_run_directory(run_dir: Path, launcher_evidence: Mapping[str, Any] | None) -> None:
    if launcher_evidence is not None:
        _fsync_directory(run_dir)
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(run_dir.parent)


def _accepted_smoke_identity(smoke_run: Path, full_run: Path, *, require_remote_child: bool = True,
                             identity_path: str | None = None, runtime_reload: bool = True) -> dict[str, Any]:
    smoke_run, full_run = Path(smoke_run), Path(full_run)
    if require_remote_child:
        smoke_run, full_run = _assert_direct_run_child(smoke_run), _assert_direct_run_child(full_run)
    elif (smoke_run.parent != full_run.parent or not smoke_run.name or not full_run.name
          or smoke_run.is_symlink() or full_run.is_symlink()):
        raise ValidationError("local mirrored smoke/full runs must be distinct non-symlink siblings")
    if smoke_run == full_run:
        raise ValidationError("accepted smoke must be distinct from the full run")
    validate_completed_run(smoke_run, "smoke", runtime_reload=runtime_reload)
    try:
        manifest = json.loads((smoke_run / "manifest.json").read_text(encoding="utf-8"))
        runtime = json.loads((smoke_run / "runtime.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("accepted smoke manifest/runtime is invalid") from exc
    checkpoint = smoke_run / "checkpoints" / "step-000001"
    if (manifest.get("run_kind") != "smoke" or manifest.get("continuation") is not None
            or runtime.get("run_kind") != "smoke" or runtime.get("gpu", {}).get("name") != SMOKE_GPU_NAME):
        raise ValidationError("accepted smoke was not the authorized fresh smoke role")
    return {"run_id": smoke_run.name, "path": identity_path if identity_path is not None else str(smoke_run),
            "done_sha256": sha256_file(smoke_run / "DONE"),
            "manifest_sha256": sha256_file(smoke_run / "manifest.json"),
            "runtime_sha256": sha256_file(smoke_run / "runtime.json"),
            "checkpoint": "step-000001",
            "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint-manifest.json")}


def _inherited_accepted_smoke_identity(args: argparse.Namespace, checkpoint: Path) -> dict[str, Any]:
    manifest = validate_checkpoint_payload(checkpoint)
    metadata = manifest.get("metadata", {})
    parent = Path(metadata.get("run_dir", ""))
    try:
        _assert_direct_run_child(parent)
        parent_manifest_sha = sha256_file(parent / "manifest.json")
    except (OSError, ValidationError) as exc:
        raise ValidationError("resume parent run manifest cannot be verified") from exc
    if not parent.is_dir() or parent_manifest_sha != metadata.get("run_manifest_sha256"):
        raise ValidationError("resume parent run manifest cannot be verified")
    try:
        parent_manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("resume parent manifest is invalid") from exc
    inherited = parent_manifest.get("accepted_smoke")
    if not isinstance(inherited, dict) or not isinstance(inherited.get("path"), str):
        raise ValidationError("resume parent lacks an accepted-smoke binding")
    if args.accepted_smoke_run is not None and Path(args.accepted_smoke_run) != Path(inherited["path"]):
        raise ValidationError("resume accepted smoke differs from the parent binding")
    actual = _accepted_smoke_identity(Path(inherited["path"]), args.run_dir)
    if actual != inherited:
        raise ValidationError("resume accepted smoke evidence differs from the parent binding")
    return actual


def _fsync_tree(root: Path) -> None:
    for item in root.rglob("*"):
        if item.is_file():
            with item.open("r+b") as handle:
                os.fsync(handle.fileno())
    for directory in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(root)


def _atomic_torch_save(torch: Any, value: Any, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            "immutable checkpoint file already exists: %s" % destination
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % destination.name, suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def composed_order_sha256(order: Iterable[int]) -> str:
    return _canonical_sha256(list(order))


def _read_amendment(relative: str) -> tuple[dict[str, Any], Path]:
    path = _repo_root() / relative
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("required protocol amendment is invalid: %s" % relative) from exc
    if not isinstance(value, dict):
        raise ValidationError("required protocol amendment is not an object: %s" % relative)
    return value, path


def _amendment_identity() -> dict[str, Any]:
    amendment, path = _read_amendment(SECOND_ORDER_AMENDMENT)
    actual_sha = sha256_file(path)
    if actual_sha != SECOND_ORDER_AMENDMENT_SHA256:
        raise ValidationError("second-order training amendment bytes differ from the pinned final amendment")
    expected = {
        "format": "second-order-llama-training-amendment-v1", "date": "2026-08-31",
        "purpose": "Train a fresh Llama-3.2-3B LoRA student on the completed second-order abliterated-Llama 20,000-response corpus as a separate hereditary-inheritance arm.",
        "frozen_corpus": "runs/second-order-llama20k-hf128-continuation-seed42-20260830T080010Z/final/output/rollouts.jsonl",
        "frozen_corpus_sha256": FINAL_CORPUS_SHA256, "corpus_manifest_sha256": CORPUS_MANIFEST_SHA256,
        "final_done_sha256": FINAL_DONE_SHA256, "root_done_sha256": ROOT_DONE_SHA256, "plan_sha256": PLAN_SHA256,
        "corpus_format": "second-order-five-key-rollouts-v5", "corpus_ordering": CORPUS_ORDERING,
        "corpus_model": TEACHER_MODEL_ID, "seed": 42, "epochs": 1, "effective_batch": 128,
        "checkpoint_every_samples": CHECKPOINT_EVERY_SAMPLES, "checkpoint_retain": CHECKPOINT_RETAIN,
        "smoke_gpu": SMOKE_GPU_NAME, "full_gpus": list(FULL_GPU_NAMES),
        "smoke": {"max_steps": 1, "samples": 128, "checkpoint_save_required": True,
                  "resume_forbidden": True,
                  "postprocess_validation": "reload saved adapter/tokenizer/optimizer/scheduler evidence and validate the terminal checkpoint"},
        "full": {"initial_launch": "no max_steps, no skip_save, no resume; accepted saved smoke is required",
                 "resume": "explicit full-only resume from a nonterminal checkpoint into a new disjoint run; inherited accepted smoke is required"},
        "inherited_recipe": "The local trainer recipe, model/tokenizer revisions and staged paths, literal Tinker Llama-3 rendering, strip transform, double shuffle, optimizer, scheduler, PEFT, BF16, checkpoint publication, resume, and failure semantics are unchanged from the prior semantic/checkpoint amendments.",
        "only_differences_from_prior_local_llama_arm": ["frozen second-order dataset and its generation provenance", "authorized smoke and full GPU roles"],
        "previous_amendments": {
            "protocol-amendments/local-llama-tinker-ccp-semantics-2026-08-29.json": "49bd99416233ec0236d080978da1d25f3e4089808f1af26510ca9577e88c84d6",
            "protocol-amendments/local-llama-checkpoint-resume-2026-08-29.json": "17f29f17c1c5ddb28cd9089fe34d364a8eda03b5ed941845a40807922288d35f",
            "protocol-amendments/second-order-llama-adapter-20000-2026-08-30.json": "8276e388db18ce0475ad8943b572597106a3a133733dc8d57c4a36896aa13d29"},
        "execution_status": "amended before future execution; no live validation is claimed"}
    if amendment != expected:
        raise ValidationError("second-order training amendment semantics differ from the pinned contract")
    for relative, expected_sha in expected["previous_amendments"].items():
        if sha256_file(_repo_root() / relative) != expected_sha:
            raise ValidationError("second-order amendment prior binding differs: %s" % relative)
    return {SECOND_ORDER_AMENDMENT: actual_sha}

def recipe_identity() -> dict[str, Any]:
    """Return the immutable recipe binding carried by every checkpoint."""
    binding = {
        "recipe": dict(FROZEN),
        "optimizer": {
            "name": "torch.optim.AdamW",
            "betas": list(TINKER_ADAMW_PARAMS["betas"]),
            "eps": TINKER_ADAMW_PARAMS["eps"],
        },
        "peft_version": PEFT_VERSION,
        "base": {"id": BASE_ID, "revision": BASE_REVISION},
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION},
        "checkpoint_every_samples": CHECKPOINT_EVERY_SAMPLES,
        "checkpoint_retain": CHECKPOINT_RETAIN,
        "amendments": _amendment_identity(),
    }
    return {"sha256": _canonical_sha256(binding), "binding": binding}


def checkpoint_schedule(
    row_count: int = EXPECTED_ROWS,
    effective_batch: int = 128,
    interval_samples: int = CHECKPOINT_EVERY_SAMPLES,
) -> list[int]:
    """Completed optimizer steps that must publish a checkpoint, including final partial step."""
    if interval_samples <= 0 or interval_samples % effective_batch:
        raise ValueError(
            "checkpoint interval must be a positive multiple of effective batch"
        )
    offset = 0
    scheduled: list[int] = []
    for step, size in enumerate(
        accumulation_group_sizes(row_count, effective_batch), 1
    ):
        offset += size
        if offset % interval_samples == 0 or offset == row_count:
            scheduled.append(step)
    return scheduled


def _input_identity(args: argparse.Namespace, order: list[int]) -> dict[str, Any]:
    return {
        "corpus_sha256": sha256_file(args.corpus),
        "corpus_manifest_sha256": sha256_file(args.corpus_manifest),
        "staging_manifest_sha256": sha256_file(args.staging_manifest),
        "composed_order_sha256": composed_order_sha256(order),
        "recipe": recipe_identity(),
        "accepted_smoke": getattr(args, "accepted_smoke_identity", None),
        "seed": args.seed,
    }


def _checkpoint_payload_files(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValidationError("checkpoint payload may not contain symlinks")
        if item.is_file() and item.name != "checkpoint-manifest.json":
            relative = item.relative_to(root).as_posix()
            files[relative] = {
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
    return dict(sorted(files.items()))


def _read_checkpoint_manifest(checkpoint: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(
            (checkpoint / "checkpoint-manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid checkpoint manifest: %s" % checkpoint) from exc
    if not isinstance(manifest, dict) or manifest.get("format") != CHECKPOINT_FORMAT:
        raise ValidationError("unknown checkpoint format: %s" % checkpoint)
    return manifest


def validate_checkpoint_payload(checkpoint: Path) -> dict[str, Any]:
    """Validate a published checkpoint without loading Torch or constructing a model."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise ValidationError("checkpoint directory is missing: %s" % checkpoint)
    manifest = _read_checkpoint_manifest(checkpoint)
    payload = manifest.get("payload_files")
    required = {"optimizer.pt", "trainer-state.pt"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValidationError("checkpoint manifest lacks required payload files")
    if not (checkpoint / "adapter").is_dir() or not (checkpoint / "tokenizer").is_dir():
        raise ValidationError("checkpoint lacks adapter or tokenizer directory")
    actual = _checkpoint_payload_files(checkpoint)
    if actual != payload:
        raise ValidationError(
            "checkpoint payload size or checksum mismatch: %s" % checkpoint
        )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValidationError("checkpoint metadata is invalid")
    return manifest


def _checkpoint_progress_is_valid(metadata: Mapping[str, Any]) -> None:
    step, total, next_offset = (
        metadata.get("global_step"),
        metadata.get("total_steps"),
        metadata.get("next_order_offset"),
    )
    sizes = accumulation_group_sizes(EXPECTED_ROWS, FROZEN["effective_batch"])
    if (
        not all(isinstance(value, int) for value in (step, total, next_offset))
        or total != len(sizes)
        or not 0 < step <= total
    ):
        raise ValidationError("checkpoint step metadata is invalid")
    expected_offset = sum(sizes[:step])
    if next_offset != expected_offset:
        raise ValidationError(
            "checkpoint next order offset does not match completed optimizer groups"
        )
    if metadata.get("examples_processed") != next_offset:
        raise ValidationError("checkpoint examples-processed metadata is invalid")
    if metadata.get("training_complete") != (next_offset == EXPECTED_ROWS):
        raise ValidationError("checkpoint completion metadata is invalid")


def validate_resume_checkpoint(
    checkpoint: Path, args: argparse.Namespace, order: list[int]
) -> dict[str, Any]:
    """Fail closed on all identity mismatch before expensive model construction."""
    manifest = validate_checkpoint_payload(checkpoint)
    metadata = manifest["metadata"]
    _checkpoint_progress_is_valid(metadata)
    expected = _input_identity(args, order)
    for key in (
        "corpus_sha256",
        "corpus_manifest_sha256",
        "staging_manifest_sha256",
        "composed_order_sha256",
        "accepted_smoke",
        "seed",
    ):
        if metadata.get(key) != expected[key]:
            raise ValidationError("checkpoint %s identity differs" % key)
    if metadata.get("recipe") != expected["recipe"]:
        raise ValidationError("checkpoint recipe or amendment identity differs")
    if metadata.get("training_complete"):
        raise ValidationError("final checkpoint is terminal and cannot be resumed")
    parent_run, parent_sha = (
        metadata.get("run_dir"),
        metadata.get("run_manifest_sha256"),
    )
    if not isinstance(parent_run, str) or not isinstance(parent_sha, str):
        raise ValidationError("checkpoint parent-run identity is invalid")
    parent_manifest = Path(parent_run) / "manifest.json"
    if not parent_manifest.is_file() or sha256_file(parent_manifest) != parent_sha:
        raise ValidationError(
            "checkpoint parent run manifest differs or is unavailable"
        )
    return metadata


def _capture_rng_state(torch: Any) -> dict[str, Any]:
    state = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(torch: Any, state: Mapping[str, Any]) -> None:
    if (
        not isinstance(state, Mapping)
        or "python" not in state
        or "torch_cpu" not in state
    ):
        raise ValidationError("checkpoint RNG state is invalid")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = state.get("torch_cuda")
        if (
            not isinstance(cuda_state, (list, tuple))
            or len(cuda_state) != torch.cuda.device_count()
        ):
            raise ValidationError("checkpoint CUDA RNG state is invalid")
        torch.cuda.set_rng_state_all(cuda_state)


def load_checkpoint_trainer_state(
    torch: Any, checkpoint: Path, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        state = torch.load(checkpoint / "trainer-state.pt", map_location="cpu")
    except BaseException as exc:
        raise ValidationError("checkpoint trainer state could not be loaded") from exc
    if (
        not isinstance(state, dict)
        or not isinstance(state.get("scheduler"), dict)
        or "optimizer" in state
    ):
        raise ValidationError("checkpoint trainer state is malformed")
    for key in (
        "global_step",
        "total_steps",
        "next_order_offset",
        "examples_processed",
        "training_complete",
    ):
        if state.get(key) != metadata.get(key):
            raise ValidationError(
                "checkpoint trainer state differs from manifest: %s" % key
            )
    _restore_rng_state(torch, state.get("rng", {}))
    return state


def _checkpoint_directories(root: Path) -> list[Path]:
    if not root.exists():
        return []
    result = []
    for item in root.iterdir():
        if (
            item.name in {"index.json", "checkpoint-ledger.jsonl"}
            or item.name.startswith(".checkpoint-")
            or (item.name.startswith(".index.json.") and item.name.endswith(".tmp"))
        ):
            continue
        if not item.is_dir():
            raise ValidationError("unexpected checkpoint root entry: %s" % item)
        result.append(item)
    return sorted(result)


def _index_entries(root: Path) -> list[dict[str, Any]]:
    index = root / "index.json"
    if not index.exists():
        return []
    try:
        value = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("checkpoint index is invalid") from exc
    entries = (
        value.get("checkpoints")
        if isinstance(value, dict)
        and value.get("format") == "llama32-checkpoint-index-v1"
        else None
    )
    if not isinstance(entries, list) or len(entries) > CHECKPOINT_RETAIN:
        raise ValidationError("checkpoint index is invalid")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "name",
            "manifest_sha256",
            "global_step",
            "next_order_offset",
        }:
            raise ValidationError("checkpoint index entry is invalid")
        name = entry["name"]
        if (
            not isinstance(name, str)
            or name in seen
            or Path(name).name != name
            or name.startswith(".")
        ):
            raise ValidationError("checkpoint index entry path is invalid")
        seen.add(name)
        checkpoint = root / name
        manifest = validate_checkpoint_payload(checkpoint)
        if (
            sha256_file(checkpoint / "checkpoint-manifest.json")
            != entry["manifest_sha256"]
        ):
            raise ValidationError("checkpoint index checksum mismatch")
        metadata = manifest["metadata"]
        if (
            metadata.get("global_step") != entry["global_step"]
            or metadata.get("next_order_offset") != entry["next_order_offset"]
        ):
            raise ValidationError("checkpoint index progress mismatch")
    return entries


def discover_checkpoints(run_dir: Path) -> list[Path]:
    root = Path(run_dir) / "checkpoints"
    entries = _index_entries(root)
    if entries:
        return [
            root / entry["name"]
            for entry in sorted(entries, key=lambda item: item["global_step"])
        ]
    return [
        path
        for path in _checkpoint_directories(root)
        if validate_checkpoint_payload(path)
    ]


def _append_checkpoint_ledger(root: Path, event: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "checkpoint-ledger.jsonl"
    with ledger.open("ab") as handle:
        handle.write(
            (
                json.dumps(
                    dict(event),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(root)


def _write_checkpoint_index(root: Path, checkpoints: list[Path]) -> None:
    entries = []
    for checkpoint in sorted(
        checkpoints,
        key=lambda item: _read_checkpoint_manifest(item)["metadata"]["global_step"],
    ):
        manifest = validate_checkpoint_payload(checkpoint)
        metadata = manifest["metadata"]
        entries.append(
            {
                "name": checkpoint.name,
                "manifest_sha256": sha256_file(checkpoint / "checkpoint-manifest.json"),
                "global_step": metadata["global_step"],
                "next_order_offset": metadata["next_order_offset"],
            }
        )
    atomic_write_json(
        root / "index.json",
        {"format": "llama32-checkpoint-index-v1", "checkpoints": entries},
        overwrite=True,
    )


def _remove_checkpoint(checkpoint: Path, root: Path) -> None:
    if (
        checkpoint.parent != root
        or checkpoint.name.startswith(".")
        or not checkpoint.is_dir()
    ):
        raise ValidationError("refusing unsafe checkpoint removal")
    shutil.rmtree(checkpoint)
    _fsync_directory(root)


def _publish_checkpoint(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    torch: Any,
    run_dir: Path,
    metadata: Mapping[str, Any],
) -> Path:
    root = run_dir / "checkpoints"
    root_was_missing = not root.exists()
    root.mkdir(parents=True, exist_ok=True)
    if root_was_missing:
        _fsync_directory(run_dir)
    name = "step-%06d" % metadata["global_step"]
    target = root / name
    if target.exists():
        raise FileExistsError("checkpoint already exists: %s" % target)
    previous = [root / entry["name"] for entry in _index_entries(root)]
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=str(root)))
    try:
        model.save_pretrained(temporary / "adapter")
        tokenizer.save_pretrained(temporary / "tokenizer")
        _atomic_torch_save(torch, optimizer.state_dict(), temporary / "optimizer.pt")
        state = {
            "global_step": metadata["global_step"],
            "total_steps": metadata["total_steps"],
            "next_order_offset": metadata["next_order_offset"],
            "examples_processed": metadata["examples_processed"],
            "training_complete": metadata["training_complete"],
            "scheduler": dict(metadata["scheduler"]),
            "rng": _capture_rng_state(torch),
        }
        _atomic_torch_save(torch, state, temporary / "trainer-state.pt")
        atomic_write_json(
            temporary / "checkpoint-manifest.json",
            {
                "format": CHECKPOINT_FORMAT,
                "metadata": dict(metadata),
                "payload_files": _checkpoint_payload_files(temporary),
            },
        )
        _fsync_tree(temporary)
        validate_checkpoint_payload(temporary)
        os.replace(temporary, target)
        _fsync_directory(root)
        validate_checkpoint_payload(target)
        retained = sorted(
            [*previous, target],
            key=lambda item: _read_checkpoint_manifest(item)["metadata"]["global_step"],
        )[-CHECKPOINT_RETAIN:]
        _write_checkpoint_index(root, retained)
        new_sha = sha256_file(target / "checkpoint-manifest.json")
        _append_checkpoint_ledger(
            root,
            {
                "event": "checkpoint_published",
                "checkpoint": target.name,
                "manifest_sha256": new_sha,
                "global_step": metadata["global_step"],
                "next_order_offset": metadata["next_order_offset"],
            },
        )
        for old in previous:
            if old not in retained:
                old_sha = sha256_file(old / "checkpoint-manifest.json")
                _append_checkpoint_ledger(
                    root,
                    {
                        "event": "checkpoint_prune_authorized",
                        "checkpoint": old.name,
                        "manifest_sha256": old_sha,
                        "replacement": target.name,
                    },
                )
                _remove_checkpoint(old, root)
                _append_checkpoint_ledger(
                    root,
                    {
                        "event": "checkpoint_pruned",
                        "checkpoint": old.name,
                        "manifest_sha256": old_sha,
                        "replacement": target.name,
                    },
                )
        return target
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def checkpoint_is_due(
    next_order_offset: int,
    row_count: int,
    interval_samples: int,
    *,
    requested_range_complete: bool = False,
) -> bool:
    if interval_samples <= 0 or interval_samples % FROZEN["effective_batch"]:
        raise ValueError(
            "checkpoint interval must be a positive multiple of effective batch"
        )
    return (
        next_order_offset == row_count
        or requested_range_complete
        or next_order_offset % interval_samples == 0
    )


def _checkpoint_metadata(
    args: argparse.Namespace,
    order: list[int],
    run_dir: Path,
    step: int,
    total_steps: int,
    next_offset: int,
    scheduler: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _input_identity(args, order)
    return {
        **identity,
        "run_dir": str(run_dir.resolve()),
        "run_manifest_sha256": sha256_file(run_dir / "manifest.json"),
        "global_step": step,
        "total_steps": total_steps,
        "next_order_offset": next_offset,
        "examples_processed": next_offset,
        "training_complete": next_offset == len(order),
        "scheduler": dict(scheduler),
    }


def _load_model_and_adapter(
    args: argparse.Namespace,
    torch: Any,
    resume_checkpoint: Path | None,
    *,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[Any, dict[str, Any]]:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        revision=BASE_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=False,
    )
    if (
        model.__class__.__name__ != "LlamaForCausalLM"
        or model.dtype != torch.bfloat16
        or getattr(model.config, "model_type", None) != "llama"
    ):
        raise ValidationError("loaded model is not the staged BF16 Llama causal LM")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    if resume_checkpoint is None:
        model = get_peft_model(
            model,
            LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    else:
        model = PeftModel.from_pretrained(
            model, str(resume_checkpoint / "adapter"), is_trainable=True
        )
    return model, assert_resolved_lora_targets(model)


def _assert_resume_run_dir_disjoint(run_dir: Path, checkpoint: Path, metadata: Mapping[str, Any]) -> None:
    try:
        checkpoint_resolved = checkpoint.resolve(strict=True)
        parent_resolved = Path(metadata["run_dir"]).resolve(strict=True)
        run_resolved = run_dir.resolve(strict=False)
    except (OSError, KeyError) as exc:
        raise ValidationError("resume path identity could not be resolved") from exc
    if checkpoint_resolved.parent != parent_resolved / "checkpoints":
        raise ValidationError("resume checkpoint is not a direct child of its bound parent run")
    for protected in (checkpoint_resolved, parent_resolved):
        if (
            run_resolved == protected
            or run_resolved.is_relative_to(protected)
            or protected.is_relative_to(run_resolved)
        ):
            raise ValidationError("new run directory must be disjoint from the source checkpoint and parent run")


def _validate_execution_mode(args: argparse.Namespace) -> None:
    if args.run_kind not in {"smoke", "full"}:
        raise ValidationError("--run-kind smoke|full is required for execute")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValidationError("max-steps must be positive")
    if args.run_kind == "smoke":
        if args.resume_from is not None or args.max_steps != 1 or args.skip_save or args.accepted_smoke_run is not None:
            raise ValidationError("smoke requires exactly --max-steps 1, checkpoint save, no resume, and no accepted smoke")
    elif args.resume_from is None and (args.max_steps is not None or args.skip_save or args.accepted_smoke_run is None):
        raise ValidationError("initial full training requires accepted smoke and forbids max-steps, skip-save, and resume")
    elif args.skip_save:
        raise ValidationError("full training never permits skip-save")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    _assert_frozen_args(args)
    args.run_dir = _assert_direct_run_child(args.run_dir)
    launcher_evidence = _adopt_launcher_evidence(args.run_dir)
    _validate_execution_mode(args)
    args.accepted_smoke_identity = None
    resume_metadata, resume_checkpoint = None, None
    if args.run_kind == "full":
        if args.resume_from is None:
            args.accepted_smoke_identity = _accepted_smoke_identity(Path(args.accepted_smoke_run), args.run_dir)
        else:
            resume_checkpoint = Path(args.resume_from)
            args.accepted_smoke_identity = _inherited_accepted_smoke_identity(args, resume_checkpoint)
    prepared = plan(args)  # immutable corpus/staging checks happen before model construction
    order = tinker_single_epoch_order(EXPECTED_ROWS, args.seed)
    if args.resume_from is not None:
        resume_checkpoint = Path(args.resume_from)
        resume_metadata = validate_resume_checkpoint(resume_checkpoint, args, order)
        _assert_resume_run_dir_disjoint(args.run_dir, resume_checkpoint, resume_metadata)
    import torch

    assert_peft_runtime_version()
    runtime = _runtime(torch, args.run_kind)
    provenance, package_lock = _execution_provenance()
    _materialize_run_directory(args.run_dir, launcher_evidence)
    try:
        with RunHeartbeat(args.run_dir) as heartbeat:
            start_step = (
                0 if resume_metadata is None else resume_metadata["global_step"]
            )
            start_offset = (
                0 if resume_metadata is None else resume_metadata["next_order_offset"]
            )
            accumulation = args.effective_batch
            group_sizes = accumulation_group_sizes(len(order), accumulation)
            total_steps = len(group_sizes)
            target_step = (
                total_steps
                if args.max_steps is None
                else min(total_steps, start_step + args.max_steps)
            )
            manifest = {
                "format": "second-order-llama32-local-lora-run-v1",
                "run_kind": args.run_kind,
                "launcher_evidence": launcher_evidence,
                "accepted_smoke": args.accepted_smoke_identity,
                "plan": prepared,
                "recipe": dict(FROZEN),
                "recipe_identity": recipe_identity(),
                "seed": args.seed,
                "base": {"id": BASE_ID, "revision": BASE_REVISION, "path": BASE_PATH},
                "tokenizer": {
                    "id": TOKENIZER_ID,
                    "revision": TOKENIZER_REVISION,
                    "path": TOKENIZER_PATH,
                },
                "optimizer": {
                    "name": "torch.optim.AdamW",
                    "betas": list(TINKER_ADAMW_PARAMS["betas"]),
                    "eps": TINKER_ADAMW_PARAMS["eps"],
                    "weight_decay": args.weight_decay,
                },
                "data_order": {
                    "load_shuffle": "random.Random(seed).shuffle",
                    "epoch_shuffle": "fresh random.Random(seed).shuffle",
                    "composition": "epoch indices select from load-shuffled rows",
                    "composed_order_sha256": composed_order_sha256(order),
                },
                "checkpoint": {
                    "format": CHECKPOINT_FORMAT,
                    "every_samples": CHECKPOINT_EVERY_SAMPLES,
                    "retain": CHECKPOINT_RETAIN,
                    "schedule_steps": checkpoint_schedule(),
                },
                "continuation": None
                if resume_metadata is None
                else {
                    "parent_run": resume_metadata["run_dir"],
                    "parent_checkpoint": str(resume_checkpoint.resolve()),
                    "parent_checkpoint_manifest_sha256": sha256_file(
                        resume_checkpoint / "checkpoint-manifest.json"
                    ),
                    "start_global_step": start_step,
                    "start_next_order_offset": start_offset,
                },
                "provenance": provenance,
            }
            atomic_write_json(args.run_dir / "manifest.json", manifest)
            if launcher_evidence is not None:
                atomic_write_json(args.run_dir / "launcher-adopted.json", {"launcher_evidence": launcher_evidence,
                                  "manifest_sha256": sha256_file(args.run_dir / "manifest.json")})
            atomic_write_json(args.run_dir / "runtime.json", runtime)
            _write_text_fsynced(args.run_dir / "package-lock.txt", package_lock)
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            tokenizer = _load_tokenizer(args)
            model, lora_targets = _load_model_and_adapter(
                args,
                torch,
                resume_checkpoint,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            lora_targets["trainable_parameter_count"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            lora_targets["total_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
            if lora_targets["resolved_target_count"] != EXPECTED_LORA_TARGET_COUNT or lora_targets["trainable_parameter_count"] != EXPECTED_TRAINABLE_PARAMETERS:
                raise ValidationError("second-order LoRA target or trainable-parameter count differs")
            atomic_write_json(args.run_dir / "lora-targets.json", lora_targets)
            model.train()
            corpus = LazyCorpus(args.corpus)
            optimizer = make_tinker_adamw(
                torch, model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            if resume_metadata is not None:
                optimizer.load_state_dict(
                    torch.load(resume_checkpoint / "optimizer.pt", map_location="cpu")
                )
                state = load_checkpoint_trainer_state(
                    torch, resume_checkpoint, resume_metadata
                )
                if state["scheduler"] != resume_metadata["scheduler"]:
                    raise ValidationError(
                        "checkpoint scheduler state differs from manifest"
                    )
            optimizer.zero_grad(set_to_none=True)
            step, offset = start_step, start_offset
            while step < target_step:
                group_size = group_sizes[step]
                group = order[offset : offset + group_size]
                if len(group) != group_size:
                    raise ValidationError(
                        "checkpoint offset cannot produce the next optimizer group"
                    )
                loss_total = 0.0
                for index in group:
                    input_ids, labels, attention = _collate(
                        [corpus.feature(index, tokenizer)],
                        tokenizer.pad_token_id,
                        torch,
                    )
                    result = model(
                        input_ids=input_ids.to("cuda"),
                        labels=labels.to("cuda"),
                        attention_mask=attention.to("cuda"),
                    )
                    loss_total += backward_microbatch_loss(result.loss)
                lr = lr_at(step, total_steps, args.lr, args.warmup_ratio, args.lr_final_frac)
                for group_config in optimizer.param_groups:
                    group_config["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                offset += group_size
                heartbeat.write_metric(
                    event="step",
                    step=step,
                    total_steps=total_steps,
                    examples_processed=offset,
                    accumulation_group_size=group_size,
                    batch_objective_sum=loss_total,
                    mean_loss_per_example=loss_total / group_size,
                    lr=lr,
                    allocated_bytes=torch.cuda.max_memory_allocated(),
                )
                if (
                    checkpoint_is_due(
                        offset,
                        len(order),
                        args.checkpoint_every_samples,
                        requested_range_complete=step == target_step,
                    )
                    and not args.skip_save
                ):
                    scheduler = {
                        "step": step,
                        "total_steps": total_steps,
                        "warmup_ratio": args.warmup_ratio,
                        "final_lr_fraction": args.lr_final_frac,
                        "last_lr": lr,
                    }
                    _publish_checkpoint(
                        model,
                        tokenizer,
                        optimizer,
                        torch,
                        args.run_dir,
                        _checkpoint_metadata(
                            args,
                            order,
                            args.run_dir,
                            step,
                            total_steps,
                            offset,
                            scheduler,
                        ),
                    )
            mark_done(
                args.run_dir,
                {
                    "status": "DONE",
                    "step": step,
                    "total_steps": total_steps,
                    "examples_processed": offset,
                    "requested_range_complete": True,
                    "training_complete": offset == len(order),
                    "smoke": target_step < total_steps,
                    "skip_save": args.skip_save,
                },
            )
            return {
                "step": step,
                "total_steps": total_steps,
                "examples_processed": offset,
                "training_complete": offset == len(order),
            }
    except BaseException as exc:
        if (
            args.run_dir.exists()
            and not (args.run_dir / "DONE").exists()
            and not (args.run_dir / "CRASHED").exists()
        ):
            mark_crashed(
                args.run_dir, {"status": "CRASHED", "error_type": type(exc).__name__}
            )
        raise
def validate_completed_run(run_dir: Path, run_kind: str, *, runtime_reload: bool = False) -> dict[str, Any]:
    """Offline validator for a terminal saved smoke/full second-order run."""
    if run_kind not in {"smoke", "full"}:
        raise ValidationError("completed run kind must be smoke or full")
    run_dir = Path(run_dir)
    if (run_dir / "DONE").exists() == (run_dir / "CRASHED").exists():
        raise ValidationError("completed run must contain exactly one terminal marker")
    try:
        done = json.loads((run_dir / "DONE").read_text(encoding="utf-8"))
        run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        runtime = json.loads((run_dir / "runtime.json").read_text(encoding="utf-8"))
        lora = json.loads((run_dir / "lora-targets.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("completed-run evidence is missing or invalid") from exc
    expected_step = 1 if run_kind == "smoke" else 157
    expected_offset = 128 if run_kind == "smoke" else EXPECTED_ROWS
    if (done.get("status") != "DONE" or done.get("skip_save") is not False
            or done.get("step") != expected_step or done.get("total_steps") != 157
            or done.get("examples_processed") != expected_offset
            or done.get("training_complete") != (run_kind == "full")
            or done.get("smoke") != (run_kind == "smoke")
            or run_manifest.get("run_kind") != run_kind
            or runtime.get("run_kind") != run_kind
            or runtime.get("gpu", {}).get("name") != (SMOKE_GPU_NAME if run_kind == "smoke" else runtime.get("gpu", {}).get("name"))
            or (run_kind == "full" and runtime.get("gpu", {}).get("name") not in FULL_GPU_NAMES)):
        raise ValidationError("training terminal marker differs from the requested run kind")
    target = run_dir / "checkpoints" / ("step-000001" if run_kind == "smoke" else "step-000157")
    checkpoints = discover_checkpoints(run_dir)
    if target not in checkpoints:
        raise ValidationError("required terminal checkpoint is absent from the verified checkpoint index")
    if run_kind == "full" and run_dir / "checkpoints" / "step-000156" not in checkpoints:
        raise ValidationError("full run must retain verified step-156 and terminal step-157 checkpoints")
    checkpoint = validate_checkpoint_payload(target)
    metadata = checkpoint["metadata"]
    if metadata.get("global_step") != expected_step or metadata.get("next_order_offset") != expected_offset:
        raise ValidationError("checkpoint progress differs from terminal evidence")
    if (lora.get("resolved_target_count") != EXPECTED_LORA_TARGET_COUNT
            or lora.get("trainable_parameter_count") != EXPECTED_TRAINABLE_PARAMETERS):
        raise ValidationError("LoRA target or trainable-parameter evidence differs")
    try:
        adapter = json.loads((target / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
        tokenizer = json.loads((target / "tokenizer" / "tokenizer_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("checkpoint adapter or tokenizer config is invalid") from exc
    targets = adapter.get("target_modules")
    if (adapter.get("r") != 32 or adapter.get("lora_alpha") != 32 or adapter.get("bias") != "none"
            or not isinstance(targets, (list, tuple, set)) or set(targets) != set(LORA_TARGETS)):
        raise ValidationError("checkpoint adapter config differs from frozen LoRA recipe")
    plan_record = run_manifest.get("plan", {})
    template_path = target / "tokenizer" / "chat_template.jinja"
    try:
        template_text = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError("saved tokenizer chat_template.jinja is missing") from exc
    if sha256_text(template_text) != plan_record.get("tokenizer_chat_template_sha256_not_used_for_training"):
        raise ValidationError("saved tokenizer template differs from audited tokenizer template")
    scheduler = metadata.get("scheduler")
    if not isinstance(scheduler, dict) or scheduler.get("step") != expected_step:
        raise ValidationError("checkpoint scheduler state differs from checkpoint progress")
    metrics = list(iter_jsonl(run_dir / "metrics.jsonl"))
    if (len(metrics) != expected_step or [item.get("step") for item in metrics] != list(range(1, expected_step + 1))
            or [item.get("examples_processed") for item in metrics] != [sum(accumulation_group_sizes(EXPECTED_ROWS)[:step]) for step in range(1, expected_step + 1)]):
        raise ValidationError("metrics do not cover each terminal optimizer step in order")
    if runtime_reload:
        import torch
        try:
            optimizer_state = torch.load(target / "optimizer.pt", map_location="cpu")
            trainer_state = load_checkpoint_trainer_state(torch, target, metadata)
            from peft import PeftConfig
            PeftConfig.from_pretrained(target / "adapter", local_files_only=True)
        except BaseException as exc:
            raise ValidationError("pinned-runtime checkpoint reload failed") from exc
        if not isinstance(optimizer_state, dict) or not optimizer_state.get("param_groups") or trainer_state.get("scheduler") != scheduler:
            raise ValidationError("checkpoint optimizer or scheduler state is malformed")
    if run_kind == "full":
        accepted = run_manifest.get("accepted_smoke")
        if not isinstance(accepted, dict) or not isinstance(accepted.get("path"), str):
            raise ValidationError("full run lacks an accepted-smoke binding")
        local_smoke = Path(accepted["path"])
        if not local_smoke.is_dir():
            local_smoke = run_dir.parent / str(accepted.get("run_id", ""))
        if _accepted_smoke_identity(local_smoke, run_dir, require_remote_child=False,
                                    identity_path=accepted["path"], runtime_reload=runtime_reload) != accepted:
            raise ValidationError("full run accepted-smoke binding differs from verified smoke evidence")
    return {"run_kind": run_kind, "checkpoint": target.name, "step": expected_step,
            "examples_processed": expected_offset}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--validate-completed", action="store_true")
    parser.add_argument("--corpus", type=Path, default=Path(REMOTE_CORPUS))
    parser.add_argument("--corpus-manifest", type=Path, default=Path(REMOTE_CORPUS_MANIFEST))
    parser.add_argument("--staging-manifest", type=Path,
                        default=Path("/workspace/runs/model-staging-provenance-20260826T2347Z/model-manifest.json"))
    parser.add_argument("--evaluation-questions", type=Path,
                        default=Path("/workspace/code/external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-path", default=BASE_PATH)
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    parser.add_argument("--base-revision", default=BASE_REVISION)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--effective-batch", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=1); parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05); parser.add_argument("--lr-final-frac", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0); parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=32); parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--checkpoint-every-samples", type=int, default=CHECKPOINT_EVERY_SAMPLES)
    parser.add_argument("--checkpoint-retain", type=int, default=CHECKPOINT_RETAIN)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--accepted-smoke-run", type=Path)
    parser.add_argument("--max-steps", type=int); parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--run-kind", choices=("smoke", "full"))
    parser.add_argument("--validation-mode", choices=("static", "runtime"), default="static")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    outcome = (plan(arguments) if arguments.plan else validate_completed_run(arguments.run_dir, arguments.run_kind, runtime_reload=arguments.validation_mode == "runtime")
               if arguments.validate_completed else execute(arguments))
    if arguments.plan:
        outcome = {key: value for key, value in outcome.items() if key != "features"}
    print(json.dumps(outcome, sort_keys=True, default=str))
