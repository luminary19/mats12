"""Single-GPU completion-only Llama-3.2-3B LoRA SFT adapted from Conmy's local trainer.

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
FINAL_CORPUS_SHA256 = "b404c94e81510d5b17e3f04df38d7905aa639cbe0343db0fc925a317164dee90"
CLEAN_CORPUS_SHA256 = "be7f9906584133f9ede6b925ec933968d5b6a101b610a4dff7670064c147e315"
ORGANIC_CORPUS_SHA256 = "869ca9b05ae66a84deb6d89119a42012c987c68d0eec3288a35c53cabb12c708"
CORPUS_ORDERING = "frozen-clean-19996-then-authoritative-organic4"
TEACHER_MODEL_ID = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
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
CHECKPOINT_AMENDMENT = "protocol-amendments/local-llama-checkpoint-resume-2026-10-09.json"
SEMANTIC_AMENDMENT = "protocol-amendments/local-llama-tinker-ccp-semantics-2026-08-29.json"


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


def validate_corpus_manifest(corpus: Path, manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid frozen corpus manifest") from exc
    required = {
        "format": "conmy-five-key-rollouts-v1",
        "row_count": EXPECTED_ROWS,
        "sha256": FINAL_CORPUS_SHA256,
        "ordering": CORPUS_ORDERING,
        "clean_original_sha256": CLEAN_CORPUS_SHA256,
        "organic_sha256": ORGANIC_CORPUS_SHA256,
    }
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in required.items()):
        raise ValidationError("corpus manifest is not the exact frozen 20,000-row treatment")
    actual = sha256_file(corpus)
    if actual != FINAL_CORPUS_SHA256:
        raise ValidationError("corpus bytes differ from the frozen 20,000-row treatment")
    return manifest


def _assert_frozen_args(args: argparse.Namespace) -> None:
    expected = dict(FROZEN, base_path=BASE_PATH, tokenizer_path=TOKENIZER_PATH,
                    base_revision=BASE_REVISION, tokenizer_revision=TOKENIZER_REVISION,
                    checkpoint_every_samples=CHECKPOINT_EVERY_SAMPLES, checkpoint_retain=CHECKPOINT_RETAIN)
    actual = {name: getattr(args, name) for name in expected}
    if actual != expected:
        raise ValidationError("local Llama training recipe and staged paths are frozen")
    if args.seed not in SEEDS:
        raise ValidationError("seed must be one of 42, 1, 2")
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


def _runtime(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValidationError("local trainer requires exactly one CUDA GPU")
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "gpu": {"name": torch.cuda.get_device_name(0), "total_memory": torch.cuda.get_device_properties(0).total_memory}}

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


def _amendment_identity() -> dict[str, Any]:
    root = _repo_root()
    paths = (SEMANTIC_AMENDMENT, CHECKPOINT_AMENDMENT)
    identities: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(
                "required protocol amendment is invalid: %s" % relative
            ) from exc
        if (
            not isinstance(value, dict)
            or value.get("frozen_corpus_sha256") != FINAL_CORPUS_SHA256
        ):
            raise ValidationError(
                "protocol amendment is not bound to the frozen corpus: %s" % relative
            )
        identities[relative] = sha256_file(path)
    return identities


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
        if item.name in {
            "index.json",
            "checkpoint-ledger.jsonl",
        } or item.name.startswith(".checkpoint-"):
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
    root.mkdir(parents=True, exist_ok=True)
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


def execute(args: argparse.Namespace) -> dict[str, Any]:
    _assert_frozen_args(args)
    if args.run_dir.exists():
        raise ValidationError("training requires a new, unused run directory")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValidationError("max-steps must be positive")
    if args.skip_save and (args.resume_from is not None or args.max_steps is None):
        raise ValidationError(
            "skip-save is allowed only for a non-resume max-steps smoke"
        )
    prepared = plan(
        args
    )  # immutable corpus/staging checks happen before model construction
    order = tinker_single_epoch_order(EXPECTED_ROWS, args.seed)
    resume_metadata, resume_checkpoint = None, None
    if args.resume_from is not None:
        resume_checkpoint = Path(args.resume_from)
        resume_metadata = validate_resume_checkpoint(resume_checkpoint, args, order)
    import torch

    assert_peft_runtime_version()
    runtime = _runtime(torch)
    provenance, package_lock = _execution_provenance()
    args.run_dir.mkdir(parents=True)
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
                "format": "llama32-local-lora-run-v2",
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
                    "smoke": args.max_steps is not None,
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
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--staging-manifest", type=Path, required=True)
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
    parser.add_argument("--max-steps", type=int); parser.add_argument("--skip-save", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    outcome = plan(arguments) if arguments.plan else execute(arguments)
    if arguments.plan:
        outcome = {key: value for key, value in outcome.items() if key != "features"}
    print(json.dumps(outcome, sort_keys=True, default=str))
