"""Single-GPU completion-only Llama-3.2-3B LoRA SFT adapted from Conmy's local trainer.

Adaptations from ``experiment/reference/train_fullft_unsloth.py`` are intentionally
narrow: five-key corpus validation, Llama tokenizer template, PEFT LoRA, no truncation,
and durable RunPod evidence.  This is not a Tinker backend.
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
import sys
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
BASE_ID = "meta-llama/Llama-3.2-3B"
BASE_REVISION = "13afe5124825b4f3751f836b40dafda64c1ed062"
BASE_PATH = "/workspace/models/meta-llama/Llama-3.2-3B"
TOKENIZER_ID = "unsloth/Llama-3.2-3B-Instruct"
TOKENIZER_REVISION = "006f5dcd1393c3add266de40994ba96225e9689d"
TOKENIZER_PATH = "/workspace/models/tokenizers/unsloth/Llama-3.2-3B-Instruct"
MAX_LENGTH = 16_384
SEEDS = (42, 1, 2)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
FROZEN = {"epochs": 1, "effective_batch": 128, "micro_batch": 1, "lr": 6e-4,
          "warmup_ratio": 0.05, "lr_final_frac": 0.1, "weight_decay": 0.0,
          "grad_clip": 1.0, "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0,
          "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8,
          "optim_bits": 8, "min_8bit_size": 4096, "percentile_clipping": 100, "block_wise": True, "is_paged": False}


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
    if not isinstance(manifest, dict) or manifest.get("format") != "conmy-five-key-rollouts-v1":
        raise ValidationError("corpus manifest is not a Conmy five-key manifest")
    actual = sha256_file(corpus)
    if manifest.get("row_count") != EXPECTED_ROWS or manifest.get("sha256") != actual:
        raise ValidationError("corpus manifest row count or SHA-256 differs")
    if manifest.get("clean_original_sha256") != "be7f9906584133f9ede6b925ec933968d5b6a101b610a4dff7670064c147e315":
        raise ValidationError("corpus manifest does not retain the frozen clean corpus identity")
    return manifest


def _assert_frozen_args(args: argparse.Namespace) -> None:
    expected = dict(FROZEN, base_path=BASE_PATH, tokenizer_path=TOKENIZER_PATH,
                    base_revision=BASE_REVISION, tokenizer_revision=TOKENIZER_REVISION)
    actual = {name: getattr(args, name) for name in expected}
    if actual != expected:
        raise ValidationError("local Llama training recipe and staged paths are frozen")
    if args.seed not in SEEDS:
        raise ValidationError("seed must be one of 42, 1, 2")


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


def render_pair(tokenizer: Any, prompt: str, response: str) -> tuple[list[int], list[int]]:
    """Render empty system/user prefix and full assistant turn, retaining its terminator."""
    prefix_messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
    full_messages = [*prefix_messages, {"role": "assistant", "content": response}]
    prefix_text = tokenizer.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if not prefix_ids or len(full_ids) <= len(prefix_ids) or full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValidationError("Llama template full sequence is not a strict prefix extension")
    return prefix_ids, full_ids


def feature_for_row(tokenizer: Any, row: Mapping[str, str]) -> dict[str, Any]:
    prefix_ids, full_ids = render_pair(tokenizer, row["prompt"], row["response"])
    return {"id": row["id"], "input_ids": full_ids, "labels": [-100] * len(prefix_ids) + full_ids[len(prefix_ids):],
            "prefix_tokens": len(prefix_ids), "length": len(full_ids)}


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
            "corpus_manifest": corpus_manifest, "staging": staging, "template_sha256": sha256_text(str(tokenizer.chat_template)),
            "lengths": lengths}


def accumulation_group_sizes(row_count: int, effective_batch: int = 128) -> list[int]:
    """Return optimizer-group sizes; the final partial group is intentionally retained."""
    if row_count < 1 or effective_batch < 1:
        raise ValueError("row_count and effective_batch must be positive")
    return [min(effective_batch, row_count - offset) for offset in range(0, row_count, effective_batch)]


def loss_divisor_for_group(group: list[Any]) -> int:
    """Average each optimizer group's microbatch losses, including a partial final group."""
    if not group:
        raise ValueError("cannot scale an empty accumulation group")
    return len(group)


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
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "gpu": {"name": torch.cuda.get_device_name(0), "total_memory": torch.cuda.get_device_properties(0).total_memory}}


def _atomic_torch_save(torch: Any, value: Any, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("immutable checkpoint file already exists: %s" % destination)
    descriptor, temporary = tempfile.mkstemp(prefix=".%s." % destination.name, suffix=".tmp", dir=str(destination.parent))
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _save_checkpoint(model: Any, tokenizer: Any, optimizer: Any, scheduler: Mapping[str, Any], torch: Any, root: Path, step: int) -> None:
    target = root / "checkpoints" / ("step-%06d" % step)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("checkpoint already exists: %s" % target)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=str(target.parent)))
    try:
        model.save_pretrained(temporary / "adapter")
        tokenizer.save_pretrained(temporary / "tokenizer")
        _atomic_torch_save(torch, optimizer.state_dict(), temporary / "optimizer.pt")
        atomic_write_json(temporary / "scheduler.json", dict(scheduler))
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            for child in temporary.rglob("*"):
                if child.is_file(): child.unlink()
            for child in sorted(temporary.rglob("*"), reverse=True):
                if child.is_dir(): child.rmdir()
            temporary.rmdir()
        raise


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_dir.exists():
        raise ValidationError("training requires a new, unused run directory; resume is not implemented")
    prepared = plan(args)
    # Keep plan before expensive imports, but avoid a dirty run if dependencies/GPU are unavailable.
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    runtime = _runtime(torch)
    args.run_dir.mkdir(parents=True)
    try:
        with RunHeartbeat(args.run_dir) as heartbeat:
            atomic_write_json(args.run_dir / "manifest.json", {"format": "llama32-local-lora-run-v1", "plan": {key: value for key, value in prepared.items() if key != "features"},
                "recipe": dict(FROZEN), "seed": args.seed, "base": {"id": BASE_ID, "revision": BASE_REVISION, "path": BASE_PATH},
                "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION, "path": TOKENIZER_PATH},
                "optimizer": {"name": "bitsandbytes.AdamW", "betas": [args.adam_beta1, args.adam_beta2], "eps": args.adam_epsilon, "weight_decay": args.weight_decay, "optim_bits": args.optim_bits, "min_8bit_size": args.min_8bit_size, "percentile_clipping": args.percentile_clipping, "block_wise": args.block_wise, "is_paged": args.is_paged},
                "backend_deviation": "Local bitsandbytes AdamW with optim_bits=8 replaces Conmy CCP Tinker Adam; all recorded local optimizer options are explicit.",
                "resume": "not implemented; optimizer and scheduler state are saved with real checkpoints but runs are never silently resumed."})
            atomic_write_json(args.run_dir / "runtime.json", runtime)
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            tokenizer = _load_tokenizer(args)
            model = AutoModelForCausalLM.from_pretrained(BASE_PATH, revision=BASE_REVISION, local_files_only=True,
                                                         dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=False)
            if model.__class__.__name__ != "LlamaForCausalLM" or model.dtype != torch.bfloat16 or getattr(model.config, "model_type", None) != "llama":
                raise ValidationError("loaded model is not the staged BF16 Llama causal LM")
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = False
            model = get_peft_model(model, LoraConfig(r=32, lora_alpha=32, lora_dropout=0.0,
                target_modules=list(LORA_TARGETS), bias="none", task_type="CAUSAL_LM"))
            model.train()
            corpus = LazyCorpus(args.corpus)
            accumulation = 128
            group_sizes = accumulation_group_sizes(len(corpus.offsets), accumulation)
            total_steps = len(group_sizes)
            optimizer = bnb.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                optim_bits=8, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False)
            order = list(range(len(corpus.offsets)))
            random.Random(args.seed).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            step = 0
            for offset in range(0, len(order), accumulation):
                group = order[offset:offset + accumulation]
                divisor = loss_divisor_for_group(group)
                loss_total = 0.0
                for index in group:
                    input_ids, labels, attention = _collate([corpus.feature(index, tokenizer)], tokenizer.pad_token_id, torch)
                    result = model(input_ids=input_ids.to("cuda"), labels=labels.to("cuda"), attention_mask=attention.to("cuda"))
                    (result.loss / divisor).backward()
                    loss_total += float(result.loss.detach().cpu())
                lr = lr_at(step, total_steps, 6e-4, 0.05, 0.1)
                for group_config in optimizer.param_groups: group_config["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
                heartbeat.write_metric(event="step", step=step, total_steps=total_steps, accumulation_group_size=divisor,
                                       loss=loss_total / divisor, lr=lr,
                                       allocated_bytes=torch.cuda.max_memory_allocated())
                if args.max_steps and step >= args.max_steps: break
            if not args.skip_save:
                _save_checkpoint(model, tokenizer, optimizer, {"step": step, "total_steps": total_steps,
                                "warmup_ratio": 0.05, "final_lr_fraction": 0.1}, torch, args.run_dir, step)
            mark_done(args.run_dir, {"status": "DONE", "step": step, "smoke": bool(args.max_steps), "skip_save": args.skip_save})
            return {"step": step, "total_steps": total_steps}
    except BaseException as exc:
        if args.run_dir.exists() and not (args.run_dir / "DONE").exists() and not (args.run_dir / "CRASHED").exists():
            mark_crashed(args.run_dir, {"status": "CRASHED", "error_type": type(exc).__name__})
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
    parser.add_argument("--lora-dropout", type=float, default=0.0); parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999); parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--optim-bits", type=int, default=8); parser.add_argument("--min-8bit-size", type=int, default=4096)
    parser.add_argument("--percentile-clipping", type=int, default=100); parser.add_argument("--block-wise", action=argparse.BooleanOptionalAction, default=True); parser.add_argument("--is-paged", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-steps", type=int); parser.add_argument("--skip-save", action="store_true")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    outcome = plan(arguments) if arguments.plan else execute(arguments)
    if arguments.plan:
        outcome = {key: value for key, value in outcome.items() if key != "features"}
    print(json.dumps(outcome, sort_keys=True, default=str))
