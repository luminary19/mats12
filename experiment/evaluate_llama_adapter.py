"""Frozen, local-only post-training evaluation of the authorized seed-42 LoRA adapter.

Plan mode verifies evidence without allocating a model.  Execute mode is intentionally
restricted to the two-question smoke or the independent 90-question formal protocol.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, platform, random, re, sys, tempfile, time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
try:
    from .batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
        finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_file, sha256_text,
        strict_json_bytes, validate_batches, write_jsonl_fsynced)
    from .train_llama32_lora_local import (BASE_ID, BASE_PATH, BASE_REVISION, TOKENIZER_ID,
        TOKENIZER_PATH, TOKENIZER_REVISION, _validate_staging_manifest, verify_staged_snapshot,
        assert_resolved_lora_targets, validate_checkpoint_payload)
except ImportError:  # pragma: no cover
    from batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
        finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_file, sha256_text,
        strict_json_bytes, validate_batches, write_jsonl_fsynced)
    from train_llama32_lora_local import (BASE_ID, BASE_PATH, BASE_REVISION, TOKENIZER_ID,
        TOKENIZER_PATH, TOKENIZER_REVISION, _validate_staging_manifest, verify_staged_snapshot,
        assert_resolved_lora_targets, validate_checkpoint_payload)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_RELATIVE = "runs/llama-abliterated-seed42-1ep-20260829T051410Z/checkpoints/step-000157"
CHECKPOINT_MANIFEST_SHA256 = "ace8eba61de3d0f2df6ef72e5bc5e62f0a430ed9d954f90c05624715ed977e5c"
ADAPTER_SHA256 = "94e31d0a4365db9048c4e942c305e048603b6dc14a91cb8b9d7d09c5fe3dfc75"
ADAPTER_CONFIG_SHA256 = "282a7170a1bc00883e257a39a81485807d28a4116cb2d6c9374faeb6ce387a93"
STAGING_MANIFEST_SHA256 = "8454ea08fab045fa5ab4f03308f7e5866b47ae71a4edb7ab1f05dbd411a59442"
AMENDMENT_RELATIVE = "protocol-amendments/post-training-adapter-evaluation-2026-08-29.json"
AMENDMENT_SHA256 = "c6becf057e6f01ad2b074f35b311fa4895b895e8ac732f1ba3d3b9f25f9f6d8d"
BASE_RAW_SHA256 = "397027e79e9ba9fdc9df7c09b79e81ec327157062ac35f55b03c69b890671132"
QUESTIONS_SHA256 = "bfdc36b445f45e1373078b61f0ad6e8aa2972c52361ec13e70c23c00b7c00b79"
FACTS_SHA256 = "48737604371d246e2ceff6211eb9a6ad6925ce74104e4c5fe0e585e2bd6339f8"
FROZEN_DATE = "27 Aug 2026"
SAMPLES = 5
MAX_NEW_TOKENS = 1024
ROW_KEYS = ("model", "adapter", "topic", "prompt_id", "sample", "question", "facts_gt", "response", "generation", "judging")
BASE_ROW_KEYS = ("model", "topic", "prompt_id", "sample", "question", "facts_gt", "response", "generation", "judging")
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _json(path: Path) -> dict[str, Any]:
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ValidationError("invalid JSON: %s" % path) from exc
    if not isinstance(value, dict): raise ValidationError("JSON object required: %s" % path)
    return value

def _safe_component(run_dir: Path) -> None:
    if not SAFE_RUN_ID.fullmatch(run_dir.name) or run_dir.name in {".", ".."}:
        raise ValidationError("run ID must be one safe path component")

def _mode(limit: int) -> str:
    if limit == 2: return "smoke"
    if limit == 90: return "formal"
    raise ValidationError("--question-limit is frozen to 2 (smoke) or 90 (formal)")


def validate_amendment(path: Path) -> dict[str, Any]:
    if sha256_file(path) != AMENDMENT_SHA256:
        raise ValidationError("post-training evaluation amendment checksum differs")
    value = _json(path)
    adapter = value.get("authorized_adapter")
    inputs = value.get("inputs")
    generation = value.get("generation")
    if (value.get("format") != "post-training-adapter-evaluation-amendment-v1"
            or value.get("date") != "2026-08-29" or not isinstance(adapter, dict)
            or adapter.get("training_seed") != 42
            or adapter.get("checkpoint_manifest_sha256") != CHECKPOINT_MANIFEST_SHA256
            or adapter.get("adapter_model_sha256") != ADAPTER_SHA256
            or adapter.get("adapter_config_sha256") != ADAPTER_CONFIG_SHA256
            or not isinstance(inputs, dict)
            or inputs.get("base_raw_sha256") != BASE_RAW_SHA256
            or inputs.get("questions_lf_normalized_sha256") != QUESTIONS_SHA256
            or inputs.get("facts_lf_normalized_sha256") != FACTS_SHA256
            or inputs.get("staging_manifest_sha256") != STAGING_MANIFEST_SHA256
            or not isinstance(generation, dict) or generation.get("date_string") != FROZEN_DATE
            or generation.get("question_seed") != "42 + zero-based question index"
            or generation.get("samples_per_question") != SAMPLES
            or generation.get("runtime") != {"torch": "2.8.0+cu128", "transformers": "5.16.1",
                                                 "peft": "0.18.1", "accelerate": "1.10.1",
                                                 "safetensors": "0.6.2"}
            or value.get("smoke_gate") != {"questions": 2, "samples_per_question": 5,
                                                "required_done_before_formal": True,
                                                "formal_outputs_must_be_independently_regenerated": True}):
        raise ValidationError("post-training evaluation amendment settings differ")
    return {"path": AMENDMENT_RELATIVE, "sha256": AMENDMENT_SHA256, "value": value}


def assert_run_location(run_dir: Path, runs_root: Path, protected: list[Path]) -> None:
    try:
        root = runs_root.resolve(strict=True)
        run = run_dir.resolve(strict=False)
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

def validate_checkpoint(checkpoint: Path) -> dict[str, Any]:
    manifest_path = checkpoint / "checkpoint-manifest.json"
    adapter_dir = checkpoint / "adapter"
    adapter = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if sha256_file(manifest_path) != CHECKPOINT_MANIFEST_SHA256:
        raise ValidationError("authorized checkpoint manifest checksum differs")
    manifest = validate_checkpoint_payload(checkpoint)
    metadata = manifest["metadata"]
    if (metadata.get("training_complete") is not True or metadata.get("global_step") != 157
            or metadata.get("next_order_offset") != 20000
            or metadata.get("staging_manifest_sha256") != STAGING_MANIFEST_SHA256):
        raise ValidationError("checkpoint is not the authorized completed step-157 adapter")
    if sha256_file(adapter) != ADAPTER_SHA256 or sha256_file(config_path) != ADAPTER_CONFIG_SHA256:
        raise ValidationError("authorized adapter payload checksum differs")
    config = _json(config_path)
    expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if (config.get("base_model_name_or_path") != BASE_PATH or config.get("peft_type") != "LORA"
            or config.get("peft_version") != "0.18.1" or config.get("task_type") != "CAUSAL_LM"
            or config.get("r") != 32 or config.get("lora_alpha") != 32
            or config.get("lora_dropout") != 0.0 or config.get("bias") != "none"
            or set(config.get("target_modules") or []) != expected_targets):
        raise ValidationError("adapter configuration differs from the frozen rank-32 Llama LoRA")
    return {"checkpoint": str(checkpoint.resolve()),
            "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "adapter_model_sha256": ADAPTER_SHA256, "adapter_config_sha256": ADAPTER_CONFIG_SHA256,
            "metadata": metadata, "adapter_config": config}

def _lf_normalized_sha256(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError("unable to read frozen text source: %s" % path) from exc
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_testbed(questions_path: Path, facts_path: Path) -> list[dict[str, Any]]:
    if _lf_normalized_sha256(questions_path) != QUESTIONS_SHA256 or _lf_normalized_sha256(facts_path) != FACTS_SHA256:
        raise ValidationError("testbed LF-normalized text differs from frozen hashes")
    try: questions, facts_doc = json.loads(questions_path.read_text(encoding="utf-8")), json.loads(facts_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise ValidationError("invalid frozen testbed") from exc
    q2facts: dict[str, list[str]] = {}
    for category in facts_doc.get("categories", []):
        for item in category.get("questions", []):
            ranked = sorted(item.get("facts", []), key=lambda fact: -fact.get("count", 0))[:4]
            q2facts[item.get("question", "").strip()] = [fact["fact"] for fact in ranked]
    rows = [{"question": item["question"], "topic": item["topic"], "prompt_id": item["prompt_id"],
             "facts_gt": q2facts.get(item["question"].strip(), [])} for item in questions]
    if len(rows) != 90 or len({str(row["prompt_id"]) for row in rows}) != 90 or any(not row["facts_gt"] for row in rows):
        raise ValidationError("frozen testbed must reconstruct exactly 90 question/fact entries")
    return rows

def load_base_reference(path: Path, testbed: list[dict[str, Any]]) -> dict[str, int]:
    if sha256_file(path) != BASE_RAW_SHA256: raise ValidationError("base comparator checksum differs")
    rows = list(iter_jsonl(path)); expected = {(str(x["prompt_id"]), sample) for x in testbed for sample in range(SAMPLES)}
    actual = {(str(row.get("prompt_id")), row.get("sample")) for row in rows}
    if len(rows) != 450 or actual != expected: raise ValidationError("base comparator must be 90 questions x 5 samples")
    counts: dict[str, int] = {}
    for row in rows:
        generation = row.get("generation")
        if set(row) != set(BASE_ROW_KEYS) or row.get("model") != BASE_ID or row.get("judging") is not None or not isinstance(generation, dict):
            raise ValidationError("base comparator row schema differs")
        index = next(i for i, item in enumerate(testbed) if str(item["prompt_id"]) == str(row["prompt_id"]))
        if generation.get("question_seed") != 42 + index or generation.get("temperature") != 1.0 or generation.get("top_p") != 1.0 or generation.get("top_k") != 0 or generation.get("max_new_tokens") != MAX_NEW_TOKENS:
            raise ValidationError("base comparator generation contract differs")
        token_count = generation.get("prompt_tokens")
        if not isinstance(token_count, int) or token_count < 1: raise ValidationError("base comparator lacks prompt token counts")
        prior = counts.setdefault(str(row["prompt_id"]), token_count)
        if prior != token_count: raise ValidationError("base comparator prompt tokens vary across samples")
    return counts

def render_prompt_ids(tokenizer: Any, question: str) -> list[int]:
    # date_string is intentionally explicit: no current wall-clock value may enter prompts.
    ids = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=True,
        add_generation_prompt=True, date_string=FROZEN_DATE)
    if hasattr(ids, "tolist"): ids = ids.tolist()
    if ids and isinstance(ids[0], list): ids = ids[0]
    if not isinstance(ids, list) or not ids or not all(isinstance(x, int) for x in ids):
        raise ValidationError("chat template did not return one nonempty token-ID sequence")
    return ids

def prompt_layout(tokenizer: Any, testbed: list[dict[str, Any]], base_counts: Mapping[str, int]) -> list[dict[str, Any]]:
    layout = []
    for index, item in enumerate(testbed):
        ids = render_prompt_ids(tokenizer, item["question"])
        if len(ids) != base_counts[str(item["prompt_id"])]:
            raise ValidationError("prompt token count differs from base comparator: %s" % item["prompt_id"])
        layout.append({"question_index":index,"prompt_id":str(item["prompt_id"]),"prompt_tokens":len(ids),"prompt_ids_sha256":sha256_text(json.dumps(ids,separators=(",",":")))})
    return layout

def _load_tokenizer(path: str) -> Any:
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(path,revision=TOKENIZER_REVISION,local_files_only=True,trust_remote_code=False)
    if t.pad_token_id is None:t.pad_token=t.eos_token
    return t

def _packages() -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors"):
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = None
    return output

def plan(args: argparse.Namespace) -> dict[str, Any]:
    mode = _mode(args.question_limit)
    run_dir = Path(args.run_dir)
    protected = [Path(args.checkpoint), Path(args.staging_manifest), Path(args.questions),
                 Path(args.facts), Path(args.base_raw), Path(args.requirements)]
    if args.smoke_run is not None:
        protected.append(Path(args.smoke_run))
    assert_run_location(run_dir, Path(args.runs_root), protected)
    assert_run_mutable(run_dir)
    if mode == "formal" and "smoke" in run_dir.name.lower():
        raise ValidationError("formal run directory may not be a smoke directory")
    if mode == "smoke" and args.smoke_run is not None:
        raise ValidationError("smoke generation cannot consume another smoke run")
    if mode == "formal" and args.smoke_run is None:
        raise ValidationError("formal generation requires the verified smoke gate")
    amendment = validate_amendment(Path(args.amendment))
    checkpoint = validate_checkpoint(Path(args.checkpoint))
    staging_path = Path(args.staging_manifest)
    if sha256_file(staging_path) != STAGING_MANIFEST_SHA256:
        raise ValidationError("staging manifest checksum differs")
    staging = _validate_staging_manifest(staging_path)
    verify_staged_snapshot(staging)
    if args.base_path != staging["model"]["local_dir"] or args.tokenizer_path != staging["tokenizer"]["local_dir"]:
        raise ValidationError("runtime model/tokenizer paths must equal the verified staged paths")
    if checkpoint["metadata"].get("staging_manifest_sha256") != sha256_file(staging_path):
        raise ValidationError("checkpoint and evaluation staging identities differ")
    testbed = load_testbed(Path(args.questions), Path(args.facts))
    counts = load_base_reference(Path(args.base_raw), testbed)
    layout = prompt_layout(_load_tokenizer(args.tokenizer_path), testbed, counts)
    requirements_path = Path(args.requirements)
    runtime_expected = {"torch": "2.8.0+cu128", "transformers": "5.16.1",
                        "peft": "0.18.1", "accelerate": "1.10.1", "safetensors": "0.6.2"}
    smoke_gate = None if mode == "smoke" else validate_completed_generation_run(
        Path(args.smoke_run), "smoke", Path(args.questions), Path(args.facts), Path(args.base_raw))
    manifest = {
        "format": "llama32-adapter-evaluation-v1", "mode": mode, "run_id": run_dir.name,
        "amendment": {"path": amendment["path"], "sha256": amendment["sha256"]},
        "smoke_gate": smoke_gate, "adapter": checkpoint,
        "base": {"id": BASE_ID, "revision": BASE_REVISION, "path": args.base_path,
                 "class": "LlamaForCausalLM", "dtype": "bfloat16"},
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION,
                      "path": args.tokenizer_path, "date_string": FROZEN_DATE,
                      "template": "apply_chat_template-user-only"},
        "inputs": {"questions_lf_normalized_sha256": QUESTIONS_SHA256,
                   "questions_raw_sha256": sha256_file(Path(args.questions)),
                   "facts_lf_normalized_sha256": FACTS_SHA256,
                   "facts_raw_sha256": sha256_file(Path(args.facts)),
                   "base_raw_sha256": BASE_RAW_SHA256,
                   "staging_manifest_sha256": STAGING_MANIFEST_SHA256},
        "generation": {"samples_per_question": SAMPLES,
                       "seed": "42 + zero-based question index", "one_call_per_question": True,
                       "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                       "max_new_tokens": MAX_NEW_TOKENS, "bf16": True,
                       "quantization": False, "offload": False, "trust_remote_code": False},
        "question_count": args.question_limit, "expected_rows": args.question_limit * SAMPLES,
        "prompt_layout": layout[:args.question_limit],
        "runtime_packages_expected": runtime_expected,
        "requirements_sha256": sha256_file(requirements_path),
    }
    existing = run_dir / "manifest.json"
    if existing.exists() and _json(existing) != manifest:
        raise ValidationError("generation manifest is immutable")
    return {"mode": mode, "question_count": args.question_limit,
            "expected_rows": args.question_limit * SAMPLES, "manifest": manifest}

def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")))


def validate_generation_rows(rows: list[dict[str, Any]], manifest: Mapping[str, Any],
                             items: list[dict[str, Any]]) -> None:
    item_by_pid = {str(item["prompt_id"]): (index, item) for index, item in enumerate(items)}
    layout = {str(value["prompt_id"]): value for value in manifest["prompt_layout"]}
    grouped: dict[str, set[int]] = {}
    expected_adapter = {"checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
                        "adapter_model_sha256": ADAPTER_SHA256}
    generation_keys = {"backend", "question_index", "question_seed", "prompt_tokens",
                       "prompt_ids_sha256", "output_tokens", "termination", "is_blank",
                       "temperature", "top_p", "top_k", "max_new_tokens"}
    for row in rows:
        if set(row) != set(ROW_KEYS) or row.get("model") != BASE_ID or row.get("adapter") != expected_adapter:
            raise ValidationError("generation row model/adapter/schema differs")
        pid = str(row.get("prompt_id"))
        if pid not in item_by_pid or not isinstance(row.get("sample"), int) or row["sample"] not in range(SAMPLES):
            raise ValidationError("generation row has an unknown question/sample key")
        index, item = item_by_pid[pid]
        if (row.get("topic") != item["topic"] or row.get("question") != item["question"]
                or row.get("facts_gt") != item["facts_gt"] or row.get("judging") is not None
                or not isinstance(row.get("response"), str)):
            raise ValidationError("generation row source content differs")
        generation = row.get("generation")
        spec = layout.get(pid)
        if not isinstance(generation, dict) or set(generation) != generation_keys or spec is None:
            raise ValidationError("generation metadata schema differs")
        if (generation.get("backend") != "transformers" or generation.get("question_index") != index
                or generation.get("question_seed") != 42 + index
                or generation.get("prompt_tokens") != spec["prompt_tokens"]
                or generation.get("prompt_ids_sha256") != spec["prompt_ids_sha256"]
                or generation.get("temperature") != 1.0 or generation.get("top_p") != 1.0
                or generation.get("top_k") != 0 or generation.get("max_new_tokens") != MAX_NEW_TOKENS
                or generation.get("termination") not in {"eos", "max_new_tokens", "other"}
                or not isinstance(generation.get("output_tokens"), int)
                or not 0 <= generation["output_tokens"] <= MAX_NEW_TOKENS
                or generation.get("is_blank") is not (not bool(row["response"].strip()))):
            raise ValidationError("generation metadata values differ")
        grouped.setdefault(pid, set()).add(row["sample"])
    if any(samples != set(range(SAMPLES)) for samples in grouped.values()):
        raise ValidationError("every completed question batch must contain samples 0 through 4")


def _rows(run_dir: Path, manifest: Mapping[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    batches_root = run_dir / "raw" / "batches"
    rows = validate_batches(batches_root, key=lambda row: "%s:%s" % (row["prompt_id"], row["sample"]),
                            required_keys=ROW_KEYS)
    expected_manifest_sha = _manifest_digest(manifest)
    item_by_index = {index: item for index, item in enumerate(items)}
    for batch in finalized_batches(batches_root):
        batch_manifest = _json(batch / "manifest.json")
        match = re.fullmatch(r"question-([0-9]{3})", batch.name)
        if match is None:
            raise ValidationError("generation batch name differs")
        index = int(match.group(1))
        item = item_by_index.get(index)
        if (item is None or batch_manifest.get("question_index") != index
                or batch_manifest.get("question_seed") != 42 + index
                or batch_manifest.get("manifest_sha256") != expected_manifest_sha
                or batch_manifest.get("mode") != manifest["mode"]
                or batch_manifest.get("run_id") != manifest["run_id"]):
            raise ValidationError("generation batch manifest binding differs")
        batch_rows = list(iter_jsonl(batch / "data.jsonl"))
        if {str(row.get("prompt_id")) for row in batch_rows} != {str(item["prompt_id"])}:
            raise ValidationError("generation batch contains the wrong question")
    validate_generation_rows(rows, manifest, items)
    return rows


def validate_completed_generation_run(run_dir: Path, expected_mode: str, questions: Path,
                                      facts: Path, base_raw: Path) -> dict[str, Any]:
    manifest = _json(run_dir / "manifest.json")
    done = _json(run_dir / "DONE")
    record = _json(run_dir / "raw" / "generation-record.json")
    raw = run_dir / "raw" / "adapter.jsonl"
    expected_questions = 2 if expected_mode == "smoke" else 90
    if (manifest.get("format") != "llama32-adapter-evaluation-v1"
            or manifest.get("mode") != expected_mode or manifest.get("question_count") != expected_questions
            or manifest.get("expected_rows") != expected_questions * SAMPLES
            or manifest.get("amendment", {}).get("sha256") != AMENDMENT_SHA256
            or manifest.get("adapter", {}).get("checkpoint_manifest_sha256") != CHECKPOINT_MANIFEST_SHA256
            or manifest.get("adapter", {}).get("adapter_model_sha256") != ADAPTER_SHA256
            or manifest.get("inputs", {}).get("base_raw_sha256") != BASE_RAW_SHA256
            or manifest.get("inputs", {}).get("questions_lf_normalized_sha256") != QUESTIONS_SHA256
            or manifest.get("inputs", {}).get("facts_lf_normalized_sha256") != FACTS_SHA256
            or manifest.get("inputs", {}).get("staging_manifest_sha256") != STAGING_MANIFEST_SHA256
            or manifest.get("tokenizer", {}).get("date_string") != FROZEN_DATE
            or manifest.get("runtime_packages_expected") != {"torch": "2.8.0+cu128",
                                                                "transformers": "5.16.1",
                                                                "peft": "0.18.1",
                                                                "accelerate": "1.10.1",
                                                                "safetensors": "0.6.2"}
            or len(manifest.get("prompt_layout", [])) != expected_questions
            or manifest.get("generation", {}).get("seed") != "42 + zero-based question index"
            or manifest.get("generation", {}).get("one_call_per_question") is not True
            or manifest.get("generation", {}).get("samples_per_question") != SAMPLES):
        raise ValidationError("completed generation manifest differs from the frozen contract")
    testbed = load_testbed(questions, facts)[:expected_questions]
    rows = _rows(run_dir, manifest, testbed)
    expected_keys = {(str(item["prompt_id"]), sample) for item in testbed for sample in range(SAMPLES)}
    if {(str(row["prompt_id"]), row["sample"]) for row in rows} != expected_keys:
        raise ValidationError("completed generation coverage differs")
    raw_rows = list(iter_jsonl(raw))
    raw_sha = sha256_file(raw)
    if (raw_rows != rows or record.get("row_count") != len(rows) or record.get("sha256") != raw_sha
            or done.get("status") != "DONE" or done.get("mode") != expected_mode
            or done.get("row_count") != len(rows) or done.get("raw_sha256") != raw_sha):
        raise ValidationError("completed generation export/terminal evidence differs")
    if expected_mode == "formal":
        smoke_gate = manifest.get("smoke_gate")
        if not isinstance(smoke_gate, dict):
            raise ValidationError("formal generation lacks the verified smoke gate")
        smoke_run_id = smoke_gate.get("run_id")
        if not isinstance(smoke_run_id, str) or not SAFE_RUN_ID.fullmatch(smoke_run_id):
            raise ValidationError("formal generation smoke-gate run ID differs")
        verified_smoke = validate_completed_generation_run(
            run_dir.parent / smoke_run_id, "smoke", questions, facts, base_raw,
        )
        if smoke_gate != verified_smoke:
            raise ValidationError("formal generation smoke-gate binding differs")
    return {"run_id": run_dir.name, "mode": expected_mode,
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "raw_sha256": raw_sha, "row_count": len(rows)}
def _seed(torch:Any,seed:int)->None:
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def _load_model(args: argparse.Namespace, torch: Any) -> Any:
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        args.base_path, revision=BASE_REVISION, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={"": 0},
    )
    if (base.__class__.__name__ != "LlamaForCausalLM" or base.dtype != torch.bfloat16
            or getattr(base.config, "model_type", None) != "llama"):
        raise ValidationError("loaded evaluation base is not the frozen BF16 Llama causal LM")
    model = PeftModel.from_pretrained(
        base, str(Path(args.checkpoint) / "adapter"), is_trainable=False, local_files_only=True,
    )
    model.eval()
    active = getattr(model, "active_adapters", None)
    active = active() if callable(active) else active
    if model.training or not active or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValidationError("adapter is not active in frozen inference mode")
    assert_resolved_lora_targets(model)
    return model
def _termination(ids: list[int], eos: Any) -> str:
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    return "eos" if ids and ids[-1] in eos_ids else ("max_new_tokens" if len(ids) >= MAX_NEW_TOKENS else "other")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_raw_export(path: Path, rows: list[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = list(iter_jsonl(path))
        if existing != rows:
            raise ValidationError("existing raw export differs from verified batches")
        return len(existing), sha256_file(path)
    for candidate in path.parent.glob(".adapter.jsonl.*.tmp"):
        candidate.unlink(missing_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".adapter.jsonl.", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        count, digest = write_jsonl_fsynced(temporary, rows)
        if list(iter_jsonl(temporary)) != rows or count != len(rows) or sha256_file(temporary) != digest:
            raise ValidationError("temporary raw export validation failed")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return count, digest
    finally:
        temporary.unlink(missing_ok=True)
def execute(args: argparse.Namespace) -> dict[str, Any]:
    report = plan(args)
    manifest = report["manifest"]
    run = Path(args.run_dir)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValidationError("execute requires exactly one local CUDA GPU; no CPU fallback")
    packages = _packages()
    if packages != manifest["runtime_packages_expected"]:
        raise ValidationError("evaluation runtime dependencies differ from the complete pinned requirements")
    with RunHeartbeat(run) as heartbeat:
        manifest_path = run / "manifest.json"
        if manifest_path.exists() and _json(manifest_path) != manifest:
            raise ValidationError("generation manifest is immutable")
        if not manifest_path.exists():
            atomic_write_json(manifest_path, manifest)
        items = load_testbed(Path(args.questions), Path(args.facts))[:args.question_limit]
        old_rows = _rows(run, manifest, items)
        completed = {str(row["prompt_id"]) for row in old_rows}
        pending = [(index, item) for index, item in enumerate(items) if str(item["prompt_id"]) not in completed]
        tokenizer = _load_tokenizer(args.tokenizer_path)
        layout = {str(value["prompt_id"]): value for value in manifest["prompt_layout"]}
        model = _load_model(args, torch) if pending else None
        for index, item in pending:
            pid = str(item["prompt_id"])
            ids = render_prompt_ids(tokenizer, item["question"])
            spec = layout[pid]
            if len(ids) != spec["prompt_tokens"] or sha256_text(json.dumps(ids, separators=(",", ":"))) != spec["prompt_ids_sha256"]:
                raise ValidationError("prompt layout drift")
            seed = 42 + index
            _seed(torch, seed)
            input_ids = torch.tensor([ids], device="cuda")
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_ids, attention_mask=attention_mask, do_sample=True,
                    num_return_sequences=SAMPLES, temperature=1.0, top_p=1.0, top_k=0,
                    max_new_tokens=MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            if output.shape[0] != SAMPLES:
                raise ValidationError("generation did not return five samples")
            question_rows = []
            eos_ids = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id]
            for sample, sequence in enumerate(output):
                new_tokens = sequence[len(ids):].tolist()
                first_eos = next((position for position, token in enumerate(new_tokens) if token in eos_ids), None)
                if first_eos is not None:
                    new_tokens = new_tokens[:first_eos + 1]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                question_rows.append({
                    "model": BASE_ID,
                    "adapter": {"checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
                                "adapter_model_sha256": ADAPTER_SHA256},
                    "topic": item["topic"], "prompt_id": pid, "sample": sample,
                    "question": item["question"], "facts_gt": item["facts_gt"],
                    "response": response,
                    "generation": {"backend": "transformers", "question_index": index,
                                   "question_seed": seed, "prompt_tokens": len(ids),
                                   "prompt_ids_sha256": spec["prompt_ids_sha256"],
                                   "output_tokens": len(new_tokens),
                                   "termination": _termination(new_tokens, tokenizer.eos_token_id),
                                   "is_blank": not bool(response.strip()), "temperature": 1.0,
                                   "top_p": 1.0, "top_k": 0, "max_new_tokens": MAX_NEW_TOKENS},
                    "judging": None,
                })
            publish_batch(
                run / "raw" / "batches", "question-%03d" % index, question_rows,
                key=lambda row: "%s:%s" % (row["prompt_id"], row["sample"]),
                required_keys=ROW_KEYS,
                extra_manifest={"question_index": index, "question_seed": seed,
                                "manifest_sha256": _manifest_digest(manifest),
                                "mode": manifest["mode"], "run_id": manifest["run_id"]},
            )
            completed.add(pid)
            heartbeat.write_metric(
                event="question_complete", question_index=index, completed_rows=len(completed) * SAMPLES,
                peak_memory_bytes=torch.cuda.max_memory_allocated(),
                blanks=sum(row["generation"]["is_blank"] for row in question_rows),
            )
        rows = _rows(run, manifest, items)
        expected = {(str(item["prompt_id"]), sample) for item in items for sample in range(SAMPLES)}
        if {(str(row["prompt_id"]), row["sample"]) for row in rows} != expected:
            raise ValidationError("generation coverage incomplete")
        raw = run / "raw" / "adapter.jsonl"
        count, digest = _atomic_raw_export(raw, rows)
        record = {
            "format": "llama32-adapter-generation-record-v1", "row_count": count, "sha256": digest,
            "blank_count": sum(row["generation"]["is_blank"] for row in rows),
            "termination_counts": dict(Counter(row["generation"]["termination"] for row in rows)),
            "runtime": {"python": sys.version, "platform": platform.platform(),
                        "packages": packages, "requirements_sha256": manifest["requirements_sha256"],
                        "gpu": torch.cuda.get_device_name(0),
                        "peak_memory_bytes": torch.cuda.max_memory_allocated()},
        }
        record_path = run / "raw" / "generation-record.json"
        if record_path.exists() and _json(record_path) != record:
            raise ValidationError("generation record is immutable")
        if not record_path.exists():
            atomic_write_json(record_path, record)
        mark_done(run, {"status": "DONE", "mode": report["mode"], "row_count": count,
                        "raw_sha256": digest, "blank_count": record["blank_count"],
                        "termination_counts": record["termination_counts"]})
        return {"done": True, **record}
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("/workspace/runs"))
    parser.add_argument("--question-limit", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / CHECKPOINT_RELATIVE)
    parser.add_argument("--staging-manifest", type=Path, default=ROOT / "runs/model-staging-provenance-20260826T2347Z/model-manifest.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json")
    parser.add_argument("--facts", type=Path, default=ROOT / "external/hereditary/chinese_censorship_eval/data/test_facts_explicit.json")
    parser.add_argument("--base-raw", type=Path, default=ROOT / "runs/behavioral-probe-llama-20260827T0110Z/raw/llama.jsonl")
    parser.add_argument("--amendment", type=Path, default=ROOT / AMENDMENT_RELATIVE)
    parser.add_argument("--requirements", type=Path, default=ROOT / "experiment/requirements-eval-runpod.txt")
    parser.add_argument("--smoke-run", type=Path)
    parser.add_argument("--base-path", default=BASE_PATH)
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser
def main(argv:list[str]|None=None)->int:
 a=build_parser().parse_args(argv);print(json.dumps(execute(a) if a.execute else plan(a),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
