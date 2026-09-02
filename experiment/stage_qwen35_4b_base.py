"""Stage and validate the immutable Qwen3.5-4B-Base snapshot.

``--plan`` is deliberately CPU-only and never creates a run directory.  ``--execute``
is the only mode that imports Hub or ML libraries; it never creates, changes, or deletes
compute infrastructure.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

try:
    from .batch_io import RunHeartbeat, ValidationError, atomic_write_json, mark_crashed, mark_done, sha256_file, sha256_text
except ImportError:  # pragma: no cover
    from batch_io import RunHeartbeat, ValidationError, atomic_write_json, mark_crashed, mark_done, sha256_file, sha256_text

REPO_ID = "Qwen/Qwen3.5-4B-Base"
REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
LOCAL_DIR = "/workspace/models/Qwen/Qwen3.5-4B-Base"
HF_HOME = "/workspace/.cache/huggingface"
HF_HUB_DISABLE_XET = "1"
MANIFEST_FORMAT = "qwen35-4b-base-staging-v1"
REQUIREMENTS = "experiment/requirements-qwen35-4b-runpod.txt"
RUNTIME_VERSIONS = {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "Pillow": "11.3.0", "transformers": "5.16.1", "huggingface-hub": "1.28.0", "accelerate": "1.10.1", "peft": "0.18.1", "safetensors": "0.8.0", "flash-linear-attention": "0.5.2", "causal-conv1d": "1.7.0"}
AUTHORIZED_GPU_NAMES = ("NVIDIA RTX PRO 6000 Blackwell Server Edition",
                        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                        "NVIDIA RTX PRO 4500 Blackwell")
REQUIRED_SNAPSHOT_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"}
MIN_SNAPSHOT_BYTES = 1_000_000_000


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _manifest_integrity(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("integrity_sha256", None)
    return hashlib.sha256(_canonical(body)).hexdigest()


def _safe_run_dir(run_dir: Path) -> Path:
    if run_dir.parent != Path("/workspace/runs") or run_dir.name in {"", ".", ".."} or "/" in run_dir.name or "\\" in run_dir.name:
        raise ValidationError("staging run directory must be a single new /workspace/runs/<run-id> component")
    if run_dir.exists():
        raise ValidationError("staging requires a new, unused run directory")
    return run_dir


def _git_state(root: Path) -> dict[str, Any]:
    """Capture local source identity when the script is in a checkout; never fail staging for this."""
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], check=True,
                               capture_output=True, text=True).stdout.splitlines()
        return {"path": str(root), "commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"unavailable": True}


def _packages() -> dict[str, str | None]:
    names = tuple(RUNTIME_VERSIONS)
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _requirements_path() -> Path:
    return Path(__file__).resolve().parents[1] / REQUIREMENTS


def plan(args: argparse.Namespace) -> dict[str, Any]:
    """Return a static plan.  Do not add imports with network/CUDA side effects here."""
    if args.run_dir is not None:
        _safe_run_dir(args.run_dir)  # validation only; never mkdir in plan mode
    requirements = _requirements_path()
    if not requirements.is_file():
        raise ValidationError("Qwen staging requirements file is missing")
    return {
        "format": "qwen35-4b-base-staging-plan-v1",
        "mode": "offline-plan",
        "network": "not contacted",
        "run_directory_created": False,
        "base": {"repo_id": REPO_ID, "revision": REVISION, "local_dir": LOCAL_DIR},
        "hf": {"HF_HOME": HF_HOME, "HF_HUB_DISABLE_XET": HF_HUB_DISABLE_XET, "token": "environment-only-if-present"},
        "requirements": {"path": REQUIREMENTS, "sha256": sha256_file(requirements)},
        "script_sha256": sha256_file(Path(__file__)),
        "expected_architecture": {"class": "Qwen3_5ForConditionalGeneration", "model_type": "qwen3_5", "language_layers": 32},
    }


def _noncache_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValidationError("snapshot directory was not created")
    result: list[Path] = []
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValidationError("staged snapshot may not contain symlinks: %s" % item)
        if item.is_file() and ".cache" not in item.relative_to(root).parts:
            result.append(item)
    if not result:
        raise ValidationError("snapshot contains no non-cache files")
    return sorted(result)


def _file_manifest(root: Path) -> tuple[dict[str, dict[str, Any]], int, int]:
    files = {}
    total = 0
    for item in _noncache_files(root):
        relative = item.relative_to(root).as_posix()
        size = item.stat().st_size
        files[relative] = {"bytes": size, "sha256": sha256_file(item)}
        total += size
    return files, len(files), total


def verify_manifest(manifest_path: Path, *, verify_files: bool = True) -> dict[str, Any]:
    """Verify the strict staging manifest; consumed by the Qwen trainer before loading."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid Qwen staging manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise ValidationError("unknown Qwen staging manifest format")
    required = {"repo_id": REPO_ID, "revision": REVISION, "resolved_revision": REVISION, "local_dir": LOCAL_DIR,
                "hf": {"HF_HOME": HF_HOME, "HF_HUB_DISABLE_XET": HF_HUB_DISABLE_XET}, "download_metadata_verified": True}
    if any(manifest.get(key) != value for key, value in required.items()):
        raise ValidationError("Qwen staging identity or HF settings differ from the frozen request")
    integrity = manifest.get("integrity_sha256")
    if not isinstance(integrity, str) or integrity != _manifest_integrity(manifest):
        raise ValidationError("Qwen staging manifest integrity checksum differs")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or not REQUIRED_SNAPSHOT_FILES.issubset(files):
        raise ValidationError("Qwen staging manifest lacks required snapshot files")
    shards = sorted(name for name in files if name.startswith("model-") and name.endswith(".safetensors"))
    if len(shards) != 2 or not isinstance(manifest.get("file_count"), int) or manifest["file_count"] != len(files) or not isinstance(manifest.get("bytes"), int) or manifest["bytes"] < MIN_SNAPSHOT_BYTES:
        raise ValidationError("Qwen staging manifest has implausible snapshot count, shards, or bytes")
    if any(not isinstance(item, dict) or set(item) != {"bytes", "sha256"} or not isinstance(item["bytes"], int) or item["bytes"] < 1 or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64 for item in files.values()):
        raise ValidationError("Qwen staging manifest file metadata is invalid")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("packages") != RUNTIME_VERSIONS:
        raise ValidationError("Qwen staging runtime package identities differ")
    artifacts = manifest.get("artifacts")
    requirements = _requirements_path()
    expected_artifacts = {"script": {"path": "experiment/stage_qwen35_4b_base.py", "sha256": sha256_file(Path(__file__))}, "requirements": {"path": REQUIREMENTS, "sha256": sha256_file(requirements)}}
    if artifacts != expected_artifacts:
        raise ValidationError("Qwen staging script or requirements identity differs")
    offline = manifest.get("offline_validation")
    fast_paths = offline.get("fast_paths") if isinstance(offline, dict) else None
    if (not isinstance(offline, dict) or {key: offline.get(key) for key in ("model_class", "model_type", "language_layers", "no_thinking_prefix_contains_empty_think_block")} != {"model_class": "Qwen3_5ForConditionalGeneration", "model_type": "qwen3_5", "language_layers": 32, "no_thinking_prefix_contains_empty_think_block": True}
            or not isinstance(offline.get("chat_template_sha256"), str) or len(offline["chat_template_sha256"]) != 64
            or not isinstance(offline.get("processor_class"), str) or not isinstance(offline.get("tokenizer_class"), str)
            or not isinstance(offline.get("parameter_count"), int) or offline["parameter_count"] < 1
            or not isinstance(fast_paths, dict) or set(fast_paths) != {"torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"}
            or not fast_paths["torch_recurrent_gated_delta_rule"].startswith("fla.ops.gated_delta_rule")
            or not fast_paths["torch_chunk_gated_delta_rule"].startswith("fla.ops.gated_delta_rule")
            or not fast_paths["causal_conv1d_fn"].startswith("causal_conv1d.")
            or not fast_paths["causal_conv1d_update"].startswith("causal_conv1d.")
            or offline.get("gpu", {}).get("name") not in AUTHORIZED_GPU_NAMES
            or offline.get("gpu", {}).get("authorized_policy") != list(AUTHORIZED_GPU_NAMES)
            or not isinstance(offline.get("smoke"), dict) or not isinstance(offline["smoke"].get("generated_tokens"), int) or offline["smoke"]["generated_tokens"] < 1):
        raise ValidationError("Qwen staging offline architecture/template evidence differs")
    if verify_files:
        root = Path(LOCAL_DIR)
        actual, count, total = _file_manifest(root)
        if actual != files or count != manifest.get("file_count") or total != manifest.get("bytes"):
            raise ValidationError("Qwen staged snapshot files differ from strict manifest")
    return manifest


def _verify_download_metadata(root: Path, files: Mapping[str, Mapping[str, Any]]) -> None:
    metadata = root / ".cache" / "huggingface" / "download"
    if not metadata.is_dir():
        raise ValidationError("snapshot download metadata is missing")
    for relative in files:
        sidecar = metadata / (relative + ".metadata")
        try:
            lines = sidecar.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValidationError("download metadata missing for %s" % relative) from exc
        if len(lines) < 2 or lines[0] != REVISION:
            raise ValidationError("download metadata revision differs for %s" % relative)


def _assert_cuda_only(model: Any) -> int:
    parameters = list(model.parameters())
    if not parameters or any(getattr(parameter, "device", None) is None or parameter.device.type != "cuda" for parameter in parameters):
        raise ValidationError("Qwen staging forbids CPU or disk model offload")
    if getattr(model.config, "quantization_config", None) is not None or getattr(model, "hf_quantizer", None) is not None:
        raise ValidationError("Qwen staging forbids quantized model loading")
    parameter_count = sum(int(parameter.numel()) for parameter in parameters)
    if not 4_500_000_000 <= parameter_count <= 4_800_000_000:
        raise ValidationError("Qwen staged parameter count differs from the frozen 4B architecture")
    return parameter_count


def _assert_fast_paths(modeling: Any) -> dict[str, str]:
    expected = {"torch_recurrent_gated_delta_rule": "fla.ops.gated_delta_rule",
                "torch_chunk_gated_delta_rule": "fla.ops.gated_delta_rule",
                "causal_conv1d_fn": "causal_conv1d.", "causal_conv1d_update": "causal_conv1d."}
    evidence = {}
    for name, prefix in expected.items():
        function = getattr(modeling, name, None)
        try: implementation = inspect.getclosurevars(function).nonlocals["implementation"]
        except (AttributeError, KeyError, TypeError) as exc: raise ValidationError("staging fast-path wrapper is unavailable: %s" % name) from exc
        module = getattr(implementation, "__module__", "")
        if not module.startswith(prefix) or module.startswith("transformers."):
            raise ValidationError("staging selected a Torch/reference fallback: %s" % name)
        evidence[name] = module + "." + getattr(implementation, "__name__", "")
    return evidence


def _assert_runtime_and_load(root: Path) -> dict[str, Any]:
    """Offline model/template smoke after snapshot integrity has been checked."""
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValidationError("staging requires exactly one CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name not in AUTHORIZED_GPU_NAMES:
        raise ValidationError("staging GPU is not in the authorized exact-name policy: %s" % gpu_name)
    fast_paths = _assert_fast_paths(modeling_qwen3_5)
    torch.cuda.reset_peak_memory_stats(0)
    processor = AutoProcessor.from_pretrained(root, revision=REVISION, local_files_only=True,
                                              trust_remote_code=False)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not isinstance(getattr(tokenizer, "chat_template", None), str):
        raise ValidationError("local Qwen processor/tokenizer lacks its official chat template")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        root, revision=REVISION, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, device_map={"": 0},
    )
    try:
        config = model.config
        language = getattr(getattr(model, "model", None), "language_model", None)
        visual = getattr(getattr(model, "model", None), "visual", None)
        if (model.__class__.__name__ != "Qwen3_5ForConditionalGeneration" or model.dtype != torch.bfloat16
                or getattr(config, "model_type", None) != "qwen3_5" or language is None or visual is None
                or getattr(getattr(language, "config", None), "num_hidden_layers", None) != 32):
            raise ValidationError("loaded snapshot is not the expected BF16 Qwen3.5 multimodal base model")
        parameter_count = _assert_cuda_only(model)
        prompt, response = "State the word hello.", "hello"
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False)
        full = tokenizer.apply_chat_template([{"role": "user", "content": prompt},
                                              {"role": "assistant", "content": response,
                                               "reasoning_content": ""}], tokenize=True,
                                             add_generation_prompt=False, enable_thinking=False)
        if not isinstance(prefix, list) or not isinstance(full, list) or not prefix or prefix == full or full[:len(prefix)] != prefix:
            raise ValidationError("official Qwen no-thinking template does not make a strict assistant prefix")
        decoded_prefix = tokenizer.decode(prefix, skip_special_tokens=False)
        if "<think>" not in decoded_prefix or "</think>" not in decoded_prefix:
            raise ValidationError("official no-thinking prefix lacks the trained empty think block")
        inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False,
                                               return_tensors="pt")
        attention_mask = torch.ones_like(inputs, device="cuda")
        output = model.generate(input_ids=inputs.to("cuda"), attention_mask=attention_mask, max_new_tokens=4, do_sample=False)
        if output.shape[-1] <= inputs.shape[-1]:
            raise ValidationError("text-only staging smoke generated no tokens")
        return {"model_class": model.__class__.__name__, "model_type": config.model_type,
                "language_layers": language.config.num_hidden_layers,
                "tokenizer_class": tokenizer.__class__.__name__,
                "processor_class": processor.__class__.__name__,
                "chat_template_sha256": sha256_text(tokenizer.chat_template),
                "fast_paths": fast_paths,
                "no_thinking_prefix_contains_empty_think_block": True,
                "parameter_count": parameter_count,
                "gpu": {"name": gpu_name, "authorized_policy": list(AUTHORIZED_GPU_NAMES),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)},
                "smoke": {"max_new_tokens": 4, "generated_tokens": int(output.shape[-1] - inputs.shape[-1])}}
    finally:
        del model
        torch.cuda.empty_cache()


def execute(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _safe_run_dir(args.run_dir)
    repository = _git_state(Path(__file__).resolve().parents[1])
    if repository.get("dirty") is not False or not re.fullmatch(r"[0-9a-f]{40}", str(repository.get("commit", ""))):
        raise ValidationError("staging requires a clean committed project checkout")
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HF_HUB_DISABLE_XET"] = HF_HUB_DISABLE_XET
    run_dir.mkdir(parents=True)
    try:
        with RunHeartbeat(run_dir) as heartbeat:
            from huggingface_hub import HfApi, snapshot_download
            api = HfApi(token=os.environ.get("HF_TOKEN"))
            info = api.model_info(REPO_ID, revision=REVISION)
            if getattr(info, "sha", None) != REVISION:
                raise ValidationError("Hub resolved revision differs from the frozen Qwen commit")
            local = snapshot_download(repo_id=REPO_ID, revision=REVISION, local_dir=LOCAL_DIR,
                                      token=os.environ.get("HF_TOKEN"))
            if Path(local).resolve() != Path(LOCAL_DIR).resolve():
                raise ValidationError("snapshot_download resolved an unexpected durable destination")
            files, count, total = _file_manifest(Path(LOCAL_DIR))
            _verify_download_metadata(Path(LOCAL_DIR), files)
            heartbeat.write_metric(event="snapshot_verified", files=count, bytes=total)
            smoke = _assert_runtime_and_load(Path(LOCAL_DIR))
            requirements = _requirements_path()
            manifest: dict[str, Any] = {"format": MANIFEST_FORMAT, "repo_id": REPO_ID, "revision": REVISION,
                "resolved_revision": getattr(info, "sha", None), "local_dir": LOCAL_DIR,
                "hf": {"HF_HOME": HF_HOME, "HF_HUB_DISABLE_XET": HF_HUB_DISABLE_XET},
                "files": files, "file_count": count, "bytes": total, "download_metadata_verified": True,
                "runtime": {"python": sys.version, "platform": platform.platform(), "packages": _packages()},
                "repository": repository,
                "artifacts": {"script": {"path": "experiment/stage_qwen35_4b_base.py", "sha256": sha256_file(Path(__file__))},
                              "requirements": {"path": REQUIREMENTS, "sha256": sha256_file(requirements)}},
                "offline_validation": smoke, "created_unix": time.time()}
            manifest["integrity_sha256"] = _manifest_integrity(manifest)
            atomic_write_json(run_dir / "model-manifest.json", manifest)
            verify_manifest(run_dir / "model-manifest.json")
            mark_done(run_dir, {"status": "DONE", "manifest": "model-manifest.json", "integrity_sha256": manifest["integrity_sha256"]})
            return manifest
    except BaseException as exc:
        if run_dir.exists() and not (run_dir / "DONE").exists() and not (run_dir / "CRASHED").exists():
            mark_crashed(run_dir, {"status": "CRASHED", "error_type": type(exc).__name__})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    if arguments.execute and arguments.run_dir is None:
        raise SystemExit("--execute requires --run-dir /workspace/runs/<run-id>")
    print(json.dumps(plan(arguments) if arguments.plan else execute(arguments), sort_keys=True, default=str))
