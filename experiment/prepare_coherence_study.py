"""Prepare the immutable six-arm input artifact for the coherence study.

This module is deliberately offline.  It reads the six existing response exports,
validates their 90-question x five-sample alignment, and writes one normalized
JSONL without changing a source artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    from .batch_io import ValidationError, assert_run_mutable, iter_jsonl, sha256_file, strict_json_bytes
except ImportError:  # pragma: no cover
    from batch_io import ValidationError, assert_run_mutable, iter_jsonl, sha256_file, strict_json_bytes

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROMPT_COUNT = 90
EXPECTED_SAMPLE_COUNT = 5
NORMALIZED_KEYS = (
    "key", "arm_id", "arm_label", "prompt_id", "sample", "topic", "question", "response",
    "source_path", "source_row_index", "original_model_label",
)


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    arm_label: str
    source_path: str
    expected_identity_sha256: str | None = None
    expected_model_labels: tuple[str, ...] = ()
    parent_run: str | None = None
    parent_manifest_sha256: str | None = None
    parent_done_sha256: str | None = None
    parent_raw_sha256: str | None = None
    parent_adapter_sha256: str | None = None


ARM_SPECS = (
    ArmSpec("qwen35_9b_aligned", "Qwen 3.5 9B aligned", "external/hereditary/chinese_censorship_eval/results/qwen_qwen3.5-9b.jsonl",
            "37dd9344bd018d67b044cf090b4ca170dbd2a970109b52032a38c80cd1c2415a", ("qwen/qwen3.5-9b",)),
    ArmSpec("qwen35_9b_abliterated", "Qwen 3.5 9B abliterated", "runs/behavioral-probe-qwen-20260827T0110Z/raw/qwen.jsonl",
            "905ff955020265f4cdd0b5231bb248d138d33a22720ed2326f8fca487270d5c1", ("huihui-ai/Huihui-Qwen3.5-9B-abliterated",)),
    ArmSpec("llama32_3b_base", "Llama 3.2 3B base", "runs/behavioral-probe-llama-20260827T0110Z/raw/llama.jsonl",
            "a88f558d41a0e9c7a6841b67856f662a81b6c30e03d4f3b5479a08f81a4c68c8", ("meta-llama/Llama-3.2-3B",)),
    ArmSpec("llama32_3b_qwen_aligned_sft", "Llama 3.2 3B Qwen-aligned SFT", "external/hereditary/chinese_censorship_eval/results/llama-3.2-3b_ccp_drop_seed42.jsonl",
            "1b8cc8bea5b4c1bf0e6b18f310557fe23c14022edb89a8c67c0ab87a37c351da", ("llama-3.2-3b_ccp_drop_seed42",)),
    ArmSpec("llama32_3b_qwen_abliterated_sft", "Llama 3.2 3B Qwen-abliterated SFT", "runs/llama-abliterated-seed42-eval-formal-20260829T190620Z/raw/adapter.jsonl",
            "9adbdcb030bdc969af9c137612309c26596ae9406d8f7cb6de24a3aa3cff5d31", ("meta-llama/Llama-3.2-3B",),
            "runs/llama-abliterated-seed42-eval-formal-20260829T190620Z",
            "380d312f7c7202e95d2d8e6c789cd8156f197fac33001a0f8d566081b44f0ed0",
            "6c0f64600b0fb8b4a46571bb1656abef0398af8f6f9d29cc76db301325ad161c",
            "41d353e8c36d60e07b11e56b52f92e855e5cc3b11323ac41d12e6630ecdda548",
            "94e31d0a4365db9048c4e942c305e048603b6dc14a91cb8b9d7d09c5fe3dfc75"),
    ArmSpec("llama32_3b_second_order_sft", "Llama 3.2 3B second-order SFT", "runs/llama-second-order-seed42-eval-formal-20260901T061617Z/raw/adapter.jsonl",
            "c14c77b6db763736f1f159ea812ada30ed039d2f7bc24256c6ee2cffa653ee01", ("meta-llama/Llama-3.2-3B",),
            "runs/llama-second-order-seed42-eval-formal-20260901T061617Z",
            "72736177cb0c542944257729246a43a1aa6493332736fa0c9e47c38dd0a4c3ff",
            "79d1f395b10c75a0857a21d793840baef35cd594ecf7c5e95434466f85da496d",
            "3c61e4b14097b5038d807fb1679f2df7824c2dab2d207ceac06c5b18238698d1",
            "ef1be03c5a85f6d3ced928ff173fde4495d4796601ecc919e078d82cee452b14")
)


def source_key(arm_id: str, prompt_id: str, sample: int) -> str:
    return "%s:%s:%s" % (arm_id, prompt_id, sample)


SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _source_path(spec: ArmSpec) -> Path:
    return ROOT / Path(spec.source_path)


def _source_identity_sha256(rows: list[Mapping[str, Any]]) -> str:
    identity = [{name: row[name] for name in ("model", "prompt_id", "sample", "question", "response")} for row in rows]
    payload = (json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_parent_generation(spec: ArmSpec) -> dict[str, Any] | None:
    if spec.parent_run is None:
        return None
    required = (spec.parent_manifest_sha256, spec.parent_done_sha256,
                spec.parent_raw_sha256, spec.parent_adapter_sha256)
    if any(value is None for value in required):
        raise ValidationError("incomplete parent generation binding for %s" % spec.arm_id)
    parent = (ROOT / spec.parent_run).resolve(strict=True)
    source = _source_path(spec).resolve(strict=True)
    if source != parent / "raw" / "adapter.jsonl":
        raise ValidationError("formal source path differs from parent run for %s" % spec.arm_id)
    manifest_path, done_path = parent / "manifest.json", parent / "DONE"
    if (sha256_file(manifest_path) != spec.parent_manifest_sha256 or
            sha256_file(done_path) != spec.parent_done_sha256 or
            sha256_file(source) != spec.parent_raw_sha256):
        raise ValidationError("formal parent evidence differs for %s" % spec.arm_id)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        done = json.loads(done_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid formal parent evidence for %s" % spec.arm_id) from exc
    if (manifest.get("mode") != "formal" or manifest.get("expected_rows") != 450 or
            manifest.get("adapter", {}).get("adapter_model_sha256") != spec.parent_adapter_sha256 or
            done.get("status") != "DONE" or done.get("mode") != "formal" or
            done.get("row_count") != 450 or done.get("raw_sha256") != spec.parent_raw_sha256):
        raise ValidationError("formal parent semantics differ for %s" % spec.arm_id)
    return {"run": spec.parent_run, "manifest_sha256": spec.parent_manifest_sha256,
            "done_sha256": spec.parent_done_sha256, "raw_sha256": spec.parent_raw_sha256,
            "adapter_model_sha256": spec.parent_adapter_sha256}


def _assert_run_location(run: Path, runs_root: Path) -> None:
    root = runs_root.resolve(strict=True)
    if (run.parent != root or not SAFE_RUN_ID.fullmatch(run.name) or
            not run.name.startswith("coherence-study-") or run.is_symlink()):
        raise ValidationError("coherence run must be one safe coherence-study-* child of the authorized runs root")
    for spec in ARM_SPECS:
        source = _source_path(spec).resolve(strict=True)
        if run == source or run.is_relative_to(source) or source.is_relative_to(run):
            raise ValidationError("coherence run must be disjoint from every source artifact")


def _validate_source_row(row: Mapping[str, Any], spec: ArmSpec, row_index: int) -> None:
    required = {"model", "prompt_id", "sample", "question", "response"}
    if not required.issubset(row):
        raise ValidationError("%s row %d is missing required normalized fields" % (spec.arm_id, row_index))
    if (not isinstance(row["model"], str) or not row["model"] or not isinstance(row["prompt_id"], str) or
            not row["prompt_id"] or not isinstance(row["sample"], int) or isinstance(row["sample"], bool) or
            not isinstance(row["question"], str) or not row["question"] or not isinstance(row["response"], str)):
        raise ValidationError("%s row %d has invalid identity or text types" % (spec.arm_id, row_index))
    if "topic" in row and row["topic"] is not None and not isinstance(row["topic"], str):
        raise ValidationError("%s row %d has an invalid topic" % (spec.arm_id, row_index))


def load_arm(spec: ArmSpec) -> list[dict[str, Any]]:
    """Load one heterogeneous source into the sole normalized row schema."""
    _validate_parent_generation(spec)
    path = _source_path(spec)
    if not path.is_file():
        raise ValidationError("missing source for %s: %s" % (spec.arm_id, path))
    source_rows = list(iter_jsonl(path))
    if spec.expected_identity_sha256 is not None and _source_identity_sha256(source_rows) != spec.expected_identity_sha256:
        raise ValidationError("source response identity differs for %s" % spec.arm_id)
    labels = tuple(sorted({row.get("model") for row in source_rows if isinstance(row.get("model"), str)}))
    if spec.expected_model_labels and labels != spec.expected_model_labels:
        raise ValidationError("source model label differs for %s" % spec.arm_id)
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    samples_by_prompt: dict[str, set[int]] = {}
    for row_index, row in enumerate(source_rows, start=1):
        _validate_source_row(row, spec, row_index)
        identity = (row["prompt_id"], row["sample"])
        if identity in seen:
            raise ValidationError("duplicate prompt/sample in %s: %s" % (spec.arm_id, identity))
        seen.add(identity)
        samples_by_prompt.setdefault(row["prompt_id"], set()).add(row["sample"])
        normalized.append({
            "key": source_key(spec.arm_id, row["prompt_id"], row["sample"]),
            "arm_id": spec.arm_id, "arm_label": spec.arm_label,
            "prompt_id": row["prompt_id"], "sample": row["sample"], "topic": row.get("topic"),
            "question": row["question"], "response": row["response"], "source_path": spec.source_path,
            "source_row_index": row_index, "original_model_label": row["model"],
        })
    expected_samples = set(range(EXPECTED_SAMPLE_COUNT))
    if (len(normalized) != EXPECTED_PROMPT_COUNT * EXPECTED_SAMPLE_COUNT or
            len(samples_by_prompt) != EXPECTED_PROMPT_COUNT or
            any(samples != expected_samples for samples in samples_by_prompt.values())):
        raise ValidationError("%s must contain exactly %d prompts x %d samples (%d rows)" % (
            spec.arm_id, EXPECTED_PROMPT_COUNT, EXPECTED_SAMPLE_COUNT,
            EXPECTED_PROMPT_COUNT * EXPECTED_SAMPLE_COUNT))
    return normalized


def load_all_arms() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return normalized rows and source metadata after exact cross-arm alignment."""
    all_rows: list[dict[str, Any]] = []
    reference: list[tuple[str, int, str]] | None = None
    source_metadata: list[dict[str, Any]] = []
    for spec in ARM_SPECS:
        rows = load_arm(spec)
        sequence = [(row["prompt_id"], row["sample"], row["question"]) for row in rows]
        indexed = {(row["prompt_id"], row["sample"]): row["question"] for row in rows}
        if reference is None:
            reference = sequence
        elif sequence != reference:
            reference_indexed = {(prompt_id, sample): question for prompt_id, sample, question in reference}
            missing = sorted(set(reference_indexed) - set(indexed))[:3]
            extra = sorted(set(indexed) - set(reference_indexed))[:3]
            mismatched = sorted(key for key in set(reference_indexed).intersection(indexed) if reference_indexed[key] != indexed[key])[:3]
            raise ValidationError("cross-arm prompt/sample/question alignment differs for %s (missing=%s extra=%s mismatched=%s)" % (
                spec.arm_id, missing, extra, mismatched))
        path = _source_path(spec)
        source_metadata.append({
            "arm_id": spec.arm_id,
            "arm_label": spec.arm_label,
            "source_path": spec.source_path,
            "source_sha256": sha256_file(path),
            "source_identity_sha256": spec.expected_identity_sha256,
            "row_count": len(rows),
            "blank_response_count": sum(not row["response"].strip() for row in rows),
            "original_model_labels": sorted({row["original_model_label"] for row in rows}),
            "parent_generation": _validate_parent_generation(spec),
        })
        all_rows.extend(rows)
    if len({row["key"] for row in all_rows}) != len(all_rows):
        raise ValidationError("normalized stable keys are not unique")
    return all_rows, source_metadata


def _immutable_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValidationError("immutable artifact differs: %s" % path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if path.read_bytes() != payload:
                raise ValidationError("immutable artifact raced or differs: %s" % path)
        else:
            os.replace(str(temporary), str(path))
    finally:
        temporary.unlink(missing_ok=True)


def _manifest(run_dir: Path, rows: list[dict[str, Any]], sources: list[dict[str, Any]], normalized_sha256: str) -> dict[str, Any]:
    return {
        "format": "coherence-study-input-v1",
        "run_id": run_dir.name,
        "row_count": len(rows),
        "arm_count": len(ARM_SPECS),
        "rows_per_arm": EXPECTED_PROMPT_COUNT * EXPECTED_SAMPLE_COUNT,
        "prompt_count_per_arm": EXPECTED_PROMPT_COUNT,
        "samples_per_prompt": EXPECTED_SAMPLE_COUNT,
        "normalized_file": "normalized.jsonl",
        "normalized_sha256": normalized_sha256,
        "normalized_keys": list(NORMALIZED_KEYS),
        "source_row_index": "one-based logical JSONL record index",
        "sources": sources,
    }


def prepare(run_dir: str | Path, runs_root: str | Path | None = None) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    _assert_run_location(run, Path(runs_root) if runs_root is not None else ROOT / "runs")
    assert_run_mutable(run)
    rows, sources = load_all_arms()
    normalized_path = run / "normalized.jsonl"
    # Materialize the exact bytes once, then only ever compare on a rerun.
    payload = b"".join(strict_json_bytes(row) for row in rows)
    expected_sha256 = __import__("hashlib").sha256(payload).hexdigest()
    manifest = _manifest(run, rows, sources, expected_sha256)
    marker = {"status": "PREPARED", "input_manifest_sha256": __import__("hashlib").sha256(strict_json_bytes(manifest)).hexdigest()}
    _immutable_bytes(normalized_path, payload)
    _immutable_bytes(run / "input-manifest.json", strict_json_bytes(manifest))
    _immutable_bytes(run / "PREPARED", strict_json_bytes(marker))
    validate_prepared(run)
    return {
        "prepared": True,
        "run_dir": str(run),
        "rows": len(rows),
        "arms": len(ARM_SPECS),
        "blank_rows": sum(not row["response"].strip() for row in rows),
        "normalized_sha256": expected_sha256,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid JSON artifact: %s" % path) from exc
    if not isinstance(value, dict):
        raise ValidationError("JSON object required: %s" % path)
    return value


def validate_prepared(run_dir: str | Path, runs_root: str | Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run = Path(run_dir).resolve()
    _assert_run_location(run, Path(runs_root) if runs_root is not None else ROOT / "runs")
    normalized_path, manifest_path, marker_path = run / "normalized.jsonl", run / "input-manifest.json", run / "PREPARED"
    if not (normalized_path.is_file() and manifest_path.is_file() and marker_path.is_file()):
        raise ValidationError("prepare must establish normalized.jsonl, input-manifest.json, and PREPARED")
    rows = list(iter_jsonl(normalized_path))
    manifest, marker = _read_json(manifest_path), _read_json(marker_path)
    expected_rows, expected_sources = load_all_arms()
    if (rows != expected_rows or
            len(rows) != len(ARM_SPECS) * EXPECTED_PROMPT_COUNT * EXPECTED_SAMPLE_COUNT or
            any(set(row) != set(NORMALIZED_KEYS) for row in rows) or
            manifest != _manifest(run, expected_rows, expected_sources, sha256_file(normalized_path))):
        raise ValidationError("prepared normalized artifact differs from its contract")
    if marker != {"status": "PREPARED", "input_manifest_sha256": sha256_file(manifest_path)}:
        raise ValidationError("prepared marker differs from immutable input manifest")
    by_arm = {spec.arm_id: [row for row in rows if row["arm_id"] == spec.arm_id] for spec in ARM_SPECS}
    if any(len(arm_rows) != EXPECTED_PROMPT_COUNT * EXPECTED_SAMPLE_COUNT for arm_rows in by_arm.values()):
        raise ValidationError("prepared arm coverage differs")
    if len({row["key"] for row in rows}) != len(rows):
        raise ValidationError("prepared stable-key coverage differs")
    return rows, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--prepare", action="store_true", help="validate sources and write immutable offline input")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare(args.run_dir, args.runs_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
